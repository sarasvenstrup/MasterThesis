import os
import sys
import hashlib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

try:
    REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
except NameError:
    REPO_ROOT = os.getcwd()

PROJECT_ROOT = os.path.abspath(os.path.join(REPO_ROOT, ".."))
THESIS_ROOT = os.path.abspath(os.path.join(REPO_ROOT, "..", ".."))

if THESIS_ROOT not in sys.path:
    sys.path.insert(0, THESIS_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Code import config
from Code.load_swapdata import my_data
from Code.model.full_model import FullModel
from Code.model.sigma_matrix import L_from_sigmas_rhos

print(f"Repo root: {REPO_ROOT}")
print(f"Active model variant from config.py: {config.VARIANT}")

# ==========================================================
# User settings
# ==========================================================
USE = "bbg"
LATENT_DIM = 2
CCY_FILTER = "EUR"   # "" for all currencies
BATCH_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAINING_ROOT = os.path.join(
    THESIS_ROOT,
    "Figures",
    "TrainingResults",
    f"dim{LATENT_DIM}_{config.VARIANT}",
)

OLD_RUN_DIR = os.path.join(TRAINING_ROOT, "ep200")
NEW_RUN_DIR = os.path.join(TRAINING_ROOT, "pricing_ep200")


# ==========================================================
# Helpers
# ==========================================================
def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_checkpoint_from_run_dir(run_dir: str, latent_dim: int):
    candidates = [
        os.path.join(run_dir, "full_checkpoint.pt"),
        os.path.join(run_dir, f"best_checkpoint_dim{latent_dim}.pt"),
        os.path.join(run_dir, f"checkpoint_dim{latent_dim}.pt"),
        os.path.join(run_dir, f"checkpoint_dim{latent_dim}_ep200.pt"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    searched = "\n".join(f"  - {p}" for p in candidates)
    raise FileNotFoundError(f"No checkpoint found in run dir:\n{run_dir}\nSearched:\n{searched}")


def describe_file(label: str, path: str):
    print(f"\n{label}")
    print("  path  :", path)
    print("  exists:", os.path.exists(path))
    if os.path.exists(path):
        print("  size  :", os.path.getsize(path))
        print("  sha256:", sha256(path))


def filter_dataset_by_currency(meta, X_tensor, ccy_filter: str):
    if not ccy_filter or str(ccy_filter).strip() == "":
        return meta.reset_index(drop=True), X_tensor

    ccy_filter = str(ccy_filter).strip().upper()
    mask = meta["ccy"].astype(str).str.upper() == ccy_filter
    n_keep = int(mask.sum())

    if n_keep == 0:
        available = sorted(meta["ccy"].astype(str).str.upper().unique().tolist())
        raise ValueError(
            f"No rows found for ccy_filter='{ccy_filter}'. "
            f"Available currencies: {available}"
        )

    meta_f = meta.loc[mask].reset_index(drop=True)
    X_tensor_f = X_tensor[mask.to_numpy()]
    print(f"Filtered dataset to currency {ccy_filter}: kept {n_keep} rows")
    return meta_f, X_tensor_f


def load_checkpoint_payload(checkpoint_path: str, device):
    raw = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(raw, dict) and "model_state_dict" in raw:
        state_dict = raw["model_state_dict"]
        model_config = raw.get("model_config", {})
        saved_variant = raw.get("variant", "unknown")
    else:
        state_dict = raw
        model_config = {}
        saved_variant = "unknown"

    if saved_variant != "unknown" and saved_variant != config.VARIANT:
        raise ValueError(
            f"Checkpoint variant '{saved_variant}' does not match active "
            f"config.VARIANT '{config.VARIANT}'."
        )

    return raw, state_dict, model_config


def sanitize_model_config(model_config: dict, latent_dim_default: int):
    allowed_keys = {
        "input_dim", "latent_dim", "tau_max", "tenors",
        "g_hidden", "h_hidden", "r_hidden",
        "g_bias", "hr_bias", "sigma_init",
        "k_z_center_init", "k_epsilon", "k_drift_scale_init", "k_learn_center",
    }

    cfg = {"latent_dim": latent_dim_default}
    if isinstance(model_config, dict):
        for k, v in model_config.items():
            if k in allowed_keys:
                cfg[k] = v
    return cfg


def build_model_from_checkpoint(checkpoint_path: str, device, latent_dim_default: int):
    raw, state_dict, model_config = load_checkpoint_payload(checkpoint_path, device)
    model_kwargs = sanitize_model_config(model_config, latent_dim_default)

    model = FullModel(**model_kwargs)
    incompat = model.load_state_dict(state_dict, strict=False)

    missing = list(incompat.missing_keys)
    unexpected = list(incompat.unexpected_keys)

    print(f"\nLoaded model from: {checkpoint_path}")
    print("  model kwargs   :", model_kwargs)
    print("  missing keys   :", missing)
    print("  unexpected keys:", unexpected)

    if missing or unexpected:
        print("  [WARN] Partial load detected.")

    model.to(device).double()
    model.eval()
    return model, state_dict


@torch.no_grad()
def get_H_outputs(model, z):
    H_out = model.H(z)

    if isinstance(H_out, tuple) and len(H_out) == 2:
        sigmas, rhos = H_out
        L = L_from_sigmas_rhos(sigmas, rhos)
        return sigmas, rhos, L

    if torch.is_tensor(H_out) and H_out.ndim == 3:
        L = H_out
        sigmas = torch.diagonal(L, dim1=1, dim2=2)
        return sigmas, None, L

    raise TypeError(
        "Unsupported model.H(z) output. Expected (sigmas, rhos) or tensor (B,d,d)."
    )


@torch.no_grad()
def get_r(model, z):
    r = model.R(z)
    if r.ndim == 2 and r.shape[-1] == 1:
        r = r.squeeze(-1)
    return r


def tensor_summary(name, x):
    x = x.detach().cpu().double().reshape(-1)
    q = torch.quantile(
        x,
        torch.tensor([0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0], dtype=x.dtype)
    )
    print(f"\n{name}")
    print(f"  mean   = {x.mean().item():.6e}")
    print(f"  std    = {x.std().item():.6e}")
    print(f"  min    = {q[0].item():.6e}")
    print(f"  q01    = {q[1].item():.6e}")
    print(f"  q05    = {q[2].item():.6e}")
    print(f"  median = {q[3].item():.6e}")
    print(f"  q95    = {q[4].item():.6e}")
    print(f"  q99    = {q[5].item():.6e}")
    print(f"  max    = {q[6].item():.6e}")


def rownorm_summary(name, x):
    vals = torch.linalg.norm(x, dim=1)
    tensor_summary(name, vals)


def get_kappa(model):
    if hasattr(model.K, "raw_kappa"):
        return float(F.softplus(model.K.raw_kappa.detach()).cpu().item())
    return np.nan


def get_theta(model):
    if hasattr(model.K, "theta"):
        return model.K.theta.detach().cpu().numpy()
    return None


def compare_state_dicts(sd_old: dict, sd_new: dict):
    print("\n" + "=" * 72)
    print("STATE_DICT COMPARISON")
    print("=" * 72)

    old_keys = set(sd_old.keys())
    new_keys = set(sd_new.keys())

    only_old = sorted(old_keys - new_keys)
    only_new = sorted(new_keys - old_keys)
    common = sorted(old_keys & new_keys)

    print("Keys only in old:", only_old)
    print("Keys only in new:", only_new)
    print("Common keys     :", len(common))

    rows = []
    all_exact = True

    for k in common:
        a = sd_old[k].detach().cpu().double()
        b = sd_new[k].detach().cpu().double()

        if a.shape != b.shape:
            rows.append({
                "name": k,
                "shape_old": tuple(a.shape),
                "shape_new": tuple(b.shape),
                "exact_equal": False,
                "max_abs_diff": np.nan,
                "mean_abs_diff": np.nan,
            })
            all_exact = False
            continue

        diff = (b - a).abs()
        exact = bool(torch.equal(a, b))
        if not exact:
            all_exact = False

        rows.append({
            "name": k,
            "shape_old": tuple(a.shape),
            "shape_new": tuple(b.shape),
            "exact_equal": exact,
            "max_abs_diff": float(diff.max().item()) if diff.numel() > 0 else 0.0,
            "mean_abs_diff": float(diff.mean().item()) if diff.numel() > 0 else 0.0,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        print("\nTop differing tensors:")
        print(
            df.sort_values(["exact_equal", "max_abs_diff"], ascending=[True, False])
              .head(20)
              .to_string(index=False)
        )

    print("\nAll common tensors exactly equal:", all_exact)
    return df


@torch.no_grad()
def compare_models_on_data(model_old, model_new, X_tensor, batch_size=256):
    z_old_list, z_new_list = [], []
    mu_old_list, mu_new_list = [], []
    r_old_list, r_new_list = [], []
    sig_old_list, sig_new_list = [], []
    L_old_list, L_new_list = [], []

    n = X_tensor.shape[0]
    for i in range(0, n, batch_size):
        xb = X_tensor[i:i + batch_size].to(DEVICE)

        z_old = model_old.encoder(xb)
        z_new = model_new.encoder(xb)

        mu_old = model_old.K(z_old)
        mu_new = model_new.K(z_new)

        sig_old, _, L_old = get_H_outputs(model_old, z_old)
        sig_new, _, L_new = get_H_outputs(model_new, z_new)

        r_old = get_r(model_old, z_old)
        r_new = get_r(model_new, z_new)

        z_old_list.append(z_old.cpu())
        z_new_list.append(z_new.cpu())
        mu_old_list.append(mu_old.cpu())
        mu_new_list.append(mu_new.cpu())
        sig_old_list.append(sig_old.cpu())
        sig_new_list.append(sig_new.cpu())
        L_old_list.append(L_old.cpu())
        L_new_list.append(L_new.cpu())
        r_old_list.append(r_old.cpu())
        r_new_list.append(r_new.cpu())

    z_old = torch.cat(z_old_list, dim=0)
    z_new = torch.cat(z_new_list, dim=0)
    mu_old = torch.cat(mu_old_list, dim=0)
    mu_new = torch.cat(mu_new_list, dim=0)
    sig_old = torch.cat(sig_old_list, dim=0)
    sig_new = torch.cat(sig_new_list, dim=0)
    L_old = torch.cat(L_old_list, dim=0)
    L_new = torch.cat(L_new_list, dim=0)
    r_old = torch.cat(r_old_list, dim=0)
    r_new = torch.cat(r_new_list, dim=0)

    dz = z_new - z_old
    dmu = mu_new - mu_old
    dsig = sig_new - sig_old
    dL = L_new - L_old
    dr = r_new - r_old

    print("\n" + "=" * 72)
    print("ENCODER COMPARISON")
    print("=" * 72)
    rownorm_summary("||z_new - z_old|| per row", dz)

    print("\n" + "=" * 72)
    print("DRIFT COMPARISON")
    print("=" * 72)
    rownorm_summary("||mu_new - mu_old|| per row", dmu)
    for d in range(mu_old.shape[1]):
        tensor_summary(f"delta mu[:, {d}]", dmu[:, d])

    print("\n" + "=" * 72)
    print("SHORT RATE COMPARISON")
    print("=" * 72)
    tensor_summary("delta r", dr)

    print("\n" + "=" * 72)
    print("DIFFUSION COMPARISON")
    print("=" * 72)
    rownorm_summary("||sig_new - sig_old|| per row", dsig)
    rownorm_summary("||vec(L_new - L_old)|| per row", dL.reshape(dL.shape[0], -1))
    for d in range(sig_old.shape[1]):
        tensor_summary(f"delta sigma[:, {d}]", dsig[:, d])

    print("\n" + "=" * 72)
    print("STABLE-K PARAMETER COMPARISON")
    print("=" * 72)
    theta_old = get_theta(model_old)
    theta_new = get_theta(model_new)
    kappa_old = get_kappa(model_old)
    kappa_new = get_kappa(model_new)

    print("theta_old:", theta_old)
    print("theta_new:", theta_new)
    if theta_old is not None and theta_new is not None:
        print("delta theta:", theta_new - theta_old)

    print("kappa_old:", kappa_old)
    print("kappa_new:", kappa_new)
    print("delta kappa:", kappa_new - kappa_old)

    print("\n" + "=" * 72)
    print("ALIGNMENT CHECKS")
    print("=" * 72)
    for d in range(mu_old.shape[1]):
        x = mu_old[:, d].numpy()
        y = mu_new[:, d].numpy()
        corr = np.corrcoef(x, y)[0, 1]
        print(f"corr(mu_old[:, {d}], mu_new[:, {d}]) = {corr:.6f}")

    corr_r = np.corrcoef(r_old.numpy(), r_new.numpy())[0, 1]
    print(f"corr(r_old, r_new) = {corr_r:.6f}")

    summary = {
        "n_obs": int(X_tensor.shape[0]),
        "mean_norm_dz": float(torch.linalg.norm(dz, dim=1).mean().item()),
        "max_norm_dz": float(torch.linalg.norm(dz, dim=1).max().item()),
        "mean_norm_dmu": float(torch.linalg.norm(dmu, dim=1).mean().item()),
        "max_norm_dmu": float(torch.linalg.norm(dmu, dim=1).max().item()),
        "mean_abs_dr": float(dr.abs().mean().item()),
        "max_abs_dr": float(dr.abs().max().item()),
        "mean_norm_dsigma": float(torch.linalg.norm(dsig, dim=1).mean().item()),
        "max_norm_dsigma": float(torch.linalg.norm(dsig, dim=1).max().item()),
        "mean_norm_dL": float(torch.linalg.norm(dL.reshape(dL.shape[0], -1), dim=1).mean().item()),
        "max_norm_dL": float(torch.linalg.norm(dL.reshape(dL.shape[0], -1), dim=1).max().item()),
        "kappa_old": kappa_old,
        "kappa_new": kappa_new,
    }

    print("\n" + "=" * 72)
    print("COMPACT SUMMARY")
    print("=" * 72)
    print(pd.DataFrame([summary]).to_string(index=False))

    return pd.DataFrame([summary])


# ==========================================================
# Run
# ==========================================================
old_ckpt = resolve_checkpoint_from_run_dir(OLD_RUN_DIR, LATENT_DIM)
new_ckpt = resolve_checkpoint_from_run_dir(NEW_RUN_DIR, LATENT_DIM)

describe_file("OLD CHECKPOINT", old_ckpt)
describe_file("NEW CHECKPOINT", new_ckpt)

raw_old, sd_old, _ = load_checkpoint_payload(old_ckpt, DEVICE)
raw_new, sd_new, _ = load_checkpoint_payload(new_ckpt, DEVICE)

state_df = compare_state_dicts(sd_old, sd_new)

meta, X_tensor, meta_full, X_tensor_full, tenors, df_wide, df_wide_all, SCALE_IS_PERCENT = my_data(use=USE)
X_tensor = X_tensor.double()
meta, X_tensor = filter_dataset_by_currency(meta, X_tensor, CCY_FILTER)

model_old, _ = build_model_from_checkpoint(old_ckpt, DEVICE, latent_dim_default=LATENT_DIM)
model_new, _ = build_model_from_checkpoint(new_ckpt, DEVICE, latent_dim_default=LATENT_DIM)

summary_df = compare_models_on_data(
    model_old=model_old,
    model_new=model_new,
    X_tensor=X_tensor,
    batch_size=BATCH_SIZE,
)

# Optional: save summaries next to new run
state_csv = os.path.join(NEW_RUN_DIR, "compare_state_dicts.csv")
summary_csv = os.path.join(NEW_RUN_DIR, "compare_summary.csv")

state_df.to_csv(state_csv, index=False)
summary_df.to_csv(summary_csv, index=False)

print("\nSaved:")
print(" ", state_csv)
print(" ", summary_csv)