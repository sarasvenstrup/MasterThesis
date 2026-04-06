# ============================= Import Packages ===============================
import math
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(4)
torch.set_num_interop_threads(2)

from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR

# ============================= Environment Setup & Imports ===============================
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()

CODE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))          # .../MasterThesis/Code
THESIS_ROOT = os.path.abspath(os.path.join(CODE_ROOT, ".."))         # .../MasterThesis

if THESIS_ROOT not in sys.path:
    sys.path.insert(0, THESIS_ROOT)
if CODE_ROOT not in sys.path:
    sys.path.insert(0, CODE_ROOT)

from Code import config
config.confirm_variant()

from Code.utils import helpers as H
from Code.load_swapdata import my_data
from Code.model.full_model import FullModel
from Code.model.sigma_matrix import L_from_sigmas_rhos

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("MPS available:", hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
print("Active model variant from config.py:", config.VARIANT)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Using device:", device)

torch.backends.mkldnn.enabled = True
USE_SET_TO_NONE = True
print("CPU threads:", torch.get_num_threads(), "interop:", torch.get_num_interop_threads())

# ==========================================================
# Settings
# ==========================================================
SHOW_PLOTS = True

LATENT_DIM = 2
BASE_EPOCHS = 200                    # checkpoint to warm-start from
CONT_EPOCHS = 200                    # continuation epochs
BATCH_SIZE = 32
EVAL_BATCH_SIZE = 256

USE = "bbg"
CCY_FILTER = "EUR"                   # pricing continuation should be currency-specific

# ---------- output folders ----------
TRAINING_ROOT = os.path.join(
    THESIS_ROOT,
    "Figures",
    "TrainingResults",
    f"dim{LATENT_DIM}_{config.VARIANT}",
)

RUN_NAME = f"pricing_dyn_ep{BASE_EPOCHS}"
RUN_DIR = os.path.join(TRAINING_ROOT, RUN_NAME)
os.makedirs(RUN_DIR, exist_ok=True)

print("Training root:", TRAINING_ROOT)
print("Run dir      :", RUN_DIR)

# ---------- model hyperparameters ----------
SIGMA_INIT = 0.015
K_DRIFT_SCALE_INIT = 0.10

# ---------- optimizer ----------
MAX_LR_KHR = 2e-5
MAX_LR_G = 2e-6   # smaller LR for decoder G when unfrozen

# ---------- warm-start / freezing ----------
USE_FIXED_CENTER = True
WARMSTART_FROM_PREVIOUS = True
FREEZE_ENCODER = True
FREEZE_DECODER_G = False

MANUAL_CENTER = np.array([0.0, 0.0], dtype=np.float32)

# ---------- loss settings ----------
# Curve MSE is in decimal rates and very small, so we rescale it to a more useful magnitude.
CURVE_LOSS_SCALE = 1e6
LAMBDA_CURVE = 2.0
LAMBDA_TRANS = 20.0
LAMBDA_CLOUD = 10.0

# ---------- transition / rollout ----------
DT_MIN = 1.0 / 365.25
DT_MAX = 40.0 / 365.25

ROLLOUT_STEPS = 4
CLOUD_THRESHOLD = 3.5
CLOUD_JITTER = 1e-8

# ---------- eval / logging ----------
VAL_FRAC = 0.15
EVAL_EVERY = 1
LOG_EVERY = 1

# ==========================================================
# General helpers
# ==========================================================
def freeze_module(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False


def summarize_load_result(tag: str, missing, unexpected):
    print(f"\n{tag}")
    print("  Missing keys   :", list(missing))
    print("  Unexpected keys:", list(unexpected))
    if len(missing) > 0 or len(unexpected) > 0:
        print("  [WARN] Partial warm-start is only OK if intentional.")


def print_trainable_parameters(model: nn.Module):
    print("\nTrainable parameters:")
    total = 0
    trainable = 0
    for name, p in model.named_parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
            print(f"  {name:40s} {tuple(p.shape)}")
    print(f"Trainable params: {trainable:,} / {total:,}\n")


def get_kappa_numpy(K_module):
    if hasattr(K_module, "raw_kappa"):
        return F.softplus(K_module.raw_kappa.detach()).cpu().numpy()
    return None


def row_finite_mask(t: torch.Tensor) -> torch.Tensor:
    return torch.isfinite(t).all(dim=1)


def resolve_base_checkpoint_path(training_root: str, latent_dim: int, base_epochs: int) -> str:
    base_dir = os.path.join(training_root, f"ep{base_epochs}")

    candidates = [
        os.path.join(base_dir, "full_checkpoint.pt"),
        os.path.join(base_dir, f"checkpoint_dim{latent_dim}_ep{base_epochs}.pt"),
        os.path.join(base_dir, f"checkpoint_dim{latent_dim}.pt"),
        os.path.join(base_dir, f"best_checkpoint_dim{latent_dim}.pt"),
    ]

    for path in candidates:
        if os.path.exists(path):
            return os.path.abspath(path)

    searched = "\n".join(f"  - {os.path.abspath(p)}" for p in candidates)
    raise FileNotFoundError(f"Base checkpoint not found. Searched:\n{searched}")


def load_state_dict_from_checkpoint(checkpoint_path: str, map_location):
    raw = torch.load(checkpoint_path, map_location=map_location, weights_only=False)

    if isinstance(raw, dict) and "model_state_dict" in raw:
        state_dict = raw["model_state_dict"]
        saved_variant = raw.get("variant", "unknown")
        if saved_variant != "unknown" and saved_variant != config.VARIANT:
            raise ValueError(
                f"Checkpoint variant '{saved_variant}' does not match active config.VARIANT '{config.VARIANT}'."
            )
    else:
        state_dict = raw

    return raw, state_dict


def get_L(model, z):
    H_out = model.H(z)

    if isinstance(H_out, tuple) and len(H_out) == 2:
        sigmas, rhos = H_out
        return L_from_sigmas_rhos(sigmas, rhos)

    if torch.is_tensor(H_out) and H_out.ndim == 3:
        return H_out

    raise TypeError(
        "Unsupported model.H(z) output. Expected either (sigmas, rhos) or tensor L with shape (B,d,d)."
    )


@torch.no_grad()
def estimate_latent_center(model: nn.Module, X: torch.Tensor, batch_size: int = 256):
    was_training = model.training
    model.eval()

    zs = []
    for i in range(0, X.shape[0], batch_size):
        xb = X[i:i + batch_size].to(device)
        z = model.encoder(xb)
        zs.append(z.detach().cpu())

    z_all = torch.cat(zs, dim=0)
    z_mean = z_all.mean(dim=0)
    z_std = z_all.std(dim=0)

    if was_training:
        model.train()

    return z_mean, z_std


@torch.no_grad()
def estimate_latent_support_stats(model: nn.Module, X: torch.Tensor, batch_size: int = 256):
    was_training = model.training
    model.eval()

    zs = []
    for i in range(0, X.shape[0], batch_size):
        xb = X[i:i + batch_size].to(device)
        z = model.encoder(xb)
        zs.append(z.detach().cpu())

    z_all = torch.cat(zs, dim=0).to(torch.float64)
    z_mean = z_all.mean(dim=0)
    z_cov = torch.cov(z_all.T)

    eye = torch.eye(z_cov.shape[0], dtype=z_cov.dtype)
    z_cov_inv = torch.linalg.inv(z_cov + CLOUD_JITTER * eye)

    if was_training:
        model.train()

    return z_mean, z_cov, z_cov_inv


@torch.no_grad()
def predict_S_hat(model: nn.Module, X_full: torch.Tensor, batch_size: int = 256) -> torch.Tensor:
    was_training = model.training
    model.eval()

    outs = []
    N = X_full.shape[0]
    for i in range(0, N, batch_size):
        xb = X_full[i:i + batch_size].to(device)
        S_hat = model(xb)
        outs.append(S_hat.detach().cpu())

    if was_training:
        model.train()

    return torch.cat(outs, dim=0)


def eval_rmse_bps(model: nn.Module, X_full: torch.Tensor, meta_full: pd.DataFrame, batch_size: int = 256):
    S_hat_all = predict_S_hat(model, X_full, batch_size=batch_size)

    mask = row_finite_mask(X_full) & row_finite_mask(S_hat_all)
    n_bad = int((~mask).sum().item())
    n_good = int(mask.sum().item())

    X_eval = X_full[mask]
    S_eval = S_hat_all[mask]
    meta_eval = meta_full.loc[mask.numpy()].reset_index(drop=True)

    rmse_per_ccy = H.rmse_bps_per_currency_paper(X_eval, S_eval, meta_eval)
    avg_rmse_bps = float(rmse_per_ccy.mean())

    return rmse_per_ccy, avg_rmse_bps, n_bad, n_good


@torch.no_grad()
def print_stable_diagnostics(model: nn.Module, X: torch.Tensor, batch_size: int = 128):
    was_training = model.training
    model.eval()

    xb = X[:batch_size].to(device)
    _, aux = model(xb, return_aux=True)

    z = aux["z"]
    sigma = aux["sigma"]
    diag = torch.diagonal(sigma, dim1=1, dim2=2)

    print("  z mean:", z.mean(dim=0).detach().cpu().numpy())
    print("  z std :", z.std(dim=0).detach().cpu().numpy())

    kappa_np = get_kappa_numpy(model.K)
    if kappa_np is not None:
        print("  kappa:", kappa_np)

    if hasattr(model.K, "theta"):
        try:
            print("  theta:", model.K.theta.detach().cpu().numpy())
        except Exception:
            pass

    print("  sigma diag mean:", diag.mean(dim=0).detach().cpu().numpy())

    if was_training:
        model.train()


@torch.no_grad()
def print_arb_diagnostics(model: nn.Module, X: torch.Tensor, batch_size: int = 16):
    was_training = model.training
    model.eval()

    xb = X[:batch_size].to(device)
    _, aux = model(xb, return_aux=True, do_arb_checks=True)
    arb = aux["arb"]

    if arb is not None:
        print("  max_abs_R mean :", arb["max_abs_R"].mean().item())
        print("  max_abs_SR mean:", arb["max_abs_SR_1to30"].mean().item())

    if was_training:
        model.train()

# ==========================================================
# Build transition pairs from dated data
# ==========================================================
@dataclass
class PairData:
    X_t: torch.Tensor
    X_tp1: torch.Tensor
    dt: torch.Tensor
    meta_pairs: pd.DataFrame


def build_transition_pairs(df_wide: pd.DataFrame, tenors, scale_is_percent: bool, ccy_filter: str | None = None) -> PairData:
    df = df_wide.copy()
    df["as_of_date"] = pd.to_datetime(df["as_of_date"])

    if ccy_filter is not None and str(ccy_filter).strip():
        c = str(ccy_filter).strip().upper()
        df = df[df["ccy"].astype(str).str.upper() == c].copy()

    df = df.sort_values(["ccy", "as_of_date"]).reset_index(drop=True)

    tenor_cols = [int(t) for t in tenors]

    X_t_list = []
    X_tp1_list = []
    dt_list = []
    rows = []

    for ccy, g in df.groupby("ccy", sort=True):
        g = g.sort_values("as_of_date").reset_index(drop=True)

        Xg = g[tenor_cols].to_numpy(dtype=np.float64)
        if scale_is_percent:
            Xg = Xg / 100.0

        dates = pd.to_datetime(g["as_of_date"]).to_numpy()

        for i in range(len(g) - 1):
            dt_days = int((dates[i + 1] - dates[i]) / np.timedelta64(1, "D"))
            if dt_days <= 0:
                continue

            dt_years = dt_days / 365.25
            if not np.isfinite(dt_years):
                continue

            X_t_list.append(Xg[i])
            X_tp1_list.append(Xg[i + 1])
            dt_list.append(dt_years)

            rows.append(
                {
                    "ccy": ccy,
                    "date_t": pd.Timestamp(g.loc[i, "as_of_date"]),
                    "date_tp1": pd.Timestamp(g.loc[i + 1, "as_of_date"]),
                    "dt_years": dt_years,
                }
            )

    if len(X_t_list) == 0:
        raise ValueError(f"No transition pairs found. ccy_filter={ccy_filter}")

    X_t = torch.tensor(np.asarray(X_t_list), dtype=torch.float64)
    X_tp1 = torch.tensor(np.asarray(X_tp1_list), dtype=torch.float64)
    dt = torch.tensor(np.asarray(dt_list), dtype=torch.float64).unsqueeze(1)
    meta_pairs = pd.DataFrame(rows)

    dt = torch.clamp(dt, min=DT_MIN, max=DT_MAX)

    return PairData(X_t=X_t, X_tp1=X_tp1, dt=dt, meta_pairs=meta_pairs)


def split_pairs_timewise(pair_data: PairData, val_frac: float = 0.15):
    meta = pair_data.meta_pairs.copy()

    train_idx = []
    val_idx = []

    for ccy, g in meta.groupby("ccy", sort=True):
        g = g.sort_values("date_t").reset_index()
        n = len(g)
        n_val = max(1, int(math.ceil(val_frac * n))) if n >= 5 else max(1, n // 5)
        n_val = min(n_val, n - 1) if n > 1 else 0

        val_rows = g.iloc[-n_val:]["index"].tolist() if n_val > 0 else []
        train_rows = g.iloc[:-n_val]["index"].tolist() if n_val > 0 else g["index"].tolist()

        train_idx.extend(train_rows)
        val_idx.extend(val_rows)

    train_idx = np.array(sorted(train_idx), dtype=int)
    val_idx = np.array(sorted(val_idx), dtype=int)

    def take_idx(t, idx):
        return t[idx] if len(idx) > 0 else t[:0]

    train = PairData(
        X_t=take_idx(pair_data.X_t, train_idx),
        X_tp1=take_idx(pair_data.X_tp1, train_idx),
        dt=take_idx(pair_data.dt, train_idx),
        meta_pairs=pair_data.meta_pairs.iloc[train_idx].reset_index(drop=True),
    )

    val = PairData(
        X_t=take_idx(pair_data.X_t, val_idx),
        X_tp1=take_idx(pair_data.X_tp1, val_idx),
        dt=take_idx(pair_data.dt, val_idx),
        meta_pairs=pair_data.meta_pairs.iloc[val_idx].reset_index(drop=True),
    )

    return train, val

# ==========================================================
# Transition / rollout losses
# ==========================================================
def mahal_sq(z: torch.Tensor, z_mean: torch.Tensor, z_cov_inv: torch.Tensor) -> torch.Tensor:
    dz = z - z_mean.unsqueeze(0)
    return torch.sum((dz @ z_cov_inv) * dz, dim=1)


def transition_nll_loss(model: nn.Module, z_t: torch.Tensor, z_tp1: torch.Tensor, dt: torch.Tensor):
    """
    Euler-Gaussian one-step transition loss:
        z_{t+1} ~ N(z_t + mu(z_t) dt, L(z_t)L(z_t)^T dt)
    """
    mu = model.K(z_t)                                      # (B,d)
    L = get_L(model, z_t)                                  # (B,d,d)

    B, d = z_t.shape
    I = torch.eye(d, device=z_t.device, dtype=z_t.dtype).unsqueeze(0)

    Sigma = L @ L.transpose(1, 2)
    Sigma = Sigma * dt.view(-1, 1, 1) + CLOUD_JITTER * I

    resid = (z_tp1 - z_t - mu * dt).unsqueeze(-1)          # (B,d,1)

    solved = torch.linalg.solve(Sigma, resid)              # (B,d,1)
    quad = torch.matmul(resid.transpose(1, 2), solved).squeeze(-1).squeeze(-1)   # (B,)
    logdet = torch.logdet(Sigma)                           # (B,)

    nll = 0.5 * (quad + logdet)
    loss = nll.mean()

    stats = {
        "quad_mean": quad.mean().detach().item(),
        "logdet_mean": logdet.mean().detach().item(),
        "nll_mean": loss.detach().item(),
    }
    return loss, stats


def rollout_cloud_penalty(
    model: nn.Module,
    z_start: torch.Tensor,
    dt: torch.Tensor,
    z_mean: torch.Tensor,
    z_cov_inv: torch.Tensor,
    n_steps: int = 4,
    threshold: float = 3.5,
):
    """
    Short rollout under Euler simulation, penalizing states that leave the empirical latent cloud.
    """
    z = z_start
    total = 0.0

    for _ in range(n_steps):
        mu = model.K(z)
        L = get_L(model, z)

        eps = torch.randn(z.shape[0], z.shape[1], device=z.device, dtype=z.dtype)
        shock = torch.matmul(L, eps.unsqueeze(-1)).squeeze(-1) * torch.sqrt(dt)

        z = z + mu * dt + shock

        d_mahal = torch.sqrt(torch.clamp(mahal_sq(z, z_mean, z_cov_inv), min=0.0))
        total = total + F.relu(d_mahal - threshold).pow(2).mean()

    return total / float(n_steps)


@torch.no_grad()
def evaluate_on_transition_loader(
    model: nn.Module,
    loader: DataLoader,
    z_mean: torch.Tensor,
    z_cov_inv: torch.Tensor,
):
    was_training = model.training
    model.eval()

    sum_curve = 0.0
    sum_trans = 0.0
    sum_cloud = 0.0
    sum_total = 0.0
    sum_exceed = 0.0
    n_obs = 0

    for xb_t, xb_tp1, dt in loader:
        xb_t = xb_t.to(device)
        xb_tp1 = xb_tp1.to(device)
        dt = dt.to(device)

        S_hat = model(xb_t)
        loss_curve = F.mse_loss(S_hat, xb_t) * CURVE_LOSS_SCALE

        z_t = model.encoder(xb_t).detach()
        z_tp1 = model.encoder(xb_tp1).detach()

        loss_trans, _ = transition_nll_loss(model, z_t, z_tp1, dt)
        loss_cloud = rollout_cloud_penalty(
            model=model,
            z_start=z_t,
            dt=dt,
            z_mean=z_mean,
            z_cov_inv=z_cov_inv,
            n_steps=ROLLOUT_STEPS,
            threshold=CLOUD_THRESHOLD,
        )

        total = LAMBDA_CURVE * loss_curve + LAMBDA_TRANS * loss_trans + LAMBDA_CLOUD * loss_cloud

        d0 = torch.sqrt(torch.clamp(mahal_sq(z_t, z_mean, z_cov_inv), min=0.0))
        frac_exceed = (d0 > CLOUD_THRESHOLD).double().mean().item()

        bs = xb_t.shape[0]
        sum_curve += loss_curve.item() * bs
        sum_trans += loss_trans.item() * bs
        sum_cloud += loss_cloud.item() * bs
        sum_total += total.item() * bs
        sum_exceed += frac_exceed * bs
        n_obs += bs

    out = {
        "loss_curve": sum_curve / max(n_obs, 1),
        "loss_trans": sum_trans / max(n_obs, 1),
        "loss_cloud": sum_cloud / max(n_obs, 1),
        "loss_total": sum_total / max(n_obs, 1),
        "frac_exceed_start": sum_exceed / max(n_obs, 1),
        "n_obs": n_obs,
    }

    if was_training:
        model.train()

    return out

# ==========================================================
# Load data
# ==========================================================
meta, X_tensor, meta_full, X_tensor_full, tenors, df_wide, df_wide_all, SCALE_IS_PERCENT = my_data(use=USE)

X_tensor = X_tensor.to(torch.float64)
X_tensor_full = X_tensor_full.to(torch.float64)

if CCY_FILTER:
    mask_curve = meta["ccy"].astype(str).str.upper() == CCY_FILTER.upper()
    meta_curve = meta.loc[mask_curve].reset_index(drop=True)
    X_curve = X_tensor[mask_curve.to_numpy()]
else:
    meta_curve = meta.copy()
    X_curve = X_tensor

if X_curve.shape[0] == 0:
    raise ValueError(f"No rows found for CCY_FILTER={CCY_FILTER}")

pair_data_all = build_transition_pairs(
    df_wide=df_wide,
    tenors=tenors,
    scale_is_percent=SCALE_IS_PERCENT,
    ccy_filter=CCY_FILTER,
)

pair_train, pair_val = split_pairs_timewise(pair_data_all, val_frac=VAL_FRAC)

print("\nTransition dataset summary:")
print("  currency filter :", CCY_FILTER)
print("  train pairs     :", len(pair_train.meta_pairs))
print("  val pairs       :", len(pair_val.meta_pairs))
print("  dt train min/max:", float(pair_train.dt.min()), float(pair_train.dt.max()))
if len(pair_val.meta_pairs) > 0:
    print("  dt val min/max  :", float(pair_val.dt.min()), float(pair_val.dt.max()))

train_dataset = TensorDataset(pair_train.X_t, pair_train.X_tp1, pair_train.dt)
val_dataset = TensorDataset(pair_val.X_t, pair_val.X_tp1, pair_val.dt)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

# ==========================================================
# Resolve and load base checkpoint
# ==========================================================
BASE_CHECKPOINT_PATH = resolve_base_checkpoint_path(
    training_root=TRAINING_ROOT,
    latent_dim=LATENT_DIM,
    base_epochs=BASE_EPOCHS,
)
print("\nBase checkpoint resolved to:")
print(BASE_CHECKPOINT_PATH)

raw_base_ckpt, warmstart_state_dict = load_state_dict_from_checkpoint(
    BASE_CHECKPOINT_PATH,
    map_location=device,
)

# ==========================================================
# Build center config from base model
# ==========================================================
k_z_center_init = None
k_learn_center = True
center_source_used = None

if USE_FIXED_CENTER:
    print("\nEstimating latent center from base checkpoint ...")

    center_model = FullModel(
        latent_dim=LATENT_DIM,
        sigma_init=SIGMA_INIT,
        k_drift_scale_init=K_DRIFT_SCALE_INIT,
        k_learn_center=True,
    ).to(device).double()

    missing_center, unexpected_center = center_model.load_state_dict(
        warmstart_state_dict,
        strict=False,
    )
    summarize_load_result("Center model warm-start:", missing_center, unexpected_center)

    z_center_mean, z_center_std = estimate_latent_center(
        center_model, X_curve, batch_size=EVAL_BATCH_SIZE
    )

    k_z_center_init = z_center_mean.numpy().astype(np.float32)
    k_learn_center = False
    center_source_used = BASE_CHECKPOINT_PATH

    print("Estimated latent center from checkpoint:", k_z_center_init)
    print("Estimated latent std from checkpoint   :", z_center_std.numpy())

    del center_model
else:
    k_z_center_init = MANUAL_CENTER.copy()
    k_learn_center = True
    center_source_used = "learned_center"

if k_z_center_init is None:
    k_z_center_init = MANUAL_CENTER.copy()
    k_learn_center = False
    center_source_used = "MANUAL_CENTER"

print("Final K center init :", k_z_center_init)
print("Final K learn_center:", k_learn_center)

# ==========================================================
# Create continuation model
# ==========================================================
model = FullModel(
    latent_dim=LATENT_DIM,
    sigma_init=SIGMA_INIT,
    k_drift_scale_init=K_DRIFT_SCALE_INIT,
    k_z_center_init=k_z_center_init,
    k_learn_center=k_learn_center,
).to(device).double()

if WARMSTART_FROM_PREVIOUS:
    missing, unexpected = model.load_state_dict(warmstart_state_dict, strict=False)
    summarize_load_result("Warm-start loaded.", missing, unexpected)

    if hasattr(model.K, "theta") and k_z_center_init is not None:
        with torch.no_grad():
            model.K.theta.copy_(
                torch.as_tensor(k_z_center_init, device=device, dtype=model.K.theta.dtype)
            )
        print("Overwrote K.theta with fixed center:", model.K.theta.detach().cpu().numpy())

# Freeze encoder and decoder G
if FREEZE_ENCODER:
    freeze_module(model.encoder)

if FREEZE_DECODER_G:
    freeze_module(model.G)

print_trainable_parameters(model)

trainable_params = [p for p in model.parameters() if p.requires_grad]
if len(trainable_params) == 0:
    raise RuntimeError("No trainable parameters left after freezing modules.")

params_khr = []
params_g = []

for name, p in model.named_parameters():
    if not p.requires_grad:
        continue
    if name.startswith("G."):
        params_g.append(p)
    else:
        params_khr.append(p)

if (not FREEZE_DECODER_G) and len(params_g) == 0:
    raise RuntimeError("FREEZE_DECODER_G=False but no trainable G parameters were found.")

# ==========================================================
# Empirical latent support stats (for cloud penalty)
# ==========================================================
z_support_mean, z_support_cov, z_support_cov_inv = estimate_latent_support_stats(
    model, X_curve, batch_size=EVAL_BATCH_SIZE
)
z_support_mean = z_support_mean.to(device)
z_support_cov_inv = z_support_cov_inv.to(device)

print("\nEmpirical latent support stats:")
print("  z mean:", z_support_mean.detach().cpu().numpy())
print("  z cov :\n", z_support_cov.detach().cpu().numpy())

# ==========================================================
# Optimizer / scheduler
# ==========================================================
optim_param_groups = []
if len(params_khr) > 0:
    optim_param_groups.append({"params": params_khr, "lr": MAX_LR_KHR})
if len(params_g) > 0:
    optim_param_groups.append({"params": params_g, "lr": MAX_LR_G})

optim = torch.optim.Adam(optim_param_groups)

scheduler = OneCycleLR(
    optim,
    max_lr=[group["lr"] for group in optim_param_groups],
    steps_per_epoch=len(train_loader),
    epochs=CONT_EPOCHS,
    pct_start=0.3,
    div_factor=10.0,
    final_div_factor=1000.0,
)

# ==========================================================
# Initial sanity check
# ==========================================================
with torch.no_grad():
    xb0 = pair_train.X_t[:8].to(device)
    S0, aux0 = model(xb0, return_aux=True)

    print("\nInitial forward OK")
    print("Initial S_hat finite:", torch.isfinite(S0).all().item())
    print("Initial z mean:", aux0["z"].mean(dim=0).cpu().numpy())
    print("Initial z std :", aux0["z"].std(dim=0).cpu().numpy())

    kappa_np = get_kappa_numpy(model.K)
    if kappa_np is not None:
        print("Initial kappa:", kappa_np)

    if hasattr(model.K, "theta"):
        try:
            print("Initial theta:", model.K.theta.detach().cpu().numpy())
        except Exception:
            pass

    diag0 = torch.diagonal(aux0["sigma"], dim1=1, dim2=2)
    print("Initial sigma diag mean:", diag0.mean(dim=0).cpu().numpy())

# ==========================================================
# CSV logger setup
# ==========================================================
csv_path = os.path.join(RUN_DIR, "train_metrics.csv")
csv_cols = [
    "epoch",
    "time_total_sec",
    "time_interval_sec",
    "train_curve_loss",
    "train_trans_loss",
    "train_cloud_loss",
    "train_total_loss",
    "val_curve_loss",
    "val_trans_loss",
    "val_cloud_loss",
    "val_total_loss",
    "val_frac_exceed_start",
    "curve_avg_rmse_bps",
    "curve_n_good",
    "curve_n_bad",
]
pd.DataFrame(columns=csv_cols).to_csv(csv_path, index=False)
print("\nLogging to:", csv_path)

# ==========================================================
# Training
# ==========================================================
best_val_total = float("inf")
best_checkpoint_path = os.path.join(RUN_DIR, f"best_checkpoint_dim{LATENT_DIM}.pt")

lrs_per_step_khr = []
lrs_per_step_g = []
history = []
nan_batches_total = 0

t0 = time.perf_counter()
t_last_log = t0

for epoch in range(CONT_EPOCHS):
    model.train()

    sum_curve = 0.0
    sum_trans = 0.0
    sum_cloud = 0.0
    sum_total = 0.0
    n_obs = 0
    nan_batches = 0
    grad_norm_epoch = []

    for xb_t, xb_tp1, dt in train_loader:
        xb_t = xb_t.to(device)
        xb_tp1 = xb_tp1.to(device)
        dt = dt.to(device)

        optim.zero_grad(set_to_none=USE_SET_TO_NONE)

        # static curve anchor
        S_hat_t = model(xb_t)
        if not torch.isfinite(S_hat_t).all():
            nan_batches += 1
            continue

        loss_curve = F.mse_loss(S_hat_t, xb_t) * CURVE_LOSS_SCALE

        # frozen encoder -> latent transitions
        with torch.no_grad():
            z_t = model.encoder(xb_t).detach()
            z_tp1 = model.encoder(xb_tp1).detach()

        loss_trans, _ = transition_nll_loss(model, z_t, z_tp1, dt)

        loss_cloud = rollout_cloud_penalty(
            model=model,
            z_start=z_t,
            dt=dt,
            z_mean=z_support_mean,
            z_cov_inv=z_support_cov_inv,
            n_steps=ROLLOUT_STEPS,
            threshold=CLOUD_THRESHOLD,
        )

        loss = (
            LAMBDA_CURVE * loss_curve
            + LAMBDA_TRANS * loss_trans
            + LAMBDA_CLOUD * loss_cloud
        )

        if not torch.isfinite(loss):
            nan_batches += 1
            continue

        loss.backward()

        all_grads_finite = True
        for p in trainable_params:
            if p.grad is not None and not torch.isfinite(p.grad).all():
                all_grads_finite = False
                break

        if not all_grads_finite:
            nan_batches += 1
            optim.zero_grad(set_to_none=USE_SET_TO_NONE)
            continue

        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        grad_norm_epoch.append(float(grad_norm))

        optim.step()
        scheduler.step()

        bs = xb_t.shape[0]
        sum_curve += loss_curve.item() * bs
        sum_trans += loss_trans.item() * bs
        sum_cloud += loss_cloud.item() * bs
        sum_total += loss.item() * bs
        n_obs += bs

        lrs_per_step_khr.append(optim.param_groups[0]["lr"] if len(optim.param_groups) > 0 else np.nan)
        lrs_per_step_g.append(optim.param_groups[1]["lr"] if len(optim.param_groups) > 1 else np.nan)

    if n_obs == 0:
        print("[ABORT] No valid batches were processed this epoch. Stopping.")
        break

    nan_batches_total += nan_batches

    train_curve_loss = sum_curve / max(n_obs, 1)
    train_trans_loss = sum_trans / max(n_obs, 1)
    train_cloud_loss = sum_cloud / max(n_obs, 1)
    train_total_loss = sum_total / max(n_obs, 1)
    mean_grad_norm = float(np.mean(grad_norm_epoch)) if len(grad_norm_epoch) > 0 else np.nan

    do_eval = ((epoch + 1) % EVAL_EVERY == 0) or (epoch == 0) or (epoch == CONT_EPOCHS - 1)
    do_log = ((epoch + 1) % LOG_EVERY == 0) or (epoch == 0) or (epoch == CONT_EPOCHS - 1)

    if do_eval:
        val_stats = evaluate_on_transition_loader(
            model=model,
            loader=val_loader,
            z_mean=z_support_mean,
            z_cov_inv=z_support_cov_inv,
        ) if len(val_dataset) > 0 else {
            "loss_curve": np.nan,
            "loss_trans": np.nan,
            "loss_cloud": np.nan,
            "loss_total": np.nan,
            "frac_exceed_start": np.nan,
            "n_obs": 0,
        }

        rmse_per_ccy, avg_rmse_bps, n_bad, n_good = eval_rmse_bps(
            model, X_curve, meta_curve, batch_size=EVAL_BATCH_SIZE
        )

        val_total = float(val_stats["loss_total"])
        if np.isfinite(val_total) and val_total < best_val_total:
            best_val_total = val_total
            torch.save(model.state_dict(), best_checkpoint_path)
            print(f"[BEST] epoch={epoch} val_total={val_total:.6f} saved to {best_checkpoint_path}")
    else:
        val_stats = {
            "loss_curve": np.nan,
            "loss_trans": np.nan,
            "loss_cloud": np.nan,
            "loss_total": np.nan,
            "frac_exceed_start": np.nan,
            "n_obs": 0,
        }
        avg_rmse_bps, n_good, n_bad = np.nan, np.nan, np.nan

    if do_log:
        t_now = time.perf_counter()
        time_total = t_now - t0
        time_interval = t_now - t_last_log
        t_last_log = t_now

        row = {
            "epoch": epoch,
            "time_total_sec": time_total,
            "time_interval_sec": time_interval,
            "train_curve_loss": train_curve_loss,
            "train_trans_loss": train_trans_loss,
            "train_cloud_loss": train_cloud_loss,
            "train_total_loss": train_total_loss,
            "val_curve_loss": val_stats["loss_curve"],
            "val_trans_loss": val_stats["loss_trans"],
            "val_cloud_loss": val_stats["loss_cloud"],
            "val_total_loss": val_stats["loss_total"],
            "val_frac_exceed_start": val_stats["frac_exceed_start"],
            "curve_avg_rmse_bps": float(avg_rmse_bps),
            "curve_n_good": int(n_good) if np.isfinite(n_good) else np.nan,
            "curve_n_bad": int(n_bad) if np.isfinite(n_bad) else np.nan,
        }
        pd.DataFrame([row], columns=csv_cols).to_csv(csv_path, mode="a", header=False, index=False)
        history.append(row)

        print(
            f"epoch={epoch:4d} "
            f"train_total={train_total_loss:.6f} "
            f"(curve={train_curve_loss:.6f}, trans={train_trans_loss:.6f}, cloud={train_cloud_loss:.6f}) | "
            f"val_total={val_stats['loss_total']:.6f} | "
            f"rmse_bps={avg_rmse_bps:.3f} | "
            f"lr={optim.param_groups[0]['lr']:.2e} | "
            f"used_obs={n_obs} nan_batches={nan_batches} total_nan_batches={nan_batches_total} | "
            f"time_total={time_total/60:.1f}min interval={time_interval/60:.1f}min | "
            f"mean_grad_norm={mean_grad_norm:.3e}"
        )

        print(
            f"  val components: curve={val_stats['loss_curve']:.6f}, "
            f"trans={val_stats['loss_trans']:.6f}, "
            f"cloud={val_stats['loss_cloud']:.6f}, "
            f"frac_exceed_start={val_stats['frac_exceed_start']:.3%}"
        )

        print_stable_diagnostics(model, X_curve)
        print_arb_diagnostics(model, X_curve)

print("Pricing continuation training done.")

# ==========================================================
# Plots
# ==========================================================
history_df = pd.DataFrame(history)

fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=160)
ax.plot(np.arange(len(lrs_per_step_khr)), lrs_per_step_khr, linewidth=1.0, label="K/H/R")
if len(lrs_per_step_g) > 0 and np.isfinite(np.asarray(lrs_per_step_g)).any():
    ax.plot(np.arange(len(lrs_per_step_g)), lrs_per_step_g, linewidth=1.0, label="G")
ax.set_xlabel("Training step (batch)")
ax.set_ylabel("Learning rate")
ax.set_title(f"Pricing continuation LR schedule — OneCycleLR (dim={LATENT_DIM})")
ax.legend()
fig.tight_layout()
lr_fig_path = os.path.join(RUN_DIR, "lr_schedule.png")
fig.savefig(lr_fig_path, dpi=300)
print("Saved LR plot:", lr_fig_path)
if SHOW_PLOTS:
    plt.show()
plt.close(fig)

if not history_df.empty:
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=160)
    ax.plot(history_df["epoch"], history_df["train_total_loss"], label="train total")
    ax.plot(history_df["epoch"], history_df["val_total_loss"], label="val total")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total objective")
    ax.set_title("Pricing continuation objective")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(RUN_DIR, "total_objective.png")
    fig.savefig(path, dpi=300)
    print("Saved total objective plot:", path)
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=160)
    ax.plot(history_df["epoch"], history_df["train_curve_loss"], label="train curve")
    ax.plot(history_df["epoch"], history_df["train_trans_loss"], label="train trans")
    ax.plot(history_df["epoch"], history_df["train_cloud_loss"], label="train cloud")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss component")
    ax.set_title("Train loss components")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(RUN_DIR, "train_loss_components.png")
    fig.savefig(path, dpi=300)
    print("Saved train component plot:", path)
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=160)
    ax.plot(history_df["epoch"], history_df["curve_avg_rmse_bps"], label="curve avg RMSE (bps)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("RMSE (bps)")
    ax.set_title("Cross-sectional curve fit during pricing continuation")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(RUN_DIR, "curve_rmse_bps.png")
    fig.savefig(path, dpi=300)
    print("Saved curve RMSE plot:", path)
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=160)
    ax.plot(history_df["epoch"], history_df["val_frac_exceed_start"], label="val frac exceed support")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Fraction")
    ax.set_title("Latent support exceedance")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(RUN_DIR, "latent_support_exceedance.png")
    fig.savefig(path, dpi=300)
    print("Saved support exceedance plot:", path)
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)

# ==========================================================
# Save trained model checkpoint
# ==========================================================
last_state_dict_path = os.path.join(RUN_DIR, f"checkpoint_dim{LATENT_DIM}.pt")
full_checkpoint_path = os.path.join(RUN_DIR, "full_checkpoint.pt")

torch.save(model.state_dict(), last_state_dict_path)
print("Saved last state_dict:", last_state_dict_path)

model_config = {
    "latent_dim": LATENT_DIM,
    "sigma_init": SIGMA_INIT,
    "k_drift_scale_init": K_DRIFT_SCALE_INIT,
    "k_z_center_init": k_z_center_init.tolist() if k_z_center_init is not None else None,
    "k_learn_center": k_learn_center,
}

training_config = {
    "epochs": CONT_EPOCHS,
    "batch_size": BATCH_SIZE,
    "max_lr_khr": MAX_LR_KHR,
    "max_lr_g": MAX_LR_G,
    "variant": config.VARIANT,
    "use_data": USE,
    "ccy_filter": CCY_FILTER,
    "center_source_used": center_source_used,
    "warmstart_from_previous": WARMSTART_FROM_PREVIOUS,
    "base_checkpoint_path": BASE_CHECKPOINT_PATH,
    "freeze_encoder": FREEZE_ENCODER,
    "freeze_decoder_g": FREEZE_DECODER_G,
    "trainable_modules": ["K", "H", "R"] + ([] if FREEZE_DECODER_G else ["G"]),
    "run_name": RUN_NAME,
    "run_dir": RUN_DIR,
    "objective": {
        "curve_loss_scale": CURVE_LOSS_SCALE,
        "lambda_curve": LAMBDA_CURVE,
        "lambda_trans": LAMBDA_TRANS,
        "lambda_cloud": LAMBDA_CLOUD,
        "rollout_steps": ROLLOUT_STEPS,
        "cloud_threshold": CLOUD_THRESHOLD,
    },
    "val_frac": VAL_FRAC,
}

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "model_config": model_config,
        "training_config": training_config,
        "latent_dim": LATENT_DIM,
        "epochs": CONT_EPOCHS,
        "use_data": USE,
        "variant": config.VARIANT,
    },
    full_checkpoint_path,
)

print("Saved full checkpoint:", full_checkpoint_path)