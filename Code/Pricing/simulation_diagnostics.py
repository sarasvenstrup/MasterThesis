import math
import os
import re
import sys
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

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
from Code.model.sigma_matrix import L_from_sigmas_rhos
from Code.utils.ode import (
    d_tau_autograd_nodewise,
    grad_and_trace_cov_hess_G,
    paper_alpha_beta_gamma_trace,
    solve_AB,
)
from Code.utils.rates import par_swap_from_discount

print(f"Repo root: {REPO_ROOT}")
print(f"Active model variant from config.py: {config.VARIANT}")

# ==========================================================
# Defaults for reconstructing model when checkpoint is a raw
# plain state_dict (e.g. best_checkpoint_dim2.pt)
# ==========================================================
DEFAULT_SIGMA_INIT = 0.0075
DEFAULT_K_DRIFT_SCALE_INIT = 0.25
DEFAULT_K_LEARN_CENTER = True
DEFAULT_K_Z_CENTER_INIT = None


# ==========================================================
# General helpers
# ==========================================================
def sanitize_tag(s: str) -> str:
    s = str(s)
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    return s.strip("_")


def safe_switch_backend(show_plots: bool):
    if show_plots:
        try:
            plt.switch_backend("TkAgg")
        except Exception as e:
            print(f"[WARN] Could not switch to TkAgg ({e}). Falling back to Agg.")
            plt.switch_backend("Agg")
    else:
        plt.switch_backend("Agg")


def resolve_checkpoint_path(repo_root: str, use: str, latent_dim: int, epochs: int) -> str:
    variant = config.VARIANT

    new_filename = f"checkpoint_dim{latent_dim}_ep{epochs}.pt"
    new_path = os.path.join(
        THESIS_ROOT,
        "Figures",
        "TrainingResults",
        f"dim{latent_dim}_{variant}",
        f"ep{epochs}",
        new_filename,
    )

    old_filename = f"fullmodel_{use}_dim{latent_dim}_ep{epochs}.pt"
    candidates = [
        new_path,
        os.path.join(repo_root, "..", "checkpoints", old_filename),
        os.path.join(repo_root, "checkpoints", old_filename),
    ]

    for path in candidates:
        if os.path.exists(path):
            return os.path.abspath(path)

    searched = "\n".join(f"  - {os.path.abspath(p)}" for p in candidates)
    raise FileNotFoundError(f"Checkpoint not found. Searched:\n{searched}")


def safe_load_state_dict(model, state_dict):
    incompat = model.load_state_dict(state_dict, strict=False)

    missing = list(incompat.missing_keys)
    unexpected = list(incompat.unexpected_keys)

    allowed_missing = set()
    allowed_unexpected = set()

    real_missing = [k for k in missing if k not in allowed_missing]
    real_unexpected = [k for k in unexpected if k not in allowed_unexpected]

    if missing:
        print(f"[load_state_dict] Missing keys: {missing}")
    if unexpected:
        print(f"[load_state_dict] Unexpected keys: {unexpected}")

    if real_missing or real_unexpected:
        raise RuntimeError(
            "Non-benign checkpoint/model mismatch detected.\n"
            f"Real missing keys: {real_missing}\n"
            f"Real unexpected keys: {real_unexpected}"
        )


def build_model_init_kwargs(
    raw_checkpoint,
    latent_dim,
    sigma_init=DEFAULT_SIGMA_INIT,
    k_drift_scale_init=DEFAULT_K_DRIFT_SCALE_INIT,
    k_z_center_init=DEFAULT_K_Z_CENTER_INIT,
    k_learn_center=DEFAULT_K_LEARN_CENTER,
):
    model_kwargs = {
        "latent_dim": latent_dim,
        "sigma_init": sigma_init,
        "k_drift_scale_init": k_drift_scale_init,
        "k_z_center_init": k_z_center_init,
        "k_learn_center": k_learn_center,
    }

    state_dict = raw_checkpoint["model_state_dict"] if isinstance(raw_checkpoint, dict) and "model_state_dict" in raw_checkpoint else raw_checkpoint

    if isinstance(raw_checkpoint, dict) and "model_config" in raw_checkpoint:
        cfg = raw_checkpoint["model_config"]

        model_kwargs["latent_dim"] = int(cfg.get("latent_dim", model_kwargs["latent_dim"]))
        model_kwargs["sigma_init"] = float(cfg.get("sigma_init", model_kwargs["sigma_init"]))
        model_kwargs["k_drift_scale_init"] = float(cfg.get("k_drift_scale_init", model_kwargs["k_drift_scale_init"]))

        if cfg.get("k_z_center_init", None) is not None:
            model_kwargs["k_z_center_init"] = np.asarray(cfg["k_z_center_init"], dtype=np.float32)

        if "k_learn_center" in cfg and cfg["k_learn_center"] is not None:
            model_kwargs["k_learn_center"] = bool(cfg["k_learn_center"])

    if model_kwargs["k_z_center_init"] is None and isinstance(state_dict, dict) and "K.theta" in state_dict:
        try:
            theta = state_dict["K.theta"].detach().cpu().numpy().astype(np.float32)
            model_kwargs["k_z_center_init"] = theta
            model_kwargs["k_learn_center"] = False
            print("Inferred fixed center from state_dict K.theta:", theta)
        except Exception as e:
            print(f"[WARN] Could not infer K.theta from state_dict: {e}")

    return model_kwargs


def load_and_setup_model(
    device,
    use,
    latent_dim,
    epochs,
    checkpoint_path=None,
    sigma_init=DEFAULT_SIGMA_INIT,
    k_drift_scale_init=DEFAULT_K_DRIFT_SCALE_INIT,
    k_z_center_init=DEFAULT_K_Z_CENTER_INIT,
    k_learn_center=DEFAULT_K_LEARN_CENTER,
):
    if latent_dim != 2:
        raise ValueError("This script currently supports only the 2-factor model (latent_dim=2).")

    if checkpoint_path is None:
        checkpoint_path = resolve_checkpoint_path(REPO_ROOT, use, latent_dim, epochs)

    raw = torch.load(checkpoint_path, map_location=device, weights_only=False)

    from Code.model.full_model import FullModel

    if isinstance(raw, dict) and "model_state_dict" in raw:
        state_dict = raw["model_state_dict"]
        saved_variant = raw.get("variant", "unknown")
        if saved_variant != "unknown" and saved_variant != config.VARIANT:
            raise ValueError(
                f"Checkpoint variant '{saved_variant}' does not match active "
                f"config.VARIANT '{config.VARIANT}'. Update Code/config.py."
            )
    else:
        state_dict = raw

    model_kwargs = build_model_init_kwargs(
        raw_checkpoint=raw,
        latent_dim=latent_dim,
        sigma_init=sigma_init,
        k_drift_scale_init=k_drift_scale_init,
        k_z_center_init=k_z_center_init,
        k_learn_center=k_learn_center,
    )

    print("Model init kwargs used for diagnostics loading:")
    print(model_kwargs)

    model = FullModel(**model_kwargs)
    safe_load_state_dict(model, state_dict)

    model.to(device).double()
    model.eval()

    print(f"Loaded model from {checkpoint_path}")
    print(f"  Active config variant: {config.VARIANT}")
    print(f"  Model dtype: {next(model.parameters()).dtype}")

    return model, checkpoint_path


def load_simulation_bundle(bundle_path, device):
    bundle = torch.load(bundle_path, map_location=device, weights_only=False)

    bundle["z_paths"] = bundle["z_paths"].to(device)
    bundle["r_paths"] = bundle["r_paths"].to(device)
    bundle["mu_paths"] = bundle["mu_paths"].to(device)
    bundle["L_paths"] = bundle["L_paths"].to(device)
    bundle["discount_paths"] = bundle["discount_paths"].to(device)
    bundle["z_train_mean"] = bundle["z_train_mean"].to(device)
    bundle["z_train_cov"] = bundle["z_train_cov"].to(device)
    bundle["decoder_tau_grid"] = bundle["decoder_tau_grid"].to(device)
    return bundle


def make_output_suffix(bundle_path, metadata):
    base = os.path.splitext(os.path.basename(bundle_path))[0].replace("simulation_bundle_", "")

    checkpoint_path = metadata.get("checkpoint_path", "")
    if checkpoint_path:
        parent = sanitize_tag(os.path.basename(os.path.dirname(checkpoint_path)))
        ckpt = sanitize_tag(os.path.splitext(os.path.basename(checkpoint_path))[0])
        tag = sanitize_tag(f"{parent}__{ckpt}")
        if tag:
            base = f"{base}__{tag}"

    return base


# ==========================================================
# Model/data helpers
# ==========================================================
@torch.no_grad()
def get_L(model, z):
    H_out = model.H(z)

    if isinstance(H_out, tuple) and len(H_out) == 2:
        sigmas, rhos = H_out
        return L_from_sigmas_rhos(sigmas, rhos)

    if torch.is_tensor(H_out) and H_out.ndim == 3:
        return H_out

    raise TypeError(
        "Unsupported model.H(z) output. Expected either "
        "(sigmas, rhos) or a tensor L of shape (B,d,d)."
    )


def load_data_for_training_cloud(use="bbg", ccy_filter=""):
    meta, X_tensor, meta_full, X_tensor_full, tenors, df_wide, df_wide_all, scale_is_percent = my_data(use=use)
    X_tensor = X_tensor.double()

    if ccy_filter and str(ccy_filter).strip():
        c = str(ccy_filter).strip().upper()
        mask = meta["ccy"].astype(str).str.upper() == c
        meta = meta.loc[mask].reset_index(drop=True)
        X_tensor = X_tensor[mask.to_numpy()]
        print(f"Filtered training cloud to {c}: kept {len(meta)} rows")

    return meta, X_tensor, tenors, scale_is_percent


def compute_latent_statistics(model, X_tensor, device, latent_dim):
    print("\n" + "=" * 60)
    print("DIAGNOSTIC: Computing training latent region statistics")
    print("=" * 60)

    z_train_list = []
    with torch.no_grad():
        for i in range(0, X_tensor.shape[0], 256):
            batch = X_tensor[i: min(i + 256, X_tensor.shape[0])].to(device)
            z_batch = model.encoder(batch)
            z_train_list.append(z_batch)

    z_train = torch.cat(z_train_list, dim=0)
    z_train_mean = z_train.mean(dim=0).detach()
    z_train_cov = torch.cov(z_train.t()).detach()
    z_train_std = z_train.std(dim=0).detach()

    print(f"Training latent cloud mean: {z_train_mean.cpu().numpy()}")
    print(f"Training latent cloud std:  {z_train_std.cpu().numpy()}")
    print("Training latent cloud range:")
    for d in range(latent_dim):
        print(f"  z[{d}]: [{z_train[:, d].min().item():.6f}, {z_train[:, d].max().item():.6f}]")
    print("=" * 60 + "\n")

    return z_train_mean, z_train_cov, z_train_std


def diagnose_G0_on_training_cloud(model, X_tensor, device, batch_size=256):
    print("\n" + "=" * 60)
    print("DIAGNOSTIC: Checking G(z,0) on training latent cloud")
    print("=" * 60)

    g0_list = []

    with torch.no_grad():
        for i in range(0, X_tensor.shape[0], batch_size):
            batch = X_tensor[i: min(i + batch_size, X_tensor.shape[0])].to(device)
            z_batch = model.encoder(batch)

            tau0 = torch.zeros(1, device=device, dtype=z_batch.dtype)
            G0_batch = model.G(z_batch, tau0)

            if G0_batch.ndim == 2:
                G0_batch = G0_batch[:, 0]
            elif G0_batch.ndim != 1:
                raise RuntimeError(f"Unexpected shape for G(z,0): {tuple(G0_batch.shape)}")

            g0_list.append(G0_batch)

    G0 = torch.cat(g0_list, dim=0)
    absG0 = G0.abs()

    print(f"G(z,0) raw min        : {G0.min().item():.6e}")
    print(f"G(z,0) raw max        : {G0.max().item():.6e}")
    print(f"|G(z,0)| min          : {absG0.min().item():.6e}")
    print(f"|G(z,0)| mean         : {absG0.mean().item():.6e}")
    print(f"|G(z,0)| median       : {absG0.median().item():.6e}")
    print(f"|G(z,0)| 1% quantile  : {torch.quantile(absG0, 0.01).item():.6e}")
    print(f"|G(z,0)| 5% quantile  : {torch.quantile(absG0, 0.05).item():.6e}")
    print(f"count(|G0| < 1e-2)    : {(absG0 < 1e-2).sum().item()}")
    print(f"count(|G0| < 1e-3)    : {(absG0 < 1e-3).sum().item()}")
    print(f"count(|G0| < 1e-4)    : {(absG0 < 1e-4).sum().item()}")
    print("=" * 60 + "\n")

    return G0


def diagnose_G0_on_simulated_paths(model, z_paths):
    print("\n" + "=" * 60)
    print("DIAGNOSTIC: Checking G(z_t,0) on simulated latent paths")
    print("=" * 60)

    n_paths, n_times, _ = z_paths.shape
    device = z_paths.device
    dtype = z_paths.dtype

    rows = []

    with torch.no_grad():
        tau0 = torch.zeros(1, device=device, dtype=dtype)

        for t in range(n_times):
            z_t = z_paths[:, t, :]
            G0_t = model.G(z_t, tau0)

            if G0_t.ndim == 2:
                G0_t = G0_t[:, 0]
            elif G0_t.ndim != 1:
                raise RuntimeError(f"Unexpected shape for G(z_t,0): {tuple(G0_t.shape)}")

            absG0_t = G0_t.abs()
            row = {
                "time_index": t,
                "G0_min": G0_t.min().item(),
                "G0_max": G0_t.max().item(),
                "absG0_min": absG0_t.min().item(),
                "absG0_mean": absG0_t.mean().item(),
                "absG0_median": absG0_t.median().item(),
                "absG0_1pct": torch.quantile(absG0_t, 0.01).item(),
                "absG0_5pct": torch.quantile(absG0_t, 0.05).item(),
                "count_absG0_lt_1e_2": (absG0_t < 1e-2).sum().item(),
                "count_absG0_lt_1e_3": (absG0_t < 1e-3).sum().item(),
                "count_absG0_lt_1e_4": (absG0_t < 1e-4).sum().item(),
                "count_absG0_lt_1e_5": (absG0_t < 1e-5).sum().item(),
            }
            rows.append(row)

            print(
                f"t_idx={t:2d} | "
                f"min |G0|={row['absG0_min']:.3e} | "
                f"1%={row['absG0_1pct']:.3e} | "
                f"5%={row['absG0_5pct']:.3e} | "
                f"<1e-3: {row['count_absG0_lt_1e_3']:3d} | "
                f"<1e-4: {row['count_absG0_lt_1e_4']:3d}"
            )

    print("=" * 60 + "\n")
    return pd.DataFrame(rows)


def analyze_paths(z_paths, r_paths, mu_paths, L_paths, latent_dim):
    print("\n" + "=" * 60)
    print("DIAGNOSTIC: Analyzing mu and L")
    print("=" * 60)

    mu_np = mu_paths.detach().cpu().numpy()
    L_np = L_paths.detach().cpu().numpy()
    z_np = z_paths.detach().cpu().numpy()
    r_np = r_paths.detach().cpu().numpy()

    print("\n--- r_t Statistics ---")
    print(f"r: mean={r_np.mean():.6f}, std={r_np.std():.6f}, min={r_np.min():.6f}, max={r_np.max():.6f}")

    print("\n--- MU (Drift) Statistics ---")
    for d in range(latent_dim):
        mu_d = mu_np[:, :, d]
        print(f"mu[{d}]: mean={mu_d.mean():.6f}, std={mu_d.std():.6f}, min={mu_d.min():.6f}, max={mu_d.max():.6f}")

    print("\n--- L (Diffusion) Statistics ---")
    for i in range(latent_dim):
        for j in range(latent_dim):
            L_ij = L_np[:, :, i, j]
            print(
                f"L[{i},{j}]: mean={L_ij.mean():.6f}, std={L_ij.std():.6f}, min={L_ij.min():.6f}, max={L_ij.max():.6f}"
            )

    print("\n--- MU Variance Analysis ---")
    mu_var_time = mu_np.var(axis=0)
    mu_var_path = mu_np.var(axis=1)
    print(f"Mean variance of mu across paths at each time step: {mu_var_time.mean():.6e}")
    print(f"Mean variance of mu across time for each path: {mu_var_path.mean():.6e}")

    print("\n--- L Frobenius Norm Analysis ---")
    L_norms = np.linalg.norm(L_np, axis=(2, 3))
    print(f"L Frobenius norm: mean={L_norms.mean():.6f}, std={L_norms.std():.6f}, min={L_norms.min():.6f}, max={L_norms.max():.6f}")

    print("\n--- Sample mu values at t=0 (first 3 paths) ---")
    for p in range(min(3, z_np.shape[0])):
        print(f"Path {p}: mu = {mu_np[p, 0, :]}")

    print("\n--- Sample mu values at final time (first 3 paths) ---")
    for p in range(min(3, z_np.shape[0])):
        print(f"Path {p}: mu = {mu_np[p, -1, :]}")

    print("\n--- Sample covariance eigenvalues Sigma=L@L.T at t=0 (first 3 paths) ---")
    for p in range(min(3, z_np.shape[0])):
        L_matrix = L_np[p, 0, :, :]
        Sigma = L_matrix @ L_matrix.T
        eigvals = np.linalg.eigvalsh(Sigma)
        print(f"Path {p}: eigenvalues = {eigvals}\nSigma =\n{Sigma}")

    mu_range = mu_np.max() - mu_np.min()
    print("\n--- MU Range Check ---")
    print(f"Overall mu range (max-min): {mu_range:.6e}")
    print(f"Is mu nearly constant? {mu_range < 1e-4}")

    print("\n--- Z-mu Correlation Analysis ---")
    for d in range(latent_dim):
        z_d_flat = z_np[:, :, d].flatten()
        mu_d_flat = mu_np[:, :, d].flatten()
        corr = np.corrcoef(z_d_flat, mu_d_flat)[0, 1]
        print(f"Correlation between z[{d}] and mu[{d}]: {corr:.6f}")

    print("=" * 60 + "\n")


def tenor_label(tenor_value):
    return f"{int(float(tenor_value))}Y"


# ==========================================================
# Decoder diagnostics
# ==========================================================
def decode_from_latent_script(model, z, tau, G_floor=1e-5, check_short_rate=True):
    if z.dim() == 1:
        z = z.unsqueeze(0)

    device = z.device
    dtype = z.dtype

    # Critical fix: diagnostics below need autograd
    z = z.to(device=device, dtype=dtype).detach().clone().requires_grad_(True)
    tau = tau.to(device=device, dtype=dtype).detach().clone().requires_grad_(True)

    if tau.ndim != 1 or tau.numel() < 2:
        raise RuntimeError("tau grid must be 1D and contain at least two points")
    if not torch.all(tau[1:] > tau[:-1]):
        raise RuntimeError("tau grid must be strictly increasing")
    if abs(float(tau[0].item())) > 1e-12:
        raise RuntimeError("tau grid must start at 0 to enforce decoder boundary conditions")

    G_vals = model.G(z, tau)
    if G_vals.dim() == 1:
        G_vals = G_vals.unsqueeze(0)

    if not torch.isfinite(G_vals).all():
        raise RuntimeError("Non-finite G_vals encountered")

    G0 = G_vals[:, 0]
    min_abs_G0 = G0.abs().min().item()
    if min_abs_G0 < G_floor:
        raise RuntimeError(
            f"G(z,0) too close to zero: min |G(z,0)| = {min_abs_G0:.3e}. "
            "Decoder ODE becomes ill-conditioned."
        )

    mu = model.K(z)
    sigma = get_L(model, z)

    r_tilde = model.R(z)
    if r_tilde.ndim == 2 and r_tilde.shape[-1] == 1:
        r_tilde = r_tilde.squeeze(-1)

    def G_single(z_single):
        return model.G(z_single.unsqueeze(0), tau).squeeze(0)

    dG_dtau = d_tau_autograd_nodewise(model.G, z, tau)
    grad_z_G, trace_cov_hess = grad_and_trace_cov_hess_G(G_single, z, sigma)

    alpha, beta, gamma = paper_alpha_beta_gamma_trace(
        G=G_vals,
        dG_dtau=dG_dtau,
        grad_z_G=grad_z_G,
        trace_cov_hess=trace_cov_hess,
        mu=mu,
        sigma=sigma,
        r_tilde=r_tilde,
    )

    if not torch.isfinite(alpha).all():
        raise RuntimeError("Non-finite alpha encountered")
    if not torch.isfinite(beta).all():
        raise RuntimeError("Non-finite beta encountered")
    if not torch.isfinite(gamma).all():
        raise RuntimeError("Non-finite gamma encountered")

    A_vals, B_vals = solve_AB(tau, alpha, beta, gamma)

    if not torch.isfinite(A_vals).all():
        raise RuntimeError("Non-finite A_vals encountered")
    if not torch.isfinite(B_vals).all():
        raise RuntimeError("Non-finite B_vals encountered")

    if not torch.allclose(A_vals[:, 0], torch.zeros_like(A_vals[:, 0]), atol=1e-8, rtol=0.0):
        raise RuntimeError("Decoder invariant violated: A(z,0) != 0")
    if not torch.allclose(B_vals[:, 0], torch.zeros_like(B_vals[:, 0]), atol=1e-8, rtol=0.0):
        raise RuntimeError("Decoder invariant violated: B(z,0) != 0")

    expo = A_vals - B_vals * G_vals
    if not torch.isfinite(expo).all():
        raise RuntimeError("Non-finite exponent encountered in bond pricing")

    expo = torch.clamp(expo, min=-80.0, max=20.0)
    P_full = torch.exp(expo)

    if not torch.isfinite(P_full).all():
        raise RuntimeError("Non-finite discount factors encountered")
    if (P_full <= 0).any():
        raise RuntimeError("Non-positive discount factors encountered")

    if not torch.allclose(P_full[:, 0], torch.ones_like(P_full[:, 0]), atol=1e-8, rtol=0.0):
        raise RuntimeError("Decoder invariant violated: P(z,0) != 1")

    short_rate_tau_used = None
    max_short_rate_err = float("nan")
    if check_short_rate and tau.numel() >= 2:
        tau1 = tau[1] - tau[0]
        short_rate_tau_used = float(tau1.item())
        f0_approx = -(torch.log(P_full[:, 1]) - torch.log(P_full[:, 0])) / tau1
        short_rate_err = (f0_approx - r_tilde).abs()
        max_short_rate_err = short_rate_err.max().item()

    diagnostics = {
        "G_range": (G_vals.min().item(), G_vals.max().item()),
        "P_range": (P_full[:, 1:].min().item(), P_full[:, 1:].max().item()),
        "min_abs_G0": min_abs_G0,
        "short_rate_tau_used": short_rate_tau_used,
        "max_short_rate_err": max_short_rate_err,
    }

    return (
        P_full.detach(),
        A_vals.detach(),
        B_vals.detach(),
        G_vals.detach(),
        mu.detach(),
        sigma.detach(),
        r_tilde.detach(),
        diagnostics,
    )


def decode_and_save_results_naive(
    model,
    z_paths,
    r_paths,
    z_train_mean,
    z_train_cov,
    decoder_tau_grid,
    annual_indices,
    device,
    times,
    tenors,
    suffix,
    out_dir,
    max_mahal=4.0,
    g0_floor=1e-5,
):
    eps_reg = 1e-8
    I_reg = torch.eye(z_train_cov.shape[0], device=device, dtype=z_train_cov.dtype)
    z_cov_inv = torch.linalg.inv(z_train_cov + eps_reg * I_reg)

    n_paths, n_times, latent_dim = z_paths.shape
    tenor_cols = [tenor_label(ten) for ten in tenors]

    swap_df_list = []
    latent_df_list = []
    mahal_df_list = []
    decoder_diag_df_list = []
    decoder_failures_list = []

    print("Naive decoding experiment...")
    t0 = time.time()

    for t in range(n_times):
        z_t = z_paths[:, t, :]
        r_t = r_paths[:, t]

        z_centered = z_t - z_train_mean
        quad = torch.sum((z_centered @ z_cov_inv) * z_centered, dim=1)
        mahal_dist = torch.sqrt(torch.clamp(quad, min=0.0))

        tau0 = torch.zeros(1, device=device, dtype=z_t.dtype)
        with torch.no_grad():
            G0_t = model.G(z_t, tau0)
            if G0_t.ndim == 2:
                G0_t = G0_t[:, 0]
            absG0_t = G0_t.abs()

        out_of_region = int((mahal_dist > max_mahal).sum().item())
        n_g0_bad_1e3 = int((absG0_t < 1e-3).sum().item())
        n_g0_bad_1e4 = int((absG0_t < 1e-4).sum().item())
        n_g0_bad_floor = int((absG0_t < g0_floor).sum().item())

        if out_of_region > 0:
            warnings.warn(
                f"At time t={times[t]:.3f}: {out_of_region}/{n_paths} paths exceed Mahalanobis {max_mahal:.2f}",
                RuntimeWarning,
            )

        batch_error = ""
        valid_decode = np.zeros(n_paths, dtype=bool)
        S_sim_np = np.full((n_paths, len(tenors)), np.nan, dtype=float)
        path_reasons = [""] * n_paths

        decoder_min_abs_G0_valid = np.nan
        decoder_max_short_rate_err_valid = np.nan

        try:
            P_full, _, _, _, _, _, _, dec_diag = decode_from_latent_script(
                model,
                z_t,
                decoder_tau_grid,
                G_floor=g0_floor,
                check_short_rate=True,
            )
            P_annual = P_full[:, annual_indices]
            S_sim = par_swap_from_discount(P_annual, tenors)

            S_sim_np = S_sim.detach().cpu().numpy()
            valid_decode[:] = True
            decoder_min_abs_G0_valid = dec_diag["min_abs_G0"]
            decoder_max_short_rate_err_valid = dec_diag["max_short_rate_err"]

        except RuntimeError as e:
            batch_error = str(e)

            min_abs_valid = np.inf
            max_sr_err_valid = -np.inf

            for p in range(n_paths):
                try:
                    P_full_p, _, _, _, _, _, _, dec_diag_p = decode_from_latent_script(
                        model,
                        z_t[p: p + 1],
                        decoder_tau_grid,
                        G_floor=g0_floor,
                        check_short_rate=True,
                    )
                    P_annual_p = P_full_p[:, annual_indices]
                    S_sim_p = par_swap_from_discount(P_annual_p, tenors)

                    S_sim_np[p, :] = S_sim_p[0].detach().cpu().numpy()
                    valid_decode[p] = True

                    min_abs_valid = min(min_abs_valid, float(dec_diag_p["min_abs_G0"]))
                    max_sr_err_valid = max(max_sr_err_valid, float(dec_diag_p["max_short_rate_err"]))

                except RuntimeError as e_p:
                    path_reasons[p] = str(e_p)
                    decoder_failures_list.append(
                        {
                            "time": float(times[t]),
                            "path_id": p,
                            "reason": str(e_p),
                            "mahal_dist": float(mahal_dist[p].item()),
                            "absG0": float(absG0_t[p].item()),
                        }
                    )

            if np.isfinite(min_abs_valid):
                decoder_min_abs_G0_valid = min_abs_valid
            if np.isfinite(max_sr_err_valid) and max_sr_err_valid > -np.inf:
                decoder_max_short_rate_err_valid = max_sr_err_valid

        n_valid = int(valid_decode.sum())
        frac_valid = n_valid / n_paths

        decoder_diag_df_list.append(
            {
                "time": float(times[t]),
                "max_mahal_dist": float(mahal_dist.max().item()),
                "mean_mahal_dist": float(mahal_dist.mean().item()),
                "frac_mahal_gt_threshold": float(out_of_region / n_paths),
                "n_paths_absG0_lt_1e_3": n_g0_bad_1e3,
                "n_paths_absG0_lt_1e_4": n_g0_bad_1e4,
                "n_paths_absG0_lt_floor": n_g0_bad_floor,
                "min_absG0_all_paths": float(absG0_t.min().item()),
                "decoder_min_G_abs0_valid_paths": decoder_min_abs_G0_valid,
                "decoder_max_short_rate_err_valid_paths": decoder_max_short_rate_err_valid,
                "n_valid_decode": n_valid,
                "frac_valid_decode": frac_valid,
                "batch_error": batch_error,
            }
        )

        for p in range(n_paths):
            swap_row = {
                "time": float(times[t]),
                "path_id": p,
                "valid_decode": bool(valid_decode[p]),
                "failure_reason": path_reasons[p],
            }
            for i, col in enumerate(tenor_cols):
                swap_row[f"swap_{col}"] = float(S_sim_np[p, i]) if np.isfinite(S_sim_np[p, i]) else np.nan
            swap_df_list.append(swap_row)

            latent_row = {
                "time": float(times[t]),
                "path_id": p,
                "r": float(r_t[p].detach().item()),
                "absG0": float(absG0_t[p].item()),
                "valid_decode": bool(valid_decode[p]),
            }
            for d in range(latent_dim):
                latent_row[f"z{d}"] = float(z_t[p, d].detach().item())
            latent_df_list.append(latent_row)

            mahal_df_list.append(
                {
                    "time": float(times[t]),
                    "path_id": p,
                    "mahal_dist": float(mahal_dist[p].detach().item()),
                    "absG0": float(absG0_t[p].item()),
                }
            )

        if t == 0 or t == n_times - 1 or t % max(1, max(1, n_times - 1) // 10) == 0:
            print(
                f"  t={times[t]:.3f} | "
                f"valid decode={n_valid}/{n_paths} | "
                f"min |G0|={absG0_t.min().item():.3e} | "
                f"max Mahalanobis={mahal_dist.max().item():.3f}"
            )

    elapsed = time.time() - t0
    print(f"Naive decoding finished in {elapsed:.2f}s")

    swap_df = pd.DataFrame(swap_df_list)
    latent_df = pd.DataFrame(latent_df_list)
    mahal_df = pd.DataFrame(mahal_df_list)
    decoder_diag_df = pd.DataFrame(decoder_diag_df_list)
    decoder_failures_df = pd.DataFrame(decoder_failures_list)

    swap_csv_path = os.path.join(out_dir, f"simulated_swap_curves_{suffix}.csv")
    latent_csv_path = os.path.join(out_dir, f"simulated_latent_{suffix}.csv")
    mahal_csv_path = os.path.join(out_dir, f"simulated_mahal_{suffix}.csv")
    decoder_diag_csv_path = os.path.join(out_dir, f"decoder_diagnostics_{suffix}.csv")
    decoder_failures_csv_path = os.path.join(out_dir, f"decoder_failures_{suffix}.csv")

    swap_df.to_csv(swap_csv_path, index=False)
    latent_df.to_csv(latent_csv_path, index=False)
    mahal_df.to_csv(mahal_csv_path, index=False)
    decoder_diag_df.to_csv(decoder_diag_csv_path, index=False)
    decoder_failures_df.to_csv(decoder_failures_csv_path, index=False)

    print(f"Saved simulated swap curves to {swap_csv_path}")
    print(f"Saved simulated latent paths to {latent_csv_path}")
    print(f"Saved Mahalanobis diagnostics to {mahal_csv_path}")
    print(f"Saved decoder diagnostics to {decoder_diag_csv_path}")
    print(f"Saved decoder failures to {decoder_failures_csv_path}")

    return swap_df, latent_df, mahal_df, decoder_diag_df, decoder_failures_df


def build_tau_grid_to_maturity(decoder_tau_grid, tau_end, device, dtype):
    tau_end = float(tau_end)
    if tau_end <= 0:
        raise ValueError(f"tau_end must be positive, got {tau_end}")

    base = decoder_tau_grid.to(device=device, dtype=dtype)
    interior = base[(base > 0.0) & (base < tau_end - 1e-12)]

    tau_grid = torch.cat(
        [
            torch.zeros(1, device=device, dtype=dtype),
            interior,
            torch.tensor([tau_end], device=device, dtype=dtype),
        ]
    )
    tau_grid = torch.unique(tau_grid, sorted=True)

    if tau_grid.numel() < 2:
        tau_grid = torch.tensor([0.0, tau_end], device=device, dtype=dtype)

    return tau_grid


def get_grid_indices_for_values(grid: torch.Tensor, values: torch.Tensor, tol: float = 1e-10):
    idx_list = []
    for v in values:
        diffs = torch.abs(grid - v)
        idx = torch.argmin(diffs)
        if diffs[idx].item() > tol:
            raise RuntimeError(
                f"Requested tau={float(v):.12f} not found on decoder grid within tolerance {tol:.1e}."
            )
        idx_list.append(int(idx.item()))
    return idx_list

def martingale_diagnostics_naive(
    model,
    z_paths,
    discount_paths,
    times,
    maturity_dates,
    out_dir,
    suffix,
    g0_floor,
    decoder_tau_grid,
    martingale_tol=0.02,
    martingale_log_every_combo=1,
    martingale_log_every_paths=100,
):
    if len(maturity_dates) == 0:
        print("No martingale dates requested; skipping discounted-bond martingale diagnostics.")
        return pd.DataFrame()

    maturity_dates = sorted(float(u) for u in maturity_dates)
    if all(u <= 0 for u in maturity_dates):
        print("All martingale dates are non-positive; skipping diagnostics.")
        return pd.DataFrame()

    print("\n" + "=" * 60)
    print("DIAGNOSTIC: Discounted-bond martingale check (exact endpoint on dense tau grid)")
    print("=" * 60)

    device = z_paths.device
    dtype = z_paths.dtype
    out_rows = []

    n_paths_local, n_times, _ = z_paths.shape

    # ------------------------------------------------------
    # Build all valid jobs first: (t_idx, t_now, U)
    # ------------------------------------------------------
    valid_jobs = []
    for t_idx, t_now in enumerate(times):
        t_now = float(t_now)
        for u in maturity_dates:
            if u > t_now + 1e-12:
                valid_jobs.append((t_idx, t_now, float(u)))

    total_jobs = len(valid_jobs)
    print(
        f"[MART] Starting martingale diagnostics with "
        f"{n_times} time points, {len(maturity_dates)} requested maturities, "
        f"{n_paths_local} paths, total jobs={total_jobs}"
    )

    t_mart0 = time.time()

    # ------------------------------------------------------
    # Precompute initial values once per maturity U
    # ------------------------------------------------------
    initial_value_by_U = {}
    print("[MART] Precomputing initial discounted bond values at t=0 ...")

    for u in maturity_dates:
        if u <= 0:
            initial_value_by_U[u] = np.nan
            continue

        try:
            tau_grid_init = build_tau_grid_to_maturity(
                decoder_tau_grid=decoder_tau_grid,
                tau_end=float(u),
                device=device,
                dtype=dtype,
            )
            tau_idx_init = get_grid_indices_for_values(
                tau_grid_init,
                torch.tensor([float(u)], device=device, dtype=dtype),
            )[0]

            P0_full, _, _, _, _, _, _, _ = decode_from_latent_script(
                model,
                z_paths[:1, 0, :],
                tau_grid_init,
                G_floor=g0_floor,
                check_short_rate=False,
            )
            initial_val = float(P0_full[0, tau_idx_init].item())
        except RuntimeError as e:
            print(f"[MART][WARN] Initial value failed for U={u:.2f}: {e}")
            initial_val = np.nan

        initial_value_by_U[u] = initial_val
        print(f"[MART] U={u:.2f} initial discounted bond value = {initial_val:.8f}")

    # ------------------------------------------------------
    # Main loop over (time, maturity) combinations
    # ------------------------------------------------------
    for job_idx, (t_idx, t_now, u) in enumerate(valid_jobs, start=1):
        tau_remaining = float(u - t_now)
        combo_start = time.time()

        if martingale_log_every_combo > 0 and (
            job_idx == 1 or
            job_idx == total_jobs or
            (job_idx % martingale_log_every_combo == 0)
        ):
            elapsed = time.time() - t_mart0
            print(
                f"[MART] job {job_idx}/{total_jobs} | "
                f"t={t_now:.3f} | U={u:.2f} | tau={tau_remaining:.3f} | "
                f"elapsed={elapsed:.1f}s"
            )

        tau_grid = build_tau_grid_to_maturity(
            decoder_tau_grid=decoder_tau_grid,
            tau_end=tau_remaining,
            device=device,
            dtype=dtype,
        )
        tau_idx = get_grid_indices_for_values(
            tau_grid,
            torch.tensor([tau_remaining], device=device, dtype=dtype),
        )[0]

        vals = []
        n_fail = 0

        for p in range(n_paths_local):
            if martingale_log_every_paths and (
                (p + 1) == 1 or
                (p + 1) == n_paths_local or
                ((p + 1) % martingale_log_every_paths == 0)
            ):
                print(
                    f"[MART]   path {p+1:4d}/{n_paths_local} | "
                    f"t={t_now:.3f} | U={u:.2f} | "
                    f"valid_so_far={len(vals)} | failed_so_far={n_fail}"
                )

            try:
                P_full, _, _, _, _, _, _, _ = decode_from_latent_script(
                    model,
                    z_paths[p: p + 1, t_idx, :],
                    tau_grid,
                    G_floor=g0_floor,
                    check_short_rate=False,
                )
                disc_val = discount_paths[p, t_idx] * P_full[0, tau_idx]
                vals.append(float(disc_val.item()))
            except RuntimeError:
                n_fail += 1

        initial_val = initial_value_by_U.get(u, np.nan)

        if len(vals) == 0:
            mean_val = np.nan
            std_val = np.nan
            sem_val = np.nan
            rel_err = np.nan
            n_valid = 0
        else:
            arr = np.asarray(vals, dtype=float)
            mean_val = float(arr.mean())
            std_val = float(arr.std(ddof=0))
            sem_val = float(std_val / math.sqrt(len(arr)))
            rel_err = (
                abs(mean_val - initial_val) / max(abs(initial_val), 1e-12)
                if np.isfinite(initial_val)
                else np.nan
            )
            n_valid = len(vals)

        out_rows.append(
            {
                "time": t_now,
                "U": float(u),
                "tau_remaining": tau_remaining,
                "disc_bond_mean": mean_val,
                "disc_bond_std": std_val,
                "disc_bond_sem": sem_val,
                "initial_disc_bond_value": initial_val,
                "relative_mean_error": rel_err,
                "n_valid_paths": n_valid,
                "frac_valid_paths": n_valid / n_paths_local,
                "n_failed_paths": n_fail,
            }
        )

        combo_elapsed = time.time() - combo_start
        total_elapsed = time.time() - t_mart0
        print(
            f"[MART] done job {job_idx}/{total_jobs} | "
            f"t={t_now:.3f} | U={u:.2f} | "
            f"valid={n_valid}/{n_paths_local} | failed={n_fail} | "
            f"rel_err={rel_err:.3%} | "
            f"combo_time={combo_elapsed:.1f}s | total_elapsed={total_elapsed:.1f}s"
        )

    mart_df = pd.DataFrame(out_rows)
    mart_csv_path = os.path.join(out_dir, f"martingale_diagnostics_{suffix}.csv")
    mart_df.to_csv(mart_csv_path, index=False)
    print(f"Saved martingale diagnostics to {mart_csv_path}")

    if not mart_df.empty:
        print("\n[MART] Summary by maturity:")
        for u in sorted(mart_df["U"].unique()):
            sub = mart_df[mart_df["U"] == u]
            finite_errs = sub["relative_mean_error"].replace([np.inf, -np.inf], np.nan).dropna()
            if len(finite_errs) == 0:
                print(f"  U={u:.2f}: no valid diagnostic points")
                continue
            max_err = float(finite_errs.max())
            mean_err = float(finite_errs.mean())
            print(f"  U={u:.2f}: mean relative error = {mean_err:.3%}, max relative error = {max_err:.3%}")
            if max_err > martingale_tol:
                warnings.warn(
                    f"Discounted-bond martingale diagnostic exceeded tolerance for U={u:.2f}: "
                    f"max relative mean error {max_err:.3%} > tol {martingale_tol:.3%}",
                    RuntimeWarning,
                )

    total_time = time.time() - t_mart0
    print(f"[MART] Finished martingale diagnostics in {total_time:.1f}s")
    print("=" * 60 + "\n")
    return mart_df

# ==========================================================
# Plot helpers
# ==========================================================
def _save_close(fig, path, dpi=200, show=False):
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Saved plot to {path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def _nearest_values(base_values, requested_values):
    base_values = np.asarray(base_values, dtype=float)
    out = []
    for v in requested_values:
        out.append(base_values[np.argmin(np.abs(base_values - v))])
    return list(dict.fromkeys([float(x) for x in out]))


def plot_g0_diagnostics(g0_df, times, out_dir, suffix, dpi=200, show=False):
    if g0_df.empty:
        return
    t = np.asarray(times)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(t, g0_df["absG0_min"].values, label="min |G0|")
    ax.plot(t, g0_df["absG0_1pct"].values, label="1% quantile")
    ax.plot(t, g0_df["absG0_5pct"].values, label="5% quantile")
    ax.set_xlabel("time")
    ax.set_ylabel("|G(z_t,0)|")
    ax.set_title("Simulated-path G(z,0) diagnostics")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save_close(fig, os.path.join(out_dir, f"plot_g0_diagnostics_{suffix}.png"), dpi=dpi, show=show)


def plot_decoder_diagnostics(decoder_diag_df, out_dir, suffix, dpi=200, show=False):
    if decoder_diag_df.empty:
        return

    t = decoder_diag_df["time"].to_numpy()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(t, decoder_diag_df["frac_valid_decode"].to_numpy(), label="frac valid decode")
    ax.plot(t, decoder_diag_df["frac_mahal_gt_threshold"].to_numpy(), label="frac mahal > threshold")
    ax.set_xlabel("time")
    ax.set_ylabel("fraction")
    ax.set_title("Decoder success and region diagnostics")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save_close(fig, os.path.join(out_dir, f"plot_decoder_validity_{suffix}.png"), dpi=dpi, show=show)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(t, decoder_diag_df["min_absG0_all_paths"].to_numpy(), label="min |G0| all paths")
    valid_col = "decoder_min_G_abs0_valid_paths"
    if valid_col in decoder_diag_df.columns:
        ax.plot(t, decoder_diag_df[valid_col].to_numpy(), label="min |G0| valid paths")
    ax.set_xlabel("time")
    ax.set_ylabel("|G0|")
    ax.set_title("Decoder G(z,0) safety diagnostics")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save_close(fig, os.path.join(out_dir, f"plot_decoder_g0_{suffix}.png"), dpi=dpi, show=show)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(t, decoder_diag_df["mean_mahal_dist"].to_numpy(), label="mean Mahalanobis")
    ax.plot(t, decoder_diag_df["max_mahal_dist"].to_numpy(), label="max Mahalanobis")
    ax.set_xlabel("time")
    ax.set_ylabel("distance")
    ax.set_title("Mahalanobis distance diagnostics")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save_close(fig, os.path.join(out_dir, f"plot_mahalanobis_{suffix}.png"), dpi=dpi, show=show)


def plot_swap_curve_snapshots(swap_df, tenors, requested_times, out_dir, suffix, dpi=200, show=False):
    if swap_df.empty:
        return

    swap_cols = [f"swap_{tenor_label(t)}" for t in tenors]
    all_times = np.sort(swap_df["time"].unique())
    plot_times = _nearest_values(all_times, requested_times)
    tenor_vals = np.asarray(tenors, dtype=float)

    fig, ax = plt.subplots(figsize=(8, 5))
    for t_sel in plot_times:
        sub = swap_df[(swap_df["time"] == t_sel) & (swap_df["valid_decode"])]
        if sub.empty:
            continue
        vals = sub[swap_cols].to_numpy(dtype=float)
        median = np.nanmedian(vals, axis=0)
        lo = np.nanquantile(vals, 0.05, axis=0)
        hi = np.nanquantile(vals, 0.95, axis=0)
        ax.plot(tenor_vals, median, label=f"t={t_sel:g} median")
        ax.fill_between(tenor_vals, lo, hi, alpha=0.12)

    ax.set_xlabel("tenor")
    ax.set_ylabel("swap rate")
    ax.set_title("Swap-curve snapshots across simulated paths")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save_close(fig, os.path.join(out_dir, f"plot_swap_snapshots_{suffix}.png"), dpi=dpi, show=show)


def plot_selected_tenor_timeseries(swap_df, selected_tenors, out_dir, suffix, dpi=200, show=False):
    if swap_df.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    times = np.sort(swap_df["time"].unique())

    for ten in selected_tenors:
        col = f"swap_{tenor_label(ten)}"
        if col not in swap_df.columns:
            continue
        med = []
        lo = []
        hi = []
        for t in times:
            vals = swap_df.loc[(swap_df["time"] == t) & (swap_df["valid_decode"]), col].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                med.append(np.nan)
                lo.append(np.nan)
                hi.append(np.nan)
            else:
                med.append(np.nanmedian(vals))
                lo.append(np.nanquantile(vals, 0.05))
                hi.append(np.nanquantile(vals, 0.95))
        med = np.asarray(med)
        lo = np.asarray(lo)
        hi = np.asarray(hi)
        ax.plot(times, med, label=tenor_label(ten))
        ax.fill_between(times, lo, hi, alpha=0.12)

    ax.set_xlabel("time")
    ax.set_ylabel("swap rate")
    ax.set_title("Selected swap tenors through time")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save_close(fig, os.path.join(out_dir, f"plot_swap_tenor_timeseries_{suffix}.png"), dpi=dpi, show=show)


def plot_martingale_diagnostics(mart_df, out_dir, suffix, dpi=200, show=False):
    if mart_df.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for U in sorted(mart_df["U"].dropna().unique()):
        sub = mart_df[mart_df["U"] == U].sort_values("time")
        ax.plot(sub["time"], sub["relative_mean_error"], label=f"U={U:g}")
    ax.set_xlabel("time")
    ax.set_ylabel("relative mean error")
    ax.set_title("Discounted-bond martingale diagnostic")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save_close(fig, os.path.join(out_dir, f"plot_martingale_error_{suffix}.png"), dpi=dpi, show=show)

    fig, ax = plt.subplots(figsize=(8, 5))
    for U in sorted(mart_df["U"].dropna().unique()):
        sub = mart_df[mart_df["U"] == U].sort_values("time")
        ax.plot(sub["time"], sub["disc_bond_mean"], label=f"U={U:g} mean")
        ax.plot(sub["time"], sub["initial_disc_bond_value"], linestyle="--", label=f"U={U:g} initial")
    ax.set_xlabel("time")
    ax.set_ylabel("discounted bond value")
    ax.set_title("Discounted-bond mean vs initial value")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)
    _save_close(fig, os.path.join(out_dir, f"plot_martingale_levels_{suffix}.png"), dpi=dpi, show=show)


# ==========================================================
# Main entry point
# ==========================================================
def run_all_diagnostics(
    bundle_path,
    device=None,
    use_saved_metadata=True,
    use="bbg",
    epochs=20,
    latent_dim=2,
    ccy_filter="",
    max_mahal=4.0,
    g0_floor=1e-5,
    martingale_dates=(5, 10, 20, 30),
    martingale_tol=0.02,
    plot_curve_times=(0, 0.5, 1.0, 2.0),
    plot_tenors=(1, 5, 10, 30),
    plot_dpi=200,
    show_plots=False,
    sigma_init=DEFAULT_SIGMA_INIT,
    k_drift_scale_init=DEFAULT_K_DRIFT_SCALE_INIT,
    k_z_center_init=DEFAULT_K_Z_CENTER_INIT,
    k_learn_center=DEFAULT_K_LEARN_CENTER,
    martingale_log_every_combo=1,
    martingale_log_every_paths=100,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    safe_switch_backend(show_plots)

    bundle = load_simulation_bundle(bundle_path, device=device)
    metadata = bundle.get("metadata", {})

    checkpoint_path = None

    if use_saved_metadata and metadata:
        use = metadata.get("use", use)
        epochs = int(metadata.get("epochs", epochs))
        latent_dim = int(metadata.get("latent_dim", latent_dim))
        ccy_filter = metadata.get("ccy_filter", ccy_filter)
        checkpoint_path = metadata.get("checkpoint_path", None)

    print("\nLoaded simulation bundle metadata:")
    for k, v in metadata.items():
        print(f"  {k}: {v}")

    model, resolved_checkpoint_path = load_and_setup_model(
        device=device,
        use=use,
        latent_dim=latent_dim,
        epochs=epochs,
        checkpoint_path=checkpoint_path,
        sigma_init=sigma_init,
        k_drift_scale_init=k_drift_scale_init,
        k_z_center_init=k_z_center_init,
        k_learn_center=k_learn_center,
    )

    z_paths = bundle["z_paths"]
    r_paths = bundle["r_paths"]
    mu_paths = bundle["mu_paths"]
    L_paths = bundle["L_paths"]
    discount_paths = bundle["discount_paths"]
    times = bundle["times"]
    z_train_mean = bundle["z_train_mean"]
    z_train_cov = bundle["z_train_cov"]
    tenors = bundle["tenors"]
    decoder_tau_grid = bundle["decoder_tau_grid"]
    annual_indices = bundle["annual_indices"]

    out_dir = os.path.dirname(bundle_path)
    suffix = make_output_suffix(bundle_path, {**metadata, "checkpoint_path": resolved_checkpoint_path})

    _, X_tensor, _, _ = load_data_for_training_cloud(use=use, ccy_filter=ccy_filter)
    compute_latent_statistics(model, X_tensor, device, latent_dim)
    diagnose_G0_on_training_cloud(model, X_tensor, device)

    analyze_paths(z_paths, r_paths, mu_paths, L_paths, latent_dim)

    g0_sim_df = diagnose_G0_on_simulated_paths(model, z_paths)
    g0_csv_path = os.path.join(out_dir, f"simulated_G0_diagnostics_{suffix}.csv")
    g0_sim_df.to_csv(g0_csv_path, index=False)
    print(f"Saved simulated G0 diagnostics to {g0_csv_path}")

    swap_df, latent_df, mahal_df, decoder_diag_df, decoder_failures_df = decode_and_save_results_naive(
        model=model,
        z_paths=z_paths,
        r_paths=r_paths,
        z_train_mean=z_train_mean,
        z_train_cov=z_train_cov,
        decoder_tau_grid=decoder_tau_grid,
        annual_indices=annual_indices,
        device=device,
        times=times,
        tenors=tenors,
        suffix=suffix,
        out_dir=out_dir,
        max_mahal=max_mahal,
        g0_floor=g0_floor,
    )

    mart_df = martingale_diagnostics_naive(
        model=model,
        z_paths=z_paths,
        discount_paths=discount_paths,
        times=times,
        maturity_dates=list(martingale_dates),
        out_dir=out_dir,
        suffix=suffix,
        g0_floor=g0_floor,
        decoder_tau_grid=decoder_tau_grid,
        martingale_tol=martingale_tol,
        martingale_log_every_combo=martingale_log_every_combo,
        martingale_log_every_paths=martingale_log_every_paths,
    )

    plot_g0_diagnostics(g0_sim_df, times, out_dir, suffix, dpi=plot_dpi, show=show_plots)
    plot_decoder_diagnostics(decoder_diag_df, out_dir, suffix, dpi=plot_dpi, show=show_plots)
    plot_swap_curve_snapshots(
        swap_df,
        tenors,
        requested_times=list(plot_curve_times),
        out_dir=out_dir,
        suffix=suffix,
        dpi=plot_dpi,
        show=show_plots,
    )
    plot_selected_tenor_timeseries(
        swap_df,
        selected_tenors=list(plot_tenors),
        out_dir=out_dir,
        suffix=suffix,
        dpi=plot_dpi,
        show=show_plots,
    )
    plot_martingale_diagnostics(mart_df, out_dir, suffix, dpi=plot_dpi, show=show_plots)

    summary = {
        "resolved_checkpoint_path": resolved_checkpoint_path,
        "bundle_path": bundle_path,
        "suffix": suffix,
        "n_paths": int(z_paths.shape[0]),
        "n_times": int(z_paths.shape[1]),
        "latent_dim": int(z_paths.shape[2]),
        "max_mahal_overall": float(mahal_df["mahal_dist"].max()) if not mahal_df.empty else np.nan,
        "min_absG0_overall": float(mahal_df["absG0"].min()) if not mahal_df.empty else np.nan,
        "mean_frac_valid_decode": float(decoder_diag_df["frac_valid_decode"].mean()) if not decoder_diag_df.empty else np.nan,
    }
    summary_df = pd.DataFrame([summary])
    summary_csv_path = os.path.join(out_dir, f"diagnostic_summary_{suffix}.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Saved diagnostic summary to {summary_csv_path}")

    return {
        "bundle": bundle,
        "model": model,
        "g0_sim_df": g0_sim_df,
        "swap_df": swap_df,
        "latent_df": latent_df,
        "mahal_df": mahal_df,
        "decoder_diag_df": decoder_diag_df,
        "decoder_failures_df": decoder_failures_df,
        "mart_df": mart_df,
        "out_dir": out_dir,
        "suffix": suffix,
        "resolved_checkpoint_path": resolved_checkpoint_path,
        "summary_df": summary_df,
    }


# ==========================================================
# Example lines to run
# ==========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bundle_path = r"C:\Users\Bruger\PycharmProjects\MasterThesis\Figures\Pricing\simulations\simulation_bundle_bbg_dim2_ep200_paths500_steps24_seed1234_euler_full_diff1.pt"

diag_out = run_all_diagnostics(
    bundle_path=bundle_path,
    device=device,
    use_saved_metadata=True,
    max_mahal=4.0,
    g0_floor=1e-5,
    martingale_dates=(5, 10, 20, 30),
    martingale_tol=0.02,
    plot_curve_times=(0, 0.5, 1.0, 2.0),
    plot_tenors=(1, 5, 10, 30),
    plot_dpi=200,
    show_plots=False,

    sigma_init=0.0075,
    k_drift_scale_init=0.25,
    k_z_center_init=np.array([-0.04631, 0.04223], dtype=np.float32),
    k_learn_center=False,

    # logging controls
    martingale_log_every_combo=1,     # print every (t,U) job
    martingale_log_every_paths=100,   # print every 100 paths inside each job
)