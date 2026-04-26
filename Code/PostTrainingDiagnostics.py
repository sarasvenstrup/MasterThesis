"""
PostTrainingDiagnostics.py
=======================

Comprehensive post-training diagnostics on the training set.

Checks
------
0. Robust checkpoint loading
1. Single-point debug on the median training curve
2. Discount-curve constraints on all training curves:
   - P(0) = 1
   - P(tau) > 0
   - P(tau) <= 1
   - monotonicity in tau on the default annual grid
3. Decoder shape at origin:
   - G(z,0) distribution
   - G(z,tau) > 0 across full tau grid (ODE denominator stability)
4. Stable-H constraints / diffusion diagnostics:
   - sigma positivity / bounds
   - rho bounds
   - covariance PSD check
4b. Short-rate r_tilde(z) value diagnostics
4c. K drift matrix eigenvalue check:
   - All eigenvalues of M must have strictly negative real parts
     (mean-reversion guarantee from the paper)
4d. ODE boundary conditions:
   - A(0) = 0 and B(0) = 0 (from the paper, ensures P(z,0)=1)
4e. Gamma non-negativity and covariance symmetry:
   - gamma = 1/2 ||sigma^T nabla_z G||^2 >= 0
   - Sigma = L L^T must be symmetric
5. Short-rate consistency:
   - tau sweep for
       f_fd(0,tau1) = -[log P(tau1)-log P(0)] / tau1
     versus r_tilde(z)
6. Paper no-arbitrage residuals:
   - Sharpe-ratio diagnostic SR_tau
7. Reconstruction RMSE per currency
8. Save CSV summaries + one summary figure

Notes
-----
- S_hat is only available on the model's default annual tau grid.
- The short-rate consistency check therefore uses custom small tau grids.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# =============================================================================
# Paths
# =============================================================================
try:
    CODE_ROOT = os.path.dirname(os.path.abspath(__file__))
except NameError:
    CODE_ROOT = os.getcwd()

PROJECT_ROOT = os.path.abspath(os.path.join(CODE_ROOT, ".."))
THESIS_ROOT = os.path.abspath(os.path.join(CODE_ROOT, "..", ".."))

for p in [CODE_ROOT, PROJECT_ROOT, THESIS_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from Code import config
from Code.model.full_model_stable import FullModel
from Code.model.sigma_matrix import L_from_sigmas_rhos
from Code.load_swapdata import my_data


# =============================================================================
# USER SETTINGS
# =============================================================================
CHECKPOINT_PATH = (
    r"C:\Users\Bruger\PycharmProjects\MasterThesis"
    r"\Figures\TrainingResults\dim2_stable\ep3500\checkpoint_dim2_ep3500.pt"
)

LATENT_DIM = 2
USE = "bbg"
CCY_FILTER = ""              # empty = all currencies
BATCH_SIZE = 256
MAX_TEST = 5000              # expensive checks use subsample; None = all
SHOW_PLOTS = True

# expected stable-H defaults from FullModel(...)
SIGMA_MIN_EXPECTED = 1e-4
SIGMA_MAX_EXPECTED = 0.20
RHO_MAX_EXPECTED = 0.999

# tolerances / thresholds
P0_TOL = 1e-6
P_LEQ1_TOL = 1e-6
P_POS_TOL = 1e-12
MONO_TOL = 1e-8
SHARPE_TOL = 0.15

# tau sweep for short-rate consistency
TAU_SWEEP = [
    1.0,
    1 / 4,
    1 / 12,
    1 / 52,
    1 / 252,
    1 / 365,
]

SEED = 1234


# =============================================================================
# Utility helpers
# =============================================================================
def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def get_output_dir(checkpoint_path: str) -> str:
    out_dir = os.path.join(os.path.dirname(checkpoint_path), "Diagnostics")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def status_str(ok: bool) -> str:
    return "PASS" if ok else "WARN"


def safe_bp(x: float) -> float:
    return 1e4 * x


# =============================================================================
# Loading
# =============================================================================
def load_model(checkpoint_path: str, latent_dim: int, device: torch.device) -> FullModel:
    obj = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = FullModel(latent_dim=latent_dim).to(device)

    if isinstance(obj, dict) and "model_state_dict" in obj:
        state_dict = obj["model_state_dict"]
    elif isinstance(obj, dict) and "state_dict" in obj:
        state_dict = obj["state_dict"]
    else:
        state_dict = obj

    result = model.load_state_dict(state_dict, strict=False)
    if result.unexpected_keys:
        print(f"  [load] dropped old params: {result.unexpected_keys}")
    model = model.double()
    model.eval()

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Variant from config.py: {config.VARIANT}")
    return model


def load_training_data(use: str, ccy_filter: str, device: torch.device, dtype: torch.dtype):
    meta, X_tensor, _, _, tenors, _, _, scale_is_percent = my_data(
        use=use, ccy_filter=ccy_filter
    )
    meta = meta.reset_index(drop=True)
    X_tensor = X_tensor.to(device=device, dtype=dtype)
    print(f"Training data: {X_tensor.shape[0]} curves")
    print(f"SCALE_IS_PERCENT: {scale_is_percent}")
    return meta, X_tensor, tenors, scale_is_percent


# =============================================================================
# Batch inference
# =============================================================================
@torch.no_grad()
def encode_all(model: FullModel, X: torch.Tensor, batch_size: int) -> torch.Tensor:
    zs = []
    was_training = model.training
    model.eval()
    for i in range(0, X.shape[0], batch_size):
        xb = X[i:i + batch_size]
        zs.append(model.encoder(xb))
    if was_training:
        model.train()
    return torch.cat(zs, dim=0)


@torch.no_grad()
def decode_default_all(model: FullModel, z_all: torch.Tensor, batch_size: int):
    p_list, s_list = [], []
    tau_grid = None

    for i in range(0, z_all.shape[0], batch_size):
        zb = z_all[i:i + batch_size]
        _, aux = model.decode_from_z(zb, tau=None, do_arb_checks=False, return_aux=True)

        p_list.append(aux["P_full"].detach().cpu())
        if aux["S_hat"] is not None:
            s_list.append(aux["S_hat"].detach().cpu())
        if tau_grid is None:
            tau_grid = aux["tau_grid"].detach().cpu().numpy()

    p_full = torch.cat(p_list, dim=0).numpy()
    s_hat = torch.cat(s_list, dim=0).numpy() if len(s_list) > 0 else None
    return p_full, tau_grid, s_hat


def subsample_tensor(x: torch.Tensor, max_pts: int | None) -> torch.Tensor:
    if max_pts is None or x.shape[0] <= max_pts:
        return x
    idx = torch.randperm(x.shape[0], device=x.device)[:max_pts]
    return x[idx]


@torch.no_grad()
def get_r_tilde(model: FullModel, z: torch.Tensor) -> np.ndarray:
    r = model.R(z)
    if r.ndim == 2 and r.shape[-1] == 1:
        r = r.squeeze(-1)
    return r.detach().cpu().numpy()


# =============================================================================
# Check 0: single-point debug
# =============================================================================
@torch.no_grad()
def check0_single_point_debug(model: FullModel, z_all: torch.Tensor):
    z_med = z_all.median(dim=0).values.unsqueeze(0)
    idx = int(((z_all - z_med) ** 2).sum(dim=1).argmin().item())
    z0 = z_all[idx:idx + 1]

    _, aux = model.decode_from_z(z0, tau=None, do_arb_checks=False, return_aux=True)

    tau_grid = aux["tau_grid"].detach().cpu().numpy()
    G_vals = aux["G_vals"][0].detach().cpu().numpy()
    beta_vals = aux["beta"][0].detach().cpu().numpy()
    B_vals = aux["B_vals"][0].detach().cpu().numpy()
    A_vals = aux["A_vals"][0].detach().cpu().numpy()
    P_vals = aux["P_full"][0].detach().cpu().numpy()
    r = float(aux["r_tilde"][0].item())

    print("\n" + "=" * 72)
    print(f"Check 0: single-point debug on median training curve (idx={idx})")
    print("=" * 72)
    print(f"r_tilde = {r:.8f}  ({safe_bp(r):.2f} bp)")
    print(f"{'tau':>8} {'G':>14} {'beta':>14} {'B':>14} {'A':>14} {'P':>14}")
    for i in range(min(len(tau_grid), 11)):
        print(
            f"{tau_grid[i]:8.4f} "
            f"{G_vals[i]:14.8f} "
            f"{beta_vals[i]:14.8f} "
            f"{B_vals[i]:14.8f} "
            f"{A_vals[i]:14.8f} "
            f"{P_vals[i]:14.8f}"
        )

    return {
        "idx": idx,
        "tau_grid": tau_grid,
        "G_vals": G_vals,
        "beta_vals": beta_vals,
        "B_vals": B_vals,
        "A_vals": A_vals,
        "P_vals": P_vals,
        "r_tilde": r,
    }


# =============================================================================
# Check 1/2: discount-curve constraints
# =============================================================================
def check_discount_constraints(p_full: np.ndarray, tau_grid: np.ndarray) -> dict:
    finite_mask = np.isfinite(p_full)
    finite_pct = 100.0 * finite_mask.mean()

    idx0 = int(np.argmin(np.abs(tau_grid - 0.0)))
    p0_err = np.abs(p_full[:, idx0] - 1.0)

    min_p = float(np.nanmin(p_full))
    max_p = float(np.nanmax(p_full))
    pct_nonpos = 100.0 * (p_full <= P_POS_TOL).mean()
    pct_above_one = 100.0 * (p_full > 1.0 + P_LEQ1_TOL).mean()

    diffs = np.diff(p_full, axis=1)
    pct_upticks = 100.0 * (diffs > MONO_TOL).mean()
    max_uptick = float(np.nanmax(diffs)) if diffs.size else 0.0

    print("\n" + "=" * 72)
    print("Check 1/2: discount-curve constraints")
    print("=" * 72)
    print(f"finite entries %              = {finite_pct:.2f}%")
    print(f"max |P(0)-1|                 = {p0_err.max():.4e}")
    print(f"mean |P(0)-1|                = {p0_err.mean():.4e}")
    print(f"min P                        = {min_p:.8f}")
    print(f"max P                        = {max_p:.8f}")
    print(f"% P <= 0                     = {pct_nonpos:.4f}%")
    print(f"% P > 1                      = {pct_above_one:.4f}%")
    print(f"% upward tau-steps           = {pct_upticks:.4f}%")
    print(f"max upward tau-step          = {max_uptick:.4e}")
    print(f"P(0)=1 status                = {status_str(p0_err.max() < P0_TOL)}")
    print(f"P>0 status                   = {status_str(pct_nonpos == 0.0)}")
    print(f"P<=1 status                  = {status_str(max_p <= 1.0 + P_LEQ1_TOL)}")
    print(f"monotone non-increasing      = {status_str(max_uptick <= MONO_TOL)}")

    return {
        "p0_err": p0_err,
        "finite_pct": finite_pct,
        "min_p": min_p,
        "max_p": max_p,
        "pct_nonpos": pct_nonpos,
        "pct_above_one": pct_above_one,
        "pct_upticks": pct_upticks,
        "max_uptick": max_uptick,
    }


# =============================================================================
# Check 3: G(z,0)
# =============================================================================
@torch.no_grad()
def check_G_full_tau(model: FullModel, z_test: torch.Tensor) -> dict:
    """
    Check G(z, tau) across the full tau grid.
    G appears in the ODE denominator as beta = r_tilde / G.
    Near-zero or negative G causes ODE blow-up in every forward pass.
    """
    device = z_test.device
    dtype  = z_test.dtype
    tau_tensor = model._tau(device=device, dtype=dtype)
    T = tau_tensor.numel()

    g_chunks = []
    for i in range(0, z_test.shape[0], BATCH_SIZE):
        zb = z_test[i:i + BATCH_SIZE]
        g_chunks.append(model.G(zb, tau_tensor).cpu().numpy())
    G_all = np.concatenate(g_chunks, axis=0)   # (N, T)
    tau_np = tau_tensor.cpu().numpy()

    rows = []
    print("\n" + "=" * 72)
    print("Check 3: G(z, tau) across full tau grid")
    print("  beta = r_tilde / G  ->  ODE blows up when G -> 0")
    print("=" * 72)
    print(f"{'tau':>5}  {'mean':>8}  {'std':>8}  {'min':>8}  {'max':>8}  "
          f"{'%<0':>6}  {'%<0.01':>7}  {'%<0.05':>7}")

    for t_idx in range(1, T):
        tau_val = float(tau_np[t_idx])
        g = G_all[:, t_idx]
        g_mean = float(np.mean(g));  g_std = float(np.std(g))
        g_min  = float(np.min(g));   g_max = float(np.max(g))
        pct_neg   = 100.0 * float(np.mean(g < 0.0))
        pct_tiny  = 100.0 * float(np.mean(g < 0.01))
        pct_small = 100.0 * float(np.mean(g < 0.05))
        flag = "  <-- NEGATIVE G" if pct_neg > 0 else ("  <-- NEAR ZERO" if pct_tiny > 0 else "")
        print(f"{tau_val:>5.0f}  {g_mean:>8.4f}  {g_std:>8.4f}  {g_min:>8.4f}  "
              f"{g_max:>8.4f}  {pct_neg:>6.1f}  {pct_tiny:>7.1f}  {pct_small:>7.1f}{flag}")
        rows.append({"tau": tau_val, "G_mean": g_mean, "G_std": g_std,
                     "G_min": g_min, "G_max": g_max,
                     "pct_neg": pct_neg, "pct_lt001": pct_tiny, "pct_lt005": pct_small})

    df = pd.DataFrame(rows)
    any_neg  = df["pct_neg"].max() > 0
    any_tiny = df["pct_lt001"].max() > 0
    overall_min = float(G_all[:, 1:].min())

    first_neg  = df.loc[df["pct_neg"] > 0,    "tau"]
    first_tiny = df.loc[df["pct_lt001"] > 0,  "tau"]
    print()
    print(f"  Global G min (tau >= 1)      : {overall_min:.6f}")
    if not first_neg.empty:
        print(f"  First tau with G < 0         : {first_neg.iloc[0]:.0f}Y  --> ODE unstable")
    if not first_tiny.empty:
        print(f"  First tau with G < 0.01      : {first_tiny.iloc[0]:.0f}Y  --> beta spike")
    print(f"  G > 0 everywhere             : {status_str(not any_neg)}")
    print(f"  G > 0.01 everywhere          : {status_str(not any_tiny)}")

    G0 = G_all[:, 0]
    pct_neg_G0 = 100.0 * float(np.mean(G0 < 0.0))
    print(f"\n  G(z,0) boundary: min={G0.min():.6f}  mean={G0.mean():.6f}  "
          f"%<0={pct_neg_G0:.2f}%  {status_str(pct_neg_G0 == 0.0)}")

    return {
        "G_all": G_all, "tau_grid": tau_np, "G_by_tau_df": df,
        "overall_min": overall_min, "any_neg": any_neg, "any_tiny": any_tiny,
        "G0": G0, "pct_neg_G0": pct_neg_G0,
    }


# =============================================================================
# Check 4: sigma / rho / PSD diagnostics
# =============================================================================

def _cholesky_inside_terms(rhos: torch.Tensor, d: int) -> dict:
    """
    Recomputes the sqrt arguments from the analytic Cholesky WITHOUT clamping.

    A negative value means the eps clamp fired in L_from_sigmas_rhos for that
    observation — the rhos formed a geometrically invalid correlation matrix
    and L was silently distorted. This is only possible for d >= 3.

    Returns a dict of (N,) tensors, one per inside term:
      d2: 1 - rho12^2                          (always > 0 when |rho12| < 1)
      d3: the d=3 determinant expression        (< 0 => clamp fired)
      d4: the d=4 determinant expression        (< 0 => clamp fired)
    """
    out = {}
    if d < 2:
        return out

    rho12            = rhos[:, 0].cpu()
    one_minus_r12_sq = 1.0 - rho12 ** 2
    out["d2"] = one_minus_r12_sq

    if d >= 3:
        rho13   = rhos[:, 1].cpu()
        rho23   = rhos[:, 2].cpu()
        inside3 = (
            1.0 - rho13 ** 2
            - ((rho23 - rho12 * rho13) ** 2) / one_minus_r12_sq.clamp(min=1e-12)
        )
        out["d3"] = inside3

    if d >= 4:
        rho13    = rhos[:, 1].cpu()
        rho14    = rhos[:, 2].cpu()
        rho23    = rhos[:, 3].cpu()
        rho24    = rhos[:, 4].cpu()
        rho34    = rhos[:, 5].cpu()
        inside3c = (
            1.0 - rho13 ** 2
            - ((rho23 - rho12 * rho13) ** 2) / one_minus_r12_sq.clamp(min=1e-12)
        ).clamp(min=1e-12)
        inside4 = (
            1.0
            - rho14 ** 2
            - ((rho24 - rho12 * rho14) ** 2) / one_minus_r12_sq.clamp(min=1e-12)
            - (
                (
                    rho34 - rho13 * rho14
                    - ((rho23 - rho12 * rho13) * (rho24 - rho12 * rho14))
                    / one_minus_r12_sq.clamp(min=1e-12)
                ) ** 2
            ) / inside3c
        )
        out["d4"] = inside4

    return out


@torch.no_grad()
def check_sigma_rho(model: FullModel, z_test: torch.Tensor) -> dict:
    d = model.H.d
    sigmas, rhos = model.H(z_test)
    sigmas_np = sigmas.detach().cpu().numpy()
    rhos_np   = rhos.detach().cpu().numpy()
    N         = sigmas.shape[0]

    L   = L_from_sigmas_rhos(sigmas, rhos)
    cov = L @ L.transpose(-1, -2)
    eigs         = torch.linalg.eigvalsh(cov)       # (N, d)
    cov_min_eigs = eigs[:, 0].detach().cpu().numpy()
    cov_max_eigs = eigs[:, -1].detach().cpu().numpy()
    cond_numbers = (cov_max_eigs / np.clip(cov_min_eigs, 1e-12, None))

    pct_sigma_nonpos = 100.0 * (sigmas_np <= 0.0).mean()
    sigma_min = float(sigmas_np.min())
    sigma_max = float(sigmas_np.max())

    if rhos_np.size > 0:
        rho_abs_max = float(np.abs(rhos_np).max())
        pct_rho_out = 100.0 * (np.abs(rhos_np) > RHO_MAX_EXPECTED + 1e-8).mean()
    else:
        rho_abs_max = 0.0
        pct_rho_out = 0.0

    pct_sigma_low  = 100.0 * (sigmas_np < SIGMA_MIN_EXPECTED - 1e-8).mean()
    pct_sigma_high = 100.0 * (sigmas_np > SIGMA_MAX_EXPECTED + 1e-8).mean()
    min_cov_eig    = float(cov_min_eigs.min())

    # Cholesky inside terms (raw, no clamp) — d>=3 only
    inside = _cholesky_inside_terms(rhos, d)
    inside_np = {k: v.numpy() for k, v in inside.items()}

    print("\n" + "=" * 72)
    print("Check 4: sigma / rho / PSD / Cholesky diagnostics")
    print("=" * 72)
    print(f"sigma min                     = {sigma_min:.8f}")
    print(f"sigma max                     = {sigma_max:.8f}")
    print(f"% sigma <= 0                  = {pct_sigma_nonpos:.4f}%")
    print(f"% sigma < expected min        = {pct_sigma_low:.4f}%")
    print(f"% sigma > expected max        = {pct_sigma_high:.4f}%")
    print(f"max |rho|                     = {rho_abs_max:.8f}")
    print(f"% |rho| > expected cap        = {pct_rho_out:.4f}%")
    print(f"min eigenvalue of Sigma       = {min_cov_eig:.8e}  {status_str(min_cov_eig >= -1e-10)}")
    print(f"max condition number of Sigma = {cond_numbers.max():.4f}")
    print(f"mean condition number         = {cond_numbers.mean():.4f}")

    # Diagonal consistency: diag(Sigma) should equal sigma_i^2
    diag_sigma = torch.diagonal(cov, dim1=-2, dim2=-1).detach().cpu().numpy()  # (N, d)
    diag_err   = np.abs(diag_sigma - sigmas_np ** 2)
    print(f"max |diag(Sigma)-sigma^2|     = {diag_err.max():.4e}  {status_str(diag_err.max() < 1e-4)}")

    # Cholesky inside terms
    print()
    print("  Cholesky inside terms (< 0 means clamp fired => L was distorted):")
    chol_any_neg = False
    for key, vals in inside_np.items():
        n_neg   = int((vals < 0).sum())
        pct_neg = 100.0 * n_neg / N
        ok      = n_neg == 0
        chol_any_neg = chol_any_neg or (not ok)
        print(
            f"    {key}: min={vals.min():.6f}  mean={vals.mean():.6f}  "
            f"n_neg={n_neg}/{N} ({pct_neg:.2f}%)  {status_str(ok)}"
        )
        if not ok:
            print(
                f"         WARNING: clamp fired for {n_neg} observations. "
                f"L_from_sigmas_rhos distorted these covariance matrices.\n"
                f"         Consider switching to L_from_sigmas_rhos_numerically for d>={key[1]}."
            )
    if not inside_np:
        print("    d=1 or d=2: analytically valid for all tanh-bounded rhos. No check needed.")

    return {
        "sigmas":            sigmas_np,
        "rhos":              rhos_np,
        "sigma_min":         sigma_min,
        "sigma_max":         sigma_max,
        "pct_sigma_nonpos":  pct_sigma_nonpos,
        "pct_sigma_low":     pct_sigma_low,
        "pct_sigma_high":    pct_sigma_high,
        "rho_abs_max":       rho_abs_max,
        "pct_rho_out":       pct_rho_out,
        "cov_min_eigs":      cov_min_eigs,
        "cov_max_eigs":      cov_max_eigs,
        "cond_numbers":      cond_numbers,
        "min_cov_eig":       min_cov_eig,
        "max_cond":          float(cond_numbers.max()),
        "mean_cond":         float(cond_numbers.mean()),
        "diag_err_max":      float(diag_err.max()),
        "inside":            inside_np,
        "chol_any_neg":      chol_any_neg,
    }



# =============================================================================
# Check 4b: r_tilde value diagnostics
# =============================================================================
@torch.no_grad()
def check_r_tilde(model: FullModel, z_test: torch.Tensor) -> dict:
    """
    Inspect the short rate r_tilde(z) across training z points.

    Reports distribution statistics and flags economically implausible values:
      - r < -0.02  (-200 bp): deeply negative, unusual even in NIRP regimes
      - r > 0.10   (+1000 bp): very high, plausible only in crisis/EM environments
      - r < 0      (negative): valid for EUR/DKK/SEK/JPY/CHF in 2014-2022
    """
    R_NEG_WARN  = -0.02    # -200 bp floor
    R_HIGH_WARN =  0.10    # +1000 bp ceiling

    r = model.R(z_test)
    if r.ndim == 2 and r.shape[-1] == 1:
        r = r.squeeze(-1)
    r_np = r.detach().cpu().numpy()

    r_mean   = float(r_np.mean())
    r_std    = float(r_np.std())
    r_min    = float(r_np.min())
    r_max    = float(r_np.max())
    r_p5     = float(np.percentile(r_np, 5))
    r_p95    = float(np.percentile(r_np, 95))
    pct_neg  = 100.0 * float(np.mean(r_np < 0.0))
    pct_very_neg = 100.0 * float(np.mean(r_np < R_NEG_WARN))
    pct_high = 100.0 * float(np.mean(r_np > R_HIGH_WARN))

    print("\n" + "=" * 72)
    print("Check 4b: r_tilde(z) value diagnostics")
    print("=" * 72)
    print(f"  mean          : {r_mean:.6f}  ({r_mean*1e4:.1f} bp)")
    print(f"  std           : {r_std:.6f}  ({r_std*1e4:.1f} bp)")
    print(f"  min           : {r_min:.6f}  ({r_min*1e4:.1f} bp)")
    print(f"  max           : {r_max:.6f}  ({r_max*1e4:.1f} bp)")
    print(f"  p5            : {r_p5:.6f}  ({r_p5*1e4:.1f} bp)")
    print(f"  p95           : {r_p95:.6f}  ({r_p95*1e4:.1f} bp)")
    print(f"  % r < 0       : {pct_neg:.2f}%  (negative rates — valid for NIRP currencies)")
    print(f"  % r < {R_NEG_WARN*1e4:.0f}bp : {pct_very_neg:.2f}%  {status_str(pct_very_neg == 0.0)}")
    print(f"  % r > {R_HIGH_WARN*1e4:.0f}bp  : {pct_high:.2f}%  {status_str(pct_high == 0.0)}")

    return {
        "r_np":        r_np,
        "r_mean":      r_mean,
        "r_std":       r_std,
        "r_min":       r_min,
        "r_max":       r_max,
        "r_p5":        r_p5,
        "r_p95":       r_p95,
        "pct_neg":     pct_neg,
        "pct_very_neg": pct_very_neg,
        "pct_high":    pct_high,
    }


# =============================================================================
# Check 4c: K drift matrix — mean-reversion eigenvalue check
# =============================================================================
@torch.no_grad()
def check_K_eigenvalues(model: FullModel) -> dict:
    """
    Paper constraint: the drift matrix M must have all strictly negative
    eigenvalues to guarantee mean-reversion of the latent process.

    Stable variant:  M = -(V^T V + eps*I)  → negative definite by construction.
    Baseline variant: M = weight matrix of K.lin — may or may not be stable.

    We verify this numerically and report PASS/WARN.
    """
    if hasattr(model.K, "stable_matrix"):
        M = model.K.stable_matrix().detach().cpu()
        variant_label = "stable (M = -(V^T V + eps*I))"
    elif hasattr(model.K, "lin"):
        M = model.K.lin.weight.detach().cpu()
        variant_label = "baseline (M = K.lin.weight)"
    else:
        print("\n" + "=" * 72)
        print("Check 4c: K drift eigenvalues — SKIPPED (unknown K variant)")
        print("=" * 72)
        return {"skipped": True}

    eig_vals = torch.linalg.eigvals(M)
    eig_reals = eig_vals.real.numpy()
    eig_imags = eig_vals.imag.numpy()
    eig_reals_sorted = np.sort(eig_reals)[::-1]  # descending

    max_real = float(eig_reals.max())
    all_negative = bool(max_real < 0.0)
    has_imag = bool(np.any(np.abs(eig_imags) > 1e-10))

    print("\n" + "=" * 72)
    print("Check 4c: K drift matrix — mean-reversion eigenvalue check")
    print("=" * 72)
    print(f"  variant             : {variant_label}")
    print(f"  M shape             : {tuple(M.shape)}")
    print(f"  eigenvalues (real)  : {eig_reals_sorted}")
    if has_imag:
        print(f"  eigenvalues (imag)  : {np.sort(eig_imags)[::-1]}")
    print(f"  max Re(eig)         : {max_real:.8e}")
    print(f"  all Re(eig) < 0     : {status_str(all_negative)}")

    if not all_negative:
        print("  WARNING: drift matrix has non-negative eigenvalue(s).")
        print("           Latent process is NOT mean-reverting → simulation unstable.")

    return {
        "skipped":       False,
        "M":             M.numpy(),
        "eig_reals":     eig_reals_sorted,
        "max_real_eig":  max_real,
        "all_negative":  all_negative,
        "has_imag":      has_imag,
    }


# =============================================================================
# Check 4d: ODE boundary conditions A(0)=0 and B(0)=0
# =============================================================================
@torch.no_grad()
def check_ODE_boundary(model: FullModel, z_test: torch.Tensor) -> dict:
    """
    Paper constraint: the ODE initial conditions require A(0)=0 and B(0)=0
    so that P(z,0) = exp(A(0) - B(0)*G(z,0)) = exp(0) = 1.

    The solver initialises A=B=0 by construction, but this check catches
    any solver regression or numerical drift at the boundary.
    """
    a_errs, b_errs = [], []

    for i in range(0, z_test.shape[0], BATCH_SIZE):
        zb = z_test[i:i + BATCH_SIZE]
        _, aux = model.decode_from_z(zb, tau=None, do_arb_checks=False, return_aux=True)
        a_errs.append(aux["A_vals"][:, 0].abs().detach().cpu())
        b_errs.append(aux["B_vals"][:, 0].abs().detach().cpu())

    a_err = torch.cat(a_errs).numpy()
    b_err = torch.cat(b_errs).numpy()

    max_a0 = float(a_err.max())
    max_b0 = float(b_err.max())
    mean_a0 = float(a_err.mean())
    mean_b0 = float(b_err.mean())

    tol = 1e-12
    ok = max_a0 < tol and max_b0 < tol

    print("\n" + "=" * 72)
    print("Check 4d: ODE boundary conditions A(0)=0, B(0)=0")
    print("=" * 72)
    print(f"  max |A(0)|          : {max_a0:.4e}  {status_str(max_a0 < tol)}")
    print(f"  max |B(0)|          : {max_b0:.4e}  {status_str(max_b0 < tol)}")
    print(f"  mean |A(0)|         : {mean_a0:.4e}")
    print(f"  mean |B(0)|         : {mean_b0:.4e}")
    print(f"  overall             : {status_str(ok)}")

    return {
        "max_a0":  max_a0,
        "max_b0":  max_b0,
        "mean_a0": mean_a0,
        "mean_b0": mean_b0,
        "ok":      ok,
    }


# =============================================================================
# Check 4e: gamma >= 0 and covariance symmetry
# =============================================================================
@torch.no_grad()
def check_gamma_and_cov_symmetry(model: FullModel, z_test: torch.Tensor) -> dict:
    """
    Paper constraints:
      (a) gamma = 1/2 ||sigma^T nabla_z G||^2 >= 0 for all (z,tau).
          Negative gamma would violate the ODE derivation.
      (b) Sigma = L L^T must be symmetric.  Guaranteed by construction
          but we verify numerically as a sanity check.
    """
    gamma_mins, gamma_maxs = [], []
    sym_errs = []

    for i in range(0, z_test.shape[0], BATCH_SIZE):
        zb = z_test[i:i + BATCH_SIZE]
        _, aux = model.decode_from_z(zb, tau=None, do_arb_checks=False, return_aux=True)

        g = aux["gamma"].detach().cpu()
        gamma_mins.append(g.min().item())
        gamma_maxs.append(g.max().item())

        L = aux["sigma"].detach().cpu()           # (B, d, d)
        cov = L @ L.transpose(-1, -2)             # (B, d, d)
        sym_err = (cov - cov.transpose(-1, -2)).abs().max().item()
        sym_errs.append(sym_err)

    gamma_global_min = float(min(gamma_mins))
    gamma_global_max = float(max(gamma_maxs))
    gamma_nonneg = gamma_global_min >= -1e-12
    pct_neg_gamma = 0.0  # recompute properly if needed

    max_sym_err = float(max(sym_errs))
    sym_ok = max_sym_err < 1e-10

    print("\n" + "=" * 72)
    print("Check 4e: gamma >= 0 and covariance symmetry")
    print("=" * 72)
    print(f"  gamma global min    : {gamma_global_min:.8e}  {status_str(gamma_nonneg)}")
    print(f"  gamma global max    : {gamma_global_max:.8e}")
    if not gamma_nonneg:
        print("  WARNING: negative gamma detected. ODE derivation assumes gamma >= 0.")
    print(f"  max |Sigma - Sigma^T| : {max_sym_err:.4e}  {status_str(sym_ok)}")

    return {
        "gamma_min":    gamma_global_min,
        "gamma_max":    gamma_global_max,
        "gamma_nonneg": gamma_nonneg,
        "max_sym_err":  max_sym_err,
        "sym_ok":       sym_ok,
    }


# =============================================================================
# Check 5: short-rate consistency tau sweep
# =============================================================================
@torch.no_grad()
def check_short_rate_tau_sweep(model: FullModel, z_test: torch.Tensor, tau_list: list[float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    eps = 1e-15
    summary_rows = []
    detail_rows = []

    for tau1 in tau_list:
        tau = torch.tensor([0.0, tau1], device=z_test.device, dtype=z_test.dtype)

        fd_all = []
        r_all = []
        err_all = []
        abs_err_all = []

        for i in range(0, z_test.shape[0], 256):
            zb = z_test[i:i + 256]
            _, aux = model.decode_from_z(zb, tau=tau, do_arb_checks=False, return_aux=True)

            P_full = aux["P_full"]
            r_tilde = aux["r_tilde"]
            if r_tilde.ndim == 2 and r_tilde.shape[-1] == 1:
                r_tilde = r_tilde.squeeze(-1)

            fd = -(torch.log(P_full[:, 1].clamp_min(eps)) - torch.log(P_full[:, 0].clamp_min(eps))) / tau1
            err = fd - r_tilde
            abs_err = err.abs()

            fd_all.append(fd.detach().cpu())
            r_all.append(r_tilde.detach().cpu())
            err_all.append(err.detach().cpu())
            abs_err_all.append(abs_err.detach().cpu())

        fd_np = torch.cat(fd_all).numpy()
        r_np = torch.cat(r_all).numpy()
        err_np = torch.cat(err_all).numpy()
        abs_err_np = torch.cat(abs_err_all).numpy()

        summary_rows.append(
            {
                "tau1": tau1,
                "mean_fd_short": float(fd_np.mean()),
                "mean_r_tilde": float(r_np.mean()),
                "mean_error": float(err_np.mean()),
                "median_error": float(np.median(err_np)),
                "mean_abs_error": float(abs_err_np.mean()),
                "median_abs_error": float(np.median(abs_err_np)),
                "max_abs_error": float(abs_err_np.max()),
                "std_error": float(err_np.std(ddof=1)),
            }
        )

        for e, ae in zip(err_np, abs_err_np):
            detail_rows.append({"tau1": tau1, "error": float(e), "abs_error": float(ae)})

    summary_df = pd.DataFrame(summary_rows).sort_values("tau1", ascending=False).reset_index(drop=True)
    detail_df = pd.DataFrame(detail_rows)

    print("\n" + "=" * 72)
    print("Check 5: short-rate consistency tau sweep")
    print("=" * 72)
    print(summary_df.to_string(index=False))

    return summary_df, detail_df


# =============================================================================
# Check 6: Sharpe-ratio diagnostic
# =============================================================================
@torch.no_grad()
def check_sharpe_ratios(model: FullModel, z_test: torch.Tensor) -> dict:
    sr_list = []
    tau_axis = None

    for i in range(0, z_test.shape[0], 256):
        zb = z_test[i:i + 256]
        _, aux = model.decode_from_z(zb, tau=None, do_arb_checks=True, return_aux=True)

        arb = aux["arb"]
        if arb is None:
            continue

        sr_list.append(arb["SR_tau"].detach().cpu().numpy())
        if tau_axis is None:
            tau_axis = arb["tau_grid"].detach().cpu().numpy()

    if len(sr_list) == 0:
        print("\nSharpe-ratio diagnostics not available.")
        return {
            "SR": None,
            "tau_axis": None,
            "max_abs_sr": np.nan,
            "mean_abs_sr": np.nan,
        }

    SR = np.concatenate(sr_list, axis=0)
    abs_sr = np.abs(SR)
    max_abs_sr = float(abs_sr.max())
    mean_abs_sr = float(abs_sr.mean())

    print("\n" + "=" * 72)
    print("Check 6: Sharpe-ratio diagnostic")
    print("=" * 72)
    print(f"max |SR|                      = {max_abs_sr:.8f}")
    print(f"mean |SR|                     = {mean_abs_sr:.8f}")
    print(f"status                        = {status_str(max_abs_sr < SHARPE_TOL)}")

    return {
        "SR": SR,
        "tau_axis": tau_axis,
        "max_abs_sr": max_abs_sr,
        "mean_abs_sr": mean_abs_sr,
    }


# =============================================================================
# Check 7: reconstruction RMSE
# =============================================================================
def check_rmse_per_currency(X_tensor: torch.Tensor, S_hat: np.ndarray, meta: pd.DataFrame, scale_is_percent: bool) -> pd.DataFrame | None:
    if S_hat is None:
        print("\nReconstruction RMSE not available: S_hat is None.")
        return None

    X_np = X_tensor.detach().cpu().numpy()
    finite_rows = np.isfinite(X_np).all(axis=1) & np.isfinite(S_hat).all(axis=1)

    X_np = X_np[finite_rows]
    S_hat = S_hat[finite_rows]
    meta_eval = meta.loc[finite_rows].reset_index(drop=True)

    bp = 100.0 if scale_is_percent else 1e4
    ccys = meta_eval["ccy"].values if "ccy" in meta_eval.columns else np.array(["ALL"] * len(meta_eval))

    rows = []
    for ccy in sorted(np.unique(ccys)):
        mask = ccys == ccy
        diff_bp = (X_np[mask] - S_hat[mask]) * bp
        rmse = float(np.sqrt(np.mean(diff_bp ** 2)))
        rows.append({"ccy": ccy, "rmse_bps": rmse, "n": int(mask.sum())})

    df = pd.DataFrame(rows).sort_values("ccy").reset_index(drop=True)

    print("\n" + "=" * 72)
    print("Check 7: reconstruction RMSE per currency")
    print("=" * 72)
    print(df.to_string(index=False))
    print(f"\nAverage RMSE across currencies: {df['rmse_bps'].mean():.4f} bp")

    return df


# =============================================================================
# Save tables
# =============================================================================
def save_outputs(
    out_dir: str,
    discount_res: dict,
    g0_res: dict,
    sigma_res: dict,
    r_res: dict,
    k_res: dict,
    ode_res: dict,
    gamma_res: dict,
    tau_summary_df: pd.DataFrame,
    tau_detail_df: pd.DataFrame,
    sharpe_res: dict,
    rmse_df: pd.DataFrame | None,
) -> None:
    # -- K eigenvalue fields --
    k_fields = {}
    if not k_res.get("skipped", True):
        k_fields["K_max_real_eig"] = k_res["max_real_eig"]
        k_fields["K_all_neg"]      = int(k_res["all_negative"])
        for i, ev in enumerate(k_res["eig_reals"]):
            k_fields[f"K_eig_real_{i}"] = float(ev)

    pd.DataFrame(
        [
            {
                "finite_pct": discount_res["finite_pct"],
                "max_abs_p0_err": float(discount_res["p0_err"].max()),
                "mean_abs_p0_err": float(discount_res["p0_err"].mean()),
                "min_p": discount_res["min_p"],
                "max_p": discount_res["max_p"],
                "pct_nonpos": discount_res["pct_nonpos"],
                "pct_above_one": discount_res["pct_above_one"],
                "pct_upticks": discount_res["pct_upticks"],
                "max_uptick": discount_res["max_uptick"],
                "g0_min":         float(g0_res["G0"].min()),
                "g0_max":         float(g0_res["G0"].max()),
                "g0_mean":        float(g0_res["G0"].mean()),
                "g0_pct_neg":     g0_res["pct_neg_G0"],
                "G_global_min":   g0_res["overall_min"],
                "G_any_neg":      int(g0_res["any_neg"]),
                "G_any_tiny":     int(g0_res["any_tiny"]),
                "sigma_min": sigma_res["sigma_min"],
                "sigma_max": sigma_res["sigma_max"],
                "pct_sigma_nonpos": sigma_res["pct_sigma_nonpos"],
                "rho_abs_max": sigma_res["rho_abs_max"],
                "pct_rho_out": sigma_res["pct_rho_out"],
                "min_cov_eig":      sigma_res["min_cov_eig"],
                "max_cond":         sigma_res["max_cond"],
                "mean_cond":        sigma_res["mean_cond"],
                "diag_err_max":     sigma_res["diag_err_max"],
                "chol_any_neg":     int(sigma_res["chol_any_neg"]),
                **{f"chol_inside_{k}_min": float(v.min()) for k, v in sigma_res["inside"].items()},
                **{f"chol_inside_{k}_pct_neg": float(100.0 * (v < 0).mean()) for k, v in sigma_res["inside"].items()},
                **k_fields,
                "ODE_max_A0":       ode_res["max_a0"],
                "ODE_max_B0":       ode_res["max_b0"],
                "ODE_boundary_ok":  int(ode_res["ok"]),
                "gamma_min":        gamma_res["gamma_min"],
                "gamma_max":        gamma_res["gamma_max"],
                "gamma_nonneg":     int(gamma_res["gamma_nonneg"]),
                "cov_max_sym_err":  gamma_res["max_sym_err"],
                "cov_sym_ok":       int(gamma_res["sym_ok"]),
                "r_mean":         r_res["r_mean"],
                "r_std":          r_res["r_std"],
                "r_min":          r_res["r_min"],
                "r_max":          r_res["r_max"],
                "r_p5":           r_res["r_p5"],
                "r_p95":          r_res["r_p95"],
                "r_pct_neg":      r_res["pct_neg"],
                "r_pct_very_neg": r_res["pct_very_neg"],
                "r_pct_high":     r_res["pct_high"],
                "max_abs_sr":     sharpe_res["max_abs_sr"],
                "mean_abs_sr": sharpe_res["mean_abs_sr"],
            }
        ]
    ).to_csv(os.path.join(out_dir, "post_training_summary_metrics.csv"), index=False)

    tau_summary_df.to_csv(os.path.join(out_dir, "short_rate_tau_sweep_summary.csv"), index=False)
    tau_detail_df.to_csv(os.path.join(out_dir, "short_rate_tau_sweep_detail.csv"), index=False)

    if rmse_df is not None:
        rmse_df.to_csv(os.path.join(out_dir, "reconstruction_rmse_per_currency.csv"), index=False)

    g0_res["G_by_tau_df"].to_csv(os.path.join(out_dir, "G_values_by_tau_training.csv"), index=False)


# =============================================================================
# Plotting
# =============================================================================
def make_summary_plot(
    out_dir: str,
    checkpoint_path: str,
    debug_res: dict,
    p_full: np.ndarray,
    tau_grid: np.ndarray,
    discount_res: dict,
    g0_res: dict,
    sigma_res: dict,
    r_res: dict,
    k_res: dict,
    ode_res: dict,
    gamma_res: dict,
    tau_summary_df: pd.DataFrame,
    sharpe_res: dict,
    rmse_df: pd.DataFrame | None,
):
    fig, axes = plt.subplots(3, 3, figsize=(19, 13))
    cp_label = os.path.basename(os.path.dirname(checkpoint_path))

    # 1. Mean/min/max discount curve
    ax = axes[0, 0]
    ax.plot(tau_grid, p_full.mean(axis=0), label="mean P", lw=2)
    ax.plot(tau_grid, p_full.min(axis=0), label="min P", lw=1)
    ax.plot(tau_grid, p_full.max(axis=0), label="max P", lw=1)
    ax.axhline(1.0, color="red", linestyle="--", lw=1)
    ax.set_title("Discount curves on training set")
    ax.set_xlabel("tau")
    ax.set_ylabel("P(z,tau)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 2. G(z,0) histogram
    ax = axes[0, 1]
    ax.hist(g0_res["G0"], bins=60, edgecolor="white")
    ax.axvline(0.0, color="red", linestyle="--", lw=1)
    ax.set_title("G(z,0) distribution")
    ax.set_xlabel("G(z,0)")
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.3)

    # 3. sigma / rho histogram
    ax = axes[0, 2]
    ax.hist(sigma_res["sigmas"].ravel(), bins=50, alpha=0.7, label="sigmas")
    if sigma_res["rhos"].size > 0:
        ax.hist(sigma_res["rhos"].ravel(), bins=50, alpha=0.7, label="rhos")
    ax.set_title("H-network outputs")
    ax.set_xlabel("value")
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 4. single-point G/beta/B profile
    ax = axes[1, 0]
    ax.plot(debug_res["tau_grid"], debug_res["G_vals"], label="G")
    ax.plot(debug_res["tau_grid"], debug_res["beta_vals"], label="beta")
    ax.plot(debug_res["tau_grid"], debug_res["B_vals"], label="B")
    ax.set_title("Median-curve profile")
    ax.set_xlabel("tau")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 5. tau-sweep signed bias
    ax = axes[1, 1]
    ax.plot(tau_summary_df["tau1"], tau_summary_df["mean_error"], marker="o", label="mean error")
    ax.plot(tau_summary_df["tau1"], tau_summary_df["median_error"], marker="s", label="median error")
    ax.axhline(0.0, color="black", linestyle="--", lw=1)
    ax.set_xscale("log")
    ax.set_title("Short-rate consistency bias")
    ax.set_xlabel("tau1")
    ax.set_ylabel("fd_short - r_tilde")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 6. tau-sweep absolute error
    ax = axes[1, 2]
    ax.plot(tau_summary_df["tau1"], tau_summary_df["mean_abs_error"], marker="o", label="mean abs error")
    ax.plot(tau_summary_df["tau1"], tau_summary_df["median_abs_error"], marker="s", label="median abs error")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Short-rate consistency abs error")
    ax.set_xlabel("tau1")
    ax.set_ylabel("|fd_short - r_tilde|")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()

    # 7. Sharpe ratios
    ax = axes[2, 0]
    if sharpe_res["SR"] is not None:
        abs_sr = np.abs(sharpe_res["SR"])
        ax.plot(sharpe_res["tau_axis"], abs_sr.mean(axis=0), label="mean |SR|")
        ax.plot(sharpe_res["tau_axis"], abs_sr.max(axis=0), label="max |SR|")
        ax.axhline(SHARPE_TOL, color="red", linestyle="--", lw=1, label=f"tol={SHARPE_TOL}")
        ax.set_title("Paper SR diagnostic")
        ax.set_xlabel("tau")
        ax.set_ylabel("|SR|")
        ax.grid(True, alpha=0.3)
        ax.legend()
    else:
        ax.axis("off")

    # 8. r_tilde histogram
    ax = axes[2, 1]
    ax.hist(r_res["r_np"] * 1e4, bins=60, edgecolor="white")
    ax.axvline(0.0,   color="red",    linestyle="--", lw=1, label="r=0")
    ax.axvline(-200,  color="orange", linestyle=":",  lw=1, label="-200bp")
    ax.axvline(1000,  color="orange", linestyle=":",  lw=1, label="+1000bp")
    ax.set_title("r_tilde distribution (bp)")
    ax.set_xlabel("r_tilde (bp)")
    ax.set_ylabel("count")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 9. RMSE per currency
    ax = axes[2, 2]
    if rmse_df is not None:
        ax.bar(rmse_df["ccy"], rmse_df["rmse_bps"])
        ax.set_title("Reconstruction RMSE per currency")
        ax.set_xlabel("currency")
        ax.set_ylabel("RMSE [bp]")
        ax.grid(True, alpha=0.3, axis="y")
    else:
        ax.axis("off")

    # 9. Summary table
    ax = axes[2, 2]
    ax.axis("off")

    summary_rows = [
        ["Metric", "Value", "Status"],
        ["max |P(0)-1|",  f"{discount_res['p0_err'].max():.2e}",  status_str(discount_res["p0_err"].max() < P0_TOL)],
        ["max P",         f"{discount_res['max_p']:.6f}",          status_str(discount_res["max_p"] <= 1.0 + P_LEQ1_TOL)],
        ["% P<=0",        f"{discount_res['pct_nonpos']:.4f}%",    status_str(discount_res["pct_nonpos"] == 0.0)],
        ["max uptick",    f"{discount_res['max_uptick']:.2e}",     status_str(discount_res["max_uptick"] <= MONO_TOL)],
        ["% G0<0",        f"{g0_res['pct_neg_G0']:.4f}%",          status_str(g0_res["pct_neg_G0"] == 0.0)],
        ["G global min",  f"{g0_res['overall_min']:.4f}",          status_str(not g0_res["any_neg"])],
        ["K mean-revert",
         f"max Re(eig)={k_res['max_real_eig']:.2e}" if not k_res.get("skipped", True) else "SKIP",
         status_str(k_res.get("all_negative", False)) if not k_res.get("skipped", True) else "N/A"],
        ["A(0)=B(0)=0",  f"{ode_res['max_a0']:.1e}/{ode_res['max_b0']:.1e}",  status_str(ode_res["ok"])],
        ["gamma >= 0",    f"min={gamma_res['gamma_min']:.2e}",     status_str(gamma_res["gamma_nonneg"])],
        ["Sigma symm",    f"err={gamma_res['max_sym_err']:.2e}",   status_str(gamma_res["sym_ok"])],
        ["min cov eig",   f"{sigma_res['min_cov_eig']:.2e}",       status_str(sigma_res["min_cov_eig"] >= -1e-10)],
        ["diag(Sigma)=σ²", f"{sigma_res['diag_err_max']:.2e}",        status_str(sigma_res["diag_err_max"] < 1e-4)],
        ["chol clamp",    "YES" if sigma_res["chol_any_neg"] else "NO", status_str(not sigma_res["chol_any_neg"])],
        ["r mean",        f"{r_res['r_mean']*1e4:.1f} bp",         ""],
        ["% r < -200bp",  f"{r_res['pct_very_neg']:.2f}%",        status_str(r_res["pct_very_neg"] == 0.0)],
        ["% r > 1000bp",  f"{r_res['pct_high']:.2f}%",            status_str(r_res["pct_high"] == 0.0)],
        ["max |SR|",      f"{sharpe_res['max_abs_sr']:.2e}" if np.isfinite(sharpe_res["max_abs_sr"]) else "N/A",
         status_str(np.isfinite(sharpe_res["max_abs_sr"]) and sharpe_res["max_abs_sr"] < SHARPE_TOL) if np.isfinite(sharpe_res["max_abs_sr"]) else "N/A"],
        ["avg RMSE",      f"{rmse_df['rmse_bps'].mean():.2f} bp" if rmse_df is not None else "N/A", ""],
    ]

    x0, x1, x2 = 0.02, 0.55, 0.86
    y = 0.97
    dy = 0.047
    for i, row in enumerate(summary_rows):
        fw = "bold" if i == 0 else "normal"
        ax.text(x0, y, row[0], transform=ax.transAxes, va="top", fontweight=fw, fontsize=8)
        ax.text(x1, y, row[1], transform=ax.transAxes, va="top", fontweight=fw, fontsize=8)
        ax.text(
            x2, y, row[2], transform=ax.transAxes, va="top", fontweight=fw, fontsize=8,
            color=("green" if row[2] == "PASS" else "crimson" if row[2] == "WARN" else "black")
        )
        y -= dy

    fig.suptitle(f"Post-training diagnostics — {cp_label}", fontsize=15)
    plt.tight_layout()

    out_path = os.path.join(out_dir, "post_training_diagnostics.png")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"\nSaved figure: {out_path}")

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


# =============================================================================
# Main
# =============================================================================
def main():
    set_seeds(SEED)
    device = get_device()
    out_dir = get_output_dir(CHECKPOINT_PATH)

    print(f"Using device: {device}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")

    model = load_model(CHECKPOINT_PATH, LATENT_DIM, device)
    dtype = next(model.parameters()).dtype

    meta, X_tensor, tenors, scale_is_percent = load_training_data(
        USE, CCY_FILTER, device, dtype
    )

    print("\nEncoding training set...")
    z_all = encode_all(model, X_tensor, BATCH_SIZE)
    print(f"z shape: {tuple(z_all.shape)}")
    print(f"z mean: {z_all.mean(dim=0).detach().cpu().numpy()}")
    print(f"z std : {z_all.std(dim=0).detach().cpu().numpy()}")

    print("\nDecoding on default annual grid...")
    p_full, tau_grid, S_hat = decode_default_all(model, z_all, BATCH_SIZE)
    print(f"default tau grid starts at: {tau_grid[:5]}")

    z_test = subsample_tensor(z_all, MAX_TEST)
    print(f"\nUsing {z_test.shape[0]} latent points for expensive checks")

    debug_res = check0_single_point_debug(model, z_all)
    discount_res = check_discount_constraints(p_full, tau_grid)
    g0_res = check_G_full_tau(model, z_test)
    sigma_res = check_sigma_rho(model, z_test)
    r_res = check_r_tilde(model, z_test)
    k_res = check_K_eigenvalues(model)
    ode_res = check_ODE_boundary(model, z_test)
    gamma_res = check_gamma_and_cov_symmetry(model, z_test)
    tau_summary_df, tau_detail_df = check_short_rate_tau_sweep(model, z_test, TAU_SWEEP)
    sharpe_res = check_sharpe_ratios(model, z_test)
    rmse_df = check_rmse_per_currency(X_tensor, S_hat, meta, scale_is_percent)

    save_outputs(
        out_dir=out_dir,
        discount_res=discount_res,
        g0_res=g0_res,
        sigma_res=sigma_res,
        r_res=r_res,
        k_res=k_res,
        ode_res=ode_res,
        gamma_res=gamma_res,
        tau_summary_df=tau_summary_df,
        tau_detail_df=tau_detail_df,
        sharpe_res=sharpe_res,
        rmse_df=rmse_df,
    )

    make_summary_plot(
        out_dir=out_dir,
        checkpoint_path=CHECKPOINT_PATH,
        debug_res=debug_res,
        p_full=p_full,
        tau_grid=tau_grid,
        discount_res=discount_res,
        g0_res=g0_res,
        sigma_res=sigma_res,
        r_res=r_res,
        k_res=k_res,
        ode_res=ode_res,
        gamma_res=gamma_res,
        tau_summary_df=tau_summary_df,
        sharpe_res=sharpe_res,
        rmse_df=rmse_df,
    )

    print("\nDone.")
    print(f"Outputs saved in: {out_dir}")


if __name__ == "__main__":
    main()