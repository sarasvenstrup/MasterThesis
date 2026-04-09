"""
check_simulated_paths.py
========================

Validation script for simulated latent paths.

What it checks
--------------
A. Simulation/pathwise sanity
   1. Finite values in z_paths, r_paths, discount_paths, P_full_paths
   2. discount_paths start at 1 and stay positive
   3. Latent excursions relative to training-cloud mean/std

B. Decoder / no-arbitrage checks on simulated states z_t
   4. P(z,0) = 1
   5. P(z,tau) > 0
   6. P(z,tau) <= 1
   7. P(z,tau) is non-increasing in tau
   8. G(z,0) distribution
   9. Edge-vs-interior failure rates

C. Diffusion checks on simulated states z_t
  10. sigma positivity / expected bounds
  11. rho bounds
  12. PSD of Sigma Sigma'

D. Consistency checks on simulated states z_t
  13. Short-rate tau sweep: f_fd(0,tau1) vs r_tilde(z)
  14. Sharpe-ratio residuals from decode_from_z(..., do_arb_checks=True)

Outputs
-------
- Printed summary in terminal
- CSV summary files in checkpoint folder
"""

import os
import sys
import numpy as np
import pandas as pd
import torch


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

from Code.Pricing.simulate_model import run_simulation
from Code.model.sigma_matrix import L_from_sigmas_rhos


# =============================================================================
# USER SETTINGS
# =============================================================================
CHECKPOINT_PATH = (
    r"C:\Users\Bruger\PycharmProjects\MasterThesis"
    r"\Figures\TrainingResults\dim2_stable\ep3500\checkpoint_dim2_ep3500.pt"
)

CCY_FILTER = "EUR"
N_PATHS = 2000
N_STEPS = 120
DT = 1 / 12
DIFFUSION_SCALE = 1.0
MAX_TEST = 3000
BATCH_SIZE = 256
SEED = 1234

# thresholds / tolerances
P0_TOL = 1e-6
P_POS_TOL = 1e-12
P_LEQ1_TOL = 1e-6
MONO_TOL = 1e-8
SIGMA_MIN_EXPECTED = 1e-4
SIGMA_MAX_EXPECTED = 0.20
RHO_MAX_EXPECTED = 0.999
PSD_TOL = -1e-10
SHARPE_TOL = 0.15

TAU_SWEEP = [
    1.0,
    1 / 4,
    1 / 12,
    1 / 52,
    1 / 252,
    1 / 365,
]


# =============================================================================
# Utilities
# =============================================================================
def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def status_str(ok: bool) -> str:
    return "PASS" if ok else "WARN"


def flatten_simulated_z(ctx: dict, device: torch.device, dtype: torch.dtype, max_pts: int) -> torch.Tensor:
    z_paths = ctx["z_paths"]
    z_flat = z_paths.reshape(-1, z_paths.shape[-1])
    if max_pts is not None and z_flat.shape[0] > max_pts:
        idx = torch.randperm(z_flat.shape[0], device=z_flat.device)[:max_pts]
        z_flat = z_flat[idx]
    return z_flat.to(device=device, dtype=dtype)


@torch.no_grad()
def decode_full(model, z_batch: torch.Tensor, batch_size: int = 256):
    p_list = []
    tau_grid = None

    for i in range(0, z_batch.shape[0], batch_size):
        zb = z_batch[i:i + batch_size]
        _, aux = model.decode_from_z(zb, tau=None, do_arb_checks=False, return_aux=True)
        p_list.append(aux["P_full"].detach().cpu())
        if tau_grid is None:
            tau_grid = aux["tau_grid"].detach().cpu().numpy()

    P_full = torch.cat(p_list, dim=0).numpy()
    return P_full, tau_grid


# =============================================================================
# A. Simulation/pathwise sanity checks
# =============================================================================
def check_finite_tensors(ctx: dict) -> dict:
    results = {}
    keys = ["z_paths", "r_paths", "discount_paths", "P_full_paths"]

    print("\n" + "=" * 72)
    print("A1: Finite-value checks")
    print("=" * 72)

    for key in keys:
        x = ctx[key]
        is_finite = bool(torch.isfinite(x).all().item())
        finite_pct = 100.0 * float(torch.isfinite(x).float().mean().item())
        x_safe = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x_min = float(x_safe.min().item())
        x_max = float(x_safe.max().item())

        print(f"{key:16s} finite% = {finite_pct:9.4f}   range = [{x_min: .6f}, {x_max: .6f}]   {status_str(is_finite)}")

        results[key] = {
            "finite_pct": finite_pct,
            "min": x_min,
            "max": x_max,
            "all_finite": is_finite,
        }

    return results


def check_discount_paths(ctx: dict) -> dict:
    D = ctx["discount_paths"].detach().cpu().numpy()

    d0_err = np.abs(D[:, 0] - 1.0)
    min_D = float(np.min(D))
    pct_nonpos = 100.0 * float(np.mean(D <= 0.0))

    print("\n" + "=" * 72)
    print("A2: Path discount-factor checks")
    print("=" * 72)
    print(f"max |D(0)-1|                  = {d0_err.max():.8e}")
    print(f"min D_t                       = {min_D:.8f}")
    print(f"% D_t <= 0                    = {pct_nonpos:.6f}%")
    print(f"status                        = {status_str(d0_err.max() < 1e-12 and min_D > 0.0)}")

    return {
        "max_abs_D0_minus_1": float(d0_err.max()),
        "min_D": min_D,
        "pct_nonpos": pct_nonpos,
    }


def check_latent_excursions(ctx: dict) -> dict:
    z = ctx["z_paths"].detach().cpu().numpy()              # (n_paths, n_times, d)
    mu = ctx["z_train_mean"].detach().cpu().numpy()        # (d,)
    sd = ctx["z_train_std"].detach().cpu().numpy()         # (d,)
    sd = np.where(sd <= 1e-12, 1.0, sd)

    z_std = np.abs((z - mu[None, None, :]) / sd[None, None, :])

    pct_out_3 = 100.0 * float(np.mean(np.any(z_std > 3.0, axis=2)))
    pct_out_5 = 100.0 * float(np.mean(np.any(z_std > 5.0, axis=2)))
    max_std_dist = float(np.max(z_std))

    print("\n" + "=" * 72)
    print("A3: Latent excursion checks")
    print("=" * 72)
    print(f"% states outside 3 std        = {pct_out_3:.4f}%")
    print(f"% states outside 5 std        = {pct_out_5:.4f}%")
    print(f"max standardized deviation    = {max_std_dist:.4f}")

    return {
        "pct_outside_3std": pct_out_3,
        "pct_outside_5std": pct_out_5,
        "max_std_dist": max_std_dist,
    }


# =============================================================================
# B. Decoder checks on simulated states
# =============================================================================
def check_discount_curve_constraints(model, z_test: torch.Tensor) -> dict:
    P_full, tau_grid = decode_full(model, z_test, batch_size=BATCH_SIZE)

    p0_err = np.abs(P_full[:, 0] - 1.0)
    pct_nonpos = 100.0 * float(np.mean(P_full <= P_POS_TOL))
    pct_above_one = 100.0 * float(np.mean(P_full > 1.0 + P_LEQ1_TOL))

    finite_pct = 100.0 * float(np.mean(np.isfinite(P_full)))

    finite_vals = P_full[np.isfinite(P_full)]
    if finite_vals.size > 0:
        min_p = float(np.min(finite_vals))
        max_p = float(np.max(finite_vals))
    else:
        min_p = np.nan
        max_p = np.nan

    P_safe = np.where(np.isfinite(P_full), P_full, np.nan)
    diffs = np.diff(P_safe, axis=1)
    upticks = diffs > MONO_TOL
    pct_upticks = 100.0 * float(np.nanmean(upticks))
    pct_curves_with_uptick = 100.0 * float(np.nanmean(np.any(upticks, axis=1)))

    finite_diffs = diffs[np.isfinite(diffs)]
    max_uptick = float(np.max(np.maximum(finite_diffs, 0.0))) if finite_diffs.size > 0 else np.nan

    print("\n" + "=" * 72)
    print("B1-B4: Discount-curve constraints on simulated states")
    print("=" * 72)
    print(f"finite%                        = {finite_pct:.4f}%")
    print(f"min P                          = {min_p:.8f}")
    print(f"max P                          = {max_p:.8f}")
    print(f"max |P(z,0)-1|                 = {p0_err.max():.8e}")
    print(f"% P <= 0                       = {pct_nonpos:.6f}%")
    print(f"% P > 1                        = {pct_above_one:.6f}%")
    print(f"% tau-upticks                  = {pct_upticks:.6f}%")
    print(f"% curves with any uptick       = {pct_curves_with_uptick:.6f}%")
    print(f"max uptick in P                = {max_uptick:.8e}")
    ok = (
        p0_err.max() < P0_TOL and
        pct_nonpos == 0.0 and
        pct_above_one == 0.0 and
        pct_curves_with_uptick == 0.0
    )
    print(f"status                         = {status_str(ok)}")

    return {
        "P_full": P_full,
        "tau_grid": tau_grid,
        "finite_pct": finite_pct,
        "min_p": min_p,
        "max_p": max_p,
        "max_p0_err": float(p0_err.max()),
        "pct_nonpos": pct_nonpos,
        "pct_above_one": pct_above_one,
        "pct_upticks": pct_upticks,
        "pct_curves_with_uptick": pct_curves_with_uptick,
        "max_uptick": max_uptick,
    }


@torch.no_grad()
def check_G0(model, z_test: torch.Tensor) -> dict:
    tau_zero = torch.zeros(1, device=z_test.device, dtype=z_test.dtype)
    G0 = model.G(z_test, tau_zero).squeeze(-1).detach().cpu().numpy()

    pct_neg = 100.0 * float(np.mean(G0 < 0.0))
    pct_tiny = 100.0 * float(np.mean(np.abs(G0) < 1e-4))

    print("\n" + "=" * 72)
    print("B5: G(z,0)")
    print("=" * 72)
    print(f"min G0                        = {G0.min():.8f}")
    print(f"max G0                        = {G0.max():.8f}")
    print(f"mean G0                       = {G0.mean():.8f}")
    print(f"std G0                        = {G0.std():.8f}")
    print(f"% G0 < 0                      = {pct_neg:.6f}%")
    print(f"% |G0| < 1e-4                 = {pct_tiny:.6f}%")
    print(f"status                        = {status_str(pct_neg == 0.0)}")

    return {
        "G0": G0,
        "pct_neg": pct_neg,
        "pct_tiny": pct_tiny,
    }


def edge_vs_interior_diagnostic(ctx: dict, z_test: torch.Tensor, discount_res: dict) -> dict:
    z_np = z_test.detach().cpu().numpy()
    P = discount_res["P_full"]

    z_mean = ctx["z_train_mean"].detach().cpu().numpy()
    z_std = ctx["z_train_std"].detach().cpu().numpy()
    z_std = np.where(z_std <= 1e-12, 1.0, z_std)

    z_std_dist = np.max(np.abs((z_np - z_mean[None, :]) / z_std[None, :]), axis=1)

    finite_bad = ~np.isfinite(P).all(axis=1)
    above1_bad = (P > 1.0 + P_LEQ1_TOL).any(axis=1)

    P_safe = np.where(np.isfinite(P), P, np.nan)
    mono_bad = np.any(np.diff(P_safe, axis=1) > MONO_TOL, axis=1)

    any_bad = finite_bad | above1_bad | mono_bad

    inside_2 = z_std_dist <= 2.0
    inside_3 = z_std_dist <= 3.0
    outside_3 = z_std_dist > 3.0

    def rate(mask):
        n = int(mask.sum())
        if n == 0:
            return np.nan
        return 100.0 * float(np.mean(any_bad[mask]))

    def rate_type(mask, bad_mask):
        n = int(mask.sum())
        if n == 0:
            return np.nan
        return 100.0 * float(np.mean(bad_mask[mask]))

    all_rate = 100.0 * float(np.mean(any_bad))
    inside_2_rate = rate(inside_2)
    inside_3_rate = rate(inside_3)
    outside_3_rate = rate(outside_3)

    print("\n" + "=" * 72)
    print("B6: Edge vs interior diagnostic")
    print("=" * 72)
    print(f"Bad-curve rate, all states       : {all_rate:.4f}%")
    print(f"Bad-curve rate, inside 2 std     : {inside_2_rate:.4f}%")
    print(f"Bad-curve rate, inside 3 std     : {inside_3_rate:.4f}%")
    print(f"Bad-curve rate, outside 3 std    : {outside_3_rate:.4f}%")
    print(f"Nonfinite curve rate             : {100.0 * float(np.mean(finite_bad)):.4f}%")
    print(f"P > 1 curve rate                 : {100.0 * float(np.mean(above1_bad)):.4f}%")
    print(f"Monotonicity-bad curve rate      : {100.0 * float(np.mean(mono_bad)):.4f}%")
    print(f"Count outside 3 std              : {int(outside_3.sum())} / {len(outside_3)}")

    return {
        "z_std_dist": z_std_dist,
        "finite_bad": finite_bad,
        "above1_bad": above1_bad,
        "mono_bad": mono_bad,
        "any_bad": any_bad,
        "bad_rate_all": all_rate,
        "bad_rate_inside_2": inside_2_rate,
        "bad_rate_inside_3": inside_3_rate,
        "bad_rate_outside_3": outside_3_rate,
        "nonfinite_rate_all": 100.0 * float(np.mean(finite_bad)),
        "above1_rate_all": 100.0 * float(np.mean(above1_bad)),
        "mono_rate_all": 100.0 * float(np.mean(mono_bad)),
        "outside_3_count": int(outside_3.sum()),
        "total_count": int(len(outside_3)),
        "nonfinite_rate_inside_3": rate_type(inside_3, finite_bad),
        "nonfinite_rate_outside_3": rate_type(outside_3, finite_bad),
        "above1_rate_inside_3": rate_type(inside_3, above1_bad),
        "above1_rate_outside_3": rate_type(outside_3, above1_bad),
        "mono_rate_inside_3": rate_type(inside_3, mono_bad),
        "mono_rate_outside_3": rate_type(outside_3, mono_bad),
    }


# =============================================================================
# C. Diffusion checks on simulated states
# =============================================================================
@torch.no_grad()
def check_sigma_rho(model, z_test: torch.Tensor) -> dict:
    sigmas, rhos = model.H(z_test)
    sigmas_np = sigmas.detach().cpu().numpy()
    rhos_np = rhos.detach().cpu().numpy()

    L = L_from_sigmas_rhos(sigmas, rhos)
    cov = L @ L.transpose(-1, -2)
    eigs = torch.linalg.eigvalsh(cov)
    cov_min_eigs = eigs[:, 0].detach().cpu().numpy()

    pct_sigma_nonpos = 100.0 * float(np.mean(sigmas_np <= 0.0))
    sigma_min = float(np.min(sigmas_np))
    sigma_max = float(np.max(sigmas_np))

    pct_sigma_low = 100.0 * float(np.mean(sigmas_np < SIGMA_MIN_EXPECTED - 1e-8))
    pct_sigma_high = 100.0 * float(np.mean(sigmas_np > SIGMA_MAX_EXPECTED + 1e-8))

    if rhos_np.size > 0:
        rho_abs_max = float(np.max(np.abs(rhos_np)))
        pct_rho_out = 100.0 * float(np.mean(np.abs(rhos_np) > RHO_MAX_EXPECTED + 1e-8))
    else:
        rho_abs_max = 0.0
        pct_rho_out = 0.0

    min_cov_eig = float(np.min(cov_min_eigs))

    print("\n" + "=" * 72)
    print("C1-C3: sigma / rho / PSD diagnostics")
    print("=" * 72)
    print(f"sigma min                     = {sigma_min:.8f}")
    print(f"sigma max                     = {sigma_max:.8f}")
    print(f"% sigma <= 0                  = {pct_sigma_nonpos:.6f}%")
    print(f"% sigma < expected min        = {pct_sigma_low:.6f}%")
    print(f"% sigma > expected max        = {pct_sigma_high:.6f}%")
    print(f"max |rho|                     = {rho_abs_max:.8f}")
    print(f"% |rho| > expected cap        = {pct_rho_out:.6f}%")
    print(f"min eigenvalue of SigmaSigma' = {min_cov_eig:.8e}")
    print(f"status                        = {status_str(min_cov_eig >= PSD_TOL)}")

    return {
        "sigmas": sigmas_np,
        "rhos": rhos_np,
        "sigma_min": sigma_min,
        "sigma_max": sigma_max,
        "pct_sigma_nonpos": pct_sigma_nonpos,
        "pct_sigma_low": pct_sigma_low,
        "pct_sigma_high": pct_sigma_high,
        "rho_abs_max": rho_abs_max,
        "pct_rho_out": pct_rho_out,
        "cov_min_eigs": cov_min_eigs,
        "min_cov_eig": min_cov_eig,
    }


# =============================================================================
# D. Consistency checks on simulated states
# =============================================================================
@torch.no_grad()
def check_short_rate_tau_sweep(model, z_test: torch.Tensor, tau_list: list[float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    eps = 1e-15
    summary_rows = []
    detail_rows = []

    for tau1 in tau_list:
        tau = torch.tensor([0.0, tau1], device=z_test.device, dtype=z_test.dtype)

        fd_all = []
        r_all = []
        err_all = []
        abs_err_all = []

        for i in range(0, z_test.shape[0], BATCH_SIZE):
            zb = z_test[i:i + BATCH_SIZE]
            _, aux = model.decode_from_z(zb, tau=tau, do_arb_checks=False, return_aux=True)

            P_full = aux["P_full"].detach().cpu().numpy()
            r_tilde = aux["r_tilde"]
            if r_tilde.ndim == 2 and r_tilde.shape[-1] == 1:
                r_tilde = r_tilde.squeeze(-1)
            r_np = r_tilde.detach().cpu().numpy()

            p0 = P_full[:, 0]
            p1 = P_full[:, 1]

            fd = np.full_like(p1, np.nan, dtype=float)
            mask = np.isfinite(p0) & np.isfinite(p1) & (p0 > 0.0) & (p1 > 0.0)
            fd[mask] = -(np.log(np.maximum(p1[mask], eps)) - np.log(np.maximum(p0[mask], eps))) / tau1

            err = fd - r_np
            abs_err = np.abs(err)

            fd_all.append(fd)
            r_all.append(r_np)
            err_all.append(err)
            abs_err_all.append(abs_err)

        fd_np = np.concatenate(fd_all)
        r_np = np.concatenate(r_all)
        err_np = np.concatenate(err_all)
        abs_err_np = np.concatenate(abs_err_all)

        summary_rows.append(
            {
                "tau1": tau1,
                "finite_fd_pct": 100.0 * float(np.mean(np.isfinite(fd_np))),
                "mean_fd_short": float(np.nanmean(fd_np)),
                "mean_r_tilde": float(np.nanmean(r_np)),
                "mean_error": float(np.nanmean(err_np)),
                "median_error": float(np.nanmedian(err_np)),
                "mean_abs_error": float(np.nanmean(abs_err_np)),
                "median_abs_error": float(np.nanmedian(abs_err_np)),
                "max_abs_error": float(np.nanmax(abs_err_np)),
                "std_error": float(np.nanstd(err_np, ddof=1)),
            }
        )

        for e, ae in zip(err_np, abs_err_np):
            detail_rows.append({"tau1": tau1, "error": float(e), "abs_error": float(ae)})

    summary_df = pd.DataFrame(summary_rows).sort_values("tau1", ascending=False).reset_index(drop=True)
    detail_df = pd.DataFrame(detail_rows)

    print("\n" + "=" * 72)
    print("D1: Short-rate consistency tau sweep")
    print("=" * 72)
    print(summary_df.to_string(index=False))

    return summary_df, detail_df


@torch.no_grad()
def check_sharpe_ratios(model, z_test: torch.Tensor) -> dict:
    sr_list = []
    tau_axis = None

    for i in range(0, z_test.shape[0], BATCH_SIZE):
        zb = z_test[i:i + BATCH_SIZE]
        _, aux = model.decode_from_z(zb, tau=None, do_arb_checks=True, return_aux=True)

        arb = aux.get("arb", None)
        if arb is None:
            continue

        sr = arb["SR_tau"].detach().cpu().numpy()
        sr_list.append(sr)

        if tau_axis is None:
            tau_axis = arb["tau_grid"].detach().cpu().numpy()

    if len(sr_list) == 0:
        print("\n" + "=" * 72)
        print("D2: Sharpe-ratio diagnostic")
        print("=" * 72)
        print("Sharpe-ratio diagnostics not available.")
        return {
            "SR": None,
            "tau_axis": None,
            "max_abs_sr": np.nan,
            "mean_abs_sr": np.nan,
        }

    SR = np.concatenate(sr_list, axis=0)
    abs_sr = np.abs(SR)
    finite_abs_sr = abs_sr[np.isfinite(abs_sr)]

    max_abs_sr = float(np.max(finite_abs_sr)) if finite_abs_sr.size > 0 else np.nan
    mean_abs_sr = float(np.mean(finite_abs_sr)) if finite_abs_sr.size > 0 else np.nan

    print("\n" + "=" * 72)
    print("D2: Sharpe-ratio diagnostic")
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
# Save results
# =============================================================================
def save_outputs(
    out_dir: str,
    finite_res: dict,
    disc_path_res: dict,
    latent_res: dict,
    discount_res: dict,
    edge_res: dict,
    g0_res: dict,
    sigma_res: dict,
    tau_summary_df: pd.DataFrame,
    tau_detail_df: pd.DataFrame,
    sharpe_res: dict,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    summary = pd.DataFrame(
        [
            {
                "finite_z_paths_pct": finite_res["z_paths"]["finite_pct"],
                "finite_r_paths_pct": finite_res["r_paths"]["finite_pct"],
                "finite_discount_paths_pct": finite_res["discount_paths"]["finite_pct"],
                "finite_P_full_paths_pct": finite_res["P_full_paths"]["finite_pct"],
                "max_abs_D0_minus_1": disc_path_res["max_abs_D0_minus_1"],
                "min_D": disc_path_res["min_D"],
                "pct_D_nonpos": disc_path_res["pct_nonpos"],
                "pct_outside_3std": latent_res["pct_outside_3std"],
                "pct_outside_5std": latent_res["pct_outside_5std"],
                "max_std_dist": latent_res["max_std_dist"],
                "discount_min_p": discount_res["min_p"],
                "discount_max_p": discount_res["max_p"],
                "discount_max_p0_err": discount_res["max_p0_err"],
                "discount_pct_nonpos": discount_res["pct_nonpos"],
                "discount_pct_above_one": discount_res["pct_above_one"],
                "discount_pct_upticks": discount_res["pct_upticks"],
                "discount_pct_curves_with_uptick": discount_res["pct_curves_with_uptick"],
                "discount_max_uptick": discount_res["max_uptick"],
                "edge_bad_rate_all": edge_res["bad_rate_all"],
                "edge_bad_rate_inside_2": edge_res["bad_rate_inside_2"],
                "edge_bad_rate_inside_3": edge_res["bad_rate_inside_3"],
                "edge_bad_rate_outside_3": edge_res["bad_rate_outside_3"],
                "edge_nonfinite_rate_all": edge_res["nonfinite_rate_all"],
                "edge_above1_rate_all": edge_res["above1_rate_all"],
                "edge_mono_rate_all": edge_res["mono_rate_all"],
                "g0_pct_neg": g0_res["pct_neg"],
                "g0_pct_tiny": g0_res["pct_tiny"],
                "sigma_min": sigma_res["sigma_min"],
                "sigma_max": sigma_res["sigma_max"],
                "pct_sigma_nonpos": sigma_res["pct_sigma_nonpos"],
                "pct_sigma_low": sigma_res["pct_sigma_low"],
                "pct_sigma_high": sigma_res["pct_sigma_high"],
                "rho_abs_max": sigma_res["rho_abs_max"],
                "pct_rho_out": sigma_res["pct_rho_out"],
                "min_cov_eig": sigma_res["min_cov_eig"],
                "max_abs_sr": sharpe_res["max_abs_sr"],
                "mean_abs_sr": sharpe_res["mean_abs_sr"],
            }
        ]
    )

    summary.to_csv(os.path.join(out_dir, "simulated_paths_summary_metrics.csv"), index=False)
    tau_summary_df.to_csv(os.path.join(out_dir, "simulated_paths_tau_sweep_summary.csv"), index=False)
    tau_detail_df.to_csv(os.path.join(out_dir, "simulated_paths_tau_sweep_detail.csv"), index=False)

    edge_df = pd.DataFrame(
        [
            {
                "bad_rate_all": edge_res["bad_rate_all"],
                "bad_rate_inside_2": edge_res["bad_rate_inside_2"],
                "bad_rate_inside_3": edge_res["bad_rate_inside_3"],
                "bad_rate_outside_3": edge_res["bad_rate_outside_3"],
                "nonfinite_rate_all": edge_res["nonfinite_rate_all"],
                "above1_rate_all": edge_res["above1_rate_all"],
                "mono_rate_all": edge_res["mono_rate_all"],
                "nonfinite_rate_inside_3": edge_res["nonfinite_rate_inside_3"],
                "nonfinite_rate_outside_3": edge_res["nonfinite_rate_outside_3"],
                "above1_rate_inside_3": edge_res["above1_rate_inside_3"],
                "above1_rate_outside_3": edge_res["above1_rate_outside_3"],
                "mono_rate_inside_3": edge_res["mono_rate_inside_3"],
                "mono_rate_outside_3": edge_res["mono_rate_outside_3"],
                "outside_3_count": edge_res["outside_3_count"],
                "total_count": edge_res["total_count"],
            }
        ]
    )
    edge_df.to_csv(os.path.join(out_dir, "simulated_paths_edge_diagnostic.csv"), index=False)

    if sharpe_res["SR"] is not None and sharpe_res["tau_axis"] is not None:
        abs_sr = np.abs(sharpe_res["SR"])
        mean_abs_sr_by_tau = np.nanmean(np.where(np.isfinite(abs_sr), abs_sr, np.nan), axis=0)
        pd.DataFrame(
            {
                "tau": sharpe_res["tau_axis"],
                "mean_abs_SR": mean_abs_sr_by_tau,
            }
        ).to_csv(os.path.join(out_dir, "simulated_paths_sharpe_by_tau.csv"), index=False)

    print(f"\nSaved summary CSVs to:\n  {out_dir}")


# =============================================================================
# Main
# =============================================================================
def main():
    set_seeds(SEED)

    print("\nRunning simulation...")
    kwargs = dict(
        checkpoint_path=CHECKPOINT_PATH,
        ccy_filter=CCY_FILTER,
        n_paths=N_PATHS,
        n_steps=N_STEPS,
        dt=DT,
        show_plot=False,
    )

    try:
        ctx = run_simulation(diffusion_scale=DIFFUSION_SCALE, **kwargs)
    except TypeError:
        ctx = run_simulation(**kwargs)

    model = ctx["model"]
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    z_test = flatten_simulated_z(ctx, device=device, dtype=dtype, max_pts=MAX_TEST)
    print(f"\nTesting on {z_test.shape[0]} simulated latent states.")

    finite_res = check_finite_tensors(ctx)
    disc_path_res = check_discount_paths(ctx)
    latent_res = check_latent_excursions(ctx)

    discount_res = check_discount_curve_constraints(model, z_test)
    edge_res = edge_vs_interior_diagnostic(ctx, z_test, discount_res)
    g0_res = check_G0(model, z_test)
    sigma_res = check_sigma_rho(model, z_test)
    tau_summary_df, tau_detail_df = check_short_rate_tau_sweep(model, z_test, TAU_SWEEP)
    sharpe_res = check_sharpe_ratios(model, z_test)

    out_dir = os.path.dirname(CHECKPOINT_PATH)
    save_outputs(
        out_dir=out_dir,
        finite_res=finite_res,
        disc_path_res=disc_path_res,
        latent_res=latent_res,
        discount_res=discount_res,
        edge_res=edge_res,
        g0_res=g0_res,
        sigma_res=sigma_res,
        tau_summary_df=tau_summary_df,
        tau_detail_df=tau_detail_df,
        sharpe_res=sharpe_res,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()