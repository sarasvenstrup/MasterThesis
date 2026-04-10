"""
check_decoder_sensitivity.py
============================

Diagnoses whether the vol surface failure is a model problem by measuring
how sensitively the decoder responds to latent factor movements across
the full maturity grid.

What it checks
--------------
S1. dG/dz (tau)     - Jacobian of the raw decoder output wrt z, by maturity
S2. dP/dz (tau)     - Jacobian of the decoded bond price wrt z, by maturity
S3. dSwapRate/dz    - Jacobian of the swap rate wrt z, by expiry and tenor
S4. Vol attribution - What fraction of the market-implied swap rate vol is
                      explained by the model's z-sensitivity at each expiry/tenor

Outputs
-------
- Terminal summary tables
- CSV files in the checkpoint directory
- Matplotlib figures (saved as PNG)
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

try:
    CODE_ROOT = os.path.dirname(os.path.abspath(__file__))
except NameError:
    CODE_ROOT = os.getcwd()

PROJECT_ROOT = os.path.abspath(os.path.join(CODE_ROOT, ".."))
THESIS_ROOT  = os.path.abspath(os.path.join(CODE_ROOT, "..", ".."))
for p in [CODE_ROOT, PROJECT_ROOT, THESIS_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from Code.Pricing.simulate_model import run_simulation


# =============================================================================
# USER SETTINGS
# =============================================================================
CHECKPOINT_PATH = (
    r"C:\Users\Bruger\PycharmProjects\MasterThesis"
    r"\Figures\TrainingResults\dim2_stable\ep3500\checkpoint_dim2_ep3500.pt"
)
CCY_FILTER  = "EUR"
N_SAMPLE    = 200      # number of training-cloud z points to evaluate at
BATCH_SIZE  = 64
ACCRUAL     = 1.0

# Swaption grid to evaluate swap rate sensitivity at
EXPIRY_TENOR_PAIRS = [
    (1, 1), (1, 5), (1, 10),
    (5, 1), (5, 5), (5, 10),
    (10, 1), (10, 5), (10, 10),
]


# =============================================================================
# Utilities
# =============================================================================
def get_tau_idx(tau_grid: np.ndarray, tau_val: float) -> int:
    idx = int(np.argmin(np.abs(tau_grid - tau_val)))
    if abs(tau_grid[idx] - tau_val) > 0.02:
        raise ValueError(f"tau={tau_val} not found in tau_grid (closest={tau_grid[idx]:.4f})")
    return idx


def sample_training_z(ctx: dict, n: int, device, dtype) -> torch.Tensor:
    """Draw n points uniformly from the training latent cloud (encoded from data)."""
    from Code.Pricing.simulate_model import compute_latent_statistics
    model   = ctx["model"]
    X       = ctx["meta"]       # use the meta df to get back X_tensor if needed

    # Use z_paths as a proxy for the training cloud (already encoded)
    z_paths = ctx["z_paths"]                      # (n_paths, n_times, d)
    z_flat  = z_paths.reshape(-1, z_paths.shape[-1])

    # Also include z0 neighbourhood: sample from a Gaussian fitted to training stats
    z_mean = ctx["z_train_mean"].to(device=device, dtype=dtype)   # (d,)
    z_std  = ctx["z_train_std"].to(device=device, dtype=dtype)    # (d,)

    rng  = torch.Generator(device=device)
    rng.manual_seed(42)
    z_gauss = z_mean + z_std * torch.randn(n, z_mean.shape[0], device=device,
                                            dtype=dtype, generator=rng)

    # Mix: half from paths, half Gaussian
    n_path = min(n // 2, z_flat.shape[0])
    idx    = torch.randperm(z_flat.shape[0])[:n_path]
    z_sel  = z_flat[idx].to(device=device, dtype=dtype)
    z_all  = torch.cat([z_sel, z_gauss[: n - n_path]], dim=0)
    return z_all[:n]


# =============================================================================
# S1 & S2 — dG/dz and dP/dz across the full tau grid
# =============================================================================
def filter_finite_z(
    model,
    z_sample: torch.Tensor,
    tau_tensor: torch.Tensor,
    batch_size: int = 64,
    p_max: float = 1.0 + 1e-4,   # P must be in (0, 1] — anything larger means ODE blew up
    g_zero_tol: float = 0.01,     # flag G values below this as near-zero
) -> tuple[torch.Tensor, float]:
    """
    Keep only z points where:
      - every P(z, tau) is finite, AND
      - every P(z, tau) <= p_max  (rules out ODE blow-ups that don't reach inf)

    Also prints G(z, tau) statistics to diagnose whether G passing through
    zero is the root cause of the ODE explosion (beta = r_tilde / G).
    """
    keep = []
    g_min_vals = []   # track smallest G seen across all z and tau

    with torch.no_grad():
        for start in range(0, z_sample.shape[0], batch_size):
            zb = z_sample[start: start + batch_size]
            _, aux = model.decode_from_z(zb, tau=tau_tensor,
                                         do_arb_checks=False, return_aux=True)
            P = aux["P_full"]                           # (B, T)
            G = aux["G_vals"]                           # (B, T)

            good = torch.isfinite(P).all(dim=1) & (P <= p_max).all(dim=1)
            keep.append(zb[good])

            if G is not None:
                g_min_vals.append(float(G[good].min().item()) if good.any() else float("nan"))

    z_valid = torch.cat(keep, dim=0)
    n_total = z_sample.shape[0]
    n_valid = z_valid.shape[0]
    pct_blown = 100.0 * (1.0 - n_valid / n_total)

    print(f"\n  ODE blow-up diagnostic:")
    print(f"    z points tested        : {n_total}")
    print(f"    z points with valid P  : {n_valid}  (P finite and <= 1)")
    print(f"    blow-up rate           : {pct_blown:.1f}%")

    if g_min_vals:
        overall_g_min = min(v for v in g_min_vals if not np.isnan(v)) if g_min_vals else float("nan")
        print(f"    min G(z,tau) seen      : {overall_g_min:.6f}")
        if overall_g_min < g_zero_tol:
            print(f"    WARNING: G reaches near-zero (< {g_zero_tol}).")
            print(f"             beta = r_tilde/G blows up when G -> 0.")
            print(f"             This is the root cause of the ODE explosion.")
            print(f"             Fix: enforce G > epsilon in DecoderG, or add")
            print(f"             a G > 0 penalty to the training loss.")

    if pct_blown > 20:
        print("    WARNING: High blow-up rate — ODE is unstable for out-of-distribution z.")
        print("             MC pricing silently discards these paths -> biased vol estimates.")

    return z_valid, pct_blown


def compute_decoder_jacobians(ctx: dict, z_sample: torch.Tensor) -> dict:
    """
    For each z in z_sample (pre-filtered to finite-P points), compute the
    Jacobian of G(z, tau) and P(z, tau) wrt z, for every tau on the grid.

    Uses torch.autograd.grad with retain_graph=True so the computation graph
    is built once per batch. tau is passed as a 1-D tensor of shape (T,) as
    DecoderG.forward expects.

    Returns
    -------
    dict with:
        tau_grid     : np.ndarray (T,)
        dG_dz_mean   : np.ndarray (T, d)   mean |dG/dz_i| per tau per factor
        dP_dz_mean   : np.ndarray (T, d)   mean |dP/dz_i| per tau per factor
        dG_dz_norm   : np.ndarray (T,)     mean ||dG/dz|| (Frobenius) per tau
        dP_dz_norm   : np.ndarray (T,)     mean ||dP/dz|| per tau
    """
    model    = ctx["model"]
    tau_grid = ctx["tau_grid"].detach().cpu().numpy()      # (T,)
    T        = len(tau_grid)
    d        = z_sample.shape[1]
    device   = z_sample.device
    dtype    = z_sample.dtype

    # Keep tau as a plain 1-D tensor — DecoderG.forward does its own expansion
    tau_tensor = ctx["tau_grid"].to(device=device, dtype=dtype)  # (T,)

    # Pre-filter: only use z where P is fully finite (ODE didn't blow up)
    z_finite, pct_blown = filter_finite_z(model, z_sample, tau_tensor, BATCH_SIZE)

    if z_finite.shape[0] == 0:
        raise RuntimeError(
            "All z points produced non-finite P. ODE is completely unstable. "
            "Check training or reduce the z range in sample_training_z."
        )

    dG_acc = np.zeros((T, d), dtype=np.float64)
    dP_acc = np.zeros((T, d), dtype=np.float64)
    n_done = 0

    for start in range(0, z_finite.shape[0], BATCH_SIZE):
        zb = z_finite[start: start + BATCH_SIZE].detach().clone().requires_grad_(True)
        B  = zb.shape[0]

        # Build G and P graphs once per batch (tau is 1-D, DecoderG expands it)
        G_all = model.G(zb, tau_tensor)                           # (B, T)
        _, aux = model.decode_from_z(
            zb, tau=tau_tensor, do_arb_checks=False, return_aux=True
        )
        P_full = aux["P_full"]                                    # (B, T)

        for t_idx in range(T):
            is_last = (t_idx == T - 1)

            grads_G = torch.autograd.grad(
                G_all[:, t_idx].sum(), zb,
                retain_graph=True,
                create_graph=False,
            )[0]                                                  # (B, d)
            dG_acc[t_idx] += grads_G.detach().abs().cpu().numpy().sum(axis=0)

            grads_P = torch.autograd.grad(
                P_full[:, t_idx].sum(), zb,
                retain_graph=not is_last,
                create_graph=False,
            )[0]                                                  # (B, d)
            dP_acc[t_idx] += grads_P.detach().abs().cpu().numpy().sum(axis=0)

        n_done += B

    dG_mean = dG_acc / n_done
    dP_mean = dP_acc / n_done

    return {
        "tau_grid":    tau_grid,
        "dG_dz_mean":  dG_mean,
        "dP_dz_mean":  dP_mean,
        "dG_dz_norm":  np.linalg.norm(dG_mean, axis=1),
        "dP_dz_norm":  np.linalg.norm(dP_mean, axis=1),
        "pct_blown_up": pct_blown,
    }



# =============================================================================
# S2b — G(z, tau) value inspection
# =============================================================================
@torch.no_grad()
def check_G_values(ctx: dict, z_sample: torch.Tensor) -> pd.DataFrame:
    """
    Inspect the actual values of G(z, tau) across the full tau grid.

    Prints per-tau statistics: mean, std, min, max, and fraction of z points
    where G is near zero (< 0.01) or negative. Near-zero / negative G is the
    direct cause of ODE blow-up because beta = r_tilde / G.
    """
    model    = ctx["model"]
    tau_grid = ctx["tau_grid"].detach().cpu().numpy()   # (T,)
    T        = len(tau_grid)
    device   = z_sample.device
    dtype    = z_sample.dtype
    tau_tensor = ctx["tau_grid"].to(device=device, dtype=dtype)

    # Collect G values: shape (N, T)
    g_chunks = []
    for start in range(0, z_sample.shape[0], BATCH_SIZE):
        zb = z_sample[start: start + BATCH_SIZE]
        G  = model.G(zb, tau_tensor)          # (B, T)
        g_chunks.append(G.cpu().numpy())
    G_all = np.concatenate(g_chunks, axis=0)  # (N, T)

    rows = []
    print("\n" + "=" * 72)
    print("S2b: G(z, tau) values — root cause of ODE blow-up if near zero")
    print("     beta = r_tilde / G  ->  beta -> inf when G -> 0")
    print("=" * 72)
    print(f"{'tau':>5}  {'mean':>9}  {'std':>9}  {'min':>9}  {'max':>9}  "
          f"{'%<0':>6}  {'%<0.01':>7}  {'%<0.05':>7}")

    for t_idx in range(1, T):   # skip tau=0
        tau_val = tau_grid[t_idx]
        g       = G_all[:, t_idx]
        g_mean  = float(np.mean(g))
        g_std   = float(np.std(g))
        g_min   = float(np.min(g))
        g_max   = float(np.max(g))
        pct_neg    = 100.0 * float(np.mean(g < 0.0))
        pct_tiny   = 100.0 * float(np.mean(g < 0.01))
        pct_small  = 100.0 * float(np.mean(g < 0.05))

        flag = ""
        if pct_neg > 0:
            flag = "  <-- NEGATIVE G"
        elif pct_tiny > 0:
            flag = "  <-- NEAR ZERO"

        print(f"{tau_val:>5.0f}  {g_mean:>9.4f}  {g_std:>9.4f}  {g_min:>9.4f}  "
              f"{g_max:>9.4f}  {pct_neg:>6.1f}  {pct_tiny:>7.1f}  {pct_small:>7.1f}{flag}")

        rows.append({
            "tau":        tau_val,
            "G_mean":     g_mean,
            "G_std":      g_std,
            "G_min":      g_min,
            "G_max":      g_max,
            "pct_neg":    pct_neg,
            "pct_lt001":  pct_tiny,
            "pct_lt005":  pct_small,
        })

    df = pd.DataFrame(rows)

    first_neg_tau = df.loc[df["pct_neg"] > 0, "tau"]
    first_tiny_tau = df.loc[df["pct_lt001"] > 0, "tau"]
    print()
    if not first_neg_tau.empty:
        print(f"  First tau with G < 0      : {first_neg_tau.iloc[0]:.0f}Y  "
              f"-> ODE blows up from here")
    if not first_tiny_tau.empty:
        print(f"  First tau with G < 0.01   : {first_tiny_tau.iloc[0]:.0f}Y  "
              f"-> beta spike starts here")

    return df


# =============================================================================
# S3 — dSwapRate/dz at each expiry x tenor
# =============================================================================
def compute_swap_rate_jacobians(ctx: dict, z_sample: torch.Tensor) -> pd.DataFrame:
    """
    For each (expiry, tenor) pair compute mean ||dS/dz|| over z_sample,
    and compare to the vol implied by the market (proxy: market_vol_bp / sqrt(expiry)).
    """
    model    = ctx["model"]
    tau_grid = ctx["tau_grid"].detach().cpu().numpy()
    device   = z_sample.device
    dtype    = z_sample.dtype

    rows = []

    for expiry, tenor in EXPIRY_TENOR_PAIRS:
        payment_taus = [expiry + ACCRUAL * j for j in range(1, tenor + 1)]
        start_tau    = float(expiry)

        try:
            pay_idx   = [get_tau_idx(tau_grid, tau) for tau in payment_taus]
            start_idx = get_tau_idx(tau_grid, start_tau)
        except ValueError as e:
            warnings.warn(str(e))
            continue

        dS_acc  = np.zeros(z_sample.shape[1], dtype=np.float64)
        S_mean  = 0.0
        n_done  = 0

        for start in range(0, z_sample.shape[0], BATCH_SIZE):
            zb = z_sample[start: start + BATCH_SIZE].detach().clone().requires_grad_(True)

            _, aux   = model.decode_from_z(zb, tau=None, do_arb_checks=False, return_aux=True)
            P_full   = aux["P_full"]                       # (B, T)

            # Skip rows where ODE blew up
            good = torch.isfinite(P_full).all(dim=1)       # (B,)
            if not good.any():
                continue
            n_good = int(good.sum())

            P_start  = P_full[good, start_idx]             # (n_good,)
            pay_dfs  = P_full[good][:, pay_idx]            # (n_good, tenor)
            annuity  = ACCRUAL * pay_dfs.sum(dim=1)        # (n_good,)
            P_end    = pay_dfs[:, -1]                      # (n_good,)
            S        = (P_start - P_end) / annuity.clamp(min=1e-12)

            S_mean += float(S.detach().mean().item()) * n_good

            # grad flows back to the full zb; index to good rows after
            grads = torch.autograd.grad(
                S.sum(), zb,
                retain_graph=False,
                create_graph=False,
            )[0]                                           # (B, d)
            dS_acc += grads[good].detach().abs().cpu().numpy().sum(axis=0)
            n_done += n_good

        dS_mean = dS_acc / n_done
        S_avg   = S_mean / n_done
        dS_norm = float(np.linalg.norm(dS_mean))

        rows.append({
            "expiry":         expiry,
            "tenor":          tenor,
            "mean_S_bp":      round(10000.0 * S_avg, 2),
            "dS_dz1":         round(float(dS_mean[0]) * 10000, 4),   # in bp per unit z
            "dS_dz2":         round(float(dS_mean[1]) * 10000, 4),
            "norm_dS_dz_bp":  round(dS_norm * 10000, 4),             # ||dS/dz|| in bp
        })

    return pd.DataFrame(rows)


# =============================================================================
# S4 — Vol attribution: compare model z-sensitivity to market vol
# =============================================================================
def vol_attribution(
    jac_df: pd.DataFrame,
    sigma_z: np.ndarray,           # (d,) training-cloud std of each factor
    market_vols_bp: dict,          # {(expiry, tenor): market_vol_bp}
) -> pd.DataFrame:
    """
    Approximate the model-implied 1-year swap rate vol contribution from each
    latent factor using a first-order delta approximation:

        sigma_S ≈ sqrt( sum_i (dS/dz_i)^2 * sigma_{z_i}^2 )

    Compare to market vol to get a coverage ratio.
    """
    rows = []
    for _, row in jac_df.iterrows():
        exp = int(row["expiry"])
        ten = int(row["tenor"])
        # dS/dz in absolute units (not bp)
        dSdz = np.array([row["dS_dz1"], row["dS_dz2"]]) / 10000.0
        # approximate 1-year vol of S implied by z dynamics
        var_S = float(np.sum((dSdz * sigma_z) ** 2))
        sigma_S_bp = 10000.0 * float(np.sqrt(max(var_S, 0.0)))

        mkt_vol = market_vols_bp.get((exp, ten), np.nan)
        coverage = sigma_S_bp / mkt_vol if mkt_vol > 0 else np.nan

        rows.append({
            "expiry":          exp,
            "tenor":           ten,
            "model_z_vol_bp":  round(sigma_S_bp, 2),
            "market_vol_bp":   round(mkt_vol, 2) if not np.isnan(mkt_vol) else np.nan,
            "coverage_ratio":  round(coverage, 3) if not np.isnan(coverage) else np.nan,
        })

    return pd.DataFrame(rows)


# =============================================================================
# Plotting
# =============================================================================
def plot_decoder_sensitivity(jac_res: dict, out_dir: str) -> None:
    tau  = jac_res["tau_grid"][1:]          # skip tau=0
    dG   = jac_res["dG_dz_norm"][1:]
    dP   = jac_res["dP_dz_norm"][1:]
    d    = jac_res["dG_dz_mean"].shape[1]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.plot(tau, dG, color="#185FA5", linewidth=2, label="||dG/dz||")
    for i in range(d):
        ax.plot(tau, jac_res["dG_dz_mean"][1:, i],
                linestyle="--", linewidth=1,
                label=f"|dG/dz{i+1}|")
    ax.set_xlabel("Maturity τ (years)")
    ax.set_ylabel("Mean absolute gradient")
    ax.set_title("S1: Decoder G(z,τ) sensitivity to z")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(tau, dP, color="#A32D2D", linewidth=2, label="||dP/dz||")
    for i in range(d):
        ax.plot(tau, jac_res["dP_dz_mean"][1:, i],
                linestyle="--", linewidth=1,
                label=f"|dP/dz{i+1}|")
    ax.set_xlabel("Maturity τ (years)")
    ax.set_ylabel("Mean absolute gradient")
    ax.set_title("S2: Bond price P(z,τ) sensitivity to z")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "decoder_sensitivity_by_tau.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_swap_rate_sensitivity(jac_df: pd.DataFrame, out_dir: str) -> None:
    pairs   = [(r["expiry"], r["tenor"]) for _, r in jac_df.iterrows()]
    norms   = [r["norm_dS_dz_bp"] for _, r in jac_df.iterrows()]
    labels  = [f"{e}Y×{t}Y" for e, t in pairs]

    colors = []
    for v in norms:
        if v < 5:
            colors.append("#A32D2D")
        elif v < 15:
            colors.append("#BA7517")
        else:
            colors.append("#185FA5")

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(labels, norms, color=colors, edgecolor="none")
    ax.axhline(5, color="#A32D2D", linestyle="--", linewidth=1, label="5 bp threshold")
    ax.set_ylabel("||dS/dz|| (bp per unit z, mean abs)")
    ax.set_title("S3: Swap rate sensitivity to latent factors by expiry×tenor")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    path = os.path.join(out_dir, "swap_rate_sensitivity.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_vol_attribution(attr_df: pd.DataFrame, out_dir: str) -> None:
    labels   = [f"{int(r.expiry)}Y×{int(r.tenor)}Y" for _, r in attr_df.iterrows()]
    model_v  = attr_df["model_z_vol_bp"].values
    market_v = attr_df["market_vol_bp"].values

    x   = np.arange(len(labels))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - w/2, market_v, w, label="Market vol", color="#185FA5", alpha=0.8)
    ax.bar(x + w/2, model_v,  w, label="Model z-sensitivity vol", color="#A32D2D", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Vol (bp)")
    ax.set_title("S4: Market vol vs model vol implied by decoder sensitivity")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = os.path.join(out_dir, "vol_attribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# =============================================================================
# Main
# =============================================================================
def main():
    torch.manual_seed(42)
    np.random.seed(42)

    print("Loading model and simulation context...")
    try:
        ctx = run_simulation(
            checkpoint_path=CHECKPOINT_PATH,
            ccy_filter=CCY_FILTER,
            n_paths=500,
            n_steps=120,
            dt=1/12,
            show_plot=False,
        )
    except TypeError:
        ctx = run_simulation(
            checkpoint_path=CHECKPOINT_PATH,
            ccy_filter=CCY_FILTER,
            n_paths=500,
            n_steps=120,
            dt=1/12,
            show_plot=False,
        )

    model  = ctx["model"]
    device = next(model.parameters()).device
    dtype  = next(model.parameters()).dtype
    out_dir = os.path.dirname(CHECKPOINT_PATH)

    print(f"\nSampling {N_SAMPLE} latent points...")
    z_sample = sample_training_z(ctx, N_SAMPLE, device, dtype)
    print(f"  z_sample shape: {tuple(z_sample.shape)}")
    print(f"  z_sample mean:  {z_sample.mean(dim=0).detach().cpu().numpy()}")
    print(f"  z_sample std:   {z_sample.std(dim=0).detach().cpu().numpy()}")

    sigma_z = ctx["z_train_std"].detach().cpu().numpy()

    # ------------------------------------------------------------------
    # S1 & S2: Decoder Jacobians across tau grid
    # ------------------------------------------------------------------
    print("\nComputing S1/S2: decoder Jacobians across tau grid...")
    jac_res = compute_decoder_jacobians(ctx, z_sample)
    df_G_vals = check_G_values(ctx, z_sample)

    tau_grid = jac_res["tau_grid"]
    print("\n" + "=" * 64)
    print("S1: ||dG/dz|| by maturity (key diagnostic)")
    print("=" * 64)
    print(f"{'tau':>6}  {'||dG/dz||':>12}  " +
          "  ".join(f"|dG/dz{i+1}|" for i in range(jac_res["dG_dz_mean"].shape[1])))
    for t_idx in range(1, len(tau_grid)):
        tau_val = tau_grid[t_idx]
        norm    = jac_res["dG_dz_norm"][t_idx]
        per_fac = "  ".join(f"{jac_res['dG_dz_mean'][t_idx, i]:>10.6f}"
                             for i in range(jac_res["dG_dz_mean"].shape[1]))
        flag    = "  <-- NEAR ZERO" if norm < 1e-4 else ""
        print(f"{tau_val:>6.1f}  {norm:>12.6f}  {per_fac}{flag}")

    print("\n" + "=" * 64)
    print("S2: ||dP/dz|| by maturity")
    print("=" * 64)
    print(f"{'tau':>6}  {'||dP/dz||':>12}")
    for t_idx in range(1, len(tau_grid)):
        tau_val = tau_grid[t_idx]
        norm    = jac_res["dP_dz_norm"][t_idx]
        flag    = "  <-- NEAR ZERO" if norm < 1e-5 else ""
        print(f"{tau_val:>6.1f}  {norm:>12.8f}{flag}")

    # ------------------------------------------------------------------
    # S3: Swap rate Jacobians
    # ------------------------------------------------------------------
    print("\nComputing S3: swap rate Jacobians by expiry×tenor...")
    jac_df = compute_swap_rate_jacobians(ctx, z_sample)

    print("\n" + "=" * 64)
    print("S3: ||dS/dz|| by expiry × tenor (in bp per unit z)")
    print("=" * 64)
    print(jac_df.to_string(index=False))

    # ------------------------------------------------------------------
    # S4: Vol attribution
    # ------------------------------------------------------------------
    # Approximate market vols from the comparison file if available,
    # otherwise use the hard-coded averages from the comparison run
    market_vols_bp = {
        (1,1): 48.0,  (1,5): 30.6,  (1,10): 25.6,
        (5,1): 25.2,  (5,5): 21.0,  (5,10): 20.0,
        (10,1): 18.1, (10,5): 17.1, (10,10): 17.9,
    }

    print("\nComputing S4: vol attribution...")
    attr_df = vol_attribution(jac_df, sigma_z, market_vols_bp)

    print("\n" + "=" * 64)
    print("S4: Vol attribution — model z-sensitivity vol vs market")
    print("    coverage_ratio = model_z_vol / market_vol")
    print("    (1.0 = perfect, <0.5 = severely underfit)")
    print("=" * 64)
    print(attr_df.to_string(index=False))

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    tau_df = pd.DataFrame({
        "tau": tau_grid[1:],
        "dG_dz_norm": jac_res["dG_dz_norm"][1:],
        "dP_dz_norm": jac_res["dP_dz_norm"][1:],
        **{f"dG_dz{i+1}": jac_res["dG_dz_mean"][1:, i]
           for i in range(jac_res["dG_dz_mean"].shape[1])},
        **{f"dP_dz{i+1}": jac_res["dP_dz_mean"][1:, i]
           for i in range(jac_res["dP_dz_mean"].shape[1])},
    })
    tau_df.to_csv(os.path.join(out_dir, "decoder_sensitivity_by_tau.csv"), index=False)
    jac_df.to_csv(os.path.join(out_dir, "swap_rate_jacobians.csv"), index=False)
    attr_df.to_csv(os.path.join(out_dir, "vol_attribution.csv"), index=False)
    df_G_vals.to_csv(os.path.join(out_dir, "G_values_by_tau.csv"), index=False)
    print(f"\nSaved CSVs to {out_dir}")

    plot_decoder_sensitivity(jac_res, out_dir)
    plot_swap_rate_sensitivity(jac_df, out_dir)
    plot_vol_attribution(attr_df, out_dir)

    # ------------------------------------------------------------------
    # Final verdict
    # ------------------------------------------------------------------
    print("\n" + "=" * 64)
    print("VERDICT")
    print("=" * 64)
    pct_blown = jac_res.get("pct_blown_up", 0.0)
    low_tau_norm  = jac_res["dP_dz_norm"][1]     # at tau=1
    high_tau_norm = jac_res["dP_dz_norm"][-1]    # at tau=max
    ratio         = high_tau_norm / max(low_tau_norm, 1e-12)
    print(f"ODE blow-up rate     : {pct_blown:.1f}% of z points")
    print(f"||dP/dz|| at tau=1Y  : {low_tau_norm:.6f}")
    print(f"||dP/dz|| at tau=30Y : {high_tau_norm:.6f}")
    print(f"Ratio (long/short)   : {ratio:.4f}")
    if pct_blown > 20:
        print("\nDIAGNOSIS: ODE INSTABILITY — the dominant problem.")
        print("  exp(A-B·G) overflows for a large fraction of simulated z states.")
        print("  MC pricing silently discards these paths, leaving a biased sample")
        print("  that underestimates vol. Fixes to consider:")
        print("  1. Clamp A and B inside solve_AB before the exp (soft fix).")
        print("  2. Add an ODE blow-up penalty to the training loss.")
        print("  3. Constrain G(z,tau) > 0 so B stays bounded (G appears in the")
        print("     denominator of alpha; if G -> 0 then alpha -> inf).")
    elif ratio < 0.1:
        print("\nDIAGNOSIS: MODEL PROBLEM — decoder is near-flat in z at long maturities.")
        print("  Retraining with a tau-weighted gradient penalty on G is needed.")
    elif ratio < 0.4:
        print("\nDIAGNOSIS: PARTIAL MODEL PROBLEM — decoder sensitivity decays with tau.")
    else:
        print("\nDIAGNOSIS: Decoder sensitivity is broadly maintained across maturities.")
        print("  The vol problem is more likely in the z dynamics (K/H networks).")

    print("\nDone.")


if __name__ == "__main__":
    main()