"""
plot_pricing_distribution.py
============================

Model vs. market *pricing* distribution of the terminal swap rate.

For every (expiry, tenor) cell on the 3x3 grid 1Y/5Y/10Y x 1Y/5Y/10Y this
script plots two densities of the centred terminal swap-rate move
    X = S_{mu,nu}(T_mu) - K      (in bp),

aggregated across a configurable list of pricing dates:

  * MODEL  — annuity-weighted distribution of the Monte Carlo paths,
             weights
                w_m = D(0,T_mu)^(m) * A_{mu,nu}^(m)(T_mu)
                      / sum_n  D(0,T_mu)^(n) * A_{mu,nu}^(n)(T_mu),
             plotted as a weighted histogram.

  * MARKET — mixture (with equal date weights) of the Bachelier
             benchmark densities
                X | date_d ~ N(0, sigma_{N,market,d}^2 * T_mu),
             evaluated on a fine bp-grid.  Plotted as a smooth line.

This is the right object to compare with: the half-straddle MC price
equals the annuity-weighted expected absolute centred move, and the
Bachelier benchmark is what the market quote implies for that same
quantity under the annuity measure.

Outputs (in --out_dir):
  pricing_distribution.png
  pricing_distribution.csv   (per cell + date summary)

Usage:
    $env:SKIP_VARIANT_CONFIRM = "1"
    python Code/Pricing/plot_pricing_distribution.py
    python Code/Pricing/plot_pricing_distribution.py --dates 2020-04-30,2024-02-29
"""

from __future__ import annotations
import argparse
import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("SKIP_VARIANT_CONFIRM", "1")

import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------------- paths
_HERE      = Path(__file__).resolve().parent
_CODE_ROOT = _HERE.parent
_PROJECT   = _CODE_ROOT.parent
for p in (str(_PROJECT), str(_CODE_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from Code.Simulation.simulate_model import run_simulation
from Code.Pricing.pricing import (
    get_grid_index_for_value,
    time0_forward_swap_and_annuity,
)
from Code.Pricing.ResultsGeneratorPricingComparison import load_market_vols


STRUCTURES = [
    (1, 1), (1, 5), (1, 10),
    (5, 1), (5, 5), (5, 10),
    (10, 1), (10, 5), (10, 10),
]

DEFAULT_DATES = [
    "2010-10-29", "2012-09-28", "2014-08-29", "2016-07-29",
    "2018-05-31", "2020-04-30", "2022-03-31", "2024-02-29",
]

NVOL_SHEETS = {"EUR": "EUR SwapNVol OIS", "USD": "USD SwapNVol OIS"}


# =====================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--checkpoint", type=str,
        default=str(_PROJECT / "Figures" / "TrainingResults"
                    / "dim4_stable" / "ep5000" / "checkpoint_dim4_ep5000.pt"))
    p.add_argument("--latent_dim",      type=int,   default=4)
    p.add_argument("--ccy",             type=str,   default="EUR")
    p.add_argument("--dates",           type=str,   default=",".join(DEFAULT_DATES),
                   help="Comma-separated pricing dates (YYYY-MM-DD).")
    p.add_argument("--n_paths",         type=int,   default=10_000)
    p.add_argument("--n_steps",         type=int,   default=120)
    p.add_argument("--dt",              type=float, default=1.0 / 12.0)
    p.add_argument("--diffusion_scale", type=float, default=1.0)
    p.add_argument("--seed",            type=int,   default=1234)
    p.add_argument("--market_data",     type=str,
                   default=str(_PROJECT / "SwapData" / "SwapVol.xlsx"))
    p.add_argument("--x_range_bp",      type=float, default=2500.0,
                   help="Half-width of the bp x-axis for the densities. "
                        "Default 2500 bp accommodates the model std of "
                        "~600-1170 bp; reduce if you want to zoom on the "
                        "market mode.")
    p.add_argument("--n_bins",          type=int,   default=100)
    p.add_argument(
        "--out_dir", type=str,
        default=str(_PROJECT / "Figures" / "PricingResults" / "Diagnostics"
                    / "EUR_dim4_stable" / "plots"))
    return p.parse_args()


# =====================================================================
def collect_cell_for_date(
    ctx: dict,
    expiry: int,
    tenor: int,
) -> dict:
    """
    For one pricing date and one (expiry, tenor) cell, returns per-path:
        S_bp, K_bp, weight_unnormalised, valid_mask
    so the caller can aggregate across dates.
    """
    times        = np.asarray(ctx["times"], dtype=float)
    tau_grid     = ctx["tau_grid"].detach().cpu().numpy()
    P_full_paths = ctx["P_full_paths"].detach().cpu().numpy()
    P_full_0     = ctx["P_full_0"].detach().cpu().numpy().reshape(-1)
    discount     = ctx["discount_paths"].detach().cpu().numpy()

    decode_idx = get_grid_index_for_value(times, float(expiry))
    # discount lives on the full simulation grid
    n_full_steps = discount.shape[1] - 1
    t_max = float(times.max())
    dt_full = t_max / max(n_full_steps, 1)
    full_exp_idx = int(round(float(expiry) / dt_full))
    D = discount[:, full_exp_idx].astype(float)

    # time-0 ATM
    q0 = time0_forward_swap_and_annuity(
        P_full_0=P_full_0, tau_grid=tau_grid,
        expiry=float(expiry), tenor=int(tenor), accrual=1.0,
    )
    F0 = float(q0["forward_swap"])
    A0 = float(q0["annuity"])
    K  = F0

    # per-path swap rate and annuity at expiry
    pay_tau_idx = [get_grid_index_for_value(tau_grid, float(j))
                   for j in range(1, tenor + 1)]
    pay_dfs = P_full_paths[:, decode_idx, :][:, pay_tau_idx]   # (n_paths, tenor)
    finite_curve = np.isfinite(P_full_paths[:, decode_idx, :]).all(axis=-1) \
                   & (P_full_paths[:, decode_idx, :] > 0).all(axis=-1) \
                   & (P_full_paths[:, decode_idx, :] <= 10.0).all(axis=-1)

    A_path = pay_dfs.sum(axis=-1)
    S_path = (1.0 - pay_dfs[:, -1]) / np.where(A_path > 0, A_path, np.nan)

    valid = (
        finite_curve
        & np.isfinite(A_path) & (A_path > 0)
        & np.isfinite(S_path)
        & np.isfinite(D) & (D > 0)
    )

    # Unnormalised annuity-measure weight (annuity numeraire change).
    # Mask BEFORE multiplying to avoid overflow on inf-decoded paths.
    weight = np.zeros_like(D, dtype=float)
    weight[valid] = D[valid] * A_path[valid]

    return {
        "expiry":  expiry,
        "tenor":   tenor,
        "S_bp":    1e4 * S_path,
        "K_bp":    1e4 * K,
        "weight":  weight,
        "valid":   valid,
        "A0":      A0,
        "F0":      F0,
    }


# =====================================================================
def gaussian_mixture_pdf(
    x_bp: np.ndarray, sigma_T_bp_list: list[float]
) -> np.ndarray:
    """Equal-weight mixture of zero-mean Gaussians with std `sigma_T_bp`."""
    out = np.zeros_like(x_bp, dtype=float)
    n = len(sigma_T_bp_list)
    if n == 0:
        return out
    for s in sigma_T_bp_list:
        if not np.isfinite(s) or s <= 0:
            continue
        out += np.exp(-0.5 * (x_bp / s) ** 2) / (s * np.sqrt(2.0 * np.pi))
    return out / n


# =====================================================================
def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    dates = [d.strip() for d in args.dates.split(",") if d.strip()]

    # --- market data --------------------------------------------------
    sheet = NVOL_SHEETS.get(args.ccy.upper(),
                            f"{args.ccy.upper()} SwapNVol OIS")
    df_market = load_market_vols(Path(args.market_data), sheet_name=sheet)

    def market_vol_bp(date_str: str, expiry: int, tenor: int) -> float | None:
        target = np.datetime64(date_str)
        sub = df_market[(df_market["expiry"] == expiry)
                        & (df_market["tenor"] == tenor)]
        if sub.empty:
            return None
        diffs = (sub["date"].values.astype("datetime64[D]")
                 - target.astype("datetime64[D]")).astype(int)
        idx = int(np.argmin(np.abs(diffs)))
        if abs(diffs[idx]) > 30:   # more than a month off → skip
            return None
        return float(sub.iloc[idx]["market_vol_bps"])

    # --- decode steps -------------------------------------------------
    expiry_years = sorted({exp for (exp, _ten) in STRUCTURES})
    decode_steps = [int(round(e / args.dt)) for e in expiry_years]

    # --- per-cell aggregation across dates ----------------------------
    # store per cell: list of dicts with X_bp arr, weight arr, sigma_T_bp
    agg = {(e, t): {"X": [], "w": [], "sigma_T_bp": [], "rows": []}
           for (e, t) in STRUCTURES}

    for date_str in dates:
        print("\n" + "=" * 70)
        print(f"as_of_date = {date_str}")
        print("=" * 70)
        try:
            ctx = run_simulation(
                use="bbg",
                latent_dim=args.latent_dim,
                checkpoint_path=args.checkpoint,
                n_paths=args.n_paths,
                n_steps=args.n_steps,
                dt=args.dt,
                as_of_date=date_str,
                ccy_filter=args.ccy,
                diffusion_scale=args.diffusion_scale,
                use_antithetic=False,
                seed=args.seed,
                show_plot=False,
                decode_steps=decode_steps,
            )
        except Exception as e:
            print(f"[skip date] {date_str}: {str(e)[:150]}")
            continue

        for (exp, ten) in STRUCTURES:
            try:
                cell = collect_cell_for_date(ctx, exp, ten)
            except Exception as e:
                print(f"  [skip cell {exp}x{ten}]: {str(e)[:120]}")
                continue
            X = cell["S_bp"] - cell["K_bp"]        # (n_paths,)
            w = cell["weight"]                     # 0 on invalid
            if w.sum() <= 0:
                print(f"  {exp}x{ten}: no valid weight; skipping date.")
                continue
            sigma_market_bp_yr = market_vol_bp(date_str, exp, ten)
            sigma_T_bp = (None if sigma_market_bp_yr is None
                          else float(sigma_market_bp_yr) * np.sqrt(exp))

            agg[(exp, ten)]["X"].append(X[cell["valid"]])
            agg[(exp, ten)]["w"].append(w[cell["valid"]])
            if sigma_T_bp is not None:
                agg[(exp, ten)]["sigma_T_bp"].append(sigma_T_bp)

            # per-(date, cell) summary row
            w_norm = w[cell["valid"]] / w[cell["valid"]].sum()
            mean_w  = float(np.sum(w_norm * X[cell["valid"]]))
            std_w   = float(np.sqrt(np.sum(w_norm * (X[cell["valid"]] - mean_w) ** 2)))
            agg[(exp, ten)]["rows"].append({
                "date":            date_str,
                "expiry":          exp, "tenor": ten,
                "K_bp":            cell["K_bp"],
                "n_valid":         int(cell["valid"].sum()),
                "mean_w_X_bp":     mean_w,
                "std_w_X_bp":      std_w,
                "sigma_market_bp": sigma_market_bp_yr,
                "sigma_T_bp_market": sigma_T_bp,
            })
            print(f"  {exp}Yx{ten}Y  K={cell['K_bp']:7.1f}bp  "
                  f"mean(w*X)={mean_w:+7.1f}bp  std(w*X)={std_w:7.1f}bp  "
                  f"sigma_market_T={'-' if sigma_T_bp is None else f'{sigma_T_bp:7.1f}bp'}")

    # --- CSV ---------------------------------------------------------
    csv_path = out_dir / "pricing_distribution.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "expiry", "tenor", "K_bp", "n_valid",
                    "mean_w_X_bp", "std_w_X_bp",
                    "sigma_market_bp_yr", "sigma_market_T_bp"])
        for (e, t), pkg in agg.items():
            for r in pkg["rows"]:
                w.writerow([r["date"], r["expiry"], r["tenor"],
                            f"{r['K_bp']:.2f}", r["n_valid"],
                            f"{r['mean_w_X_bp']:.3f}", f"{r['std_w_X_bp']:.3f}",
                            ("" if r["sigma_market_bp"] is None
                             else f"{r['sigma_market_bp']:.3f}"),
                            ("" if r["sigma_T_bp_market"] is None
                             else f"{r['sigma_T_bp_market']:.3f}")])
    print(f"\nSaved per-date summary: {csv_path}")

    # --- plot helpers ------------------------------------------------
    x_grid = np.linspace(-args.x_range_bp, args.x_range_bp, 1001)
    edges  = np.linspace(-args.x_range_bp, args.x_range_bp, args.n_bins + 1)
    bin_width = edges[1] - edges[0]
    x_centers = 0.5 * (edges[:-1] + edges[1:])

    # Pre-compute per-cell normalised arrays once
    cell_data: dict[tuple, dict] = {}
    for (exp, ten) in STRUCTURES:
        pkg = agg[(exp, ten)]
        if not pkg["X"]:
            cell_data[(exp, ten)] = None
            continue
        X_all = np.concatenate(pkg["X"])
        n_dates = len(pkg["X"])
        date_w_normed = [w_arr / w_arr.sum() / n_dates
                         for w_arr in pkg["w"]]
        w_all = np.concatenate(date_w_normed)
        cell_data[(exp, ten)] = {"X": X_all, "w": w_all,
                                 "sigma_T_bp": pkg["sigma_T_bp"]}

    # ---- helper: draw one panel -------------------------------------
    def draw_panel(ax, exp, ten, show_weighted: bool, show_unweighted: bool) -> None:
        d = cell_data.get((exp, ten))
        if d is None:
            ax.set_title(f"{exp}Yx{ten}Y  (no data)")
            ax.set_axis_off()
            return
        X_all, w_all = d["X"], d["w"]

        if show_weighted:
            hist_w, _ = np.histogram(X_all, bins=edges, weights=w_all)
            ax.bar(x_centers, hist_w / bin_width, width=bin_width,
                   color="tomato", alpha=0.55, edgecolor="darkred", linewidth=0.4,
                   label="model — annuity-weighted")

        if show_unweighted:
            hist_uw, _ = np.histogram(X_all, bins=edges, density=False)
            uw_density = hist_uw / (hist_uw.sum() * bin_width)
            ax.step(x_centers, uw_density, where="mid",
                    color="darkorange", linewidth=1.5, alpha=0.9,
                    label="model — unweighted (path count)")

        sigma_T_bp_list = d["sigma_T_bp"]
        if sigma_T_bp_list:
            pdf_mkt = gaussian_mixture_pdf(x_grid, sigma_T_bp_list)
            ax.plot(x_grid, pdf_mkt, color="steelblue", linewidth=2.0,
                    label=f"market Bachelier\n(n={len(sigma_T_bp_list)} dates)")

        ax.axvline(0.0, color="black", linewidth=0.7, linestyle=":")
        ax.set_title(rf"${exp}\mathrm{{Y}}\times{ten}\mathrm{{Y}}$", fontsize=11)
        ax.set_xlabel(r"$S(T_\mu)-K$  (bp)")
        ax.set_ylabel("density")
        ax.grid(alpha=0.3)

    # ---- Figure 1: weighted only ------------------------------------
    fig1, axes1 = plt.subplots(3, 3, figsize=(12, 10), sharex=True)
    for ax, (exp, ten) in zip(axes1.flatten(), STRUCTURES):
        draw_panel(ax, exp, ten, show_weighted=True, show_unweighted=False)
        if (exp, ten) == STRUCTURES[0]:
            ax.legend(fontsize=8, loc="upper left")
    fig1.suptitle(
        f"Pricing distribution — annuity-weighted MC vs. market Bachelier\n"
        f"{args.ccy} dim{args.latent_dim} stable, "
        f"{len(dates)} dates, {args.n_paths} paths/date",
        fontsize=11)
    fig1.tight_layout(rect=(0, 0, 1, 0.95))
    out1 = out_dir / "pricing_distribution_weighted.png"
    fig1.savefig(out1, dpi=200); plt.close(fig1)
    print(f"Saved (weighted):       {out1}")

    # ---- Figure 2: unweighted only ----------------------------------
    fig2, axes2 = plt.subplots(3, 3, figsize=(12, 10), sharex=True)
    for ax, (exp, ten) in zip(axes2.flatten(), STRUCTURES):
        draw_panel(ax, exp, ten, show_weighted=False, show_unweighted=True)
        if (exp, ten) == STRUCTURES[0]:
            ax.legend(fontsize=8, loc="upper left")
    fig2.suptitle(
        f"Pricing distribution — unweighted MC (path count) vs. market Bachelier\n"
        f"{args.ccy} dim{args.latent_dim} stable, "
        f"{len(dates)} dates, {args.n_paths} paths/date",
        fontsize=11)
    fig2.tight_layout(rect=(0, 0, 1, 0.95))
    out2 = out_dir / "pricing_distribution_unweighted.png"
    fig2.savefig(out2, dpi=200); plt.close(fig2)
    print(f"Saved (unweighted):     {out2}")

    # ---- Figure 3: combined overlay ---------------------------------
    fig3, axes3 = plt.subplots(3, 3, figsize=(12, 10), sharex=True)
    for ax, (exp, ten) in zip(axes3.flatten(), STRUCTURES):
        draw_panel(ax, exp, ten, show_weighted=True, show_unweighted=True)
        if (exp, ten) == STRUCTURES[0]:
            ax.legend(fontsize=8, loc="upper left")
    fig3.suptitle(
        f"Pricing distribution — weighted vs. unweighted vs. market Bachelier\n"
        f"{args.ccy} dim{args.latent_dim} stable, "
        f"{len(dates)} dates, {args.n_paths} paths/date",
        fontsize=11)
    fig3.tight_layout(rect=(0, 0, 1, 0.95))
    out3 = out_dir / "pricing_distribution_combined.png"
    fig3.savefig(out3, dpi=200); plt.close(fig3)
    print(f"Saved (combined):       {out3}")

    # ---- Figure 4: cluster diagnostic for (10Y × 5Y) ---------------
    diag_cell = (10, 5)
    d_diag = cell_data.get(diag_cell)
    if d_diag is not None and agg[diag_cell]["X"]:
        X_d  = d_diag["X"]
        w_d  = d_diag["w"]

        # Identify the boundary between the two visible clusters:
        # use the weighted median as a simple split point.
        sort_idx   = np.argsort(X_d)
        w_sorted   = w_d[sort_idx]
        cum_w      = np.cumsum(w_sorted)
        split_bp   = float(X_d[sort_idx[np.searchsorted(cum_w, 0.5 * cum_w[-1])]])
        lower_mask = X_d <= split_bp
        upper_mask = ~lower_mask

        # Re-collect per-path annuity + discount from the LAST date that
        # had valid data for this cell (use stored raw arrays from agg).
        pkg_diag = agg[diag_cell]
        # Use the last date's data for the diagnostic scatter
        X_last = pkg_diag["X"][-1]
        w_last = pkg_diag["w"][-1]
        lo = X_last <= split_bp
        hi = ~lo

        fig4, axes4 = plt.subplots(1, 3, figsize=(14, 4))

        for ax, (vals, ylabel, title) in zip(
            axes4,
            [
                (w_last,          r"unnorm. weight $D \cdot \mathcal{A}$",
                 "Pricing weight"),
                (w_last / (w_last + 1e-30),  # placeholder — replaced below
                 r"path index",   ""),         # see below
            ] + [(None, None, None)],
        ):
            pass   # filled explicitly below

        # Panel A: pricing weight distribution
        ax = axes4[0]
        ax.hist(w_last[lo],  bins=60, alpha=0.6, color="steelblue",
                density=True, label=f"lower cluster (X≤{split_bp:.0f}bp)")
        ax.hist(w_last[hi],  bins=60, alpha=0.6, color="tomato",
                density=True, label=f"upper cluster (X>{split_bp:.0f}bp)")
        ax.set_xlabel(r"unnorm. weight $D_{0,T}\cdot\mathcal{A}(T)$")
        ax.set_ylabel("density")
        ax.set_title("Pricing weights by cluster")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # Panel B: swap-rate X coloured by normalised weight magnitude
        ax = axes4[1]
        w_norm_last = w_last / w_last.sum()
        sc = ax.scatter(np.arange(len(X_last)), X_last,
                        c=w_norm_last, cmap="RdYlBu_r",
                        s=1.5, alpha=0.5, rasterized=True)
        plt.colorbar(sc, ax=ax, label="norm. weight")
        ax.axhline(split_bp, color="black", linewidth=0.8, linestyle="--",
                   label=f"cluster split = {split_bp:.0f} bp")
        ax.set_xlabel("path index")
        ax.set_ylabel(r"$S(T_\mu)-K$  (bp)")
        ax.set_title("Per-path X coloured by pricing weight")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # Panel C: weighted vs unweighted density for this cell only
        ax = axes4[2]
        hist_w_d, _ = np.histogram(X_d, bins=edges, weights=w_d)
        ax.fill_between(x_centers, hist_w_d / bin_width,
                        step="mid", alpha=0.5, color="tomato",
                        label="annuity-weighted")
        hist_uw_d, _ = np.histogram(X_d, bins=edges, density=False)
        ax.step(x_centers, hist_uw_d / (hist_uw_d.sum() * bin_width),
                where="mid", color="darkorange", linewidth=1.5,
                label="unweighted")
        sigma_list = agg[diag_cell]["sigma_T_bp"]
        if sigma_list:
            ax.plot(x_grid, gaussian_mixture_pdf(x_grid, sigma_list),
                    color="steelblue", linewidth=2, label="market Bachelier")
        ax.axvline(split_bp, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel(r"$S(T_\mu)-K$  (bp)")
        ax.set_ylabel("density")
        ax.set_title(r"$10\mathrm{Y}\times5\mathrm{Y}$ — weighted vs. unweighted")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        fig4.suptitle(
            rf"Cluster diagnostic: $10\mathrm{{Y}}\times5\mathrm{{Y}}$  "
            f"(last date = {dates[-1] if dates else '?'}, "
            f"split at {split_bp:.0f} bp)",
            fontsize=11)
        fig4.tight_layout()
        out4 = out_dir / "pricing_distribution_cluster_diagnostic.png"
        fig4.savefig(out4, dpi=200); plt.close(fig4)
        print(f"Saved (cluster diag):   {out4}")
    else:
        print("Cluster diagnostic skipped: no data for (10Y×5Y).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


