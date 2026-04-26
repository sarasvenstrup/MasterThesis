# -*- coding: utf-8 -*-
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()

PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
THESIS_ROOT  = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
for p in [THESIS_ROOT, PROJECT_ROOT, SCRIPT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from Code.Simulation.simulate_model import run_simulation
from Code.Pricing.pricing import (
    time0_forward_swap_and_annuity,
    swap_from_discount_curve_at_expiry,
    atm_swaption_mc_price_from_simulation,
    get_grid_index_for_value,
)
from Code.utils.helpers import PlotConfig, save_figure, instantaneous_forward

# =============================================================================
# USER SETTINGS
# =============================================================================
CHECKPOINT_PATH = (
    r"C:\Users\Bruger\PycharmProjects\MasterThesis\Figures\TrainingResults"
    r"\dim2_stable\ep5000\checkpoint_dim2_ep5000.pt"
)
CCY_FILTER   = "EUR"
AS_OF_DATE   = "2014-12-31"
N_PATHS      = 2000
N_STEPS      = 120          # 10 years at monthly dt
DT           = 1 / 12

EXPIRIES     = [1, 5, 10]
TENORS       = [1, 5, 10]

OUT_DIR      = os.path.join(THESIS_ROOT, "Figures", "Simulation")
os.makedirs(OUT_DIR, exist_ok=True)
TOL_P0       = 1e-6
TOL_MONOTONE = 1e-10        # fix 3: avoid fp noise false-positives

CFG = PlotConfig(figures_dir=OUT_DIR, use_tag=AS_OF_DATE or "latest", dpi=200)

# =============================================================================
# Styling & print helpers
# =============================================================================
BLUE  = "#2c4f8c"
RED   = "#8c2c2c"
GREY  = "#555555"
SEP   = "-" * 70
SEP2  = "=" * 70

def hdr(t): print(f"\n{SEP2}\n  {t}\n{SEP2}")
def sub(t): print(f"\n{SEP}\n  {t}\n{SEP}")
def pf(ok, label, good, bad):
    print(f"  [{'PASS' if ok else 'FAIL'}]  {label}: {good if ok else bad}")
    return ok

def _np(t):
    if hasattr(t, "detach"):
        return t.detach().cpu().numpy()
    return np.asarray(t)


# =============================================================================
# FIG 1  —  Discount curve + instantaneous forward curve
# =============================================================================
def fig1_initial_curve(ctx):
    hdr("1. INITIAL CURVE  (t = 0)")

    P0_t  = ctx["P_full_0"]
    tau_t = ctx["tau_grid"]
    P0    = _np(P0_t)[0]
    tau   = _np(tau_t)
    date_str = str(ctx["meta_row"].get("as_of_date", AS_OF_DATE))[:10]

    fwd = _np(instantaneous_forward(P0_t[:, 1:], tau_t[1:]))[0]

    # ── console ──────────────────────────────────────────────────────────────
    print(f"\n  Date : {date_str}")
    print(f"  P(0,1) = {P0[1]:.6f}   P(0,10) = {P0[10]:.6f}"
          f"   P(0,30) = {P0[-1]:.6f}")

    rows = []
    print(f"\n  {'Expiry':>7}  {'Tenor':>6}  {'F0 (bp)':>9}  {'A0':>8}"
          f"  {'P(0,Te)':>9}  {'P(0,Te+n)':>10}")
    print(f"  {'-'*7}  {'-'*6}  {'-'*9}  {'-'*8}  {'-'*9}  {'-'*10}")

    for exp in EXPIRIES:
        for ten in TENORS:
            if exp + ten > int(tau[-1]):
                continue
            q = time0_forward_swap_and_annuity(P0, tau, exp, ten)
            rows.append({
                "Expiry (Y)": exp, "Tenor (Y)": ten,
                "F0 (bp)":    round(q["forward_swap"] * 10000, 2),
                "Annuity A0": round(q["annuity"], 6),
                "P(0, Te)":   round(q["P_start"], 6),
                "P(0, Te+n)": round(q["P_end"],   6),
            })
            print(f"  {exp:>6}Y  {ten:>5}Y"
                  f"  {q['forward_swap']*10000:>9.1f}"
                  f"  {q['annuity']:>8.4f}"
                  f"  {q['P_start']:>9.6f}"
                  f"  {q['P_end']:>10.6f}")

    # ── plot: two panels, clean line style ────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(tau, P0, "o-", markersize=4, linewidth=1.8, color=BLUE)
    axes[0].set_xlabel("Maturity τ (years)")
    axes[0].set_ylabel("P(0, τ)")
    axes[0].set_title(f"Discount curve  —  {date_str}")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(tau[1:], fwd * 100, "o-", markersize=4,
                 linewidth=1.8, color=RED)
    axes[1].set_xlabel("Maturity τ (years)")
    axes[1].set_ylabel("Instantaneous forward rate (%)")
    axes[1].set_title(f"Implied forward curve  —  {date_str}")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    save_figure(fig, CFG, "fig1_initial_curve")
    plt.close(fig)

    return pd.DataFrame(rows)


# =============================================================================
# FIG 2  —  Latent drift vs training cloud  +  curve violation rate
#           Two lines on one plot — shows the causal link
# =============================================================================
def fig2_latent_violations(ctx):
    hdr("2. LATENT DRIFT & CURVE CONSISTENCY")

    z_paths  = _np(ctx["z_paths"])       # (N, T+1, d)
    z_mean   = _np(ctx["z_train_mean"])  # (d,)
    z_std    = _np(ctx["z_train_std"])   # (d,)
    P_paths  = _np(ctx["P_full_paths"])  # (N, T+1, tau_max+1)
    P0       = _np(ctx["P_full_0"])[0]
    tau      = _np(ctx["tau_grid"])
    times    = ctx["times"]
    n_paths, n_times, _ = P_paths.shape

    # ── fix 4: non-finite reporting ──────────────────────────────────────
    sub("2a. Non-finite entries in simulated discount curves")
    finite_mask       = np.isfinite(P_paths)             # (N, T+1, tau_max+1)
    frac_nonfinite    = 1.0 - float(np.mean(finite_mask))
    finite_curve_mask = np.all(finite_mask, axis=2)      # (N, T+1) — True if full curve finite
    frac_bad_curves   = 1.0 - float(np.mean(finite_curve_mask))
    print(f"  Non-finite P entries  : {100*frac_nonfinite:.3f}%")
    print(f"  Curves with any non-finite value: {100*frac_bad_curves:.2f}%")

    # ── initial curve checks ──────────────────────────────────────────────
    sub("2b. Initial curve (t = 0)")
    diffs0  = np.diff(P0)
    init_p0_ok   = pf(abs(P0[0] - 1.0) < TOL_P0,
                      "P(0,0) == 1", f"{P0[0]:.8f}",
                      f"{P0[0]:.8f}  (dev {abs(P0[0]-1):.2e})")
    init_pos_ok  = pf(np.all(P0 > 0), "All P(0,τ) > 0",
                      "yes", f"min = {P0.min():.6f}")
    init_mono_ok = pf(np.all(diffs0 < 0), "P(0,·) strictly decreasing",
                      "yes",
                      f"{np.sum(diffs0 >= 0)} violations")

    # ── simulated curve checks (fix 2: return sim checks) ────────────────
    sub("2c. Simulated paths — discount curve validity")

    # only check finite curves
    P_fin = P_paths.copy()
    P_fin[~finite_curve_mask] = np.nan   # NaN out bad curves for diff

    p0_dev      = np.abs(P_paths[:, :, 0] - 1.0)
    sim_p0_ok   = pf(float(np.nanmax(p0_dev)) < TOL_P0,
                     "P(t,0) == 1 on all paths",
                     f"max dev = {np.nanmax(p0_dev):.2e}",
                     f"max dev = {np.nanmax(p0_dev):.2e}")

    n_nonpos    = int(np.sum(P_paths[finite_curve_mask.repeat(
                    P_paths.shape[2]).reshape(P_paths.shape)] <= 0))
    sim_pos_ok  = pf(n_nonpos == 0, "P(t,τ) > 0 on finite curves",
                     "yes", f"{n_nonpos} non-positive entries")

    n_above1    = int(np.nansum(P_paths > 1.0 + TOL_P0))
    sim_le1_ok  = pf(n_above1 == 0, "P(t,τ) ≤ 1 everywhere",
                     "yes",
                     f"{n_above1} entries above 1  "
                     f"({100*n_above1/P_paths.size:.2f}%)")

    # monotonicity only on finite curves (fix 3: TOL_MONOTONE=1e-10)
    path_diffs      = np.diff(P_fin, axis=2)             # (N, T+1, tau_max)
    n_mono_viol     = int(np.nansum(path_diffs > TOL_MONOTONE))
    frac_mono_viol  = float(np.mean(
        np.any(path_diffs > TOL_MONOTONE, axis=2) & finite_curve_mask
    ))
    sim_mono_ok     = pf(n_mono_viol == 0,
                         "P(t,·) decreasing on all finite paths",
                         "yes",
                         f"{n_mono_viol} violations  "
                         f"({100*frac_mono_viol:.1f}% of curves)")

    # ── short-rate consistency ────────────────────────────────────────────
    sub("2d. Short-rate consistency  r_model vs −log P(t,1)")
    r_paths  = _np(ctx["r_paths"])
    tau1_idx = get_grid_index_for_value(tau, 1.0)
    P_tau1   = np.clip(P_paths[:, :, tau1_idx], 1e-14, None)
    r_curve  = -np.log(P_tau1) / 1.0
    diff_bp  = (r_paths - r_curve) * 10000
    rmse_bp  = float(np.sqrt(np.nanmean(diff_bp ** 2)))
    mae_bp   = float(np.nanmean(np.abs(diff_bp)))
    sr_ok    = pf(rmse_bp < 10.0,
                  "r_model ≈ −log P(t,1)  (RMSE < 10 bp)",
                  f"RMSE = {rmse_bp:.2f} bp,  MAE = {mae_bp:.2f} bp",
                  f"RMSE = {rmse_bp:.2f} bp — ODE drifting")

    # ── latent support ────────────────────────────────────────────────────
    sub("2e. Latent support  (% paths outside ±2σ training cloud)")
    outside      = np.abs(z_paths - z_mean) > 2.0 * z_std   # (N,T+1,d)
    frac_out     = np.mean(np.any(outside, axis=2), axis=0)  # (T+1,)
    frac_vio     = np.mean(
        np.any(path_diffs > TOL_MONOTONE, axis=2) & finite_curve_mask,
        axis=0
    )                                                          # (T+1,)

    print(f"\n  {'t':>5}  {'% outside ±2σ':>15}  {'% mono viol':>13}")
    print(f"  {'-'*5}  {'-'*15}  {'-'*13}")
    for s in [s for s in [0, 12, 24, 60, 120] if s < len(times)]:
        print(f"  {times[s]:>5.1f}  {frac_out[s]*100:>14.1f}%"
              f"  {frac_vio[s]*100:>12.1f}%")

    lat_ok = pf(frac_out[-1] < 0.30,
                f"< 30% paths outside ±2σ at T={N_STEPS*DT:.0f}Y",
                f"{frac_out[-1]*100:.1f}%  — within support",
                f"{frac_out[-1]*100:.1f}%  — extrapolating")

    # ── plot: two lines, one panel ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(times, frac_out * 100, linewidth=2.0, color=BLUE,
            label="% paths outside ±2σ latent training support")
    ax.plot(times, frac_vio * 100, linewidth=2.0, color=RED,
            linestyle="--",
            label="% paths with monotonicity violation in P(t, ·)")
    ax.set_xlabel("Simulation time (years)")
    ax.set_ylabel("% of paths")
    ax.set_title("Latent extrapolation and curve violations rise together")
    ax.legend(fontsize=9)
    ax.set_ylim(-2, 105)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_figure(fig, CFG, "fig2_latent_violations")
    plt.close(fig)

    return {
        "init_p0_ok":  init_p0_ok,  "init_pos_ok": init_pos_ok,
        "init_mono_ok": init_mono_ok,
        "sim_p0_ok":   sim_p0_ok,   "sim_pos_ok":  sim_pos_ok,
        "sim_le1_ok":  sim_le1_ok,  "sim_mono_ok": sim_mono_ok,
        "frac_mono_viol": frac_mono_viol,
        "frac_nonfinite": frac_nonfinite,
        "short_rate_rmse_bp": rmse_bp,
        "lat_ok": lat_ok,
    }


# =============================================================================
# FIG 3  —  Short-rate fan chart
# =============================================================================
def fig3_short_rate_fan(ctx):
    hdr("3. SHORT-RATE DISTRIBUTION")

    r_paths = _np(ctx["r_paths"])
    times   = ctx["times"]

    # guard against non-finite
    r_paths = np.where(np.isfinite(r_paths), r_paths, np.nan)

    pcts  = [5, 25, 50, 75, 95]
    bands = np.nanpercentile(r_paths * 100, pcts, axis=0)

    print(f"\n  {'t':>5}  {'Mean':>8}  {'Std':>7}  {'p5':>8}"
          f"  {'p95':>8}  {'P(r<0)':>8}")
    print(f"  {'-'*5}  {'-'*8}  {'-'*7}  {'-'*8}  {'-'*8}  {'-'*8}")
    for s in [s for s in [0, 12, 24, 60, 120] if s < r_paths.shape[1]]:
        r = r_paths[:, s]
        print(f"  {times[s]:>5.1f}"
              f"  {np.nanmean(r)*100:>7.3f}%"
              f"  {np.nanstd(r)*100:>6.3f}%"
              f"  {np.nanpercentile(r*100,5):>7.3f}%"
              f"  {np.nanpercentile(r*100,95):>7.3f}%"
              f"  {np.nanmean(r<0)*100:>7.1f}%")

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.fill_between(times, bands[0], bands[4],
                    alpha=0.18, color=BLUE, label="5–95%")
    ax.fill_between(times, bands[1], bands[3],
                    alpha=0.35, color=BLUE, label="25–75%")
    ax.plot(times, bands[2], linewidth=2.0, color=BLUE, label="Median")
    ax.axhline(0, linestyle="--", linewidth=0.9, color="black", alpha=0.5)
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Short rate r(t)  (%)")
    ax.set_title("Short-rate fan chart  —  percentile bands across paths")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_figure(fig, CFG, "fig3_short_rate_fan")
    plt.close(fig)


# =============================================================================
# FIG 4  —  Forward swap rate distribution at expiry
#           Descriptive under the simulated measure.
#           True drift = annuity-weighted discounted mean  (fix 1)
# =============================================================================
def fig4_swap_distributions(ctx, df_t1):
    hdr("4. FORWARD SWAP RATE DISTRIBUTION AT EXPIRY")

    print(
        "\n  Descriptive section under the simulated (risk-neutral) measure."
        "\n  Annuity-weighted drift: weighted_mean = Σ(D_T·A_T·F_T) / Σ(D_T·A_T)"
        "\n  This equals F_0 if the model is correctly risk-neutral.\n"
    )

    P_paths   = _np(ctx["P_full_paths"])      # (N, T+1, tau_max+1)
    tau       = _np(ctx["tau_grid"])
    disc      = _np(ctx["discount_paths"])    # (N, T+1)
    n_paths   = P_paths.shape[0]

    # rebuild t0 lookup from table
    t0 = {
        (int(r["Expiry (Y)"]), int(r["Tenor (Y)"])): r["F0 (bp)"] / 10000
        for _, r in df_t1.iterrows()
    }

    print(f"  {'Pair':>8}  {'K(bp)':>7}  {'Unwt mean(bp)':>14}"
          f"  {'Wt mean(bp)':>12}  {'Drift(bp)':>10}"
          f"  {'Std(bp)':>8}  {'%ITM':>6}")
    print(f"  {'-'*8}  {'-'*7}  {'-'*14}  {'-'*12}  {'-'*10}"
          f"  {'-'*8}  {'-'*6}")

    rows   = []
    pairs  = []
    drifts = []
    stds   = []

    for exp in EXPIRIES:
        exp_step = int(round(exp / DT))
        if exp_step >= P_paths.shape[1]:
            continue
        P_at_exp   = P_paths[:, exp_step, :]    # (N, tau_max+1)
        D_at_exp   = disc[:, exp_step]           # (N,)

        for ten in TENORS:
            if exp + ten > int(tau[-1]):
                continue
            K = t0.get((exp, ten))
            if K is None:
                continue

            F_T_list, A_T_list, D_T_list = [], [], []
            for pi in range(n_paths):
                if not np.all(np.isfinite(P_at_exp[pi])):
                    continue
                try:
                    res = swap_from_discount_curve_at_expiry(
                        P_at_exp[pi], tau, tenor=ten)
                    if np.isfinite(res["swap_rate"]) and \
                       np.isfinite(res["annuity"]) and \
                       np.isfinite(D_at_exp[pi]):
                        F_T_list.append(res["swap_rate"])
                        A_T_list.append(res["annuity"])
                        D_T_list.append(float(D_at_exp[pi]))
                except Exception:
                    pass

            if not F_T_list:
                continue

            F_T = np.array(F_T_list)
            A_T = np.array(A_T_list)
            D_T = np.array(D_T_list)

            # fix 1: annuity-measure weighted mean
            weights      = D_T * A_T
            wt_mean      = float(np.sum(weights * F_T) / np.sum(weights))
            unwt_mean    = float(F_T.mean())
            drift_bp     = (wt_mean - K) * 10000
            std_bp       = float(F_T.std() * 10000)
            pct_itm      = float(np.mean(F_T > K) * 100)
            label        = f"{exp}Y×{ten}Y"

            pairs.append(label)
            drifts.append(drift_bp)
            stds.append(std_bp)

            rows.append({
                "Expiry (Y)": exp, "Tenor (Y)": ten,
                "K (bp)":            round(K * 10000, 2),
                "Unweighted mean (bp)": round(unwt_mean * 10000, 2),
                "Annuity-wtd mean (bp)": round(wt_mean * 10000, 2),
                "Drift (bp)":        round(drift_bp, 2),
                "Std F_T (bp)":      round(std_bp, 2),
                "% ITM":             round(pct_itm, 1),
                "N valid":           len(F_T_list),
            })

            print(f"  {label:>8}  {K*10000:>7.1f}"
                  f"  {unwt_mean*10000:>14.1f}"
                  f"  {wt_mean*10000:>12.1f}"
                  f"  {drift_bp:>+10.1f}"
                  f"  {std_bp:>8.1f}  {pct_itm:>5.1f}%")

    mean_abs = float(np.mean(np.abs(drifts))) if drifts else float("nan")
    print(f"\n  Mean |annuity-weighted drift| : {mean_abs:.1f} bp"
          f"  ({'OK' if mean_abs < 20 else 'WARN — pricing biased'})")

    # ── plot: grouped bar chart — drift and std per pair ─────────────────
    n      = len(pairs)
    x      = np.arange(n)
    width  = 0.38

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # left: drift
    colors_drift = [RED if d < 0 else BLUE for d in drifts]
    axes[0].bar(x, drifts, width=0.6, color=colors_drift, alpha=0.85,
                edgecolor="white")
    axes[0].axhline(0, color="black", linewidth=0.9)
    axes[0].axhline(20,  color=GREY, linewidth=0.8, linestyle=":",
                    label="+20 bp threshold")
    axes[0].axhline(-20, color=GREY, linewidth=0.8, linestyle=":")
    axes[0].set_xticks(x); axes[0].set_xticklabels(pairs, rotation=30, ha="right")
    axes[0].set_ylabel("Drift  E[F_T]_wt − K  (bp)")
    axes[0].set_title("Annuity-weighted drift per (expiry × tenor)\n"
                      "Blue = paths above strike, Red = below")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, axis="y", alpha=0.3)

    # right: std of F_T
    axes[1].bar(x, stds, width=0.6, color=BLUE, alpha=0.75,
                edgecolor="white")
    axes[1].set_xticks(x); axes[1].set_xticklabels(pairs, rotation=30, ha="right")
    axes[1].set_ylabel("Std of F_T  (bp)")
    axes[1].set_title("Dispersion of realized swap rate at expiry\n"
                      "Captures the vol the model generates")
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    save_figure(fig, CFG, "fig4_swap_distributions")
    plt.close(fig)

    return pd.DataFrame(rows)


# =============================================================================
# FIG 5  —  Implied vol bar chart grouped by expiry
# =============================================================================
def fig5_vol_surface(ctx):
    hdr("5. MC SWAPTION PRICES & IMPLIED NORMAL VOLS")

    print(f"\n  ATM payer swaptions, Bachelier (normal) vol."
          f"\n  EUR ATM reference: 1Y expiry ~40–80 bp, 5Y ~20–35 bp, "
          f"10Y ~15–25 bp.\n")
    print(f"  {'Pair':>8}  {'F0(bp)':>7}  {'Vol(bp)':>8}"
          f"  {'Rel SE':>7}  {'N valid':>8}")
    print(f"  {'-'*8}  {'-'*7}  {'-'*8}  {'-'*7}  {'-'*8}")

    tau_max = int(_np(ctx["tau_grid"])[-1])
    rows    = []

    # collect by expiry group for grouped bar chart
    groups  = {exp: {"labels": [], "vols": [], "ses": []} for exp in EXPIRIES}

    for exp in EXPIRIES:
        for ten in TENORS:
            if exp + ten > tau_max:
                continue
            if int(round(exp / DT)) >= ctx["P_full_paths"].shape[1]:
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = atm_swaption_mc_price_from_simulation(
                        ctx=ctx, expiry=exp, tenor=ten,
                        payer=True, accrual=1.0, notional=1.0,
                    )
                iv   = res["implied_normal_vol"] * 10000
                rse  = (res["mc_stderr"] / res["mc_price"]
                        if res["mc_price"] > 1e-16 else np.nan)
                f0   = res["quote"]["forward_swap"] * 10000
                nval = int(res["valid_mask"].sum())

                groups[exp]["labels"].append(f"{ten}Y tenor")
                groups[exp]["vols"].append(iv if np.isfinite(iv) else 0.0)
                groups[exp]["ses"].append(rse * 100 if np.isfinite(rse) else np.nan)

                rows.append({
                    "Expiry (Y)": exp, "Tenor (Y)": ten,
                    "F0 (bp)":        round(f0, 2),
                    "MC Price":       round(res["mc_price"], 8),
                    "MC Stderr":      round(res["mc_stderr"], 8),
                    "Rel SE (%)":     round(rse * 100, 2) if np.isfinite(rse) else np.nan,
                    "Impl Vol (bp)":  round(iv, 2) if np.isfinite(iv) else np.nan,
                    "N valid":        nval,
                })
                print(f"  {exp}Y×{ten}Y  {f0:>7.1f}  {iv:>8.2f}"
                      f"  {rse:>6.1%}  {nval:>5}/{N_PATHS}")
            except Exception as e:
                print(f"  {exp}Y×{ten}Y  ERROR: {e}")

    # ── plot: grouped bar chart, one bar-group per expiry ────────────────
    n_exp    = len(EXPIRIES)
    n_ten    = len(TENORS)
    bar_w    = 0.22
    x_base   = np.arange(n_ten)
    offsets  = np.linspace(-(n_exp-1)/2, (n_exp-1)/2, n_exp) * bar_w
    colors   = [BLUE, "#4a7c2c", RED]

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (exp, off, col) in enumerate(zip(EXPIRIES, offsets, colors)):
        g = groups[exp]
        if not g["vols"]:
            continue
        xs = x_base[:len(g["vols"])] + off
        ax.bar(xs, g["vols"], width=bar_w, color=col, alpha=0.82,
               edgecolor="white", label=f"{exp}Y expiry")

    tenor_labels = [f"{t}Y tenor" for t in TENORS]
    ax.set_xticks(x_base)
    ax.set_xticklabels(tenor_labels)
    ax.set_ylabel("Implied normal vol (bp)")
    ax.set_title("Model-implied ATM normal vol  —  grouped by option expiry")
    ax.legend(title="Expiry", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    save_figure(fig, CFG, "fig5_vol_surface")
    plt.close(fig)

    return pd.DataFrame(rows)


# =============================================================================
# EXCEL  —  three sheets
# =============================================================================
def write_excel(df_t1, df_t2, df_t3):
    path = os.path.join(OUT_DIR, f"diagnostics_{AS_OF_DATE}.xlsx")
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df_t1.to_excel(writer, sheet_name="ForwardRates",    index=False)
        df_t2.to_excel(writer, sheet_name="SwapDistribution", index=False)
        df_t3.to_excel(writer, sheet_name="ImpliedVols",      index=False)
        for sheet in writer.sheets.values():
            for col in sheet.columns:
                w = max(
                    (len(str(c.value)) if c.value is not None else 0)
                    for c in col
                )
                sheet.column_dimensions[col[0].column_letter].width = \
                    min(w + 3, 32)
    print(f"\n  Excel → {path}")


# =============================================================================
# SUMMARY
# =============================================================================
def section_summary(cons, df_t2, df_t3):
    hdr("SUMMARY")

    drifts  = df_t2["Drift (bp)"].abs().tolist() if df_t2 is not None else []
    vols    = df_t3["Impl Vol (bp)"].dropna().tolist() if df_t3 is not None else []
    rel_ses = (df_t3["Rel SE (%)"].dropna() / 100).tolist() if df_t3 is not None else []

    print()
    print(f"  Non-finite P entries (sim)       : {cons['frac_nonfinite']*100:.3f}%")
    print(f"  Curves with monotonicity violation: "
          f"{cons['frac_mono_viol']*100:.1f}%")
    print(f"  Short-rate RMSE vs curve-implied : "
          f"{cons['short_rate_rmse_bp']:.2f} bp")
    print(f"  Mean |annuity-wt drift|          : "
          f"{np.mean(drifts):.1f} bp" if drifts else
          "  Mean |annuity-wt drift|          : n/a")
    print(f"  Implied vol range                : "
          f"[{min(vols):.1f}, {max(vols):.1f}] bp" if vols else
          "  Implied vol range                : n/a")
    print(f"  Max relative MC stderr           : "
          f"{max(rel_ses):.1%}" if rel_ses else
          "  Max relative MC stderr           : n/a")

    # fix 2: summary uses simulated checks
    all_pass = (
        cons["init_p0_ok"]   and cons["init_pos_ok"]  and
        cons["init_mono_ok"] and cons["sim_p0_ok"]    and
        cons["sim_pos_ok"]   and cons["sim_le1_ok"]   and
        cons["sim_mono_ok"]  and
        cons["short_rate_rmse_bp"] < 10.0 and
        cons["lat_ok"] and
        (not drifts  or np.mean(drifts)  < 20.0) and
        (not rel_ses or max(rel_ses)     < 0.05)
    )
    print()
    if all_pass:
        print("  *** ALL CHECKS PASSED — model ready for market vol comparison ***")
    else:
        print("  *** ONE OR MORE CHECKS FAILED — review above before proceeding ***")


# =============================================================================
# MAIN
# =============================================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(SEP2)
    print("  PRE-PRICING DIAGNOSTICS  —  FINAL")
    print(f"  Checkpoint : {os.path.basename(CHECKPOINT_PATH)}")
    print(f"  Date       : {AS_OF_DATE}  |  CCY : {CCY_FILTER}")
    print(f"  Paths      : {N_PATHS}  |  Steps : {N_STEPS}"
          f"  |  Horizon : {N_STEPS*DT:.0f}Y")
    print(SEP2)

    ctx  = run_simulation(
        checkpoint_path=CHECKPOINT_PATH,
        ccy_filter=CCY_FILTER,
        as_of_date=AS_OF_DATE,
        n_paths=N_PATHS,
        n_steps=N_STEPS,
        dt=DT,
        show_plot=False,
    )

    df_t1 = fig1_initial_curve(ctx)
    cons  = fig2_latent_violations(ctx)
    fig3_short_rate_fan(ctx)
    df_t2 = fig4_swap_distributions(ctx, df_t1)
    df_t3 = fig5_vol_surface(ctx)

    write_excel(df_t1, df_t2, df_t3)
    section_summary(cons, df_t2, df_t3)

    print(f"\n{SEP2}\n  Output → {OUT_DIR}\n{SEP2}\n")


if __name__ == "__main__":
    main()