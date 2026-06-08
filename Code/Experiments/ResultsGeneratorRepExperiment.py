# =============================================================================
# ResultsGeneratorRepExperiment.py
#
# Reads completed prevalence-sweep CSVs and produces a combined thesis figure:
#
#   Fig 1 — side-by-side: baseline (left) and stable (right)
#       Each panel shows RMSE_negative and RMSE_positive vs prevalence.
#
#   Fig 2 — baseline only (cleaner single-panel for thesis use)
#
# Usage:
#   python Code/Experiments/ResultsGeneratorRepExperiment.py
#   python Code/Experiments/ResultsGeneratorRepExperiment.py --dim 2 --epochs 2000 --n-train 2000
# =============================================================================

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Code.load_swapdata import set_paper_theme

_paper_palette = set_paper_theme()

REGIME      = "any_negative"
DEFAULT_DIM = 2
DEFAULT_EP  = 2000
DEFAULT_N   = 2000


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim",     type=int, default=DEFAULT_DIM)
    ap.add_argument("--epochs",  type=int, default=DEFAULT_EP)
    ap.add_argument("--n-train", type=int, default=DEFAULT_N)
    ap.add_argument("--regime",  default=REGIME)
    return ap.parse_args()


def load_csv(regime, dim, ep, n, model_type):
    folder = f"{regime}_dim{dim}_ep{ep}_N{n}_{model_type}"
    path = os.path.join(REPO_ROOT, "Figures", "Experiments", "PrevalenceSweep",
                        folder, "sweep_results.csv")
    if not os.path.exists(path):
        print(f"  [missing] {path}")
        return None
    df = pd.read_csv(path)
    print(f"  [loaded]  {folder}  ({len(df)} rows)")
    return df


def plot_single(ax, df, title, color_reg=None, color_non=None):
    """Plot regime + non-regime RMSE on one axis."""
    if color_reg is None:
        color_reg = _paper_palette[0]
    if color_non is None:
        color_non = _paper_palette[4]

    if df is None:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return

    _keep = [0.05, 0.15, 0.30, 0.50, 0.75, 0.90]
    df = df[df["p_target"].isin(_keep)]
    ps = df["p_target"] * 100
    ax.plot(ps, df["rmse_bps_eval_regime"],    "o-", color=color_reg,
            lw=1.8, ms=5, label="Negative-rate curves")
    ax.plot(ps, df["rmse_bps_eval_nonregime"], "s--", color=color_non,
            lw=1.4, ms=4, label="Normal-rate curves")

    ax.set_xlabel("Share of negative curves in training (%)")
    ax.set_ylabel("RMSE (bps)")
    ax.set_title(title)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))


def main():
    args = parse_args()

    print("Loading CSVs …")
    df_base   = load_csv(args.regime, args.dim, args.epochs, args.n_train, "baseline")
    df_stable = load_csv(args.regime, args.dim, args.epochs, args.n_train, "stable")

    out_dir = os.path.join(REPO_ROOT, "Figures", "Experiments", "PrevalenceSweep",
                           f"plots_dim{args.dim}_ep{args.epochs}_N{args.n_train}")
    os.makedirs(out_dir, exist_ok=True)

    # ── Figure 1: side-by-side baseline (dim=2) vs stable (dim=4), ep3500 N1500
    df_fig1_base   = load_csv(args.regime, 2, 3500, 1500, "baseline")
    df_fig1_stable = load_csv(args.regime, 4, 3500, 1500, "stable")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    plot_single(axes[0], df_fig1_base,
                f"Baseline  ($\\ell=2$)")
    plot_single(axes[1], df_fig1_stable,
                f"Stable  ($\\ell=4$)")
    axes[1].set_ylabel("")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               bbox_to_anchor=(0.5, -0.02), ncol=2, borderaxespad=0)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    p1 = os.path.join(out_dir, "prevalence_baseline_vs_stable.png")
    fig.savefig(p1, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p1}")

    # ── Figure 2: baseline only, ep3500 N1500 (thesis-quality single panel) ──
    df_thesis = load_csv(args.regime, args.dim, 3500, 1500, "baseline")
    if df_thesis is not None:
        _keep = [0.05, 0.15, 0.30, 0.50, 0.75, 0.90]
        df_thesis = df_thesis[df_thesis["p_target"].isin(_keep)]
        fig2, ax2 = plt.subplots(figsize=(7, 4.5))
        ax2.plot(df_thesis["p_target"] * 100, df_thesis["rmse_bps_eval_regime"],
                 "o-", color=_paper_palette[0], lw=1.8, ms=5,
                 label="Negative-rate curves")
        ax2.plot(df_thesis["p_target"] * 100, df_thesis["rmse_bps_eval_nonregime"],
                 "s--", color=_paper_palette[4], lw=1.4, ms=4,
                 label="Normal-rate curves")
        ax2.set_xlabel("Share of negative curves in training (%)")
        ax2.set_ylabel("RMSE (bps)")
        ax2.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
        leg = ax2.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0)
        fig2.tight_layout()
        p2 = os.path.join(out_dir, "prevalence_baseline_only.png")
        fig2.savefig(p2, dpi=300, bbox_inches="tight")
        plt.close(fig2)
        print(f"Saved: {p2}")

    # ── Print summary table ────────────────────────────────────────────────
    for label, df in [("Baseline", df_base), ("Stable", df_stable)]:
        if df is None:
            continue
        print(f"\n{label}:")
        print(df[["p_target", "n_regime_train",
                   "rmse_bps_eval_regime", "rmse_bps_eval_nonregime"]].to_string(index=False))


if __name__ == "__main__":
    main()
