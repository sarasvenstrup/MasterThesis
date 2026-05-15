"""
Generate thesis comparison figures for the pricing chapter.

Figures produced:
  fig_pricing_comparison_timeseries.pdf  -- 9-cell vol error over time, base vs constant MPR
  fig_pricing_heatmap_comparison.pdf     -- side-by-side MAE heatmaps (base vs constant MPR)
  fig_pricing_scatter_cmpr.pdf           -- constant MPR model vs market scatter (9 cells)
  fig_pricing_forward_bias_cmpr.pdf      -- forward bias over time, constant MPR

Run from repo root:
    python Code/Pricing/_make_comparison_figures.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import Normalize
from matplotlib import cm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

BASE_CSV  = os.path.join(PROJECT_ROOT, "Figures", "pricing", "eval_base",         "per_cell_final.csv")
CMPR_CSV  = os.path.join(PROJECT_ROOT, "Figures", "pricing", "eval_constant_mpr",  "per_cell_final.csv")
OUT_DIR   = os.path.join(PROJECT_ROOT, "Figures", "pricing", "comparison")
os.makedirs(OUT_DIR, exist_ok=True)

# ── load ───────────────────────────────────────────────────────────────────────
df_base = pd.read_csv(BASE_CSV);  df_base["date"] = pd.to_datetime(df_base["date"])
df_cmpr = pd.read_csv(CMPR_CSV);  df_cmpr["date"] = pd.to_datetime(df_cmpr["date"])

EXPIRY_VALS = [1, 5, 10]
TENOR_VALS  = [1, 5, 10]

def get(df, e, t):
    return df[(df["expiry"]==e) & (df["tenor"]==t)].sort_values("date")

# ── Figure 1: Vol error over time — base (grey) / bounded lambda (orange) / constant MPR (coloured)
fig, axes = plt.subplots(3, 3, figsize=(13, 9))
row_colors = ["#2563eb", "#16a34a", "#dc2626"]  # per expiry row (Constant MPR)

for i, e in enumerate(EXPIRY_VALS):
    for j, t in enumerate(TENOR_VALS):
        ax = axes[i][j]
        b  = get(df_base, e, t)
        c  = get(df_cmpr, e, t)

        if len(b) == 0 and len(c) == 0:
            ax.set_visible(False); continue

        # Shade test region
        test_c = c[c["split"]=="test"]["date"]
        if len(test_c):
            ax.axvspan(test_c.min(), test_c.max(), alpha=0.07, color="#f59e0b",
                       label="Test period" if (i==0 and j==0) else "")

        ax.axhline(0, color="black", lw=0.7, ls="--")

        # Base model — thin grey
        if len(b):
            ax.plot(b["date"], b["vol_error_bp"],
                    color="#9ca3af", lw=0.9, alpha=0.85,
                    label="Base" if (i==0 and j==0) else "")

        # Constant MPR — coloured solid
        if len(c):
            ax.plot(c["date"], c["vol_error_bp"],
                    color=row_colors[i], lw=1.2, alpha=0.95,
                    label="Const. MPR" if (i==0 and j==0) else "")

        # Stats annotations
        mae_b = b["vol_error_bp"].abs().mean() if len(b) else float("nan")
        mae_c = c["vol_error_bp"].abs().mean() if len(c) else float("nan")
        ax.text(0.03, 0.97,
                f"Base MAE={mae_b:.0f} bp\nCMPR MAE={mae_c:.0f} bp",
                transform=ax.transAxes, fontsize=6.5, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.8))

        ax.set_title(f"{e}Yx{t}Y", fontsize=9, fontweight="bold")
        ax.set_ylabel("Error (bp)", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

# Single legend on first subplot
handles, labels = axes[0][0].get_legend_handles_labels()
axes[0][0].legend(handles, labels, fontsize=7, loc="lower right", framealpha=0.85)

fig.suptitle("Swaption Vol Error Over Time: Base Model vs Constant MPR\n"
             "(amber = test set; dashed = zero error)",
             fontsize=11, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.96])
out1 = os.path.join(OUT_DIR, "fig_pricing_comparison_timeseries.pdf")
fig.savefig(out1, bbox_inches="tight")
fig.savefig(out1.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out1}")

# ── Figure 2: Three-panel MAE heatmaps (base / bounded lambda / constant MPR) ──
def mae_grid(df, split=None):
    g = np.full((3, 3), np.nan)
    for i, e in enumerate(EXPIRY_VALS):
        for j, t in enumerate(TENOR_VALS):
            sub = df[(df["expiry"]==e) & (df["tenor"]==t)]
            if split:
                sub = sub[sub["split"]==split]
            if len(sub):
                g[i, j] = sub["vol_error_bp"].abs().mean()
    return g

for split_key, split_label in [(None, "All Dates"), ("test", "Test Set"), ("train", "Train Set")]:
    gb = mae_grid(df_base, split_key)
    gc = mae_grid(df_cmpr, split_key)

    # Common colour scale
    vmax = np.nanmax([gb, gc])
    norm = Normalize(vmin=0, vmax=vmax)
    cmap = cm.YlOrRd

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8))

    for ax, g, title in [(ax1, gb, "Base Model"), (ax2, gc, "Constant MPR")]:
        im = ax.imshow(g, cmap=cmap, norm=norm, aspect="auto")
        ax.set_xticks(range(3)); ax.set_yticks(range(3))
        ax.set_xticklabels([f"{t}Y" for t in TENOR_VALS], fontsize=10)
        ax.set_yticklabels([f"{e}Y" for e in EXPIRY_VALS], fontsize=10)
        ax.set_xlabel("Tenor", fontsize=10); ax.set_ylabel("Expiry", fontsize=10)
        for ii in range(3):
            for jj in range(3):
                if not np.isnan(g[ii, jj]):
                    ax.text(jj, ii, f"{g[ii,jj]:.0f}", ha="center", va="center",
                            fontsize=11, fontweight="bold",
                            color="white" if g[ii,jj] > vmax*0.6 else "black")
        overall = np.nanmean(g)
        ax.set_title(f"{title}\n(overall MAE = {overall:.0f} bp)", fontsize=10)

    plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=[ax1, ax2],
                 label="Vol MAE (bp)", shrink=0.8)
    fig.suptitle(f"Per-Cell ATM Vol MAE (bp) — {split_label}, EUR",
                 fontsize=11, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])

    tag = split_key if split_key else "all"
    out2 = os.path.join(OUT_DIR, f"fig_pricing_heatmap_comparison_{tag}.pdf")
    fig.savefig(out2, bbox_inches="tight")
    fig.savefig(out2.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out2}")

# ── Figure 3: Constant MPR scatter — model vs market ──────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(11, 9))
for i, e in enumerate(EXPIRY_VALS):
    for j, t in enumerate(TENOR_VALS):
        ax  = axes[i][j]
        sub = df_cmpr[(df_cmpr["expiry"]==e) & (df_cmpr["tenor"]==t)]
        if len(sub) == 0:
            ax.set_visible(False); continue

        for sp, c, mk in [("train", "#16a34a", "o"), ("test", "#2563eb", "^")]:
            ss = sub[sub["split"]==sp]
            ax.scatter(ss["mkt_bp"], ss["sigma_str_bp"], s=14, alpha=0.75,
                       color=c, marker=mk, rasterized=True, label=sp)

        lo = min(sub["mkt_bp"].min(), sub["sigma_str_bp"].min()) * 0.90
        hi = max(sub["mkt_bp"].max(), sub["sigma_str_bp"].max()) * 1.10
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)

        s_all = sub["vol_error_bp"].abs().mean()
        s_tst = sub[sub["split"]=="test"]["vol_error_bp"].abs().mean()
        ax.text(0.05, 0.95,
                f"MAE = {s_all:.0f} bp\ntest = {s_tst:.0f} bp",
                transform=ax.transAxes, fontsize=7, va="top",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))

        ax.set_title(f"{e}Yx{t}Y", fontsize=9, fontweight="bold")
        ax.set_xlabel("Market vol (bp)", fontsize=7)
        ax.set_ylabel("Model vol (bp)", fontsize=7)
        ax.tick_params(labelsize=7)

axes[0][0].legend(fontsize=7, loc="lower right")
fig.suptitle(r"Constant MPR: Model vs Market ATM Straddle Vol"
             "\n(circles = train, triangles = test; dashed = 45° line)",
             fontsize=11, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.96])
out3 = os.path.join(OUT_DIR, "fig_pricing_scatter_cmpr.pdf")
fig.savefig(out3, bbox_inches="tight")
fig.savefig(out3.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out3}")

# ── Figure 4: Forward bias over time — Constant MPR ───────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(13, 9))
for i, e in enumerate(EXPIRY_VALS):
    for j, t in enumerate(TENOR_VALS):
        ax  = axes[i][j]
        sub = get(df_cmpr, e, t)
        if len(sub) == 0:
            ax.set_visible(False); continue

        test_sub = sub[sub["split"]=="test"]["date"]
        if len(test_sub):
            ax.axvspan(test_sub.min(), test_sub.max(), alpha=0.07, color="#f59e0b")

        ax.plot(sub["date"], sub["forward_bias_bp"], color="#7c3aed", lw=1.1)
        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.fill_between(sub["date"], sub["forward_bias_bp"], 0,
                        where=sub["forward_bias_bp"] > 0, alpha=0.12, color="#dc2626")
        ax.fill_between(sub["date"], sub["forward_bias_bp"], 0,
                        where=sub["forward_bias_bp"] < 0, alpha=0.12, color="#2563eb")

        mean_fwd = sub["forward_bias_bp"].mean()
        ax.text(0.03, 0.97, f"mean = {mean_fwd:+.0f} bp",
                transform=ax.transAxes, fontsize=7, va="top",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))

        ax.set_title(f"{e}Yx{t}Y", fontsize=9, fontweight="bold")
        ax.set_ylabel("Fwd bias (bp)", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

fig.suptitle("Constant MPR: Forward Bias $(V_\\mathrm{pay} - V_\\mathrm{rec})/A_0$"
             " (target: 0 bp, amber = test)",
             fontsize=11, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.96])
out4 = os.path.join(OUT_DIR, "fig_pricing_forward_bias_cmpr.pdf")
fig.savefig(out4, bbox_inches="tight")
fig.savefig(out4.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out4}")

print(f"\nAll comparison figures written to: {OUT_DIR}")
