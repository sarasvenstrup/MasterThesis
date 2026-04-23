import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import seaborn as sns
import matplotlib as mpl
from cycler import cycler

# ============================================================
# Config — adjust to match your Training.py settings
# ============================================================
LATENT_DIM = 2
EPOCHS     = 3500       # match your ep{EPOCHS} folder
USE        = "bbg"
VARIANT    = "stable"   # match config.VARIANT

SHOW_PLOTS = True

# ============================================================
# Simulation overlay — set to True to enable Plot 3
# ============================================================
PLOT_SIMULATION = True

# These must match what was passed to run_simulation()
SIM_CCY     = "EUR"   # ccy_filter used in simulate_model.py (or "all")
SIM_N_PATHS = 500     # n_paths
SIM_N_STEPS = 24      # n_steps

# ============================================================
# Paths
# ============================================================
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(
    REPO_ROOT, "Figures", "TrainingResults",
    f"dim{LATENT_DIM}_{VARIANT}", f"ep{EPOCHS}"
)
SIM_DIR = FIGURES_DIR  # simulation CSVs are saved alongside the checkpoint

TRAINING_CSV = os.path.join(FIGURES_DIR, f"latent_z_{USE}_dim{LATENT_DIM}_ep{EPOCHS}.csv")
SIM_SUMMARY  = os.path.join(SIM_DIR, f"latent_sim_summary_{SIM_CCY}_npaths{SIM_N_PATHS}_nsteps{SIM_N_STEPS}.csv")
SIM_SUBSET   = os.path.join(SIM_DIR, f"latent_sim_subset_{SIM_CCY}_npaths{SIM_N_PATHS}_nsteps{SIM_N_STEPS}.csv")

# ============================================================
# Theme + currency colors
# ============================================================
CCY_ORDER = ["AUD", "CAD", "DKK", "EUR", "JPY", "NOK", "SEK", "GBP", "USD"]

def set_paper_theme():
    sns.set_theme(context="paper", style="darkgrid", font_scale=1.05)
    full_palette = sns.color_palette("tab20b", 20)
    selected_indices = [0, 1, 2, 3, 12, 13, 14, 15, 8]
    palette = [full_palette[i] for i in selected_indices]
    mpl.rcParams.update({
        "figure.dpi": 180,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.edgecolor": "0.8",
        "axes.linewidth": 1.0,
        "axes.grid": True,
        "grid.color": "0.9",
        "grid.linewidth": 1.0,
        "font.size": 11,
        "axes.labelcolor": "0.2",
        "xtick.color": "0.2",
        "ytick.color": "0.2",
        "legend.frameon": False,
        "lines.linewidth": 1.6,
        "lines.markersize": 5.0,
    })
    mpl.rcParams["axes.prop_cycle"] = cycler(color=palette)
    return palette

palette = set_paper_theme()
CCY_COLOR = {ccy: palette[i % len(palette)] for i, ccy in enumerate(CCY_ORDER)}

os.makedirs(FIGURES_DIR, exist_ok=True)

# ============================================================
# Load training data
# ============================================================
print("Loading training latent CSV:", TRAINING_CSV)
df = pd.read_csv(TRAINING_CSV, parse_dates=["as_of_date"])
print(f"  {len(df)} rows | currencies: {sorted(df['ccy'].unique())}")
print(f"  Date range: {df['as_of_date'].min().date()} — {df['as_of_date'].max().date()}")

# ============================================================
# Plot 1 — All currencies, colored by currency
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))

for ccy in CCY_ORDER:
    sub = df[df["ccy"] == ccy]
    if sub.empty:
        continue
    ax.scatter(
        sub["z_1"], sub["z_2"],
        s=8, alpha=0.5, linewidths=0,
        color=CCY_COLOR[ccy], label=ccy, rasterized=True
    )

ax.set_xlabel("$z_1$")
ax.set_ylabel("$z_2$")
ax.set_title(f"Latent space — all currencies  (dim={LATENT_DIM}, ep={EPOCHS})")
ax.legend(loc="best", markerscale=2.5, title="Currency")

fig.tight_layout()
out1 = os.path.join(FIGURES_DIR, f"latent_by_currency_{USE}_dim{LATENT_DIM}_ep{EPOCHS}.png")
fig.savefig(out1)
print("Saved:", out1)
if SHOW_PLOTS:
    plt.show()
plt.close(fig)

# ============================================================
# Plot 2 — Per-currency subplots, colored by date
# ============================================================
ccys_present = [c for c in CCY_ORDER if c in df["ccy"].values]
n = len(ccys_present)
ncols = 3
nrows = int(np.ceil(n / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows))
axes = axes.flatten()

date_min = df["as_of_date"].min()
date_max = df["as_of_date"].max()
norm = mcolors.Normalize(
    vmin=pd.Timestamp(date_min).timestamp(),
    vmax=pd.Timestamp(date_max).timestamp()
)
cmap_date = cm.plasma

for i, ccy in enumerate(ccys_present):
    ax = axes[i]
    sub = df[df["ccy"] == ccy].sort_values("as_of_date")
    ts = sub["as_of_date"].apply(lambda d: pd.Timestamp(d).timestamp()).values
    ax.scatter(
        sub["z_1"], sub["z_2"],
        c=ts, cmap=cmap_date, norm=norm,
        s=8, alpha=0.7, linewidths=0, rasterized=True
    )
    ax.set_title(ccy, fontsize=12, fontweight="bold")
    ax.set_xlabel("$z_1$")
    ax.set_ylabel("$z_2$")

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

cbar_ax = fig.add_axes([1.01, 0.15, 0.02, 0.7])
sm = cm.ScalarMappable(cmap=cmap_date, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)
tick_ts = np.linspace(pd.Timestamp(date_min).timestamp(), pd.Timestamp(date_max).timestamp(), 6)
cbar.set_ticks(tick_ts)
cbar.set_ticklabels([pd.Timestamp(t, unit="s").strftime("%Y") for t in tick_ts])
cbar.set_label("Date", rotation=270, labelpad=15)

fig.suptitle(
    f"Latent trajectories per currency — colored by date  (dim={LATENT_DIM}, ep={EPOCHS})",
    fontsize=13, y=1.02
)
fig.tight_layout()
out2 = os.path.join(FIGURES_DIR, f"latent_by_date_{USE}_dim{LATENT_DIM}_ep{EPOCHS}.png")
fig.savefig(out2, bbox_inches="tight")
print("Saved:", out2)
if SHOW_PLOTS:
    plt.show()
plt.close(fig)

# ============================================================
# Plot 3 — Training cloud vs simulated paths (optional)
# ============================================================
if not PLOT_SIMULATION:
    print("\n[SKIP] Plot 3: PLOT_SIMULATION=False")
elif not os.path.exists(SIM_SUMMARY) or not os.path.exists(SIM_SUBSET):
    print(
        f"\n[SKIP] Plot 3: simulation CSVs not found. Expected:\n"
        f"  {SIM_SUMMARY}\n"
        f"  {SIM_SUBSET}\n"
        f"Run simulate_model.py first, then re-run this script."
    )
else:
    print("\nLoading simulation CSVs...")
    df_summary = pd.read_csv(SIM_SUMMARY)
    df_subset  = pd.read_csv(SIM_SUBSET)
    print(f"  Summary: {len(df_summary)} timesteps")
    print(f"  Subset:  {len(df_subset['path'].unique())} paths")

    sim_times = df_summary["time"].values
    sim_norm  = mcolors.Normalize(vmin=sim_times.min(), vmax=sim_times.max())
    sim_cmap  = cm.autumn

    fig, ax = plt.subplots(figsize=(8, 6))

    # --- Background: full training cloud ---
    # If filtered to one currency, dim others and highlight that currency
    if SIM_CCY != "all" and SIM_CCY in df["ccy"].values:
        df_bg  = df[df["ccy"] != SIM_CCY]
        df_ccy = df[df["ccy"] == SIM_CCY]
        ax.scatter(
            df_bg["z_1"], df_bg["z_2"],
            s=5, alpha=0.12, linewidths=0,
            color="grey", rasterized=True, label="Other currencies (training)"
        )
        ax.scatter(
            df_ccy["z_1"], df_ccy["z_2"],
            s=6, alpha=0.35, linewidths=0,
            color=CCY_COLOR.get(SIM_CCY, "steelblue"),
            rasterized=True, label=f"{SIM_CCY} (training)"
        )
    else:
        ax.scatter(
            df["z_1"], df["z_2"],
            s=5, alpha=0.15, linewidths=0,
            color="grey", rasterized=True, label="Training cloud"
        )

    # --- Simulated path subset: thin lines, each segment colored by time ---
    first_path = True
    for path_id in df_subset["path"].unique():
        path = df_subset[df_subset["path"] == path_id].sort_values("time")
        for t_idx in range(len(path) - 1):
            t_val = path["time"].iloc[t_idx]
            color = sim_cmap(sim_norm(t_val))
            ax.plot(
                path["z_1"].iloc[t_idx:t_idx + 2],
                path["z_2"].iloc[t_idx:t_idx + 2],
                color=color, alpha=0.35, linewidth=0.7,
                label="Simulated paths" if (first_path and t_idx == 0) else None
            )
        first_path = False

    # --- Percentile bands (filled polygons going forward then back) ---
    ax.fill(
        np.concatenate([df_summary["z1_p5"],  df_summary["z1_p95"] [::-1]]),
        np.concatenate([df_summary["z2_p5"],  df_summary["z2_p95"] [::-1]]),
        alpha=0.10, color="black", label="5–95th pct band"
    )
    ax.fill(
        np.concatenate([df_summary["z1_p25"], df_summary["z1_p75"][::-1]]),
        np.concatenate([df_summary["z2_p25"], df_summary["z2_p75"][::-1]]),
        alpha=0.18, color="black", label="25–75th pct band"
    )

    # --- Mean path ---
    ax.plot(
        df_summary["z1_mean"], df_summary["z2_mean"],
        color="black", linewidth=2.0, zorder=5, label="Mean path"
    )

    # --- Start point ---
    ax.scatter(
        df_summary["z1_mean"].iloc[0],
        df_summary["z2_mean"].iloc[0],
        color="green", s=80, zorder=6, marker="o", label="Start $z_0$"
    )

    # --- Colorbar for simulation time ---
    sm_sim = cm.ScalarMappable(cmap=sim_cmap, norm=sim_norm)
    sm_sim.set_array([])
    cbar_sim = fig.colorbar(sm_sim, ax=ax, pad=0.02)
    cbar_sim.set_label("Simulation time (years)", rotation=270, labelpad=15)

    ax.set_xlabel("$z_1$")
    ax.set_ylabel("$z_2$")
    ccy_label = SIM_CCY if SIM_CCY != "all" else "all currencies"
    ax.set_title(
        f"Training cloud vs simulated paths — {ccy_label}\n"
        f"({SIM_N_PATHS} paths, {SIM_N_STEPS} steps, dt=1/12)"
    )
    ax.legend(loc="best", markerscale=1.5)

    fig.tight_layout()
    out3 = os.path.join(
        FIGURES_DIR,
        f"latent_sim_vs_training_{SIM_CCY}_npaths{SIM_N_PATHS}_nsteps{SIM_N_STEPS}.png"
    )
    fig.savefig(out3, bbox_inches="tight")
    print("Saved:", out3)
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)