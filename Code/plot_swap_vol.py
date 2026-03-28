# Code/plot_volatility_process.py
# Saves to: Figures/<USE>/swap_vol/
# Computes EWMA vol from the SAME swap data you train on (no changes to my_data)

import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from typing import List, Optional

# -----------------------------
# Repo root setup (same as training script)
# -----------------------------
try:
    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    REPO_ROOT = os.getcwd()

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Code.utils import helpers as H
from Code.load_swapdata import my_data, custom_palette, TARGET_TENORS


# ============================================================
# EWMA VOLATILITY (computed locally — no change to my_data)
# ============================================================
def compute_ewma_vol_panel(
    meta: pd.DataFrame,
    X_tensor: torch.Tensor,
    half_life_months: float = 13.5,   # half-life parameterization
    init_window: int = 12,
    annualize: bool = False,
) -> torch.Tensor:
    """
    Computes EWMA volatility per currency and tenor.
    Input X_tensor must be decimals (as in training script).
    Returns V_tensor in bps (per month) unless annualize=True (bps per year).
    """
    X = X_tensor.detach().cpu().numpy()  # decimals
    N, d = X.shape

    lam = 2.0 ** (-1.0 / float(half_life_months))

    # sort by currency + date (critical for correct EWMA per currency)
    order = meta.sort_values(["ccy", "as_of_date"]).index.to_numpy()
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))

    meta_s = meta.loc[order].reset_index(drop=True)
    X_s = X[order]
    V_s = np.zeros_like(X_s)

    for ccy, idx in meta_s.groupby("ccy").indices.items():
        idx = np.asarray(idx, dtype=int)
        S = X_s[idx]  # (T,d)
        T = S.shape[0]
        if T < 2:
            continue

        # monthly change in bps (since S is decimals)
        dS = np.full_like(S, np.nan)
        dS[1:] = 10000.0 * (S[1:] - S[:-1])

        # init variance using first init_window changes
        m = min(int(init_window), T - 1)
        init_var = np.nanvar(dS[1 : m + 1], axis=0, ddof=1)
        init_var = np.maximum(init_var, 1e-12)

        var = np.zeros_like(S)
        var[0] = init_var

        for t in range(1, T):
            var[t] = lam * var[t - 1] + (1.0 - lam) * (dS[t] ** 2)

        V = np.sqrt(var)  # bps per month
        if annualize:
            V *= np.sqrt(12.0)  # bps per year (monthly data)

        V_s[idx] = V

    V = V_s[inv]
    return torch.from_numpy(V).to(dtype=X_tensor.dtype)


# ============================================================
# Plotting helpers
# ============================================================
def build_currency_color_map():
    ccy_order = ["AUD", "CAD", "DKK", "EUR", "JPY", "NOK", "SEK", "GBP", "USD"]
    return {ccy: custom_palette[i % len(custom_palette)] for i, ccy in enumerate(ccy_order)}


def vol_to_long_df(meta: pd.DataFrame, V_tensor: torch.Tensor, tenors: List[int]) -> pd.DataFrame:
    V = V_tensor.detach().cpu().numpy()

    df = meta.copy().reset_index(drop=True)
    for j, t in enumerate(tenors):
        df[f"V_{t}Y"] = V[:, j]

    long = df.melt(
        id_vars=["as_of_date", "ccy"],
        value_vars=[f"V_{t}Y" for t in tenors],
        var_name="tenor",
        value_name="vol_bps",
    )
    long["tenor"] = (
        long["tenor"]
        .str.replace("V_", "", regex=False)
        .str.replace("Y", "", regex=False)
        .astype(int)
    )
    long["as_of_date"] = pd.to_datetime(long["as_of_date"])
    return long.sort_values(["ccy", "tenor", "as_of_date"]).reset_index(drop=True)



# ============================================================
# MAIN
# ============================================================
def main():
    # --- User option: show plots interactively? ---
    SHOW_PLOTS = True  # Set to False to only save plots

    USE = "bbg"  # same as training

    # ---- User knobs ----
    HALF_LIFE_MONTHS = 13.5   # ~lambda 0.95
    INIT_WINDOW = 12
    ANNUALIZE = False          # <-- set True to annualize

    # Save to Figures/bbg/swap_vol/
    BASE_DIR = os.path.join(REPO_ROOT, "Figures", USE)
    FIGURES_DIR = os.path.join(BASE_DIR, "swap_vol")
    os.makedirs(FIGURES_DIR, exist_ok=True)

    currency_color_map = build_currency_color_map()

    plot_cfg = H.PlotConfig(
        figures_dir=FIGURES_DIR,
        use_tag=USE,
        currency_colors=currency_color_map,
        dpi=300,
    )

    # --------------------------------------------------------
    # LOAD DATA EXACTLY LIKE TRAINING SCRIPT
    # --------------------------------------------------------
    meta, X_tensor, meta_full, X_tensor_full, tenors, df_wide, df_wide_all, SCALE_IS_PERCENT = my_data(use=USE)

    # --------------------------------------------------------
    # COMPUTE VOLATILITY
    # --------------------------------------------------------
    V_tensor = compute_ewma_vol_panel(
        meta=meta,
        X_tensor=X_tensor,
        half_life_months=HALF_LIFE_MONTHS,
        init_window=INIT_WINDOW,
        annualize=ANNUALIZE,
    )

    # Sanity check: print scale
    V_np = V_tensor.detach().cpu().numpy()
    print("Vol stats (mean/std/min/max):",
          float(np.nanmean(V_np)),
          float(np.nanstd(V_np)),
          float(np.nanmin(V_np)),
          float(np.nanmax(V_np)))

    vol_long = vol_to_long_df(meta, V_tensor, TARGET_TENORS)

    # --------------------------------------------------------
    # Plot per tenor (all currencies)
    # --------------------------------------------------------
    ann_tag = "ann" if ANNUALIZE else "mth"

    if ANNUALIZE:
        ylab = "Volatility (bps per year)"
    else:
        ylab = "Volatility (bps per month)"

    for t in TARGET_TENORS:
        df_t = vol_long[vol_long["tenor"] == t].copy()
        if df_t.empty:
            continue

        fig, ax = plt.subplots(figsize=(10.5, 4.2), dpi=160)

        for ccy, g in df_t.groupby("ccy"):
            g = g.sort_values("as_of_date")
            ax.plot(
                g["as_of_date"], g["vol_bps"],
                label=ccy,
                color=currency_color_map.get(ccy),
                alpha=0.9,
            )

        ax.set_title(f"EWMA volatility — {t}Y swap")
        ax.set_xlabel("Date")
        ax.set_ylabel(ylab)

        # --- Automatically adapt y-limits nicely ---
        y_max = np.nanmax(df_t["vol_bps"].values)
        ax.set_ylim(0, y_max * 1.1)

        ax.grid(True)
        ax.legend(ncol=5, frameon=False)

        plot_path = os.path.join(FIGURES_DIR, f"{ann_tag}_vol_time_by_ccy_{t}Y_hl{HALF_LIFE_MONTHS:g}.png")
        fig.savefig(plot_path, dpi=300)
        print(f"Saved volatility plot for {t}Y swap to {plot_path}")
        if SHOW_PLOTS:
            plt.show()
        plt.close(fig)


if __name__ == "__main__":
    main()