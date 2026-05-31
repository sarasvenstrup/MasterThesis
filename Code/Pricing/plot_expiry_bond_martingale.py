"""
Plot the expiry-bond martingale identity:

    P(z_0, T_mu + k)   vs   E[ D(0, T_mu) * P(z_{T_mu}, k) ]

for each option expiry T_mu in {1, 5, 10} years, across k = 0,...,10.

Reads `E_expiry_bond_martingale.csv` produced by
dynamic_consistency_diagnostics.py and writes
`plots/E_expiry_bond_martingale.png` next to it. Style and figure
dimensions match the other diagnostic plots (B, F, G) in that script.

Usage (from project root):
    python Code/Pricing/plot_expiry_bond_martingale.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
_DIAG_DIR = _REPO / "Figures" / "PricingResults" / "Diagnostics" / "EUR_dim4_stable"


def main() -> int:
    csv_path = _DIAG_DIR / "E_expiry_bond_martingale.csv"
    out_path = _DIAG_DIR / "plots" / "E_expiry_bond_martingale.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        print(f"E_expiry_bond_martingale.csv not found at {csv_path}")
        return 1

    df = pd.read_csv(csv_path)
    expiries = sorted(df["expiry"].unique())

    # Aggregate across pricing dates: mean target, mean simulated, and
    # cross-date std of the simulated value for shaded uncertainty.
    g = df.groupby(["expiry", "k", "T_k"]).agg(
        P0=("P0_Tk", "mean"),
        MC=("MC_mean", "mean"),
        MC_std=("MC_mean", "std"),
    ).reset_index()

    fig, axes = plt.subplots(1, len(expiries), figsize=(11, 3.5), sharey=False)
    if len(expiries) == 1:
        axes = [axes]

    for ax, T_mu in zip(axes, expiries):
        sub = g[g["expiry"] == T_mu].sort_values("k")
        if sub.empty:
            continue
        k = sub["k"].values
        P0 = sub["P0"].values
        MC = sub["MC"].values
        MC_std = sub["MC_std"].fillna(0.0).values

        ax.plot(k, P0, marker="o", color="steelblue",
                label=r"$P(z_0,\,T_\mu+k)$")
        ax.plot(k, MC, marker="s", linestyle="--", color="tomato",
                label=r"$\mathbb{E}[D(0,T_\mu)\,P(z_{T_\mu},k)]$")
        ax.fill_between(k, MC - MC_std, MC + MC_std,
                        color="tomato", alpha=0.15)
        ax.set_title(rf"$T_\mu = {int(T_mu)}$Y")
        ax.set_xlabel(r"$k$ (years past $T_\mu$)")
        ax.set_ylabel("Bond price")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
