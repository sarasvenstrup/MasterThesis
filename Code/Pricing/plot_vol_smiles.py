"""
plot_vol_smiles.py
==================

Volatility smile diagnostic for the four-dimensional stable model.

For one chosen as-of date and the standard 3x3 grid of expiry-tenor cells
(1Y/5Y/10Y x 1Y/5Y/10Y), this script:

  1. runs one Monte Carlo simulation of the latent dynamics for that date,
  2. for each cell, prices a strike grid around the model-implied ATM
     forward swap rate F_0,
  3. inverts each price to a normal (Bachelier) implied volatility,
  4. plots vol vs. (K - F_0) for every cell.

Reuses the existing pricing engine (`Code.Pricing.pricing`) and the
simulator (`Code.Simulation.simulate_model.run_simulation`); no new
pricing logic is introduced. The output is saved to
`Figures/PricingResults/Diagnostics/EUR_dim4_stable/plots/`.

Usage (PowerShell):
    $env:SKIP_VARIANT_CONFIRM = "1"
    python Code/Pricing/plot_vol_smiles.py
    python Code/Pricing/plot_vol_smiles.py --as_of_date 2024-02-29
"""

from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

# Safety net so right-click runs do not block on the interactive
# variant-confirm prompt. Has no effect if config.confirm_variant() is
# not called in this script's import chain.
os.environ.setdefault("SKIP_VARIANT_CONFIRM", "1")

import csv

import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------------- paths
_HERE        = Path(__file__).resolve().parent
_CODE_ROOT   = _HERE.parent
_PROJECT     = _CODE_ROOT.parent
for p in (str(_PROJECT), str(_CODE_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from Code.Simulation.simulate_model import run_simulation
from Code.Pricing.pricing import (
    time0_forward_swap_and_annuity,
    swaption_mc_price_from_simulation,
    implied_bachelier_vol,
)


STRUCTURES = [
    (1, 1), (1, 5), (1, 10),
    (5, 1), (5, 5), (5, 10),
    (10, 1), (10, 5), (10, 10),
]

# Strike offsets relative to F_0, in basis points.
# +/- 100 bp around ATM, 9 points.
DEFAULT_OFFSETS_BP = np.array([-100, -75, -50, -25, 0, 25, 50, 75, 100], dtype=float)

# Pricing dates to loop over by default. Picked to span the EUR sample
# (October 2010 - February 2024) including rising, low and inverted rate
# regimes. Edit this list to add/remove dates.
DEFAULT_DATES = [
    "2010-10-29",
    "2014-08-29",
    "2016-07-29",
    "2018-05-31",
    "2020-04-30",
    "2022-03-31",
    "2024-02-29",
]


# =====================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--checkpoint",
        type=str,
        default=str(_PROJECT / "Figures" / "TrainingResults"
                    / "dim4_stable" / "ep5000" / "checkpoint_dim4_ep5000.pt"),
        help="Path to trained model checkpoint.",
    )
    p.add_argument("--latent_dim", type=int, default=4)
    p.add_argument("--ccy",        type=str, default="EUR")
    p.add_argument("--dates",      type=str, default=",".join(DEFAULT_DATES),
                   help="Comma-separated list of pricing dates (YYYY-MM-DD). "
                        "One smile figure is produced per date.")
    p.add_argument("--n_paths",    type=int, default=10_000)
    p.add_argument("--n_steps",    type=int, default=120,
                   help="Total simulation steps. 120 monthly = 10 years, "
                        "enough to price all 1/5/10Y expiries.")
    p.add_argument("--dt",         type=float, default=1.0 / 12.0)
    p.add_argument("--strike_offsets_bp", type=str,
                   default=",".join(str(int(x)) for x in DEFAULT_OFFSETS_BP),
                   help="Comma-separated strike offsets in bp.")
    p.add_argument(
        "--out_dir",
        type=str,
        default=str(_PROJECT / "Figures" / "PricingResults" / "Diagnostics"
                    / "EUR_dim4_stable" / "plots"),
        help="Where to save the smile figure.",
    )
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--diffusion_scale", type=float, default=1.0,
                   help="Multiplier on the latent diffusion. Default 1.0 "
                        "(physical scale). Lower values (e.g. 0.5) reduce "
                        "decoder-overflow path loss at the cost of biasing "
                        "the simulated dynamics.")
    return p.parse_args()


# =====================================================================
def price_smile_for_cell(
    ctx: dict,
    expiry: int,
    tenor: int,
    offsets_bp: np.ndarray,
) -> dict:
    """
    Price one expiry-tenor cell at every strike on the offset grid, and
    invert each price to a normal vol.

    Straddle convention: at every strike we price BOTH legs on the same MC
    paths and average,
        V_str(K) = 0.5 * (V_payer(K) + V_receiver(K)).
    At K = F_0 this is the usual ATM straddle (the existing
    `atm_swaption_straddle_mc_price_from_simulation` reduces to this), and
    the average cancels the directional forward-centring bias to first
    order. To invert against Bachelier we use put-call parity exactly:
        V_str(K) = V_pay_Bachelier(K, sigma) - 0.5 * A_0 * (F_0 - K),
    so the synthetic payer-equivalent price
        V_pay_eq = V_str(K) + 0.5 * A_0 * (F_0 - K)
    can be fed to the standard `implied_bachelier_vol` with payer=True.
    """
    P_full_0 = ctx["P_full_0"].detach().cpu().numpy()
    tau_grid = ctx["tau_grid"].detach().cpu().numpy()
    q0 = time0_forward_swap_and_annuity(
        P_full_0=P_full_0, tau_grid=tau_grid,
        expiry=float(expiry), tenor=int(tenor), accrual=1.0,
    )
    F0 = float(q0["forward_swap"])
    A0 = float(q0["annuity"])

    strikes_bp = offsets_bp + F0 * 1e4    # absolute strike in bp
    strikes    = strikes_bp / 1e4         # decimal

    vols_bp = np.full_like(strikes, np.nan, dtype=float)
    prices  = np.full_like(strikes, np.nan, dtype=float)
    stderrs = np.full_like(strikes, np.nan, dtype=float)
    n_valid = np.zeros_like(strikes, dtype=int)
    n_total = 0
    for i, K in enumerate(strikes):
        try:
            res_pay = swaption_mc_price_from_simulation(
                ctx=ctx, expiry=float(expiry), tenor=int(tenor),
                strike=float(K), payer=True,
                accrual=1.0, notional=1.0, verbose=False,
            )
            res_rec = swaption_mc_price_from_simulation(
                ctx=ctx, expiry=float(expiry), tenor=int(tenor),
                strike=float(K), payer=False,
                accrual=1.0, notional=1.0, verbose=False,
            )
            # Average the two legs (same paths) to cancel forward-centring
            # bias. At K=F0 this is exactly the ATM straddle.
            v_str = 0.5 * (res_pay["mc_price"] + res_rec["mc_price"])
            # Both legs use the same valid_mask intersection through the
            # underlying pricer; take the smaller of the two reported counts.
            n_ok  = min(int(res_pay["valid_mask"].sum()),
                        int(res_rec["valid_mask"].sum()))
            # Conservative SE of the average: 0.5 * sqrt(se_p^2 + se_r^2)
            # (upper bound; tight bound would need pv arrays). Cheap and
            # close enough for a diagnostic plot.
            se_str = 0.5 * float(np.sqrt(res_pay["mc_stderr"]**2 +
                                         res_rec["mc_stderr"]**2))
            prices[i]  = v_str
            stderrs[i] = se_str
            n_valid[i] = n_ok
            n_total    = int(len(res_pay["valid_mask"]))

            # Put-call parity gives the equivalent payer market price
            v_pay_eq = v_str + 0.5 * A0 * (F0 - float(K))
            iv = implied_bachelier_vol(
                market_price=v_pay_eq,
                forward=F0, strike=float(K), expiry=float(expiry),
                annuity=A0, notional=1.0, payer=True,
            )
            if iv is not None and np.isfinite(iv):
                vols_bp[i] = float(iv) * 1e4
        except Exception as e:
            print(f"  [skip] {expiry}Yx{tenor}Y K={K*1e4:.1f}bp: {str(e)[:120]}")

    return {
        "expiry": expiry,
        "tenor": tenor,
        "F0_bp": F0 * 1e4,
        "A0": A0,
        "offsets_bp": offsets_bp,
        "strikes_bp": strikes_bp,
        "prices": prices,
        "stderrs": stderrs,
        "n_valid": n_valid,
        "n_total": n_total,
        "vols_bp": vols_bp,
    }


# =====================================================================
def run_one_date(
    as_of_date: str,
    args: argparse.Namespace,
    offsets_bp: np.ndarray,
    out_dir: Path,
) -> None:
    """Simulate + price + plot for a single as-of-date."""
    print("\n" + "=" * 70)
    print(f"Pricing volatility smiles for as_of_date = {as_of_date}")
    print("=" * 70)
    print(f"  paths={args.n_paths}, steps={args.n_steps}, dt={args.dt}, "
          f"diffusion_scale={args.diffusion_scale}, "
          f"strikes (bp offsets) = {offsets_bp.tolist()}")

    # Only decode the time steps we actually price at (expiries 1Y, 5Y, 10Y).
    # Step 0 is auto-included by run_simulation. This avoids decoding the
    # full n_paths x (n_steps+1) grid (~1.2M states for the defaults) when
    # pricing only needs ~3 slices per path.
    expiry_years = sorted({exp for (exp, _ten) in STRUCTURES})
    decode_steps = [int(round(e / args.dt)) for e in expiry_years]

    ctx = run_simulation(
        use="bbg",
        latent_dim=args.latent_dim,
        checkpoint_path=args.checkpoint,
        n_paths=args.n_paths,
        n_steps=args.n_steps,
        dt=args.dt,
        as_of_date=as_of_date,
        ccy_filter=args.ccy,
        diffusion_scale=args.diffusion_scale,
        use_antithetic=False,
        seed=args.seed,
        show_plot=False,
        decode_steps=decode_steps,
    )

    tau_grid_np = ctx["tau_grid"].detach().cpu().numpy()
    print(f"  tau_grid: n={len(tau_grid_np)}, "
          f"min={tau_grid_np.min():.3f}, max={tau_grid_np.max():.3f}")

    smiles = []
    for (exp, ten) in STRUCTURES:
        print(f"Pricing smile for {exp}Y x {ten}Y ...")
        smile = price_smile_for_cell(ctx, exp, ten, offsets_bp)
        smiles.append(smile)
        good = np.isfinite(smile["vols_bp"]).sum()
        print(f"  F_0 = {smile['F0_bp']:.1f} bp, A_0 = {smile['A0']:.4f}, "
              f"{good}/{len(offsets_bp)} strikes successfully inverted.")
        for j, off in enumerate(smile["offsets_bp"]):
            print(f"    off={off:+6.1f}bp  K={smile['strikes_bp'][j]:+7.1f}bp  "
                  f"price={smile['prices'][j]:.3e}  "
                  f"se={smile['stderrs'][j]:.2e}  "
                  f"n_ok={int(smile['n_valid'][j])}/{smile['n_total']}  "
                  f"sigmaN={smile['vols_bp'][j]:7.2f}bp")

    # ------------------------------------------------------------ CSV dump
    csv_path = out_dir / f"vol_smiles_{as_of_date}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["expiry", "tenor", "F0_bp", "A0", "offset_bp",
                    "strike_bp", "price", "stderr",
                    "n_valid", "n_total", "sigma_N_bp"])
        for s in smiles:
            for j, off in enumerate(s["offsets_bp"]):
                w.writerow([
                    s["expiry"], s["tenor"],
                    f"{s['F0_bp']:.4f}", f"{s['A0']:.8f}",
                    f"{off:.2f}", f"{s['strikes_bp'][j]:.4f}",
                    f"{s['prices'][j]:.10e}", f"{s['stderrs'][j]:.6e}",
                    int(s["n_valid"][j]), s["n_total"],
                    f"{s['vols_bp'][j]:.4f}",
                ])
    print(f"Saved CSV: {csv_path}")

    # ------------------------------------------------------------ plot
    fig, axes = plt.subplots(3, 3, figsize=(11, 9), sharex=True)
    axes = axes.flatten()
    for ax, smile in zip(axes, smiles):
        offs = smile["offsets_bp"]
        vols = smile["vols_bp"]
        F0   = smile["F0_bp"]
        ax.plot(offs, vols, marker="o", color="tomato", linewidth=1.5)
        ax.axvline(0.0, color="grey", linestyle=":", linewidth=0.8)
        ax.set_title(rf"${smile['expiry']}$Y$\times{smile['tenor']}$Y  "
                     rf"($F_0={F0:.0f}$ bp)", fontsize=10)
        ax.set_xlabel(r"$K - F_0$ (bp)")
        ax.set_ylabel(r"$\sigma_N$ (bp)")
        ax.grid(alpha=0.3)

    fig.suptitle(
        f"Model-implied normal vol smiles — {as_of_date}",
        fontsize=13,
    )
    fig.tight_layout()
    out_png = out_dir / f"vol_smiles_{as_of_date}.png"
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_png}")


def main() -> int:
    args = parse_args()
    offsets_bp = np.array(
        [float(x) for x in args.strike_offsets_bp.split(",") if x.strip()],
        dtype=float,
    )
    dates = [d.strip() for d in args.dates.split(",") if d.strip()]
    if not dates:
        print("No dates supplied (--dates is empty).")
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for d in dates:
        try:
            run_one_date(d, args, offsets_bp, out_dir)
        except Exception as e:
            print(f"[skip] {d}: {str(e)[:200]}")

    print(f"\nDone. Figures in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
