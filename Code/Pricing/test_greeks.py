"""Quick sanity check for bachelier_greeks() + model-implied Greeks from checkpoint."""
import os
import sys
import math

try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.getcwd()

for _p in [_HERE, os.path.join(_HERE, ".."), os.path.join(_HERE, "..", "..")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd
from Code.Pricing.pricing import (
    bachelier_greeks, bachelier_price,
    run_simulation, atm_swaption_mc_price_from_simulation,
    quote_swaption_time0,
)

# =============================================================================
# PART 1 — Analytic sanity check  (no model needed)
# =============================================================================

F = 0.03; K = 0.03; sigma = 0.006; T = 2.0; A = 8.0; N = 1.0

g = bachelier_greeks(F, K, sigma, T, A, N, payer=True)
price = bachelier_price(F, K, sigma, T, A, N, payer=True)

ATM_price_formula = N * A * sigma * math.sqrt(T) / math.sqrt(2 * math.pi)
ATM_delta_formula = 0.5 * N * A
ATM_vega_formula  = N * A * math.sqrt(T / (2 * math.pi))
ATM_theta_formula = N * A * sigma / (2 * math.sqrt(2 * math.pi * T))

print("=== ATM Bachelier Greeks Sanity Check ===")
print(f"Price  : {price:.8f}  expected={ATM_price_formula:.8f}  ok={abs(price - ATM_price_formula) < 1e-12}")
print(f"Delta  : {g['delta']:.8f}  expected={ATM_delta_formula:.8f}  ok={abs(g['delta'] - ATM_delta_formula) < 1e-12}")
print(f"Vega   : {g['vega']:.8f}  expected={ATM_vega_formula:.8f}   ok={abs(g['vega'] - ATM_vega_formula) < 1e-12}")
print(f"Gamma  : {g['gamma']:.8f}")
print(f"Theta  : {g['theta']:.8f}  expected={ATM_theta_formula:.8f}  ok={abs(g['theta'] - ATM_theta_formula) < 1e-12}")
print(f"DV01   : {g['dv01']:.8f}  (= Delta * 1e-4 = {ATM_delta_formula * 1e-4:.8f})")
print(f"Vanna  : {g['vanna']:.10f}  (should be 0 at ATM)")
print(f"Volga  : {g['volga']:.10f}  (should be 0 at ATM)")

g_rec = bachelier_greeks(F, K, sigma, T, A, N, payer=False)
pcp_delta = g['delta'] - g_rec['delta']
print(f"\nPut-call parity: delta_pay - delta_rec = {pcp_delta:.8f}  expected={N*A:.8f}  ok={abs(pcp_delta - N*A) < 1e-12}")

g_itm = bachelier_greeks(0.03, 0.025, sigma, T, A, N, payer=True)
print(f"\nITM (F=3%, K=2.5%): delta={g_itm['delta']:.6f}  vega={g_itm['vega']:.6f}  (vega < ATM vega: {g_itm['vega'] < g['vega']})")

_all_ok = all([
    abs(price - ATM_price_formula) < 1e-12,
    abs(g['delta'] - ATM_delta_formula) < 1e-12,
    abs(g['vega'] - ATM_vega_formula) < 1e-12,
    abs(g['theta'] - ATM_theta_formula) < 1e-12,
    abs(g['vanna']) < 1e-14,
    abs(g['volga']) < 1e-14,
    abs(pcp_delta - N*A) < 1e-12,
])
print("\nAll checks passed!" if _all_ok else "Some checks FAILED.")

# =============================================================================
# PART 2 — Model-implied Greeks from checkpoint
#           Uses the stage-2 checkpoint; falls back to stage-1 if not found.
#           Edit CHECKPOINT, AS_OF_DATE and CELLS below.
# =============================================================================

if __name__ == "__main__":

    _THESIS_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))

    # ── SETTINGS ─────────────────────────────────────────────────────────────
    CHECKPOINT = os.path.join(
        _THESIS_ROOT, "Figures", "Pricing", "stage2_checkpoints",
        "checkpoint_stage2_dim4_ep500.pt"
    )
    # Fallback to stage-1 if stage-2 not yet available
    if not os.path.exists(CHECKPOINT):
        CHECKPOINT = os.path.join(
            _THESIS_ROOT, "Figures", "TrainingResults",
            "dim4_stable", "ep3500", "checkpoint_dim4_ep3500.pt"
        )

    AS_OF_DATE = "2018-06-29"   # ISO date — must be in the swap data

    # (option_maturity_years, swap_tenor_years) cells to evaluate
    CELLS = [
        (1, 5),
        (2, 5),
        (5, 5),
        (5, 10),
        (10, 5),
        (10, 10),
    ]

    N_PATHS = 2000
    N_STEPS = 120       # 10 yr monthly
    DT      = 1 / 12
    CCY     = "EUR"
    # ─────────────────────────────────────────────────────────────────────────

    print("\n" + "=" * 70)
    print("MODEL-IMPLIED BACHELIER GREEKS")
    print("=" * 70)
    print(f"  Checkpoint : {CHECKPOINT}")
    print(f"  Date       : {AS_OF_DATE}")
    print(f"  n_paths    : {N_PATHS}")
    print("=" * 70)

    ctx = run_simulation(
        checkpoint_path = CHECKPOINT,
        ccy_filter      = CCY,
        as_of_date      = AS_OF_DATE,
        n_paths         = N_PATHS,
        n_steps         = N_STEPS,
        dt              = DT,
        show_plot       = False,
    )

    rows = []
    for expiry, tenor in CELLS:
        try:
            # 1) MC price → implied vol
            res = atm_swaption_mc_price_from_simulation(
                ctx=ctx, expiry=expiry, tenor=tenor,
                payer=True, accrual=1.0, notional=1.0,
            )
            iv = res["implied_normal_vol"]
            if iv is None or not np.isfinite(iv):
                print(f"  [{expiry}Yx{ten}Y]  implied vol unavailable — skipping")
                continue

            # 2) Time-0 forward & annuity
            quote = res["quote"]
            F0    = quote["forward_swap"]
            A0    = quote["annuity"]

            # 3) Bachelier Greeks at model-implied vol
            gk = bachelier_greeks(
                forward    = F0,
                strike     = F0,      # ATM
                normal_vol = iv,
                expiry     = expiry,
                annuity    = A0,
                notional   = 1.0,
                payer      = True,
            )

            rows.append({
                "expiry"     : expiry,
                "tenor"      : tenor,
                "F0 (bp)"    : round(F0 * 10_000, 1),
                "A0"         : round(A0, 4),
                "IV (bp)"    : round(iv * 10_000, 1),
                "MC SE (bp)" : round(res["mc_stderr"] / max(gk["vega"], 1e-16) * 10_000, 2),
                "delta"      : round(gk["delta"], 6),
                "vega"       : round(gk["vega"],  6),
                "gamma"      : round(gk["gamma"], 8),
                "theta"      : round(gk["theta"], 6),
                "dv01"       : round(gk["dv01"],  8),
                "vanna"      : round(gk["vanna"], 8),
                "volga"      : round(gk["volga"], 8),
            })

        except Exception as exc:
            print(f"  [{expiry}Yx{tenor}Y]  ERROR: {exc}")

    if rows:
        df_gk = pd.DataFrame(rows)
        print("\n" + df_gk.to_string(index=False))

        # save to CSV next to the checkpoint
        out_csv = os.path.join(
            os.path.dirname(CHECKPOINT),
            f"greeks_{AS_OF_DATE}.csv"
        )
        df_gk.to_csv(out_csv, index=False)
        print(f"\n  Saved → {out_csv}")
    else:
        print("  No valid Greeks computed.")
