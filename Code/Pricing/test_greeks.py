"""Quick sanity check for bachelier_greeks()."""
import sys, math
sys.path.insert(0, r"C:\Users\Bruger\PycharmProjects\MasterThesis")
from Code.Pricing.pricing import bachelier_greeks, bachelier_price

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

# put-call parity: delta_pay - delta_rec = N * A
g_rec = bachelier_greeks(F, K, sigma, T, A, N, payer=False)
pcp_delta = g['delta'] - g_rec['delta']
print(f"\nPut-call parity: delta_pay - delta_rec = {pcp_delta:.8f}  expected={N*A:.8f}  ok={abs(pcp_delta - N*A) < 1e-12}")

# ITM test
g_itm = bachelier_greeks(0.03, 0.025, sigma, T, A, N, payer=True)
print(f"\nITM (F=3%, K=2.5%): delta={g_itm['delta']:.6f}  vega={g_itm['vega']:.6f}  (vega < ATM vega: {g_itm['vega'] < g['vega']})")

print("\nAll checks passed!" if all([
    abs(price - ATM_price_formula) < 1e-12,
    abs(g['delta'] - ATM_delta_formula) < 1e-12,
    abs(g['vega'] - ATM_vega_formula) < 1e-12,
    abs(g['theta'] - ATM_theta_formula) < 1e-12,
    abs(g['vanna']) < 1e-14,
    abs(g['volga']) < 1e-14,
    abs(pcp_delta - N*A) < 1e-12,
]) else "Some checks FAILED.")

