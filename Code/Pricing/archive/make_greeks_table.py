"""
make_greeks_table.py
====================
Generate the ATM Bachelier Greeks table (tab_atm_greeks.tex) for the
Pricing chapter.

The table is computed analytically from the closed-form ATM formulas:

    Δ    = ½ N A₀
    Vega = N A₀ √T / √(2π)
    Γ    = N A₀ / (σ √T √(2π))
    Θ    = N A₀ σ / (2 √T √(2π))   [satisfies Θ = ½ σ² Γ exactly]

Annuities use the proper expiry-dependent definition
    A₀ = δ Σⱼ P(0, Tₑ + j δ)
with annual compounding on a flat r% curve.

Usage
-----
    python make_greeks_table.py                  # uses defaults below
    python make_greeks_table.py --r 0.03 --sigma_bp 60 --out_dir /path/to/dir
"""

import argparse
import math
import os

# ── Defaults ─────────────────────────────────────────────────────────────────
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.getcwd()

_THESIS_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
DEFAULT_OUT_DIR = os.path.join(_THESIS_ROOT, "Figures", "Pricing")

CELLS = [(1, 5), (2, 5), (5, 5), (1, 10), (2, 10), (5, 10)]
DEFAULT_R        = 0.03    # flat annual rate (annual compounding)
DEFAULT_SIGMA_BP = 60      # normal vol in basis points
DEFAULT_NOTIONAL = 1.0
DEFAULT_DELTA    = 1       # accrual period (years)


# ── Core calculation ──────────────────────────────────────────────────────────

def discount(T: float, r: float) -> float:
    """Annual-compounding discount factor."""
    return (1.0 + r) ** (-T)


def atm_greeks_row(Te: int, Ts: int, r: float, sigma: float,
                   notional: float = 1.0, delta: int = 1) -> dict:
    """
    Compute ATM Bachelier Greeks for one (Te, Ts) cell.

    Parameters
    ----------
    Te     : option expiry in years
    Ts     : swap tenor in years
    r      : flat annual rate (annual compounding), e.g. 0.03
    sigma  : normal volatility in absolute decimal, e.g. 0.006
    notional : notional
    delta  : accrual period (years)
    """
    phi0 = 1.0 / math.sqrt(2.0 * math.pi)    # φ(0)
    sqT  = math.sqrt(Te)

    # Payment dates: Te+δ, Te+2δ, …, Te+Ts
    pay_dates = [Te + delta * j for j in range(1, Ts // delta + 1)]
    pays = [discount(t, r) for t in pay_dates]
    A0   = delta * sum(pays)

    P_start = discount(Te, r) if Te > 0 else 1.0
    P_end   = pays[-1]
    F0      = (P_start - P_end) / A0

    Delta = notional * A0 * 0.5
    Vega  = notional * A0 * sqT * phi0
    Gamma = notional * A0 * phi0 / (sigma * sqT)
    Theta = notional * A0 * sigma * phi0 / (2.0 * sqT)
    DV01  = Delta * 1e-4

    # Sanity: Theta == ½ σ² Gamma
    assert abs(Theta - 0.5 * sigma ** 2 * Gamma) < 1e-10, \
        f"Theta-Gamma identity violated for ({Te}Y×{Ts}Y)"

    return dict(Te=Te, Ts=Ts, A0=A0, F0=F0, Delta=Delta,
                Vega=Vega, Gamma=Gamma, Theta=Theta, DV01=DV01)


# ── LaTeX renderer ────────────────────────────────────────────────────────────

def render_latex(rows: list, r: float, sigma_bp: int) -> str:
    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{ATM Bachelier Greeks for a representative swaption grid "
        r"(\(\mathcal N=1\), \(\sigma_N=" + str(sigma_bp) + r"\) bp"
        r"\(=" + f"{sigma_bp/10000:.4f}".rstrip('0') + r"\), "
        r"\(\mathcal A_0=\delta\sum_{j=1}^{T_s}(1+"
        + f"{r:.2f}".rstrip('0').rstrip('.')
        + r")^{-(T_e+j\delta)}\) on a flat "
        + f"{r*100:.0f}" + r"\% annually-compounded curve).}"
    )
    lines.append(r"\label{tab:atm_greeks}")
    lines.append(r"\begin{tabular}{@{}ccrrrrr@{}}")
    lines.append(r"\toprule")
    lines.append(
        r"\(T_e\) & \(T_s\) & \(\mathcal A_0\) & \(\Delta\) "
        r"& Vega & \(\Gamma\) & \(\Theta\) \\"
    )
    lines.append(r"\midrule")
    for row in rows:
        lines.append(
            f"{row['Te']}Y & {row['Ts']}Y "
            f"& {row['A0']:.3f} "
            f"& {row['Delta']:.3f} "
            f"& {row['Vega']:.3f} "
            f"& {row['Gamma']:.1f} "
            f"& {row['Theta']:.5f} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\begin{tablenotes}")
    lines.append(r"\small")
    lines.append(
        r"\item \textit{Notes.} "
        r"\(\mathcal A_0\) uses the expiry-dependent definition "
        r"\(\mathcal A_0=\delta\sum_{j=1}^{T_s}P(0,T_e+j\delta)\); "
        r"it differs across rows with the same \(T_s\). "
        r"At the money: \(\Delta=\tfrac12\mathcal N\mathcal A_0\), "
        r"Vanna\(=\)Volga\(=0\), and \(\Theta=\tfrac12\sigma_N^2\Gamma\) "
        r"(verified for every row). "
        r"\(\Theta\) is the partial derivative \(\partial V/\partial T_e\) "
        r"with all other inputs fixed; its small magnitude reflects "
        r"\(\sigma_N=" + str(sigma_bp) + r"\) bp. "
        r"Model-implied Greeks are obtained by substituting "
        r"\(F_0\), \(\mathcal A_0\), and the Monte Carlo implied vol "
        r"\(\widehat\sigma_{N,\mathrm{mod}}\) into the same formulas."
    )
    lines.append(r"\end{tablenotes}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ── Entry point ───────────────────────────────────────────────────────────────

def run(
    r: float = DEFAULT_R,
    sigma_bp: int = DEFAULT_SIGMA_BP,
    notional: float = DEFAULT_NOTIONAL,
    cells: list = None,
    out_dir: str = DEFAULT_OUT_DIR,
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    if cells is None:
        cells = CELLS
    sigma = sigma_bp / 10_000.0

    rows = [atm_greeks_row(Te, Ts, r=r, sigma=sigma, notional=notional)
            for Te, Ts in cells]

    # Print to console
    header = f"{'Te':>3} {'Ts':>3}  {'A0':>7}  {'F0 (bp)':>8}  {'Delta':>7}  {'Vega':>7}  {'Gamma':>7}  {'Theta':>9}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(f"{row['Te']:>2}Y {row['Ts']:>2}Y  "
              f"{row['A0']:7.4f}  "
              f"{row['F0']*10000:8.1f}  "
              f"{row['Delta']:7.4f}  "
              f"{row['Vega']:7.4f}  "
              f"{row['Gamma']:7.2f}  "
              f"{row['Theta']:9.6f}")

    latex = render_latex(rows, r=r, sigma_bp=sigma_bp)

    out_path = os.path.join(out_dir, "tab_atm_greeks.tex")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(latex)
    print(f"\nSaved → {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ATM Greeks LaTeX table")
    parser.add_argument("--r",        type=float, default=DEFAULT_R,
                        help=f"Flat annual rate (default: {DEFAULT_R})")
    parser.add_argument("--sigma_bp", type=int,   default=DEFAULT_SIGMA_BP,
                        help=f"Normal vol in bp (default: {DEFAULT_SIGMA_BP})")
    parser.add_argument("--out_dir",  type=str,   default=DEFAULT_OUT_DIR,
                        help=f"Output directory (default: {DEFAULT_OUT_DIR})")
    args = parser.parse_args()

    run(r=args.r, sigma_bp=args.sigma_bp, out_dir=args.out_dir)


