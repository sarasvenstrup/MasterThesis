#!/usr/bin/env python3
"""
make_diagnostics.py
===================
Convert ``sim_diagnostics.csv`` (produced by ``ResultsGeneratorSimulationExperiment.py``)
into LaTeX ``tabular`` fragments that can be ``\\input{}``-ed inside table
environments in the thesis chapter.

Outputs:
  1. A full simulation diagnostics table.
  2. Optionally, a smaller table containing only the in-sample reconstruction RMSE.

Usage
-----
    python make_diagnostics.py CSV_PATH TEX_PATH

Example
-------
    python make_diagnostics.py \\
        Figures/Simulation/sim_diagnostics.csv \\
        Figures/Simulation/sim_diagnostics_table.tex

Optional RMSE-only table
------------------------
    python make_diagnostics.py \\
        Figures/Simulation/sim_diagnostics.csv \\
        Figures/Simulation/sim_diagnostics_table.tex \\
        --rmse-tex-path Figures/Simulation/sim_insample_rmse_table.tex

The LaTeX fragments expect ``\\usepackage{booktabs}`` in the preamble.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def fmt_sci(value: float, decimals: int = 2) -> str:
    """Format ``value`` as ``\\(a.bb\\times 10^{c}\\)``."""
    if value == 0.0:
        return r"\(0\)"
    sign = "-" if value < 0 else ""
    abs_v = abs(value)
    exponent = int(f"{abs_v:e}".split("e")[1])
    mantissa = abs_v / 10 ** exponent
    return rf"\({sign}{mantissa:.{decimals}f}\times 10^{{{exponent}}}\)"


def fmt_num(value: float, decimals: int) -> str:
    return rf"\({value:.{decimals}f}\)"


def fmt_pct(value: float, decimals: int = 2) -> str:
    return rf"\({value:.{decimals}f}\%\)"


def parse_terminal_z_std(text: str) -> dict[str, float]:
    """Parse 'z1=...; z2=...; ...' into {'z1': float, 'z2': float, ...}."""
    out: dict[str, float] = {}
    for piece in text.split(";"):
        if "=" not in piece:
            continue
        k, v = piece.split("=", 1)
        out[k.strip()] = float(v.strip())
    return out


def parse_pct_string(text: str) -> float:
    """Parse '37.97%' -> 37.97."""
    return float(text.strip().rstrip("%"))


def parse_range(text: str) -> tuple[float, float]:
    """Parse '[a, b]' -> (a, b)."""
    m = re.match(r"\s*\[\s*([^,]+),\s*([^\]]+)\s*\]\s*", text)
    if not m:
        raise ValueError(f"Cannot parse range: {text!r}")
    return float(m.group(1)), float(m.group(2))


def fmt_sigma_range(text: str) -> str:
    """Sigma ranges, 4 decimals, no percent sign: \\([0.0288,\\,1.7269]\\)"""
    a, b = parse_range(text)
    return rf"\([{a:.4f},\,{b:.4f}]\)"


def fmt_r_range(text: str) -> str:
    """Short-rate range with explicit % inside the brackets, 2 decimals."""
    a, b = parse_range(text)
    return rf"\([{a:.2f}\%,\,{b:.2f}\%]\)"


def fmt_z_std(value: float) -> str:
    """Choose between scientific and fixed formatting automatically."""
    if abs(value) >= 1e3 or 0 < abs(value) < 1e-3:
        return fmt_sci(value, decimals=2)
    return fmt_num(value, decimals=3)


# ---------------------------------------------------------------------------
# Build rows in the exact layout used by the chapter
# ---------------------------------------------------------------------------
def build_rows(d: dict[str, tuple[str, str]]) -> list[list[tuple[str, str, str]]]:
    """
    Build the full diagnostics table groups.

    Each group is separated by a ``\\midrule`` in the final LaTeX table.

    Parameters
    ----------
    d : dict
        Maps metric name to (baseline_str, stable_str) pairs.

    Returns
    -------
    list of groups, each a list of (label, baseline_cell, stable_cell) tuples.
    """
    groups: list[list[tuple[str, str, str]]] = []

    # ── Group 1: Scale + finiteness + reconstruction quality ─────────────
    g1: list[tuple[str, str, str]] = []

    b, s = d["Max |z| across all paths"]
    g1.append((
        r"Maximum \(|z|\) across all paths",
        fmt_sci(float(b)),
        fmt_num(float(s), 2),
    ))

    if "Fraction with |z_T| < 10" in d:
        b, s = d["Fraction with |z_T| < 10"]
        g1.append((
            r"Fraction with \(|z_T|<10\)",
            rf"\({parse_pct_string(b):.1f}\%\)",
            rf"\({parse_pct_string(s):.1f}\%\)",
        ))

    if "In-sample reconstruction RMSE (bps)" in d:
        b, s = d["In-sample reconstruction RMSE (bps)"]
        g1.append((
            r"In-sample reconstruction RMSE",
            rf"\({float(b):.2f}\) bps",
            rf"\({float(s):.2f}\) bps",
        ))

    groups.append(g1)

    # ── Group 2: Drift-matrix eigenvalues ────────────────────────────────
    eig_rows: list[tuple[str, str, str]] = []
    for k, lbl in [
        ("Re(lambda_1) of M", r"\(\operatorname{Re}(\lambda_1(M))\)"),
        ("Re(lambda_2) of M", r"\(\operatorname{Re}(\lambda_2(M))\)"),
    ]:
        b, s = d[k]
        eig_rows.append((lbl, fmt_num(float(b), 3), fmt_num(float(s), 3)))
    groups.append(eig_rows)

    # ── Group 3: Terminal latent-factor dispersion ───────────────────────
    b, s = d["Terminal z std"]
    bz = parse_terminal_z_std(b)
    sz = parse_terminal_z_std(s)

    z_rows: list[tuple[str, str, str]] = []
    for key in ["z1", "z2"]:
        if key in bz and key in sz:
            sub = key[1:]
            z_rows.append((
                rf"Terminal \(z_{sub}\) standard deviation",
                fmt_z_std(bz[key]),
                fmt_z_std(sz[key]),
            ))
    groups.append(z_rows)

    # ── Group 4: Terminal short-rate distribution ────────────────────────
    r_rows: list[tuple[str, str, str]] = []

    b, s = d["Terminal r mean"]
    r_rows.append((
        r"Terminal \(r\) mean",
        fmt_pct(parse_pct_string(b)),
        fmt_pct(parse_pct_string(s)),
    ))

    b, s = d["Terminal r std"]
    r_rows.append((
        r"Terminal \(r\) standard deviation",
        fmt_pct(parse_pct_string(b)),
        fmt_pct(parse_pct_string(s)),
    ))

    b, s = d["r range (%)"]
    r_rows.append((
        r"Range of \(r\)",
        fmt_r_range(b),
        fmt_r_range(s),
    ))

    groups.append(r_rows)

    # ── Group 5: Volatility ranges ───────────────────────────────────────
    sig_rows: list[tuple[str, str, str]] = []

    for key, lbl in [
        ("sigma_1 range (train cloud)",   r"\(\sigma_1\) range on training cloud"),
        ("sigma_2 range (train cloud)",   r"\(\sigma_2\) range on training cloud"),
        ("sigma_1 range (expanded grid)", r"\(\sigma_1\) range on expanded grid"),
        ("sigma_2 range (expanded grid)", r"\(\sigma_2\) range on expanded grid"),
    ]:
        b, s = d[key]
        sig_rows.append((lbl, fmt_sigma_range(b), fmt_sigma_range(s)))

    groups.append(sig_rows)

    return groups


# ---------------------------------------------------------------------------
# Render full diagnostics table
# ---------------------------------------------------------------------------
def render(groups: list[list[tuple[str, str, str]]]) -> str:
    """
    Render the full diagnostics table as a LaTeX tabular fragment.

    Parameters
    ----------
    groups : list
        Output of build_rows().

    Returns
    -------
    str
        LaTeX tabular fragment.
    """
    lines: list[str] = []

    lines.append("% Auto-generated by make_diagnostics.py — do not edit by hand.")
    lines.append(r"\begin{tabular}{@{}lcc@{}}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Metric} & \textbf{Baseline} & \textbf{Stable} \\")
    lines.append(r"\midrule")

    for i, group in enumerate(groups):
        if i > 0:
            lines.append(r"\midrule")
        for label, b, s in group:
            lines.append(f"{label} & {b} & {s} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Render RMSE-only table
# ---------------------------------------------------------------------------
def render_insample_rmse(d: dict[str, tuple[str, str]]) -> str:
    """
    Render a small LaTeX table with only the in-sample reconstruction RMSE.

    Parameters
    ----------
    d : dict
        Maps metric name to (baseline_str, stable_str) pairs.

    Returns
    -------
    str
        LaTeX tabular fragment.

    Raises
    ------
    KeyError
        If the RMSE row is absent from d.
    """
    key = "In-sample reconstruction RMSE (bps)"
    if key not in d:
        raise KeyError(key)

    b, s = d[key]

    lines: list[str] = []
    lines.append("% Auto-generated by make_diagnostics.py — do not edit by hand.")
    lines.append(r"\begin{tabular}{@{}lc@{}}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Variant} & \textbf{In-sample RMSE (bps)} \\")
    lines.append(r"\midrule")
    lines.append(rf"Baseline & \({float(b):.2f}\) \\")
    lines.append(rf"Stable & \({float(s):.2f}\) \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent

    default_csv_path = project_root / "Figures" / "Simulation" / "sim_diagnostics.csv"
    default_tex_path = project_root / "Figures" / "Simulation" / "sim_diagnostics_table.tex"
    default_rmse_tex_path = project_root / "Figures" / "Simulation" / "sim_insample_rmse_table.tex"

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    ap.add_argument(
        "csv_path",
        type=Path,
        nargs="?",
        default=default_csv_path,
        help="Input CSV: sim_diagnostics.csv",
    )
    ap.add_argument(
        "tex_path",
        type=Path,
        nargs="?",
        default=default_tex_path,
        help="Output full LaTeX tabular fragment",
    )
    ap.add_argument(
        "--rmse-tex-path",
        type=Path,
        default=default_rmse_tex_path,
        help="Output path for the in-sample RMSE-only LaTeX table",
    )

    args = ap.parse_args()

    if not args.csv_path.is_file():
        print(f"ERROR: CSV not found: {args.csv_path}", file=sys.stderr)
        print("Run ResultsGeneratorSimulationExperiment.py first to generate sim_diagnostics.csv.", file=sys.stderr)
        return 2

    df = pd.read_csv(args.csv_path)

    if not {"Metric", "Baseline", "Stable"}.issubset(df.columns):
        print("ERROR: CSV must have columns Metric, Baseline, Stable.", file=sys.stderr)
        return 2

    d = {
        row["Metric"]: (str(row["Baseline"]), str(row["Stable"]))
        for _, row in df.iterrows()
    }

    try:
        groups = build_rows(d)
    except KeyError as e:
        print(f"ERROR: missing required CSV row {e}.", file=sys.stderr)
        return 2

    tex = render(groups)
    args.tex_path.parent.mkdir(parents=True, exist_ok=True)
    args.tex_path.write_text(tex, encoding="utf-8")
    print(f"Wrote {args.tex_path}")

    try:
        rmse_tex = render_insample_rmse(d)
    except KeyError as e:
        print(f"ERROR: missing required CSV row for RMSE table {e}.", file=sys.stderr)
        return 2

    args.rmse_tex_path.parent.mkdir(parents=True, exist_ok=True)
    args.rmse_tex_path.write_text(rmse_tex, encoding="utf-8")
    print(f"Wrote {args.rmse_tex_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())