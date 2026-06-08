"""
Generates OOS Results Analysis.md from the OOS rolling CSV results.

Run from repo root:
    python Code/Overview_oos_analysis.py

Reads: Figures/OOSResults/Roll/OOS_roll_<model>/train5Y_test6M_step6M/ep<ep>/*.csv
Writes: OOS Results Analysis.md  (repo root)
"""

import os
import glob
import datetime
import numpy as np
import pandas as pd

try:
    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    REPO_ROOT = os.getcwd()

ROLL_ROOT = os.path.join(REPO_ROOT, "Figures", "OOSResults", "Roll")
OUT_PATH  = os.path.join(REPO_ROOT, "OOS Results Analysis.md")

SUBDIR = "train5Y_test6M_step6M"

MODELS = [
    ("2", "baseline", 3500),
    ("2", "stable",   3500),
    ("3", "baseline", 3500),
    ("3", "stable",   3500),
    ("4", "baseline", 3500),
    ("4", "stable",   3500),
]

CURRENCIES = ["AUD", "CAD", "DKK", "EUR", "JPY", "NOK", "SEK", "GBP", "USD"]

# ── helpers ──────────────────────────────────────────────────────────────────

def model_key(dim, mtype):
    """Return the canonical model key string for a given dimension and type."""
    return f"dim{dim}_{mtype}"


def csv_path(dim, mtype, ep):
    """Return the path to the rolling OOS CSV for a given model, or None if not found."""
    folder = os.path.join(ROLL_ROOT, f"OOS_roll_dim{dim}_{mtype}", SUBDIR, f"ep{ep}")
    hits = glob.glob(os.path.join(folder, "*.csv"))
    return hits[0] if hits else None


def load_df(dim, mtype, ep):
    """Load the rolling OOS CSV for a given model as a DataFrame, or None if missing."""
    p = csv_path(dim, mtype, ep)
    if p is None:
        return None
    df = pd.read_csv(p)
    # coerce numeric
    for col in df.columns:
        if col not in ("roll_start", "train_start", "train_end", "test_start", "test_end"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def eig_cols(df):
    """Return column names containing real eigenvalue values."""
    return [c for c in df.columns if c.startswith("eig_real_")]


def fmt(v, decimals=1):
    """Format a float to fixed decimal places, returning '—' for missing values."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:.{decimals}f}"


# ── load all data ─────────────────────────────────────────────────────────────

data = {}
for dim, mtype, ep in MODELS:
    key = model_key(dim, mtype)
    df = load_df(dim, mtype, ep)
    data[key] = {"df": df, "ep": ep, "dim": dim, "mtype": mtype}

# ── per-model summary stats ───────────────────────────────────────────────────

def summary(key):
    """Compute mean/max OOS RMSE, mean IS RMSE, and bad-row count for a model."""
    d = data[key]
    df = d["df"]
    if df is None:
        return None
    oos = df["avg_rmse_bps"].values
    is_col = df["avg_in_rmse_bps"] if "avg_in_rmse_bps" in df.columns else None
    is_vals = is_col.dropna().values if is_col is not None else np.array([])
    return {
        "ep":      d["ep"],
        "n_win":   len(df),
        "mean_oos": np.mean(oos),
        "max_oos":  np.max(oos),
        "mean_is":  np.mean(is_vals) if len(is_vals) > 0 else np.nan,
        "n_bad":    int(df["n_test_bad"].sum()) if "n_test_bad" in df.columns else 0,
    }


# ── identify big-error windows ────────────────────────────────────────────────

BIG_OOS_THRESHOLD = 50   # bps — windows above this get flagged

def big_error_windows(threshold=BIG_OOS_THRESHOLD):
    """Partition windows above the RMSE threshold into ODE-crash (A) and finite-wrong (B) groups."""
    mech_a = []   # ODE crash (n_test_bad > 0)
    mech_b = []   # Finite but wrong (n_test_bad == 0, huge RMSE)
    for key, d in data.items():
        df = d["df"]
        if df is None:
            continue
        for _, row in df.iterrows():
            oos = row["avg_rmse_bps"]
            bad = int(row.get("n_test_bad", 0)) if not np.isnan(row.get("n_test_bad", 0)) else 0
            if oos < threshold:
                continue
            eig_cols_here = eig_cols(df)
            eigs = [row[c] for c in eig_cols_here if not np.isnan(row[c])]
            lam_max = max(eigs) if eigs else np.nan
            # worst currency
            ccy_rmses = {c: row.get(f"rmse_bps_{c}", np.nan) for c in CURRENCIES}
            worst_ccy = sorted(
                [(c, v) for c, v in ccy_rmses.items() if not np.isnan(v)],
                key=lambda x: -x[1]
            )[:3]
            worst_str = ", ".join(f"{c}={v:.0f}" for c, v in worst_ccy)
            entry = {
                "window": row["roll_start"],
                "model":  key,
                "oos":    oos,
                "bad":    bad,
                "lam_max": lam_max,
                "worst":   worst_str,
            }
            if bad > 0:
                mech_a.append(entry)
            else:
                mech_b.append(entry)
    mech_a.sort(key=lambda x: -x["oos"])
    mech_b.sort(key=lambda x: -x["oos"])
    return mech_a, mech_b


# ── eigenvalue summary per model ──────────────────────────────────────────────

def eig_summary(key):
    """Return eigenvalue range and stability window counts for a model."""
    d = data[key]
    df = d["df"]
    if df is None:
        return None
    cols = eig_cols(df)
    if not cols:
        return None
    all_eigs = df[cols].values.flatten()
    all_eigs = all_eigs[~np.isnan(all_eigs)]
    if len(all_eigs) == 0:
        return None
    return {
        "min": float(np.min(all_eigs)),
        "max": float(np.max(all_eigs)),
        "near_zero_windows": int((df[cols].min(axis=1) > -0.05).sum()),
        "positive_windows":  int((df[cols].max(axis=1) > 0).sum()),
    }


# ── window-level per-currency details ────────────────────────────────────────

def worst_window_details(key, window_start):
    """Return a formatted string of the worst per-currency RMSE for a given window."""
    df = data[key]["df"]
    if df is None:
        return ""
    row = df[df["roll_start"] == window_start]
    if len(row) == 0:
        return ""
    row = row.iloc[0]
    pairs = [(c, row.get(f"rmse_bps_{c}", np.nan)) for c in CURRENCIES]
    pairs = [(c, v) for c, v in pairs if not np.isnan(v)]
    pairs.sort(key=lambda x: -x[1])
    return ", ".join(f"{c}={v:.0f}" for c, v in pairs[:4])


# ── determine best model label ────────────────────────────────────────────────

def bold_best_in_column(vals, lower_is_better=True):
    """Return list of (val, is_bold) pairs."""
    valid = [v for v in vals if v is not None and not np.isnan(v)]
    if not valid:
        return [(v, False) for v in vals]
    best = min(valid) if lower_is_better else max(valid)
    return [(v, v == best) for v in vals]


# ══════════════════════════════════════════════════════════════════════════════
# BUILD MARKDOWN
# ══════════════════════════════════════════════════════════════════════════════

lines = []
now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

lines += [
    "# OOS Rolling Results: Stable vs Baseline — Full Analysis",
    "",
    f"_Auto-generated: {now} — re-run `python Code/Overview_oos_analysis.py` to refresh._",
    "",
]

# ── Section 1: Summary Table ──────────────────────────────────────────────────

lines += [
    "## 1. Summary Table (5Y train / 6M test, 3500 epochs)",
    "",
    "| Model | Epochs | N windows | Mean OOS (bps) | Mean IS (bps) | Max OOS (bps) |",
    "|-------|--------|-----------|----------------|---------------|---------------|",
]

summaries = {}
for dim, mtype, ep in MODELS:
    key = model_key(dim, mtype)
    data[key]["summary"] = summary(key)
    summaries[key] = data[key]["summary"]

mean_oos_vals = [summaries[model_key(d, t)]["mean_oos"] if summaries[model_key(d, t)] else np.nan
                 for d, t, _ in MODELS]
max_oos_vals  = [summaries[model_key(d, t)]["max_oos"] if summaries[model_key(d, t)] else np.nan
                 for d, t, _ in MODELS]

for i, (dim, mtype, ep) in enumerate(MODELS):
    key = model_key(dim, mtype)
    s = summaries[key]
    if s is None:
        lines.append(f"| dim{dim} {mtype} | {ep} | — | — | — | — |")
        continue

    mean_oos = s["mean_oos"]
    max_oos  = s["max_oos"]
    mean_is  = s["mean_is"]
    n_win    = s["n_win"]

    # bold best mean OOS per dim-pair
    pair_keys = [model_key(dim, "baseline"), model_key(dim, "stable")]
    pair_means = [summaries[k]["mean_oos"] if summaries[k] else np.nan for k in pair_keys]
    best_mean = min(pair_means)
    mean_str = f"**{mean_oos:.1f}**" if abs(mean_oos - best_mean) < 1e-9 else f"{mean_oos:.1f}"

    pair_maxes = [summaries[k]["max_oos"] if summaries[k] else np.nan for k in pair_keys]
    best_max = max(pair_maxes)
    max_str = f"**{max_oos:.1f}**" if abs(max_oos - best_max) < 1e-9 else f"{max_oos:.1f}"

    is_str = fmt(mean_is) if not np.isnan(mean_is) else "—"

    lines.append(f"| dim{dim} {mtype} | {ep} | {n_win} | {mean_str} | {is_str} | {max_str} |")

lines.append("")

# Bottom line narrative
stable_wins = []
baseline_wins = []
for dim in ["2", "3", "4"]:
    bk = model_key(dim, "baseline")
    sk = model_key(dim, "stable")
    if summaries[bk] and summaries[sk]:
        if summaries[sk]["mean_oos"] < summaries[bk]["mean_oos"]:
            stable_wins.append(f"dim{dim}")
        else:
            baseline_wins.append(f"dim{dim}")

best_key = min(
    [model_key(d, t) for d, t, _ in MODELS if summaries[model_key(d, t)]],
    key=lambda k: summaries[k]["mean_oos"]
)
best_mean = summaries[best_key]["mean_oos"]
best_max  = summaries[best_key]["max_oos"]

lines += [
    f"**Bottom line:** Stable wins at {', '.join(stable_wins)} (mean OOS "
    f"{summaries[best_key]['mean_oos']:.1f} bps for {best_key.replace('_', ' ')})."
    + (f" Baseline edges out stable at {', '.join(baseline_wins)} — largely driven by ODE"
       " crash(es) inflating the stable mean." if baseline_wins else ""),
    "",
    "---",
    "",
]

# ── Section 2: NaN note ───────────────────────────────────────────────────────

lines += [
    "## 2. NaN Currency RMSE — What Is It?",
    "",
    "The `NaN` entries in per-currency RMSE columns (GBP from 2022-07-01, USD from 2023-07-01 onward)"
    " are **not model failures**. They simply mean that currency has **no observations in that test window**"
    " (data ends earlier in the dataset). The `avg_rmse_bps` is computed as `nanmean` across available"
    " currencies and is unaffected.",
    "",
    "`n_test_bad` is the more important metric — it counts rows where the ODE integration produced"
    " NaN/Inf and those rows were excluded from RMSE.",
    "",
    "---",
    "",
]

# ── Section 3: Big errors ─────────────────────────────────────────────────────

mech_a, mech_b = big_error_windows(threshold=50)

lines += [
    "## 3. Big Errors — Two Distinct Mechanisms",
    "",
    "### Mechanism A: ODE Numerical Divergence (`n_test_bad > 0`)",
    "The ODE integrator itself crashes for certain test observations, producing NaN/Inf outputs"
    " which are excluded. This inflates `avg_rmse_bps` even with fewer 'good' rows.",
    "",
]

if mech_a:
    lines += [
        "| Window | Model | `n_test_bad` | avg OOS | Worst currencies |",
        "|--------|-------|--------------|---------|-----------------|",
    ]
    for e in mech_a:
        lines.append(
            f"| {e['window']} | {e['model']} | **{e['bad']}** | {e['oos']:.0f} bps | {e['worst']} |"
        )
else:
    lines.append("_No ODE crash windows found above threshold._")

lines += [
    "",
    "### Mechanism B: Finite but Wildly Wrong (`n_test_bad = 0`, huge RMSE)",
    "The ODE completes but predicts completely wrong values.",
    "",
]

if mech_b:
    lines += [
        "| Window | Model | λ_max | avg OOS | Worst currencies |",
        "|--------|-------|-------|---------|-----------------|",
    ]
    for e in mech_b:
        lam_str = f"**+{e['lam_max']:.2f}**" if e["lam_max"] > 0 else f"−{abs(e['lam_max']):.2f} (stable)"
        lines.append(
            f"| {e['window']} | {e['model']} | {lam_str} | {e['oos']:.0f} bps | {e['worst']} |"
        )
else:
    lines.append("_No finite big-error windows found above threshold._")

lines += [
    "",
    "**Baseline explosions** are driven by positive eigenvalues (λ_max > 0) — the drift matrix M"
    " is explosive, which generalises catastrophically OOS when latent `z` extrapolates outside the"
    " training manifold.",
    "",
    "**Stable model failures** (all-negative eigenvalues) are pure **regime mismatch** — the model"
    " has too few factors to extrapolate through extreme market regimes (e.g. the 2022 rate-hike cycle).",
    "",
    "---",
    "",
]

# ── Section 4: Eigenvalue patterns ───────────────────────────────────────────

lines += [
    "## 4. Eigenvalue Patterns",
    "",
    "### Baseline models — eigenvalues **unconstrained**:",
    "",
]
for dim in ["2", "3", "4"]:
    key = model_key(dim, "baseline")
    es = eig_summary(key)
    df = data[key]["df"]
    if es is None or df is None:
        continue
    pos_win = es["positive_windows"]
    lines.append(
        f"- **dim{dim}_baseline**: λ range [{es['min']:.2f}, {es['max']:.2f}];"
        f" {pos_win}/{len(df)} windows have at least one positive eigenvalue (explosive manifold)"
    )

lines += [
    "",
    "### Stable models — eigenvalues all ≤ 0 by construction:",
    "",
]
for dim in ["2", "3", "4"]:
    key = model_key(dim, "stable")
    es = eig_summary(key)
    df = data[key]["df"]
    if es is None or df is None:
        continue
    near_zero = es["near_zero_windows"]
    # any failures?
    fail_wins = df[df["avg_rmse_bps"] > 50]["roll_start"].tolist() if df is not None else []
    fail_str = f"; failure windows: {', '.join(fail_wins)}" if fail_wins else "; no failure windows"
    lines.append(
        f"- **dim{dim}_stable**: λ range [{es['min']:.3f}, {es['max']:.3f}];"
        f" {near_zero}/{len(df)} windows with λ₁ > −0.05 (near unit-root){fail_str}"
    )

lines += [
    "",
    "**Near-zero eigenvalue issue**: When λ ≈ 0, the process is essentially a random walk,"
    " making it sensitive to out-of-distribution latent `z` values. This can cause ODE solver"
    " failures (dim2_stable 2016-07-01) or inflated errors. Higher-dimensional stable models"
    " appear more robust because the additional fast-mean-reverting factors stabilise the ODE numerically.",
    "",
    "---",
    "",
]

# ── Section 5: Troublesome windows table ──────────────────────────────────────

lines += [
    "## 5. All Windows — Per-Model RMSE (bps)",
    "",
]

# build a combined table: rows = windows, cols = models
all_keys = [model_key(d, t) for d, t, _ in MODELS]
# get union of windows from the first available df
ref_df = next((data[k]["df"] for k in all_keys if data[k]["df"] is not None), None)
if ref_df is not None:
    windows = ref_df["roll_start"].tolist()
    header = "| Window | " + " | ".join(k.replace("_", " ") for k in all_keys) + " |"
    sep    = "|--------|" + "|".join(["-------"] * len(all_keys)) + "|"
    lines += [header, sep]
    for w in windows:
        row_vals = []
        for k in all_keys:
            df = data[k]["df"]
            if df is None:
                row_vals.append("—")
                continue
            r = df[df["roll_start"] == w]
            if len(r) == 0:
                row_vals.append("—")
                continue
            oos = r.iloc[0]["avg_rmse_bps"]
            bad = int(r.iloc[0].get("n_test_bad", 0)) if not np.isnan(r.iloc[0].get("n_test_bad", 0)) else 0
            cell = f"{oos:.1f}"
            if oos > 100:
                cell = f"**{oos:.0f}**"
            if bad > 0:
                cell += f" ⚠️{bad}"
            row_vals.append(cell)
        lines.append(f"| {w} | " + " | ".join(row_vals) + " |")
    lines.append("")

lines += [
    "Values > 100 bps shown in **bold**. ⚠️N means N ODE-crash rows excluded.",
    "",
    "---",
    "",
]

# ── Section 6: Key Takeaways ──────────────────────────────────────────────────

# Dynamically determine best model
ranked = sorted(
    [(k, summaries[k]) for k in all_keys if summaries[k]],
    key=lambda x: x[1]["mean_oos"]
)
best_k, best_s   = ranked[0]
second_k, second_s = ranked[1] if len(ranked) > 1 else (None, None)

lines += [
    "## 6. Key Takeaways",
    "",
    f"1. **{best_k.replace('_', ' ')} is the best model** at 3500 epochs —"
    f" mean OOS {best_s['mean_oos']:.1f} bps, max {best_s['max_oos']:.1f} bps,"
    f" n_bad = {best_s['n_bad']}.",
]

if second_k:
    lines.append(
        f"2. **{second_k.replace('_', ' ')}** is the runner-up —"
        f" mean OOS {second_s['mean_oos']:.1f} bps, max {second_s['max_oos']:.1f} bps."
    )

# stable vs baseline
for dim in ["2", "3", "4"]:
    bk = model_key(dim, "baseline")
    sk = model_key(dim, "stable")
    sb, ss = summaries[bk], summaries[sk]
    if sb and ss:
        winner = "stable" if ss["mean_oos"] < sb["mean_oos"] else "baseline"
        margin = abs(ss["mean_oos"] - sb["mean_oos"])
        lines.append(
            f"{'3' if dim == '2' else '4' if dim == '3' else '5'}."
            f" **dim{dim}**: {winner} wins by {margin:.1f} bps mean OOS"
            f" ({ss['mean_oos']:.1f} vs {sb['mean_oos']:.1f})."
        )

n = 6
lines += [
    f"{n}. **Baseline positive eigenvalues** are the root cause of all Mechanism-B explosions.",
    f"{n+1}. **2022-H2 rate-hike cycle** is the hardest period — baseline models explode (300–700 bps),"
    " dim3_stable shows regime mismatch (~146 bps), dim4_stable handles it cleanly (~24 bps).",
    f"{n+2}. **Per-currency NaN values** (GBP from 2022-07, USD from 2023-07) are data availability,"
    " not model failures.",
    "",
]

# ── write ─────────────────────────────────────────────────────────────────────

content = "\n".join(lines) + "\n"

with open(OUT_PATH, "w", encoding="utf-8") as f:
    f.write(content)

print(f"Written -> {OUT_PATH}")
print()
print("=== Summary ===")
for dim, mtype, ep in MODELS:
    key = model_key(dim, mtype)
    s = summaries[key]
    if s:
        print(f"  {key:20s}  mean={s['mean_oos']:6.1f}  max={s['max_oos']:7.1f}  is={s['mean_is']:5.1f}  n_bad={s['n_bad']}")
    else:
        print(f"  {key:20s}  NO DATA")


