"""
Generates Figures/OOSResults/Roll/OOS_RUNS_STATUS.md from the run_manifest.json files.
Run from repo root: python Code/gen_oos_status.py
"""
import json
import os
import datetime

try:
    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    REPO_ROOT = os.getcwd()

ROLL_ROOT = os.path.join(REPO_ROOT, "Figures", "OOSResults", "Roll")
OUT_PATH  = os.path.join(ROLL_ROOT, "OOS_RUNS_STATUS.md")

RUNS_5Y6M = [
    ("dim2_baseline", 3500),
    ("dim2_stable",   3500),
    ("dim3_baseline", 3500),
    ("dim3_stable",   3500),
    ("dim4_baseline", 3500),
    ("dim4_stable",   3500),
]

def load(model, subdir, ep):
    p = os.path.join(ROLL_ROOT, f"OOS_roll_{model}", subdir, f"ep{ep}", "run_manifest.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def make_row(model, subdir, ep):
    m = load(model, subdir, ep)
    ep_str = f"**{ep}**"
    if m is None:
        return f"| {model} | {ep_str} | - | - | MISSING |"
    started  = m.get("run_started", "?")
    finished = m.get("run_finished", None)
    n_win    = m.get("n_windows", "?")
    win_done = len(m.get("window_results", {}))
    if finished:
        status  = f"{win_done}/{n_win} done"
        fin_str = finished
    else:
        status  = f"{win_done}/{n_win} IN PROGRESS"
        fin_str = "still running"
    return f"| {model} | {ep_str} | {started} | {fin_str} | {status} |"


lines = [
    "# OOS Rolling Run Status",
    "",
    f"_Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} -- re-run `python Code/gen_oos_status.py` after pulling to refresh._",
    "",
    "## train5Y / test6M / step6M (main comparison)",
    "",
    "| Model | Epochs | Started | Finished | Windows |",
    "|---|---|---|---|---|",
]
for model, ep in RUNS_5Y6M:
    lines.append(make_row(model, "train5Y_test6M_step6M", ep))


missing = [
    f"{model} ep{ep}"
    for model, ep in RUNS_5Y6M
    if load(model, "train5Y_test6M_step6M", ep) is None
]

lines += [
    "",
    "## Summary",
    f"- Complete: {len(RUNS_5Y6M) - len(missing)}/{len(RUNS_5Y6M)} main runs",
]
if missing:
    lines.append(f"- MISSING: {', '.join(missing)}")
else:
    lines.append("- All main runs complete!")

content = "\n".join(lines) + "\n"

with open(OUT_PATH, "w", encoding="utf-8") as f:
    f.write(content)

print(f"Written -> {OUT_PATH}")
print(content)

