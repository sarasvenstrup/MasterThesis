"""
Generates Figures/OOSResults/Roll/RUNS_STATUS.md from the run_manifest.json files.
Run from repo root: python Code/Overview_oos_status.py
"""
import json
import os
import datetime

try:
    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    REPO_ROOT = os.getcwd()

ROLL_ROOT = os.path.join(REPO_ROOT, "Figures", "OOSResults", "Roll")
TRAIN_ROOT = os.path.join(REPO_ROOT, "Figures", "TrainingResults")
OUT_PATH  = os.path.join(ROLL_ROOT, "RUNS_STATUS.md")

RUNS_5Y6M = [
    ("dim2_baseline", 3500),
    ("dim2_stable",   3500),
    ("dim3_baseline", 3500),
    ("dim3_stable",   3500),
    ("dim4_baseline", 3500),
    ("dim4_stable",   3500),
]

TRAIN_RUNS = [
    ("dim2_baseline", 3500),
    ("dim2_stable",   3500),
    ("dim3_baseline", 3500),
    ("dim3_stable",   3500),
    ("dim4_baseline", 3500),
    ("dim4_stable",   3500),
    ("dim2_baseline", 5000),
    ("dim2_stable",   5000),
    ("dim3_baseline", 5000),
    ("dim3_stable",   5000),
    ("dim4_baseline", 5000),
    ("dim4_stable",   5000),
    ("dim4_stable_joint",   5000),
]


def load_oos(model, subdir, ep):
    p = os.path.join(ROLL_ROOT, f"OOS_roll_{model}", subdir, f"ep{ep}", "run_manifest.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def load_train(model, ep):
    p = os.path.join(TRAIN_ROOT, model, f"ep{ep}", "run_config.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def make_oos_row(model, subdir, ep):
    m = load_oos(model, subdir, ep)
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


def make_train_row(model, ep):
    m = load_train(model, ep)
    ep_str = f"**{ep}**"
    if m is None:
        return f"| {model} | {ep_str} | MISSING |"

    # Check for completion by looking for the checkpoint file
    # The training scripts (Training.py, Training_stable.py) save the final
    # checkpoint with a name like: checkpoint_dim{LATENT_DIM}_ep{EPOCHS}.pt
    latent_dim = m.get("latent_dim")
    if latent_dim is None:
        # Fallback if latent_dim is not in run_config.json
        try:
            latent_dim = int(model.split('_')[0].replace('dim', ''))
        except (ValueError, IndexError):
            return f"| {model} | {ep_str} | UNKNOWN (bad model name) |"

    ckpt_path = os.path.join(TRAIN_ROOT, model, f"ep{ep}", f"checkpoint_dim{latent_dim}_ep{ep}.pt")
    status = "DONE" if os.path.exists(ckpt_path) else "IN PROGRESS"

    return f"| {model} | {ep_str} | {status} |"


lines = [
    "# OOS Rolling Run Status",
    "",
    f"_Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} -- re-run `python Code/Overview_oos_status.py` after pulling to refresh._",
    "",
    "## train5Y / test6M / step6M (main comparison)",
    "",
    "| Model | Epochs | Started | Finished | Windows |",
    "|---|---|---|---|---|",
]
for model, ep in RUNS_5Y6M:
    lines.append(make_oos_row(model, "train5Y_test6M_step6M", ep))


lines += [
    "",
    "## Normal Training Run Status",
    "",
    "| Model | Epochs | Status |",
    "|---|---|---|",
]
for model, ep in TRAIN_RUNS:
    lines.append(make_train_row(model, ep))


missing_oos = [
    f"{model} ep{ep}"
    for model, ep in RUNS_5Y6M
    if load_oos(model, "train5Y_test6M_step6M", ep) is None
]

missing_train = [
    f"{model} ep{ep}"
    for model, ep in TRAIN_RUNS
    if load_train(model, ep) is None
]


lines += [
    "",
    "## Summary",
    f"- Complete OOS: {len(RUNS_5Y6M) - len(missing_oos)}/{len(RUNS_5Y6M)} main runs",
    f"- Complete Train: {len(TRAIN_RUNS) - len(missing_train)}/{len(TRAIN_RUNS)} main runs",
]
if missing_oos:
    lines.append(f"- MISSING OOS: {', '.join(missing_oos)}")
else:
    lines.append("- All OOS runs complete!")

if missing_train:
    lines.append(f"- MISSING Train: {', '.join(missing_train)}")
else:
    lines.append("- All training runs complete!")


content = "\n".join(lines) + "\n"

with open(OUT_PATH, "w", encoding="utf-8") as f:
    f.write(content)

print(f"Written -> {OUT_PATH}")
print(content)

