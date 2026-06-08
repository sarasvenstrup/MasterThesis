# =============================================================================
# Overnight sweep — Prevalence experiment
#
# Runs Representation_Experiment.py sequentially for specific (dim, model_type)
# pairs. To run both simultaneously, open two terminals and paste the commands
# printed at startup.
#
# Results are saved to:
#   Figures/Experiments/PrevalenceSweep/
#     any_negative_dim{d}_ep{E}_N{N}_{model_type}/
#
# Launch from repo root (or right-click → Run in PyCharm):
#   python Code/Experiments/Overnight_Representation_Experiment.py
# =============================================================================

import os
import sys
import subprocess
import time

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

# ── Sweep configuration ────────────────────────────────────────────────────
RUNS = [
    (2, "baseline"),
    (4, "stable"),
]
REGIME      = "any_negative"
EPOCHS      = 5000
N_TRAIN     = 2000
LR          = 1e-3
BATCH_SIZE  = 32
SEED        = 0
PREVALENCES = "0.00,0.05,0.10,0.15,0.20,0.30,0.40,0.50,0.60,0.75,1.00"
# ──────────────────────────────────────────────────────────────────────────

EXPERIMENT_SCRIPT = os.path.join(HERE, "Representation_Experiment.py")


def make_cmd(latent_dim: int, model_type: str) -> list:
    return [
        sys.executable, EXPERIMENT_SCRIPT,
        "--regime",      REGIME,
        "--latent-dim",  str(latent_dim),
        "--model-type",  model_type,
        "--epochs",      str(EPOCHS),
        "--n-train",     str(N_TRAIN),
        "--lr",          str(LR),
        "--batch-size",  str(BATCH_SIZE),
        "--seed",        str(SEED),
        "--prevalences", PREVALENCES,
    ]


def run_one(latent_dim: int, model_type: str) -> bool:
    label = f"dim={latent_dim}  model={model_type}"
    print(f"\n{'='*65}", flush=True)
    print(f"  Starting: {label}", flush=True)
    print(f"  epochs={EPOCHS}  lr={LR}  n_train={N_TRAIN}  seed={SEED}", flush=True)
    print(f"{'='*65}\n", flush=True)

    t0 = time.perf_counter()
    result = subprocess.run(make_cmd(latent_dim, model_type), check=False)
    elapsed = (time.perf_counter() - t0) / 60

    if result.returncode == 0:
        print(f"\n  Finished {label} in {elapsed:.1f} min", flush=True)
        return True
    else:
        print(f"\n  WARNING: {label} exited with code {result.returncode} "
              f"after {elapsed:.1f} min", flush=True)
        return False


def main():
    sweep_t0 = time.perf_counter()

    print(f"Overnight prevalence sweep: {len(RUNS)} runs (sequential)", flush=True)
    print(f"  regime={REGIME}  epochs={EPOCHS}  n_train={N_TRAIN}  lr={LR}\n", flush=True)

    print("  To run both simultaneously, paste these into two separate terminals:")
    for latent_dim, model_type in RUNS:
        print(f"    {' '.join(make_cmd(latent_dim, model_type))}")
    print(flush=True)

    runs_done = runs_failed = 0
    for latent_dim, model_type in RUNS:
        ok = run_one(latent_dim, model_type)
        runs_done += 1
        if not ok:
            runs_failed += 1
        elapsed_total = (time.perf_counter() - sweep_t0) / 60
        remaining = len(RUNS) - runs_done
        eta = (elapsed_total / runs_done) * remaining if runs_done else 0
        print(f"  Progress: {runs_done}/{len(RUNS)} done  "
              f"({runs_failed} failed)  elapsed={elapsed_total:.0f}min  ETA={eta:.0f}min",
              flush=True)

    elapsed_total = (time.perf_counter() - sweep_t0) / 60
    print(f"\n{'='*65}", flush=True)
    print(f"Sweep complete: {runs_done} runs in {elapsed_total:.0f} min  "
          f"({runs_failed} failed)", flush=True)
    print(f"{'='*65}", flush=True)


if __name__ == "__main__":
    main()
