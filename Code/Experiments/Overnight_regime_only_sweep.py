# =============================================================================
# Overnight sweep — Regime-only experiment across dims and model types
#
# Runs Experiment_regime_only.py for every combination of:
#   latent_dim   in [2, 3, 4]
#   model_type   in ["baseline", "stable"]
#
# Why lr=1e-3 instead of the default 1e-2:
#   The any_negative regime has only ~235 training curves → 8 batches/epoch.
#   At lr=1e-2 one unlucky gradient spike can drive weights to NaN; the NaN
#   guard then skips every subsequent batch, training goes silent but "rmse=0"
#   and the final eval returns NaN.  lr=1e-3 still converges at 5000 epochs
#   and avoids this for all dim/regime combinations.
#
# Results are saved to:
#   Figures/Experiments/RegimeOnly/dim{d}_ep{E}_{model_type}/
#
# Launch from repo root:
#   python Code/Experiments/Overnight_regime_only_sweep.py
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
LATENT_DIMS  = [2, 3, 4]
MODEL_TYPES  = ["baseline", "stable"]
EPOCHS       = 5000
LR           = 1e-3      # lower than default 1e-2 — prevents NaN divergence
BATCH_SIZE   = 32
SEED         = 0
REGIMES      = "normal_positive,any_negative,inverted"   # skip deeply_negative (too few curves)
# ──────────────────────────────────────────────────────────────────────────

EXPERIMENT_SCRIPT = os.path.join(HERE, "Experiment_regime_only.py")


def run_one(latent_dim: int, model_type: str) -> bool:
    """Run a single dim/model_type combo. Returns True if successful."""
    label = f"dim={latent_dim}  model={model_type}"
    print(f"\n{'='*65}", flush=True)
    print(f"  Starting: {label}", flush=True)
    print(f"  epochs={EPOCHS}  lr={LR}  seed={SEED}", flush=True)
    print(f"{'='*65}\n", flush=True)

    cmd = [
        sys.executable, EXPERIMENT_SCRIPT,
        "--latent-dim",  str(latent_dim),
        "--model-type",  model_type,
        "--epochs",      str(EPOCHS),
        "--lr",          str(LR),
        "--batch-size",  str(BATCH_SIZE),
        "--seed",        str(SEED),
        "--regimes",     REGIMES,
    ]

    t0 = time.perf_counter()
    result = subprocess.run(cmd, check=False)
    elapsed = (time.perf_counter() - t0) / 60

    if result.returncode == 0:
        print(f"\n  Finished {label} in {elapsed:.1f} min", flush=True)
        return True
    else:
        print(f"\n  WARNING: {label} exited with code {result.returncode} "
              f"after {elapsed:.1f} min", flush=True)
        return False


def main():
    total = len(LATENT_DIMS) * len(MODEL_TYPES)
    runs_done, runs_failed = 0, 0
    sweep_t0 = time.perf_counter()

    print(f"Overnight regime-only sweep: {total} runs", flush=True)
    print(f"  dims={LATENT_DIMS}  model_types={MODEL_TYPES}", flush=True)
    print(f"  epochs={EPOCHS}  lr={LR}", flush=True)
    print(f"  Output root: Figures/Experiments/RegimeOnly/\n", flush=True)

    for dim in LATENT_DIMS:
        for model_type in MODEL_TYPES:
            ok = run_one(dim, model_type)
            runs_done += 1
            if not ok:
                runs_failed += 1
            elapsed_total = (time.perf_counter() - sweep_t0) / 60
            remaining = total - runs_done
            eta = (elapsed_total / runs_done) * remaining if runs_done else 0
            print(f"  Progress: {runs_done}/{total} done  "
                  f"({runs_failed} failed)  "
                  f"elapsed={elapsed_total:.0f}min  ETA={eta:.0f}min",
                  flush=True)

    elapsed_total = (time.perf_counter() - sweep_t0) / 60
    print(f"\n{'='*65}", flush=True)
    print(f"Sweep complete: {runs_done} runs in {elapsed_total:.0f} min  "
          f"({runs_failed} failed)", flush=True)
    print(f"{'='*65}", flush=True)


if __name__ == "__main__":
    main()
