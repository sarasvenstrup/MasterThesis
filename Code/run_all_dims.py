"""
Runner script: runs Training_baseline.py sequentially for each latent dim.

  Stage 1 — Training_baseline.py (baseline): LATENT_DIM = 3, 2, 4, 1  ep=5000

Baseline uses a frozen model file (full_model_baseline.py) so that stable
development can never affect baseline results.

Run from the repo root:
    python Code/run_all_dims.py
"""
import re
import subprocess
import sys
import os

# ── config ────────────────────────────────────────────────────────────────────
try:
    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    REPO_ROOT = os.getcwd()

TRAINING_BASELINE_PATH = os.path.join(REPO_ROOT, "Code", "Training_baseline.py")

STAGES = [
    {
        "name":    "Training (baseline)",
        "script":  TRAINING_BASELINE_PATH,
        "dims":    [3, 2, 4, 1],
        "epochs":  5000,
    },
]

# ── helpers ───────────────────────────────────────────────────────────────────
def patch_latent_dim(script_path: str, dim: int) -> str:
    """Replace LATENT_DIM = <any int> in script. Returns original source."""
    with open(script_path, "r") as f:
        original = f.read()
    patched = re.sub(r"^(LATENT_DIM\s*=\s*)\d+", rf"\g<1>{dim}", original, flags=re.MULTILINE)
    with open(script_path, "w") as f:
        f.write(patched)
    return original

def patch_epochs(script_path: str, epochs: int) -> str:
    """Replace EPOCHS = <any int> in script. Returns original source."""
    with open(script_path, "r") as f:
        original = f.read()
    patched = re.sub(r"^(EPOCHS\s*=\s*)\d+", rf"\g<1>{epochs}", original, flags=re.MULTILINE)
    with open(script_path, "w") as f:
        f.write(patched)
    return original

def restore_source(script_path: str, original: str):
    with open(script_path, "w") as f:
        f.write(original)

# ── main loop ─────────────────────────────────────────────────────────────────
total_runs = sum(len(s["dims"]) for s in STAGES)
run_num = 0

for stage in STAGES:
    for dim in stage["dims"]:
        run_num += 1
        print(f"\n{'='*60}")
        print(f"  [{run_num}/{total_runs}] {stage['name']}  LATENT_DIM={dim}")
        print(f"{'='*60}\n")

        original_script = patch_latent_dim(stage["script"], dim)
        original_epochs = None
        if "epochs" in stage:
            original_epochs = patch_epochs(stage["script"], stage["epochs"])

        env = os.environ.copy()
        env["PYTHONPATH"] = REPO_ROOT

        try:
            result = subprocess.run(
                [sys.executable, stage["script"]],
                cwd=REPO_ROOT,
                env=env,
                text=True,
            )
        finally:
            restore_source(stage["script"], original_script)
            if original_epochs is not None:
                restore_source(stage["script"], original_epochs)

        if result.returncode != 0:
            print(f"\n[ERROR] {stage['name']} failed for LATENT_DIM={dim} "
                  f"(exit code {result.returncode}). Stopping.")
            sys.exit(result.returncode)

        print(f"\n[DONE] {stage['name']}  LATENT_DIM={dim} finished successfully.\n")

print("\nAll runs complete!")
