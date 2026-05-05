"""
Runner script: runs missing OOS roll experiments and full IS training for stable ℓ=3,4.

  Stage 1 — Training (stable):   LATENT_DIM = 4        (ep5000)
  Stage 2 — Training (stable):   LATENT_DIM = 3        (ep5000)
  Stage 3 — OOS roll (baseline): LATENT_DIM = 3
  Stage 4 — OOS roll (stable):   LATENT_DIM = 2, 3, 4

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

OOS_ROLL_PATH  = os.path.join(REPO_ROOT, "Code", "OutOfSampleRoll.py")
TRAINING_PATH  = os.path.join(REPO_ROOT, "Code", "Training_stable.py")

STAGES = [
    # ── Stage 1: IS training — stable ℓ=4 ───────────────────────────────────
    {
        "name":          "Training (stable ℓ=4)",
        "script":        TRAINING_PATH,
        "dims":          [4],
        "model_variant": "stable",
        "patches":       {"EPOCHS": 5000},   # override default 2000 → 5000
    },
    # ── Stage 2: IS training — stable ℓ=3 ───────────────────────────────────
    {
        "name":          "Training (stable ℓ=3)",
        "script":        TRAINING_PATH,
        "dims":          [3],
        "model_variant": "stable",
        "patches":       {"EPOCHS": 5000},   # override default 2000 → 5000
    },
    # ── Stage 3: OOS roll — baseline ℓ=3 ────────────────────────────────────
    {
        "name":          "OOS roll (baseline ℓ=3)",
        "script":        OOS_ROLL_PATH,
        "dims":          [3],
        "model_variant": "baseline",
        "patches":       {},          # EPOCHS already 3500 in OutOfSampleRoll.py
    },
    # ── Stage 4: OOS roll — stable ℓ=2,3,4 ─────────────────────────────────
    {
        "name":          "OOS roll (stable)",
        "script":        OOS_ROLL_PATH,
        "dims":          [2, 3, 4],
        "model_variant": "stable",
        "patches":       {},
    },
]

# ── helpers ───────────────────────────────────────────────────────────────────
def patch_script(script_path: str, dim: int, extra_patches: dict) -> str:
    """Patch LATENT_DIM and any extra variables in script. Returns original source."""
    with open(script_path, "r") as f:
        original = f.read()
    patched = re.sub(r"^(LATENT_DIM\s*=\s*)\d+", rf"\g<1>{dim}", original, flags=re.MULTILINE)
    for var, value in extra_patches.items():
        patched = re.sub(rf"^({var}\s*=\s*)\S+", rf"\g<1>{value}", patched, flags=re.MULTILINE)
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

        original_script = patch_script(stage["script"], dim, stage["patches"])

        env = os.environ.copy()
        env["PYTHONPATH"]           = REPO_ROOT
        env["SKIP_VARIANT_CONFIRM"] = "1"
        env["MODEL_VARIANT"]        = stage["model_variant"]

        try:
            result = subprocess.run(
                [sys.executable, stage["script"]],
                cwd=REPO_ROOT,
                env=env,
                text=True,
            )
        finally:
            restore_source(stage["script"], original_script)

        if result.returncode != 0:
            print(f"\n[ERROR] {stage['name']} failed for LATENT_DIM={dim} "
                  f"(exit code {result.returncode}). Stopping.")
            sys.exit(result.returncode)

        print(f"\n[DONE] {stage['name']}  LATENT_DIM={dim} finished successfully.\n")

print("\nAll runs complete!")
