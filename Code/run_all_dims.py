"""
Runner script: runs OOS stages sequentially.

  Stage 1 — OutOfSampleSplit.py: LATENT_DIM = 2, 3, 4
  Stage 2 — OutOfSampleRoll.py:  LATENT_DIM = 2, 3, 4

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

STAGES = [
    {
        "name":   "RepairSplitDim2",
        "script": os.path.join(REPO_ROOT, "Code", "repair_split_dim2.py"),
        "dims":   [2],        # single run, no LATENT_DIM patching needed
        "no_patch": True,     # script has LATENT_DIM hardcoded
    },
    {
        "name":   "OutOfSampleSplit",
        "script": os.path.join(REPO_ROOT, "Code", "OutOfSampleSplit.py"),
        "dims":   [3, 4],
    },
]

# ── helpers ───────────────────────────────────────────────────────────────────
def patch_latent_dim(script_path: str, dim: int) -> str:
    """Replace LATENT_DIM = <any int> with LATENT_DIM = dim. Returns original source."""
    with open(script_path, "r") as f:
        original = f.read()
    patched = re.sub(r"^(LATENT_DIM\s*=\s*)\d+", rf"\g<1>{dim}", original, flags=re.MULTILINE)
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

        no_patch = stage.get("no_patch", False)
        if no_patch:
            original_source = None
        else:
            original_source = patch_latent_dim(stage["script"], dim)

        env = os.environ.copy()
        env["PYTHONPATH"] = REPO_ROOT

        try:
            result = subprocess.run(
                [sys.executable, stage["script"]],
                cwd=REPO_ROOT,
                env=env,
            )
        finally:
            if original_source is not None:
                restore_source(stage["script"], original_source)

        if result.returncode != 0:
            print(f"\n[ERROR] {stage['name']} failed for LATENT_DIM={dim} "
                  f"(exit code {result.returncode}). Stopping.")
            sys.exit(result.returncode)

        print(f"\n[DONE] {stage['name']}  LATENT_DIM={dim} finished successfully.\n")

print("\nAll runs complete!")
