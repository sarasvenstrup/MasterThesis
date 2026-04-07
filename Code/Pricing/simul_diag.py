import os
import sys
import tempfile
import subprocess
from pathlib import Path

import torch

# ==========================================================
# User settings
# ==========================================================
USE = "bbg"
LATENT_DIM = 2
EPOCHS = 200
IDX_CHOICE = 0
CCY_FILTER = "EUR"
SEED = 1234

USE_PRICING_CHECKPOINT = True
PRICING_RUN_NAME = "pricing_dyn_ep200"
EXPLICIT_CHECKPOINT_PATH = None  # set to a string path to override

N_PATHS = 500
N_STEPS = 120
DT = 1.0 / 12.0
DISCRETIZATION = "euler"
SIM_MODE = "full"
DIFFUSION_SCALE = 0.5

# Diagnostics settings
USE_SAVED_METADATA = True
MAX_MAHAL = 4.0
G0_FLOOR = 1e-5
MARTINGALE_DATES = (5, 10, 20, 30)
MARTINGALE_TOL = 0.02
PLOT_CURVE_TIMES = (0, 0.5, 1.0, 2.0)
PLOT_TENORS = (1, 5, 10, 30)
PLOT_DPI = 200
SHOW_PLOTS = False
MARTINGALE_LOG_EVERY_COMBO = 1
MARTINGALE_LOG_EVERY_PATHS = 100

# ==========================================================
# Path setup
# ==========================================================
try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    SCRIPT_DIR = Path.cwd()

# Expected local layout when you place this file in Code/Pricing or elsewhere inside project
REPO_ROOT = Path.cwd()
for candidate in [SCRIPT_DIR, Path.cwd(), Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()]:
    if (candidate / "Code").exists() and (candidate / "Figures").exists():
        REPO_ROOT = candidate
        break
    if candidate.name == "Pricing" and (candidate.parent.parent / "Figures").exists():
        REPO_ROOT = candidate.parent.parent
        break

CODE_ROOT = REPO_ROOT / "Code"
PRICING_ROOT = CODE_ROOT / "Pricing"

for p in [str(REPO_ROOT), str(CODE_ROOT), str(PRICING_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from Code.Pricing.simulate_model_naive import run_simulation


# ==========================================================
# Helpers
# ==========================================================
def patch_bundle_metadata(bundle_path: Path, expected: dict) -> dict:
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
    metadata = dict(bundle.get("metadata", {}))
    metadata.update(expected)
    bundle["metadata"] = metadata
    torch.save(bundle, bundle_path)
    return metadata


def make_temp_diagnostics_script(original_path: Path) -> Path:
    text = original_path.read_text(encoding="utf-8")

    injection = '''\n# Driver override inserted automatically\n_override_bundle_path = os.environ.get("SIM_BUNDLE_PATH", "").strip()\nif _override_bundle_path:\n    print(f"[driver] Overriding BUNDLE_PATH from environment: {_override_bundle_path}")\n    BUNDLE_PATH = _override_bundle_path\n'''

    marker = "\n# ==========================================================\n# Run\n# ==========================================================\n"
    if marker in text and injection not in text:
        text = text.replace(marker, injection + marker, 1)
    else:
        raise RuntimeError(
            "Could not patch simulation_diagnostics.py automatically. "
            "Expected the '# Run' marker to be present."
        )

    fd, temp_path = tempfile.mkstemp(prefix="diag_driver_", suffix=".py")
    os.close(fd)
    temp_path = Path(temp_path)
    temp_path.write_text(text, encoding="utf-8")
    return temp_path


# ==========================================================
# Main
# ==========================================================
def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 72)
    print("SIMULATION STAGE")
    print("=" * 72)
    print(f"Repo root: {REPO_ROOT}")
    print(f"Pricing root: {PRICING_ROOT}")
    print(f"Device: {device}")
    print(f"Pricing checkpoint mode: {USE_PRICING_CHECKPOINT}")
    print(f"Pricing run name: {PRICING_RUN_NAME}")
    print(f"Diffusion scale: {DIFFUSION_SCALE}")

    sim_out = run_simulation(
        use=USE,
        latent_dim=LATENT_DIM,
        epochs=EPOCHS,
        checkpoint_path=EXPLICIT_CHECKPOINT_PATH,
        use_pricing_checkpoint=USE_PRICING_CHECKPOINT,
        pricing_run_name=PRICING_RUN_NAME,
        n_paths=N_PATHS,
        n_steps=N_STEPS,
        dt=DT,
        idx_choice=IDX_CHOICE,
        ccy_filter=CCY_FILTER,
        discretization=DISCRETIZATION,
        sim_mode=SIM_MODE,
        diffusion_scale=DIFFUSION_SCALE,
        seed=SEED,
        device=device,
        save_bundle=True,
    )

    bundle_path = Path(sim_out["bundle_path"]).resolve()
    print(f"\nBundle path returned by simulator:\n  {bundle_path}")

    expected_metadata = {
        "use": USE,
        "latent_dim": int(LATENT_DIM),
        "epochs": int(EPOCHS),
        "n_paths": int(N_PATHS),
        "n_steps": int(N_STEPS),
        "dt": float(DT),
        "idx_choice": int(IDX_CHOICE),
        "ccy_filter": CCY_FILTER,
        "discretization": DISCRETIZATION,
        "sim_mode": SIM_MODE,
        "diffusion_scale": float(DIFFUSION_SCALE),
        "seed": int(SEED),
        "use_pricing_checkpoint": bool(USE_PRICING_CHECKPOINT),
        "pricing_run_name": PRICING_RUN_NAME if PRICING_RUN_NAME is not None else "",
    }
    patched_metadata = patch_bundle_metadata(bundle_path, expected_metadata)

    print("\nPatched bundle metadata used for diagnostics:")
    for k in [
        "checkpoint_path",
        "use_pricing_checkpoint",
        "pricing_run_name",
        "diffusion_scale",
        "sim_mode",
        "discretization",
        "n_paths",
        "n_steps",
        "seed",
    ]:
        print(f"  {k}: {patched_metadata.get(k)}")

    print("\n" + "=" * 72)
    print("DIAGNOSTICS STAGE")
    print("=" * 72)

    diagnostics_source = PRICING_ROOT / "simulation_diagnostics.py"
    if not diagnostics_source.exists():
        raise FileNotFoundError(f"Could not find diagnostics source file: {diagnostics_source}")

    temp_diag = make_temp_diagnostics_script(diagnostics_source)
    try:
        env = os.environ.copy()
        env["SIM_BUNDLE_PATH"] = str(bundle_path)
        env.setdefault("PYTHONIOENCODING", "utf-8")

        print("Running diagnostics on the exact bundle path above...")
        print(f"Temporary diagnostics launcher: {temp_diag}")
        result = subprocess.run(
            [sys.executable, str(temp_diag)],
            cwd=str(REPO_ROOT),
            env=env,
            check=False,
        )
    finally:
        try:
            temp_diag.unlink(missing_ok=True)
        except Exception:
            pass

    if result.returncode != 0:
        print(f"\nDiagnostics exited with code {result.returncode}.")
        return result.returncode

    print("\nFinished successfully.")
    print(f"Bundle diagnosed: {bundle_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
