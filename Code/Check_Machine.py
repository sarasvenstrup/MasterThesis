# Check_Machine.py
# Run from repo root or from inside Code:
#   python Code/Check_Machine.py
#
# Purpose:
#   Compare machine/environment/data/model behavior across computers.
#   Useful for debugging why one machine gets NaN batches and another does not.

import os
import sys
import platform
import hashlib
import subprocess
import json
import random

import numpy as np
import pandas as pd
import torch


# ============================================================
# 1. Find repo root
# ============================================================

def find_repo_root():
    candidates = [
        os.getcwd(),
        os.path.dirname(os.path.abspath(__file__)),
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ]

    for c in candidates:
        if os.path.isdir(os.path.join(c, "Code")):
            return c

    raise RuntimeError(
        "Could not find repo root. Run this script from the MasterThesis folder "
        "or place it inside the Code folder."
    )


REPO_ROOT = find_repo_root()

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

print("Repo root:", REPO_ROOT)


# ============================================================
# 2. Imports from your project
# ============================================================

from Code.load_swapdata import my_data, TARGET_TENORS
from Code import config
from Code.model.full_model_stable import FullModel


# ============================================================
# 3. Helper functions
# ============================================================

def sha256_array(x: np.ndarray) -> str:
    x = np.ascontiguousarray(x)
    return hashlib.sha256(x.view(np.uint8)).hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def safe_git_info():
    out = {}

    try:
        out["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        out["git_commit"] = "Could not read git commit"

    try:
        out["git_status"] = subprocess.check_output(
            ["git", "status", "--short"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        out["git_status"] = "Could not read git status"

    return out


def print_header(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def tensor_report(name, x: torch.Tensor):
    x_cpu = x.detach().cpu()
    x_np = np.ascontiguousarray(x_cpu.numpy())

    row_bad_count = None
    if x_cpu.ndim >= 2:
        row_bad_count = (~torch.isfinite(x_cpu).all(dim=1)).sum().item()

    print(f"{name} shape:", tuple(x.shape))
    print(f"{name} dtype:", x.dtype)
    print(f"{name} all finite:", torch.isfinite(x_cpu).all().item())
    if row_bad_count is not None:
        print(f"{name} bad rows:", row_bad_count)

    finite_mask = torch.isfinite(x_cpu)

    if finite_mask.any():
        finite_vals = x_cpu[finite_mask]
        print(f"{name} finite min:", float(finite_vals.min()))
        print(f"{name} finite max:", float(finite_vals.max()))
        print(f"{name} finite mean:", float(finite_vals.mean()))
    else:
        print(f"{name} finite min:", None)
        print(f"{name} finite max:", None)
        print(f"{name} finite mean:", None)

    print(f"{name} sha256:", sha256_array(x_np))


def meta_hash_report(name, meta: pd.DataFrame):
    tmp = meta.copy()
    tmp["as_of_date"] = pd.to_datetime(tmp["as_of_date"]).dt.strftime("%Y-%m-%d")
    csv_text = tmp.to_csv(index=False)

    print(f"{name} shape:", tmp.shape)
    print(f"{name} columns:", list(tmp.columns))
    print(f"{name} first rows:")
    print(tmp.head(5).to_string(index=False))
    print(f"{name} last rows:")
    print(tmp.tail(5).to_string(index=False))
    print(f"{name} sha256:", sha256_text(csv_text))


def mps_available():
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def mps_built():
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_built()


# ============================================================
# 4. Reproducibility settings
# ============================================================

SEED = 0

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

torch.set_num_threads(4)
torch.set_num_interop_threads(2)

if hasattr(torch.backends, "mkldnn"):
    torch.backends.mkldnn.enabled = False

# Same device logic as your main rolling script:
# This does NOT choose MPS.
device_main = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# ============================================================
# 5. Machine/environment report
# ============================================================

print_header("MACHINE / ENVIRONMENT")

print("Repo root:", REPO_ROOT)
print("Platform:", sys.platform)
print("Platform full:", platform.platform())
print("Machine:", platform.machine())
print("Processor:", platform.processor())
print("Python:", sys.version)
print("Torch:", torch.__version__)
print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
print("CUDA available:", torch.cuda.is_available())
print("MPS built:", mps_built())
print("MPS available:", mps_available())
print("Device according to main script:", device_main)
print("Default dtype:", torch.get_default_dtype())
print("PYTORCH_ENABLE_MPS_FALLBACK:", os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "<not set>"))

if hasattr(torch.backends, "mkldnn"):
    print("MKLDNN enabled:", torch.backends.mkldnn.enabled)
else:
    print("MKLDNN enabled: not available")

try:
    print("\nTorch build config:")
    print(torch.__config__.show())
except Exception as e:
    print("Could not print torch config:", repr(e))


# ============================================================
# 6. Git/config report
# ============================================================

print_header("GIT / CONFIG")

git_info = safe_git_info()
print("Git commit:", git_info["git_commit"])
print("Git status:")
print(git_info["git_status"] if git_info["git_status"] else "Clean working tree")

try:
    config.confirm_variant()
except Exception as e:
    print("config.confirm_variant() failed or was skipped:", repr(e))

try:
    print("Config variant:", config.VARIANT)
except Exception:
    print("Config variant: could not read")


# ============================================================
# 7. Load data exactly like main rolling script
# ============================================================

print_header("DATA LOAD / HASHES")

USE = "bbg"

meta, X_tensor, meta_full, X_tensor_full, tenors, df_wide, df_wide_all, SCALE_IS_PERCENT = my_data(use=USE)

# Same as your rolling OOS script:
X_tensor = X_tensor_full.float()
meta = meta_full.copy()
meta["as_of_date"] = pd.to_datetime(meta["as_of_date"])
meta = meta.reset_index(drop=True)

print("USE:", USE)
print("TARGET_TENORS:", TARGET_TENORS)
print("tenors returned:", tenors)
print("SCALE_IS_PERCENT:", SCALE_IS_PERCENT)
print("df_wide shape:", df_wide.shape)
print("df_wide_all shape:", df_wide_all.shape)

tensor_report("X_tensor", X_tensor)
meta_hash_report("meta", meta)

assert len(meta) == X_tensor.shape[0], "meta and X_tensor length mismatch"


# ============================================================
# 8. Rolling-window first-window check
# ============================================================

print_header("FIRST ROLLING WINDOW CHECK")

TRAIN_YEARS = 5
TEST_MONTHS = 6
STEP_MONTHS = 6
MIN_TRAIN_OBS = 200

date_min = max(meta["as_of_date"].min(), pd.Timestamp("2010-01-01"))
date_max = meta["as_of_date"].max()

first_test_start = date_min + pd.DateOffset(years=TRAIN_YEARS)
train_start = first_test_start - pd.DateOffset(years=TRAIN_YEARS)
train_end = first_test_start - pd.DateOffset(days=1)
test_end = first_test_start + pd.DateOffset(months=TEST_MONTHS) - pd.DateOffset(days=1)

m_train = (meta["as_of_date"] >= train_start) & (meta["as_of_date"] <= train_end)
m_test = (meta["as_of_date"] >= first_test_start) & (meta["as_of_date"] <= test_end)

X_train = X_tensor[m_train.values]
X_test = X_tensor[m_test.values]

meta_train = meta.loc[m_train.values].reset_index(drop=True)
meta_test = meta.loc[m_test.values].reset_index(drop=True)

print("Overall date range:", date_min.date(), "to", date_max.date())
print("Train window:", train_start.date(), "to", train_end.date())
print("Test window:", first_test_start.date(), "to", test_end.date())
print("n_train:", len(X_train))
print("n_test:", len(X_test))

tensor_report("X_train_first_window", X_train)
tensor_report("X_test_first_window", X_test)
meta_hash_report("meta_train_first_window", meta_train)

if len(X_train) < MIN_TRAIN_OBS:
    print("WARNING: First training window has fewer observations than MIN_TRAIN_OBS")


# ============================================================
# 9. Small model forward-pass smoke test
# ============================================================

print_header("MODEL FORWARD SMOKE TEST")

LATENT_DIM = 4
BATCH_SIZE = 32

xb_cpu = X_train[:BATCH_SIZE].clone().contiguous()

print("Smoke batch shape:", tuple(xb_cpu.shape))
print("Smoke batch finite:", torch.isfinite(xb_cpu).all().item())
print("Smoke batch sha256:", sha256_array(xb_cpu.numpy()))


def smoke_test_on_device(dev: torch.device):
    print("\n--- Smoke test device:", dev, "---")

    result = {
        "device": str(dev),
        "success": False,
        "output_finite": None,
        "output_sha256": None,
        "error": None,
    }

    try:
        torch.manual_seed(SEED)
        np.random.seed(SEED)

        model = FullModel(latent_dim=LATENT_DIM).to(dev)
        model.eval()

        xb = xb_cpu.to(dev)

        print("Requested device:", dev)
        print("Model parameter device:", next(model.parameters()).device)
        print("Batch device:", xb.device)

        with torch.no_grad():
            out = model(xb)

        out_cpu = out.detach().cpu()

        output_finite = torch.isfinite(out_cpu).all().item()
        output_sha = sha256_array(out_cpu.numpy())

        print("Output shape:", tuple(out_cpu.shape))
        print("Output finite:", output_finite)

        finite_mask = torch.isfinite(out_cpu)
        if finite_mask.any():
            finite_vals = out_cpu[finite_mask]
            print("Output finite min:", float(finite_vals.min()))
            print("Output finite max:", float(finite_vals.max()))
            print("Output finite mean:", float(finite_vals.mean()))
        else:
            print("Output finite min:", None)
            print("Output finite max:", None)
            print("Output finite mean:", None)

        print("Output sha256:", output_sha)

        result["success"] = True
        result["output_finite"] = bool(output_finite)
        result["output_sha256"] = output_sha

    except Exception as e:
        print("FAILED on device", dev)
        print(type(e).__name__ + ":", str(e))
        result["error"] = type(e).__name__ + ": " + str(e)

    return result


smoke_results = []

# Always test CPU.
smoke_results.append(smoke_test_on_device(torch.device("cpu")))

# Also test MPS if available on Mac.
# This is diagnostic only. Your main script does not choose MPS unless you explicitly change the device logic.
if mps_available():
    smoke_results.append(smoke_test_on_device(torch.device("mps")))

# Also test CUDA if available.
if torch.cuda.is_available():
    smoke_results.append(smoke_test_on_device(torch.device("cuda")))


# ============================================================
# 10. Mini training stress test
# ============================================================

print_header("MINI TRAINING STRESS TEST")

from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR
import torch.nn as nn


def mini_train_stress_test(dev, epochs=100, batch_size=32, max_lr=1e-3):
    print("\n--- Mini training on:", dev, "---")

    result = {
        "device": str(dev),
        "epochs_requested": epochs,
        "batch_size": batch_size,
        "max_lr": max_lr,
        "success": False,
        "total_nan_batches": None,
        "first_failure": None,
        "final_epoch": None,
        "final_loss": None,
        "error": None,
    }

    try:
        torch.manual_seed(SEED)
        np.random.seed(SEED)

        model = FullModel(latent_dim=LATENT_DIM).to(dev)
        model.train()

        print("Requested device:", dev)
        print("Model parameter device before training:", next(model.parameters()).device)

        loader = DataLoader(
            TensorDataset(X_train),
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        )

        optim = torch.optim.Adam(model.parameters(), lr=max_lr)

        scheduler = OneCycleLR(
            optim,
            max_lr=max_lr,
            steps_per_epoch=len(loader),
            epochs=epochs,
            pct_start=0.3,
            div_factor=1.0,
            final_div_factor=3000.0,
        )

        loss_fn = nn.MSELoss()

        total_nan_batches = 0
        first_failure = None
        final_loss = None
        final_epoch = None

        for epoch in range(epochs):
            nan_batches = 0
            used_batches = 0
            epoch_loss_sum = 0.0
            epoch_obs = 0

            for batch_idx, (xb_cpu_batch,) in enumerate(loader):
                xb = xb_cpu_batch.to(dev)

                if epoch == 0 and batch_idx == 0:
                    print("First batch device:", xb.device)
                    print("Model parameter device first batch:", next(model.parameters()).device)

                optim.zero_grad(set_to_none=True)

                try:
                    S_hat = model(xb)
                except Exception as e:
                    nan_batches += 1
                    msg = f"Forward error at epoch={epoch}, batch={batch_idx}: {e}"
                    print("[Forward error]", msg)
                    if first_failure is None:
                        first_failure = msg
                    continue

                if not torch.isfinite(S_hat).all():
                    nan_batches += 1
                    msg = f"S_hat NaN/Inf at epoch={epoch}, batch={batch_idx}"
                    print("[S_hat NaN/Inf]", msg)
                    print("Input min/max:", xb.min().item(), xb.max().item())

                    finite_vals = S_hat[torch.isfinite(S_hat)]
                    if finite_vals.numel() > 0:
                        print("Finite S_hat min/max:", finite_vals.min().item(), finite_vals.max().item())
                    else:
                        print("All S_hat values are NaN/Inf")

                    if first_failure is None:
                        first_failure = msg
                    continue

                loss = loss_fn(S_hat, xb)

                if not torch.isfinite(loss):
                    nan_batches += 1
                    msg = f"Loss NaN/Inf at epoch={epoch}, batch={batch_idx}, loss={loss}"
                    print("[Loss NaN/Inf]", msg)
                    if first_failure is None:
                        first_failure = msg
                    continue

                loss.backward()

                bad_grad = False
                bad_grad_name = None

                for name, p in model.named_parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        bad_grad = True
                        bad_grad_name = name
                        break

                if bad_grad:
                    nan_batches += 1
                    msg = f"Bad gradient at epoch={epoch}, batch={batch_idx}, param={bad_grad_name}"
                    print("[Bad grad]", msg)
                    if first_failure is None:
                        first_failure = msg
                    continue

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optim.step()
                scheduler.step()

                used_batches += 1
                epoch_loss_sum += float(loss.detach().cpu()) * xb.shape[0]
                epoch_obs += xb.shape[0]

            total_nan_batches += nan_batches
            final_epoch = epoch

            if epoch_obs > 0:
                final_loss = epoch_loss_sum / epoch_obs

            if epoch == 0 or (epoch + 1) % 10 == 0:
                print(
                    f"epoch={epoch:4d} "
                    f"used_batches={used_batches:3d} "
                    f"nan_batches={nan_batches:3d} "
                    f"total_nan_batches={total_nan_batches:3d} "
                    f"final_loss={final_loss} "
                    f"lr={optim.param_groups[0]['lr']:.2e}"
                )

            if total_nan_batches > 0:
                print("Stopping early because NaNs appeared.")
                break

        print("Mini stress test finished. Total NaN batches:", total_nan_batches)

        result["success"] = True
        result["total_nan_batches"] = int(total_nan_batches)
        result["first_failure"] = first_failure
        result["final_epoch"] = int(final_epoch) if final_epoch is not None else None
        result["final_loss"] = float(final_loss) if final_loss is not None else None

    except Exception as e:
        print("FAILED mini training on device", dev)
        print(type(e).__name__ + ":", str(e))
        result["error"] = type(e).__name__ + ": " + str(e)

    return result


mini_train_results = []

# Always test CPU.
mini_train_results.append(mini_train_stress_test(torch.device("cpu")))

# Also test MPS if available on Mac.
if mps_available():
    mini_train_results.append(mini_train_stress_test(torch.device("mps")))

# Also test CUDA if available.
if torch.cuda.is_available():
    mini_train_results.append(mini_train_stress_test(torch.device("cuda")))


# ============================================================
# 11. Final summary
# ============================================================

print_header("SUMMARY")

meta_for_hash = meta.copy()
meta_for_hash["as_of_date"] = pd.to_datetime(meta_for_hash["as_of_date"]).dt.strftime("%Y-%m-%d")

summary = {
    "platform": sys.platform,
    "platform_full": platform.platform(),
    "machine": platform.machine(),
    "processor": platform.processor(),
    "python": sys.version.split()[0],
    "torch": torch.__version__,
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "cuda_available": torch.cuda.is_available(),
    "mps_built": mps_built(),
    "mps_available": mps_available(),
    "device_main": str(device_main),
    "mkldnn_enabled": torch.backends.mkldnn.enabled if hasattr(torch.backends, "mkldnn") else None,
    "pytorch_enable_mps_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "<not set>"),
    "git_commit": git_info["git_commit"],
    "git_status": git_info["git_status"],
    "config_variant": getattr(config, "VARIANT", "could not read"),
    "scale_is_percent": bool(SCALE_IS_PERCENT),
    "x_shape": tuple(X_tensor.shape),
    "x_sha256": sha256_array(np.ascontiguousarray(X_tensor.detach().cpu().numpy())),
    "meta_sha256": sha256_text(meta_for_hash.to_csv(index=False)),
    "first_train_shape": tuple(X_train.shape),
    "first_train_sha256": sha256_array(np.ascontiguousarray(X_train.detach().cpu().numpy())),
    "smoke_batch_sha256": sha256_array(xb_cpu.numpy()),
    "smoke_results": smoke_results,
    "mini_train_results": mini_train_results,
}

print(json.dumps(summary, indent=2))

print("\nCompare these values between machines:")
print("x_sha256:", summary["x_sha256"])
print("meta_sha256:", summary["meta_sha256"])
print("first_train_sha256:", summary["first_train_sha256"])
print("smoke_batch_sha256:", summary["smoke_batch_sha256"])