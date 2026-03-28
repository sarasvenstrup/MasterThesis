import os
import sys
import time
import json
import math
import random
import traceback
import importlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
torch.set_num_threads(4)
torch.set_num_interop_threads(2)

import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import TensorDataset, DataLoader


# =============================================================================
# Environment setup
# =============================================================================
try:
    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    REPO_ROOT = os.getcwd()

PROJECT_ROOT = os.path.abspath(os.path.join(REPO_ROOT, ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Code import config
from Code.utils import helpers as H
from Code.load_swapdata import my_data
import Code.model.full_model as full_model_module


# =============================================================================
# Settings
# =============================================================================
SHOW_PLOTS = False

USE = "bbg"
LATENT_DIM = 2

EPOCHS = 1000
BATCH_SIZE = 32
EVAL_BATCH_SIZE = 256

MAX_LR = 1e-3
GRAD_CLIP = 1.0
WEIGHT_DECAY = 0.0

SEEDS = [0, 1, 2, 3, 4]

# IMPORTANT:
# Adjust these if your config.VARIANT strings are different.
VARIANT_MAP = {
    "baseline": "baseline",
    "stable": "stable",
}

# Chronological split
TRAIN_END = "2018-12-31"
VAL_END = "2020-12-31"

# If auto-detection fails, set DATE_COL manually.
DATE_COL = None

# Logging
PRINT_EVERY = 10

# Output folders
OUTPUT_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "Results",
    f"compare_baseline_stable_use_{USE}_dim{LATENT_DIM}_ep{EPOCHS}"
)
os.makedirs(OUTPUT_ROOT, exist_ok=True)

PLOTS_DIR = os.path.join(OUTPUT_ROOT, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

CHECKPOINTS_DIR = os.path.join(OUTPUT_ROOT, "checkpoints")
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

torch.backends.mkldnn.enabled = True
USE_SET_TO_NONE = True

device = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    else torch.device("cpu")
)

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("MPS available:", hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
print("Using device:", device)
print("CPU threads:", torch.get_num_threads(), "interop:", torch.get_num_interop_threads())


# =============================================================================
# Utilities
# =============================================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def row_finite_mask(t: torch.Tensor) -> torch.Tensor:
    return torch.isfinite(t).all(dim=1)


def detect_date_col(meta: pd.DataFrame) -> str:
    if DATE_COL is not None:
        if DATE_COL not in meta.columns:
            raise KeyError(f"DATE_COL='{DATE_COL}' not found in meta columns: {list(meta.columns)}")
        return DATE_COL

    candidates = [
        "date", "Date", "DATE",
        "asof", "AsOf", "as_of", "trade_date",
        "timestamp", "Timestamp", "obs_date",
    ]
    for col in candidates:
        if col in meta.columns:
            return col

    # Fallback: first column that can parse to datetime with decent success
    for col in meta.columns:
        parsed = pd.to_datetime(meta[col], errors="coerce")
        if parsed.notna().mean() > 0.90:
            return col

    raise ValueError(
        "Could not auto-detect date column in meta. "
        f"Available columns: {list(meta.columns)}. Set DATE_COL manually."
    )


def mask_tensor(X: torch.Tensor, mask: pd.Series) -> torch.Tensor:
    idx = torch.from_numpy(mask.to_numpy()).bool()
    return X[idx]


def make_split(meta: pd.DataFrame, X: torch.Tensor):
    date_col = detect_date_col(meta)
    meta = meta.copy()
    meta["_date"] = pd.to_datetime(meta[date_col], errors="coerce")

    if meta["_date"].isna().any():
        n_bad = int(meta["_date"].isna().sum())
        raise ValueError(f"Found {n_bad} rows with invalid dates in column '{date_col}'.")

    train_mask = meta["_date"] <= pd.Timestamp(TRAIN_END)
    val_mask = (meta["_date"] > pd.Timestamp(TRAIN_END)) & (meta["_date"] <= pd.Timestamp(VAL_END))
    test_mask = meta["_date"] > pd.Timestamp(VAL_END)

    if train_mask.sum() == 0 or val_mask.sum() == 0 or test_mask.sum() == 0:
        raise ValueError(
            "One of train/val/test splits is empty. "
            f"train={int(train_mask.sum())}, val={int(val_mask.sum())}, test={int(test_mask.sum())}"
        )

    split = {
        "train_meta": meta.loc[train_mask].reset_index(drop=True),
        "val_meta": meta.loc[val_mask].reset_index(drop=True),
        "test_meta": meta.loc[test_mask].reset_index(drop=True),
        "train_X": mask_tensor(X, train_mask),
        "val_X": mask_tensor(X, val_mask),
        "test_X": mask_tensor(X, test_mask),
        "date_col": date_col,
    }
    return split


def make_loader(X: torch.Tensor, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    ds = TensorDataset(X)
    gen = torch.Generator()
    gen.manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        generator=gen,
    )


def build_model(variant_key: str) -> nn.Module:
    if variant_key not in VARIANT_MAP:
        raise KeyError(f"Unknown variant_key='{variant_key}'. Available: {list(VARIANT_MAP.keys())}")

    config.VARIANT = VARIANT_MAP[variant_key]

    # Reload to be safe in case FullModel selects submodules at import time.
    importlib.reload(full_model_module)

    model = full_model_module.FullModel(latent_dim=LATENT_DIM).to(device)
    return model


@torch.no_grad()
def predict_S_hat(model: nn.Module, X: torch.Tensor, batch_size: int = 256) -> torch.Tensor:
    was_training = model.training
    model.eval()

    outs = []
    N = X.shape[0]
    for i in range(0, N, batch_size):
        xb = X[i:i + batch_size].to(device)
        try:
            S_hat = model(xb)
            outs.append(S_hat.detach().cpu())
        except Exception:
            # ODE / assertion failure during eval – fill NaN so row is dropped downstream
            outs.append(torch.full((xb.shape[0], X.shape[1]), float("nan")))

    if was_training:
        model.train()

    return torch.cat(outs, dim=0)


def eval_rmse_bps(model: nn.Module, X_full: torch.Tensor, meta_full: pd.DataFrame, batch_size: int = 256):
    """
    Returns:
      rmse_per_ccy (pd.Series)
      avg_rmse_bps (float)
      n_bad (int)
      n_good (int)
    """
    S_hat_all = predict_S_hat(model, X_full, batch_size=batch_size)

    mask = row_finite_mask(X_full) & row_finite_mask(S_hat_all)
    n_bad = int((~mask).sum().item())
    n_good = int(mask.sum().item())

    X_eval = X_full[mask]
    S_eval = S_hat_all[mask]
    meta_eval = meta_full.loc[mask.numpy()].reset_index(drop=True)

    rmse_per_ccy = H.rmse_bps_per_currency_paper(X_eval, S_eval, meta_eval)
    avg_rmse_bps = (
        float(rmse_per_ccy.drop("Average", errors="ignore").mean())
        if len(rmse_per_ccy) > 0 else np.nan
    )
    return rmse_per_ccy, avg_rmse_bps, n_bad, n_good


def save_json(path: str, obj: dict) -> None:
    def convert(v):
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, (np.bool_,)):
            return bool(v)
        return v

    clean = {k: convert(v) for k, v in obj.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2)


def fmt_mean_std(series: pd.Series, digits: int = 3) -> str:
    series = pd.to_numeric(series, errors="coerce").dropna()
    if len(series) == 0:
        return ""
    mean = series.mean()
    std = series.std(ddof=1) if len(series) > 1 else 0.0
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def write_latex_table(df: pd.DataFrame, path: str, index: bool = False) -> None:
    latex = df.to_latex(index=index, escape=False)
    with open(path, "w", encoding="utf-8") as f:
        f.write(latex)


# =============================================================================
# Training
# =============================================================================
def train_one_run(
    variant_key: str,
    seed: int,
    split: dict,
) -> tuple[dict, pd.DataFrame]:
    set_seed(seed)

    run_name = f"{variant_key}_seed_{seed}"
    run_dir = os.path.join(OUTPUT_ROOT, run_name)
    os.makedirs(run_dir, exist_ok=True)

    history_csv = os.path.join(run_dir, "history.csv")
    metrics_json = os.path.join(run_dir, "metrics.json")
    best_ckpt = os.path.join(CHECKPOINTS_DIR, f"{run_name}_best.pt")
    last_ckpt = os.path.join(CHECKPOINTS_DIR, f"{run_name}_last.pt")
    per_currency_csv = os.path.join(run_dir, "test_rmse_by_currency.csv")

    train_loader = make_loader(split["train_X"], BATCH_SIZE, shuffle=True, seed=seed)

    model = build_model(variant_key)
    model.train()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=MAX_LR,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = OneCycleLR(
        optimizer,
        max_lr=MAX_LR,
        steps_per_epoch=len(train_loader),
        epochs=EPOCHS,
        pct_start=0.3,
        div_factor=1.0,
        final_div_factor=3000.0,
    )

    loss_fn = nn.MSELoss()

    history_rows = []
    total_nan_batches = 0
    total_failed_batches = 0
    total_ode_failures = 0

    best_val_rmse = np.inf
    best_epoch = -1
    time_to_best_epoch_sec = np.nan

    t0 = time.perf_counter()
    completed = True
    fatal_error = ""

    try:
        for epoch in range(EPOCHS):
            epoch_start = time.perf_counter()

            running = 0.0
            n_obs = 0
            nan_batches = 0
            failed_batches = 0
            ode_failures = 0
            grad_norms = []

            model.train()

            for (xb_cpu,) in train_loader:
                xb = xb_cpu.to(device)

                optimizer.zero_grad(set_to_none=USE_SET_TO_NONE)

                try:
                    S_hat = model(xb)
                    loss = loss_fn(S_hat, xb)
                except Exception as e:
                    failed_batches += 1
                    msg = str(e).lower()
                    if any(tok in msg for tok in ["ode", "integrat", "dopri", "rk", "solver"]):
                        ode_failures += 1
                    continue

                if not torch.isfinite(loss):
                    nan_batches += 1
                    continue

                try:
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)

                    if not torch.isfinite(grad_norm):
                        failed_batches += 1
                        optimizer.zero_grad(set_to_none=USE_SET_TO_NONE)
                        continue

                    optimizer.step()
                    scheduler.step()

                    running += float(loss.detach().cpu()) * xb.shape[0]
                    n_obs += xb.shape[0]
                    grad_norms.append(float(grad_norm.detach().cpu()) if torch.is_tensor(grad_norm) else float(grad_norm))

                except Exception as e:
                    failed_batches += 1
                    msg = str(e).lower()
                    if any(tok in msg for tok in ["ode", "integrat", "dopri", "rk", "solver"]):
                        ode_failures += 1
                    optimizer.zero_grad(set_to_none=USE_SET_TO_NONE)
                    continue

            total_nan_batches += nan_batches
            total_failed_batches += failed_batches
            total_ode_failures += ode_failures

            epoch_mse = running / max(n_obs, 1)
            epoch_rmse = math.sqrt(epoch_mse) if epoch_mse >= 0 else np.nan
            epoch_time_sec = time.perf_counter() - epoch_start
            cumulative_time_sec = time.perf_counter() - t0
            mean_grad_norm = float(np.mean(grad_norms)) if len(grad_norms) else np.nan

            # Validation every epoch
            val_rmse_per_ccy, val_avg_rmse_bps, val_n_bad, val_n_good = eval_rmse_bps(
                model,
                split["val_X"],
                split["val_meta"],
                batch_size=EVAL_BATCH_SIZE,
            )

            improved = np.isfinite(val_avg_rmse_bps) and (val_avg_rmse_bps < best_val_rmse)
            if improved:
                best_val_rmse = float(val_avg_rmse_bps)
                best_epoch = epoch
                time_to_best_epoch_sec = cumulative_time_sec

                torch.save({
                    "model_state_dict": model.state_dict(),
                    "variant": variant_key,
                    "seed": seed,
                    "epoch": epoch,
                    "best_val_rmse_bps": best_val_rmse,
                    "latent_dim": LATENT_DIM,
                    "use_data": USE,
                    "config_variant_value": config.VARIANT,
                }, best_ckpt)

            row = {
                "variant": variant_key,
                "seed": seed,
                "epoch": epoch,
                "train_mse": epoch_mse,
                "train_rmse": epoch_rmse,
                "val_avg_rmse_bps": float(val_avg_rmse_bps),
                "val_n_good": int(val_n_good),
                "val_n_bad": int(val_n_bad),
                "epoch_time_sec": epoch_time_sec,
                "cumulative_time_sec": cumulative_time_sec,
                "mean_grad_norm": mean_grad_norm,
                "nan_batches": nan_batches,
                "failed_batches": failed_batches,
                "ode_failures": ode_failures,
                "lr": optimizer.param_groups[0]["lr"],
                "best_val_so_far_bps": float(best_val_rmse) if np.isfinite(best_val_rmse) else np.nan,
            }
            history_rows.append(row)

            if (epoch == 0) or ((epoch + 1) % PRINT_EVERY == 0) or (epoch == EPOCHS - 1):
                print(
                    f"[{variant_key:8s} seed={seed}] "
                    f"epoch={epoch:4d} "
                    f"train_rmse={epoch_rmse:.6e} "
                    f"val_avg_rmse_bps={val_avg_rmse_bps:.3f} "
                    f"best_val={best_val_rmse:.3f} "
                    f"lr={optimizer.param_groups[0]['lr']:.2e} "
                    f"nan_batches={nan_batches} "
                    f"failed_batches={failed_batches} "
                    f"ode_failures={ode_failures} "
                    f"time_total={cumulative_time_sec/60:.1f}min"
                )

        torch.save({
            "model_state_dict": model.state_dict(),
            "variant": variant_key,
            "seed": seed,
            "epoch": EPOCHS - 1,
            "latent_dim": LATENT_DIM,
            "use_data": USE,
            "config_variant_value": config.VARIANT,
        }, last_ckpt)

    except Exception as e:
        completed = False
        fatal_error = "".join(traceback.format_exception_only(type(e), e)).strip()
        print(f"[FATAL] {run_name}: {fatal_error}")

        # save partial state if possible
        try:
            torch.save({
                "model_state_dict": model.state_dict(),
                "variant": variant_key,
                "seed": seed,
                "epoch": len(history_rows) - 1,
                "latent_dim": LATENT_DIM,
                "use_data": USE,
                "config_variant_value": config.VARIANT,
            }, last_ckpt)
        except Exception:
            pass

    # Save history even if failed
    history_df = pd.DataFrame(history_rows)
    history_df.to_csv(history_csv, index=False)

    # Use best checkpoint if it exists, otherwise current model state
    if os.path.exists(best_ckpt):
        ckpt = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

    train_rmse_per_ccy, train_avg_rmse_bps, train_n_bad, train_n_good = eval_rmse_bps(
        model, split["train_X"], split["train_meta"], batch_size=EVAL_BATCH_SIZE
    )
    val_rmse_per_ccy, val_avg_rmse_bps, val_n_bad, val_n_good = eval_rmse_bps(
        model, split["val_X"], split["val_meta"], batch_size=EVAL_BATCH_SIZE
    )
    test_rmse_per_ccy, test_avg_rmse_bps, test_n_bad, test_n_good = eval_rmse_bps(
        model, split["test_X"], split["test_meta"], batch_size=EVAL_BATCH_SIZE
    )

    total_train_time_sec = time.perf_counter() - t0
    mean_epoch_time_sec = (
        float(history_df["epoch_time_sec"].mean()) if len(history_df) else np.nan
    )

    metrics = {
        "variant": variant_key,
        "seed": seed,
        "completed": completed,
        "fatal_error": fatal_error,
        "best_epoch": int(best_epoch) if best_epoch >= 0 else -1,
        "train_avg_rmse_bps": float(train_avg_rmse_bps),
        "val_avg_rmse_bps": float(val_avg_rmse_bps),
        "test_avg_rmse_bps": float(test_avg_rmse_bps),
        "train_n_good": int(train_n_good),
        "train_n_bad": int(train_n_bad),
        "val_n_good": int(val_n_good),
        "val_n_bad": int(val_n_bad),
        "test_n_good": int(test_n_good),
        "test_n_bad": int(test_n_bad),
        "total_train_time_sec": float(total_train_time_sec),
        "mean_epoch_time_sec": float(mean_epoch_time_sec),
        "time_to_best_epoch_sec": float(time_to_best_epoch_sec) if np.isfinite(time_to_best_epoch_sec) else np.nan,
        "total_nan_batches": int(total_nan_batches),
        "total_failed_batches": int(total_failed_batches),
        "total_ode_failures": int(total_ode_failures),
        "history_csv": history_csv,
        "best_ckpt": best_ckpt if os.path.exists(best_ckpt) else "",
        "last_ckpt": last_ckpt if os.path.exists(last_ckpt) else "",
    }

    save_json(metrics_json, metrics)

    # Save per-currency test RMSE  (drop the "Average" sentinel row)
    test_ccy_df = (
        test_rmse_per_ccy
        .drop("Average", errors="ignore")
        .rename("rmse_bps")
        .reset_index()
    )
    test_ccy_df.columns = ["currency", "rmse_bps"]
    test_ccy_df["variant"] = variant_key
    test_ccy_df["seed"] = seed
    test_ccy_df = test_ccy_df[["variant", "seed", "currency", "rmse_bps"]]
    test_ccy_df.to_csv(per_currency_csv, index=False)

    return metrics, test_ccy_df


# =============================================================================
# Aggregation tables
# =============================================================================
def make_summary_tables(summary_runs: pd.DataFrame, summary_by_currency: pd.DataFrame) -> None:
    # Overall summary
    overall = pd.DataFrame({
        "Model": sorted(summary_runs["variant"].unique()),
    })

    rows = []
    for variant, g in summary_runs.groupby("variant", sort=True):
        rows.append({
            "Model": variant,
            "Train RMSE (bps)": fmt_mean_std(g["train_avg_rmse_bps"]),
            "Val RMSE (bps)": fmt_mean_std(g["val_avg_rmse_bps"]),
            "Test RMSE (bps)": fmt_mean_std(g["test_avg_rmse_bps"]),
            "Total time (min)": fmt_mean_std(g["total_train_time_sec"] / 60.0, digits=2),
            "Time to best epoch (min)": fmt_mean_std(g["time_to_best_epoch_sec"] / 60.0, digits=2),
            "Mean epoch time (s)": fmt_mean_std(g["mean_epoch_time_sec"], digits=2),
            "Completed runs": f"{int(g['completed'].sum())}/{len(g)}",
            "NaN batches": f"{int(g['total_nan_batches'].sum())}",
            "Failed batches": f"{int(g['total_failed_batches'].sum())}",
            "ODE failures": f"{int(g['total_ode_failures'].sum())}",
        })
    overall_table = pd.DataFrame(rows)
    overall_csv = os.path.join(OUTPUT_ROOT, "table_overall.csv")
    overall_tex = os.path.join(OUTPUT_ROOT, "table_overall.tex")
    overall_table.to_csv(overall_csv, index=False)
    write_latex_table(overall_table, overall_tex, index=False)

    # Per-currency summary
    pivot_rows = []
    currency_order = sorted(summary_by_currency["currency"].unique())
    for variant, g in summary_by_currency.groupby("variant", sort=True):
        row = {"Model": variant}
        for ccy in currency_order:
            s = g.loc[g["currency"] == ccy, "rmse_bps"]
            row[ccy] = fmt_mean_std(s)
        row["Average"] = fmt_mean_std(g.groupby("seed")["rmse_bps"].mean())
        pivot_rows.append(row)

    currency_table = pd.DataFrame(pivot_rows)
    currency_csv = os.path.join(OUTPUT_ROOT, "table_currency.csv")
    currency_tex = os.path.join(OUTPUT_ROOT, "table_currency.tex")
    currency_table.to_csv(currency_csv, index=False)
    write_latex_table(currency_table, currency_tex, index=False)

    # Raw summaries
    summary_runs.to_csv(os.path.join(OUTPUT_ROOT, "summary_runs.csv"), index=False)
    summary_by_currency.to_csv(os.path.join(OUTPUT_ROOT, "summary_by_currency.csv"), index=False)


# =============================================================================
# Plots
# =============================================================================
def plot_val_curves(summary_runs: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5), dpi=160)

    for variant in sorted(summary_runs["variant"].unique()):
        curves = []
        for _, row in summary_runs.loc[summary_runs["variant"] == variant].iterrows():
            hist = pd.read_csv(row["history_csv"])
            hist = hist.sort_values("epoch")
            y = hist["val_avg_rmse_bps"].to_numpy()
            if len(y) < EPOCHS:
                pad = np.full(EPOCHS - len(y), np.nan)
                y = np.concatenate([y, pad])
            curves.append(y[:EPOCHS])

        arr = np.vstack(curves)
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0)
        epochs = np.arange(len(mean))

        plt.plot(epochs, mean, label=variant)
        plt.fill_between(epochs, mean - std, mean + std, alpha=0.20)

    plt.xlabel("Epoch")
    plt.ylabel("Validation average RMSE (bps)")
    plt.title("Validation RMSE convergence")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "val_rmse_convergence.png")
    plt.savefig(path, dpi=300)
    print("Saved:", path)
    if SHOW_PLOTS:
        plt.show()
    plt.close()


def plot_test_rmse_boxplot(summary_runs: pd.DataFrame) -> None:
    variants = sorted(summary_runs["variant"].unique())
    data = [
        summary_runs.loc[summary_runs["variant"] == v, "test_avg_rmse_bps"].dropna().to_numpy()
        for v in variants
    ]

    plt.figure(figsize=(6.5, 4.5), dpi=160)
    plt.boxplot(data, labels=variants)
    plt.ylabel("Test average RMSE (bps)")
    plt.title("Final test RMSE across seeds")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "test_rmse_boxplot.png")
    plt.savefig(path, dpi=300)
    print("Saved:", path)
    if SHOW_PLOTS:
        plt.show()
    plt.close()


def plot_runtime_boxplot(summary_runs: pd.DataFrame) -> None:
    variants = sorted(summary_runs["variant"].unique())
    data = [
        (summary_runs.loc[summary_runs["variant"] == v, "total_train_time_sec"].dropna() / 60.0).to_numpy()
        for v in variants
    ]

    plt.figure(figsize=(6.5, 4.5), dpi=160)
    plt.boxplot(data, labels=variants)
    plt.ylabel("Total training time (minutes)")
    plt.title("Training time across seeds")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "runtime_boxplot.png")
    plt.savefig(path, dpi=300)
    print("Saved:", path)
    if SHOW_PLOTS:
        plt.show()
    plt.close()


def plot_time_to_best(summary_runs: pd.DataFrame) -> None:
    variants = sorted(summary_runs["variant"].unique())
    data = [
        (summary_runs.loc[summary_runs["variant"] == v, "time_to_best_epoch_sec"].dropna() / 60.0).to_numpy()
        for v in variants
    ]

    plt.figure(figsize=(6.5, 4.5), dpi=160)
    plt.boxplot(data, labels=variants)
    plt.ylabel("Time to best validation epoch (minutes)")
    plt.title("Time to best checkpoint")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "time_to_best_epoch_boxplot.png")
    plt.savefig(path, dpi=300)
    print("Saved:", path)
    if SHOW_PLOTS:
        plt.show()
    plt.close()


def plot_per_currency(summary_by_currency: pd.DataFrame) -> None:
    agg = (
        summary_by_currency
        .groupby(["variant", "currency"])["rmse_bps"]
        .agg(["mean", "std"])
        .reset_index()
    )

    variants = sorted(agg["variant"].unique())
    currencies = sorted(agg["currency"].unique())

    x = np.arange(len(currencies))
    width = 0.35 if len(variants) == 2 else 0.8 / max(len(variants), 1)

    plt.figure(figsize=(10, 5), dpi=160)

    for i, variant in enumerate(variants):
        sub = agg.loc[agg["variant"] == variant].set_index("currency").reindex(currencies)
        means = sub["mean"].to_numpy()
        stds = sub["std"].fillna(0.0).to_numpy()
        xpos = x + (i - (len(variants) - 1) / 2.0) * width
        plt.bar(xpos, means, width=width, yerr=stds, capsize=3, label=variant)

    plt.xticks(x, currencies)
    plt.ylabel("Test RMSE (bps)")
    plt.title("Per-currency test RMSE")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "per_currency_rmse.png")
    plt.savefig(path, dpi=300)
    print("Saved:", path)
    if SHOW_PLOTS:
        plt.show()
    plt.close()


def plot_stability_counts(summary_runs: pd.DataFrame) -> None:
    agg = (
        summary_runs
        .groupby("variant")[["total_nan_batches", "total_failed_batches", "total_ode_failures"]]
        .sum()
        .reset_index()
    )

    variants = agg["variant"].tolist()
    x = np.arange(len(variants))
    width = 0.25

    plt.figure(figsize=(8, 5), dpi=160)
    plt.bar(x - width, agg["total_nan_batches"], width=width, label="NaN batches")
    plt.bar(x, agg["total_failed_batches"], width=width, label="Failed batches")
    plt.bar(x + width, agg["total_ode_failures"], width=width, label="ODE failures")

    plt.xticks(x, variants)
    plt.ylabel("Count across all seeds")
    plt.title("Stability diagnostics")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "stability_counts.png")
    plt.savefig(path, dpi=300)
    print("Saved:", path)
    if SHOW_PLOTS:
        plt.show()
    plt.close()


# =============================================================================
# Main
# =============================================================================
def main():
    print("\nLoading data...")
    meta, X_tensor, meta_full, X_tensor_full, tenors, df_wide, df_wide_all, SCALE_IS_PERCENT = my_data(use=USE)
    X_tensor = X_tensor.float()

    split = make_split(meta, X_tensor)

    split_info = pd.DataFrame({
        "split": ["train", "val", "test"],
        "n_rows": [
            split["train_X"].shape[0],
            split["val_X"].shape[0],
            split["test_X"].shape[0],
        ],
        "date_min": [
            split["train_meta"]["_date"].min(),
            split["val_meta"]["_date"].min(),
            split["test_meta"]["_date"].min(),
        ],
        "date_max": [
            split["train_meta"]["_date"].max(),
            split["val_meta"]["_date"].max(),
            split["test_meta"]["_date"].max(),
        ],
    })
    split_info.to_csv(os.path.join(OUTPUT_ROOT, "split_info.csv"), index=False)
    print(split_info)

    summary_runs_records = []
    summary_by_currency_frames = []

    for variant_key in VARIANT_MAP.keys():
        for seed in SEEDS:
            print("\n" + "=" * 100)
            print(f"RUN: variant={variant_key} seed={seed}")
            print("=" * 100)

            metrics, per_currency_df = train_one_run(
                variant_key=variant_key,
                seed=seed,
                split=split,
            )

            summary_runs_records.append(metrics)
            summary_by_currency_frames.append(per_currency_df)

    summary_runs = pd.DataFrame(summary_runs_records)
    summary_by_currency = pd.concat(summary_by_currency_frames, ignore_index=True)

    make_summary_tables(summary_runs, summary_by_currency)

    plot_val_curves(summary_runs)
    plot_test_rmse_boxplot(summary_runs)
    plot_per_currency(summary_by_currency)
    plot_runtime_boxplot(summary_runs)
    plot_time_to_best(summary_runs)
    plot_stability_counts(summary_runs)

    print("\nDone.")
    print("Output root:", OUTPUT_ROOT)


if __name__ == "__main__":
    main()