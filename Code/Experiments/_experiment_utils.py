# =============================================================================
# Shared utilities for the representation experiment.
#
# Provides training, evaluation, and curve-type classification utilities
# used by Representation_Experiment.py and Overnight_Representation_Experiment.py.
#
# The representation experiment trains fresh models across varying shares of
# negative-rate curves in the training set (total size held fixed) to test
# whether poor reconstruction of negative-rate curves is caused by their
# scarcity in training or by a structural limitation of the encoder.
# =============================================================================

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Repo path setup so this works whether run as a script or imported
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(REPO_ROOT, ".."))
for p in (PROJECT_ROOT, REPO_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from Code.model.full_model import FullModel  # noqa: E402

# Stable variant imported inside train_baseline on demand to avoid
# requiring config.VARIANT to be set at module-import time.


# -----------------------------------------------------------------------------
# Curve-type classification
# -----------------------------------------------------------------------------
def compute_regime_flags(X_np: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Classify swap curves into mutually interpretable curve types.

    Parameters
    ----------
    X_np : (N, M) array of par swap rates in decimals, tenors ordered short to long.

    Returns
    -------
    Dict of boolean masks of length N:
      - any_negative   : at least one tenor is below zero
      - deeply_negative: at least 7 of 8 tenors are below zero
      - crossing       : has both negative and positive tenors
      - inverted       : short rate (1Y) strictly above long rate (30Y)
      - normal_positive: not inverted and not any_negative
    """
    short = X_np[:, 0]
    long_ = X_np[:, -1]

    any_negative = (X_np < 0).any(axis=1)
    n_neg_per_row = (X_np < 0).sum(axis=1)
    deeply_negative = n_neg_per_row >= 7
    crossing = any_negative & ~deeply_negative
    inverted = short > long_
    normal_positive = (~inverted) & (~any_negative)

    return {
        "any_negative": any_negative,
        "deeply_negative": deeply_negative,
        "crossing": crossing,
        "inverted": inverted,
        "normal_positive": normal_positive,
    }


# -----------------------------------------------------------------------------
# Training configuration and core training loop
# -----------------------------------------------------------------------------
@dataclass
class TrainConfig:
    """Training hyperparameters for a single run in the representation experiment."""
    latent_dim: int = 2
    epochs: int = 2000
    batch_size: int = 32
    fixed_lr: float = 1e-2
    seed: int = 0
    log_every: int = 50          # print full loss line every this many epochs
    heartbeat_every: int = 10    # print short progress tick every this many epochs
    target_mse: float = 1e-8    # stop early if training MSE drops below this
    model_type: str = "baseline"  # "baseline" or "stable"


def _row_finite_mask(t: torch.Tensor) -> torch.Tensor:
    return torch.isfinite(t).all(dim=1)


def train_baseline(
    X_train: torch.Tensor,
    cfg: TrainConfig,
    device: Optional[torch.device] = None,
    tag: str = "",
) -> Tuple[nn.Module, list]:
    """
    Train a fresh model on X_train with a fixed learning rate.

    Supports both the baseline (FullModel) and stable (FullModelStable)
    architectures via cfg.model_type. Includes gradient clipping and NaN
    guards so it can be called repeatedly inside the experiment sweep loop.

    Returns (model, train_loss_history).
    """
    if device is None:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Drop any non-finite curves defensively
    mask = _row_finite_mask(X_train)
    X_train = X_train[mask].float()
    if X_train.shape[0] == 0:
        raise ValueError("X_train is empty after dropping non-finite rows.")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    if cfg.model_type == "stable":
        from Code import config as _cfg_mod
        from Code.model.full_model_stable import FullModel as FullModelStable
        _old_variant = _cfg_mod.VARIANT
        _cfg_mod.VARIANT = "stable"
        model = FullModelStable(latent_dim=cfg.latent_dim).to(device)
        _cfg_mod.VARIANT = _old_variant
    else:
        model = FullModel(latent_dim=cfg.latent_dim).to(device)
    model.train()

    optim = torch.optim.Adam(model.parameters(), lr=cfg.fixed_lr)
    loss_fn = nn.MSELoss()

    loader = DataLoader(
        TensorDataset(X_train),
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
    )
    n_batches = len(loader)

    print(
        f"  [{tag}] starting training: N={X_train.shape[0]}, "
        f"batches/epoch={n_batches}, epochs={cfg.epochs}, "
        f"latent_dim={cfg.latent_dim}, lr={cfg.fixed_lr}, device={device}",
        flush=True,
    )

    history = []
    t0 = time.perf_counter()
    for epoch in range(cfg.epochs):
        model.train()
        running, n_obs = 0.0, 0
        for b_idx, (xb_cpu,) in enumerate(loader):
            xb = xb_cpu.to(device)
            optim.zero_grad(set_to_none=True)
            try:
                S_hat = model(xb)
            except Exception:
                continue
            if not torch.isfinite(S_hat).all():
                continue
            loss = loss_fn(S_hat, xb)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            bad_grad = False
            for p in model.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    bad_grad = True
                    break
            if bad_grad:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            running += float(loss.detach().cpu()) * xb.shape[0]
            n_obs += xb.shape[0]

            # First-epoch heartbeat so the user sees something within seconds
            if epoch == 0 and (b_idx == 0 or (b_idx + 1) % max(1, n_batches // 4) == 0):
                print(
                    f"  [{tag}] epoch 0 batch {b_idx+1}/{n_batches} "
                    f"loss={float(loss.detach().cpu()):.3e}",
                    flush=True,
                )

        epoch_mse = running / max(n_obs, 1)
        history.append(epoch_mse)

        elapsed = time.perf_counter() - t0
        do_full_log = (
            (epoch + 1) % cfg.log_every == 0
            or epoch == 0
            or epoch == cfg.epochs - 1
        )
        do_heartbeat = (
            cfg.heartbeat_every > 0
            and (epoch + 1) % cfg.heartbeat_every == 0
            and not do_full_log
        )

        if do_full_log:
            # Rough ETA based on average epoch time so far
            avg_per_epoch = elapsed / (epoch + 1)
            eta = avg_per_epoch * (cfg.epochs - epoch - 1)
            print(
                f"  [{tag}] epoch {epoch+1:4d}/{cfg.epochs} "
                f"train_rmse={epoch_mse**0.5:.3e}  "
                f"elapsed={elapsed/60:.1f}min  ETA={eta/60:.1f}min",
                flush=True,
            )
        elif do_heartbeat:
            print(
                f"  [{tag}] .. epoch {epoch+1}/{cfg.epochs} "
                f"rmse={epoch_mse**0.5:.2e}",
                flush=True,
            )

        if cfg.target_mse > 0 and epoch_mse <= cfg.target_mse and n_obs > 0:
            print(f"  [{tag}] target MSE reached at epoch {epoch}; stopping early.", flush=True)
            break

    print(f"  [{tag}] training done in {(time.perf_counter()-t0)/60:.1f} min "
          f"(final train_rmse={history[-1]**0.5:.3e})", flush=True)
    return model, history


# -----------------------------------------------------------------------------
# Retry wrapper — guards against stuck initialisations in the stable model
# -----------------------------------------------------------------------------

def train_with_retry(
    X_train: torch.Tensor,
    cfg: TrainConfig,
    device: Optional[torch.device] = None,
    tag: str = "",
    stuck_threshold_bps: float = 25.0,
    max_retries: int = 2,
) -> Tuple[nn.Module, list]:
    """
    Run train_baseline up to (1 + max_retries) times and return the best result.

    The stable model occasionally converges to a poor local minimum for certain
    random initialisations. If training RMSE exceeds stuck_threshold_bps, the
    run is retried with an incremented seed. Returns the run with the lowest
    training RMSE.

    stuck_threshold_bps : RMSE threshold above which a retry is triggered (default 25 bps).
    max_retries         : maximum number of additional attempts (default 2).
    """
    if device is None:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    best_model, best_history, best_rmse = None, None, float("inf")

    for attempt in range(max_retries + 1):
        attempt_seed = cfg.seed + attempt

        if attempt > 0:
            print(
                f"  [{tag}] RETRY {attempt}/{max_retries} — seed={attempt_seed} "
                f"(prev train RMSE={best_rmse:.1f} bps > threshold={stuck_threshold_bps:.0f} bps)",
                flush=True,
            )

        # Build a config with the incremented seed for this attempt
        attempt_cfg = TrainConfig(
            latent_dim=cfg.latent_dim,
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            fixed_lr=cfg.fixed_lr,
            seed=attempt_seed,
            log_every=cfg.log_every,
            heartbeat_every=cfg.heartbeat_every,
            target_mse=cfg.target_mse,
            model_type=cfg.model_type,
        )

        model, history = train_baseline(X_train, attempt_cfg, device=device, tag=tag)

        # Evaluate on training data to detect stuck runs
        train_rmse = rmse_bps_overall(model, X_train, device)

        if train_rmse < best_rmse:
            best_model, best_history, best_rmse = model, history, train_rmse

        if best_rmse <= stuck_threshold_bps:
            if attempt > 0:
                print(f"  [{tag}] retry succeeded — train RMSE={best_rmse:.1f} bps", flush=True)
            break
    else:
        print(
            f"  [{tag}] WARNING: all {max_retries + 1} attempts above threshold "
            f"({stuck_threshold_bps:.0f} bps). Best train RMSE={best_rmse:.1f} bps.",
            flush=True,
        )

    return best_model, best_history


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------
@torch.no_grad()
def predict_S_hat(model: nn.Module, X: torch.Tensor, device: torch.device, batch_size: int = 256) -> torch.Tensor:
    """Run batched inference and return reconstructed swap curves on CPU."""
    model.eval()
    outs = []
    for i in range(0, X.shape[0], batch_size):
        xb = X[i:i + batch_size].to(device)
        outs.append(model(xb).detach().cpu())
    return torch.cat(outs, dim=0) if outs else torch.zeros((0, X.shape[1]))


def rmse_bps_overall(model: nn.Module, X: torch.Tensor, device: torch.device) -> float:
    """Single aggregate RMSE in bps over all rows of X (decimals -> bps)."""
    if X.shape[0] == 0:
        return float("nan")
    S_hat = predict_S_hat(model, X, device)
    mask = _row_finite_mask(X) & _row_finite_mask(S_hat)
    if mask.sum() == 0:
        return float("nan")
    err = (S_hat[mask] - X[mask]).numpy()
    return float(np.sqrt(np.mean(err ** 2)) * 1e4)


def rmse_bps_per_curve(model: nn.Module, X: torch.Tensor, device: torch.device) -> np.ndarray:
    """Per-curve RMSE in bps. Returns array of shape (N,) with NaN for non-finite rows."""
    if X.shape[0] == 0:
        return np.zeros((0,))
    S_hat = predict_S_hat(model, X, device)
    err = (S_hat - X).numpy()
    out = np.sqrt(np.mean(err ** 2, axis=1)) * 1e4
    mask = (~np.isfinite(X.numpy()).all(axis=1)) | (~np.isfinite(S_hat.numpy()).all(axis=1))
    out[mask] = np.nan
    return out

