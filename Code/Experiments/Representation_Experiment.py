# =============================================================================
# Experiment 1 — Prevalence sweep
#
# Question: is the baseline's poor fit of negative-rate (or inverted) curves
# driven by their SCARCITY in the training data, or by a STRUCTURAL inability
# of the linear encoder to represent them?
#
# Design:
#   - Pick a target regime ("any_negative" or "inverted").
#   - Set aside a fixed held-out evaluation set of regime curves (eval_frac of
#     all regime curves, sampled with a fixed seed). These curves NEVER enter
#     any training set, so RMSE numbers across the sweep are comparable.
#   - For each target prevalence p in PREVALENCES:
#       * Build a training set of fixed total size N_TRAIN, containing
#         round(p * N_TRAIN) regime curves (sampled from the regime pool minus
#         the held-out set, with replacement if needed) and the rest from the
#         non-regime pool.
#       * Train a fresh baseline FullModel.
#       * Evaluate average RMSE (bps) on the held-out regime set AND on a
#         held-out non-regime set (sanity control).
#   - Save a CSV + a quick PNG of RMSE_regime vs p.
#
# Interpretation:
#   - Monotone, plateauing decrease in RMSE_regime as p grows -> scarcity is
#     the dominant cause.
#   - Flat / weakly decreasing RMSE_regime even at p=0.5 or 0.75 -> structural
#     limitation; more regime data does not help.
# =============================================================================

import argparse
import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# Force unbuffered stdout so progress is visible in real time even when
# stdout is piped / redirected (e.g. from a JetBrains run config).
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

# Path setup
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(REPO_ROOT, ".."))
for p in (PROJECT_ROOT, REPO_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from Code.load_swapdata import my_data  # noqa: E402
from Code.Experiments._experiment_utils import (  # noqa: E402
    TrainConfig,
    compute_regime_flags,
    train_with_retry,
    rmse_bps_overall,
)


def parse_args():
    ap = argparse.ArgumentParser(description="Prevalence sweep experiment")
    ap.add_argument("--regime", choices=["any_negative", "inverted"], default="any_negative",
                    help="Which regime's prevalence to sweep over.")
    ap.add_argument("--latent-dim", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-train", type=int, default=2000,
                    help="Total training-set size, held constant across prevalences.")
    ap.add_argument("--eval-frac", type=float, default=0.30,
                    help="Fraction of regime curves held out for evaluation.")
    ap.add_argument("--prevalences", type=str, default="0.05,0.15,0.30,0.50,0.75",
                    help="Comma-separated target regime fractions for the training set.")
    ap.add_argument("--use", default="bbg", choices=["bbg", "test"])
    ap.add_argument("--model-type", default="baseline", choices=["baseline", "stable"],
                    help="Which model architecture to train: 'baseline' (FullModel) or "
                         "'stable' (FullModelStable with constrained K/H/R).")
    ap.add_argument("--tag", default="", help="Suffix added to output folder name.")
    return ap.parse_args()


def sample_indices(pool: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample n indices from pool; with replacement only if pool is too small."""
    if len(pool) == 0:
        raise ValueError("Cannot sample from empty pool.")
    replace = n > len(pool)
    return rng.choice(pool, size=n, replace=replace)


def main():
    args = parse_args()
    prevalences = [float(x) for x in args.prevalences.split(",")]

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Device: {device}")
    print(f"Args: {vars(args)}")

    # ------------------------------------------------------------------
    # Load data (in-sample window, post-2010, complete curves only)
    # ------------------------------------------------------------------
    meta, X_tensor, *_ = my_data(use=args.use)
    X_tensor = X_tensor.float()
    X_np = X_tensor.numpy()
    N = X_np.shape[0]
    print(f"Loaded {N} curves with shape {X_np.shape}")

    flags = compute_regime_flags(X_np)
    regime_mask = flags[args.regime]
    n_regime_total = int(regime_mask.sum())
    n_nonregime_total = int((~regime_mask).sum())
    print(f"Regime '{args.regime}': {n_regime_total} curves "
          f"({100*n_regime_total/N:.1f}%); non-regime: {n_nonregime_total}")

    if n_regime_total < 30:
        raise RuntimeError(f"Too few regime curves ({n_regime_total}) to run a meaningful sweep.")

    # ------------------------------------------------------------------
    # Build fixed held-out evaluation sets (regime + non-regime)
    # ------------------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    regime_idx = np.where(regime_mask)[0]
    nonregime_idx = np.where(~regime_mask)[0]

    # Both eval sets are the same size so RMSE numbers are directly comparable.
    # We use the regime (minority class) count to set the size — it is the
    # binding constraint.  The non-regime pool is large enough that holding out
    # the same number has negligible effect on training.
    n_eval_regime    = max(10, int(round(args.eval_frac * n_regime_total)))
    n_eval_nonregime = n_eval_regime   # equal size for fair comparison

    eval_regime_idx    = rng.choice(regime_idx,    size=n_eval_regime,    replace=False)
    eval_nonregime_idx = rng.choice(nonregime_idx, size=n_eval_nonregime, replace=False)

    # Pools used for training (eval rows excluded)
    train_regime_pool    = np.setdiff1d(regime_idx,    eval_regime_idx)
    train_nonregime_pool = np.setdiff1d(nonregime_idx, eval_nonregime_idx)

    X_eval_regime    = X_tensor[eval_regime_idx]
    X_eval_nonregime = X_tensor[eval_nonregime_idx]
    print(f"Held-out eval sets: regime={len(eval_regime_idx)}, non-regime={len(eval_nonregime_idx)} (equal size)")
    print(f"Training pools:     regime={len(train_regime_pool)}, non-regime={len(train_nonregime_pool)}")

    # ------------------------------------------------------------------
    # Output dir
    # ------------------------------------------------------------------
    out_dir = os.path.join(
        REPO_ROOT, "Figures", "Experiments", "PrevalenceSweep",
        f"{args.regime}_dim{args.latent_dim}_ep{args.epochs}_N{args.n_train}_{args.model_type}{('_' + args.tag) if args.tag else ''}",
    )
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "run_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    csv_path = os.path.join(out_dir, "sweep_results.csv")

    cols = ["p_target", "p_actual", "n_train", "n_regime_train", "n_nonregime_train",
            "rmse_bps_eval_regime", "rmse_bps_eval_nonregime", "rmse_bps_train_overall"]
    pd.DataFrame(columns=cols).to_csv(csv_path, index=False)
    print(f"Logging to {csv_path}")

    # ------------------------------------------------------------------
    # Sweep
    # ------------------------------------------------------------------
    train_cfg = TrainConfig(
        latent_dim=args.latent_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        fixed_lr=args.lr,
        seed=args.seed,
        model_type=args.model_type,
    )

    rows = []
    for i, p in enumerate(prevalences):
        n_reg = int(round(p * args.n_train))
        n_non = args.n_train - n_reg
        n_reg = min(n_reg, len(train_regime_pool) * 10)  # cap if absurd
        n_non = min(n_non, len(train_nonregime_pool) * 10)

        sub_rng = np.random.default_rng(args.seed + 1000 * (i + 1))
        reg_pick = sample_indices(train_regime_pool, n_reg, sub_rng) if n_reg > 0 else np.array([], dtype=int)
        non_pick = sample_indices(train_nonregime_pool, n_non, sub_rng) if n_non > 0 else np.array([], dtype=int)
        train_idx = np.concatenate([reg_pick, non_pick])
        sub_rng.shuffle(train_idx)

        X_train = X_tensor[train_idx]
        p_actual = n_reg / max(args.n_train, 1)
        tag = f"p={p:.2f}"
        print(f"\n=== Run {i+1}/{len(prevalences)}: target p={p:.2f}  "
              f"(n_regime={n_reg}, n_nonregime={n_non}) ===")

        model, _ = train_with_retry(X_train, train_cfg, device=device, tag=tag)

        rmse_eval_reg = rmse_bps_overall(model, X_eval_regime, device)
        rmse_eval_non = rmse_bps_overall(model, X_eval_nonregime, device)
        rmse_train = rmse_bps_overall(model, X_train, device)

        print(f"  RMSE (bps) — eval_regime={rmse_eval_reg:.2f}  "
              f"eval_nonregime={rmse_eval_non:.2f}  train={rmse_train:.2f}")

        row = {
            "p_target": p,
            "p_actual": p_actual,
            "n_train": int(args.n_train),
            "n_regime_train": int(n_reg),
            "n_nonregime_train": int(n_non),
            "rmse_bps_eval_regime": rmse_eval_reg,
            "rmse_bps_eval_nonregime": rmse_eval_non,
            "rmse_bps_train_overall": rmse_train,
        }
        rows.append(row)
        pd.DataFrame([row], columns=cols).to_csv(csv_path, mode="a", header=False, index=False)

    df = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.0, 4.5), dpi=160)
    ax.plot(df["p_target"], df["rmse_bps_eval_regime"], "o-", label=f"Eval RMSE ({args.regime})")
    ax.plot(df["p_target"], df["rmse_bps_eval_nonregime"], "s--",
            color="0.4", label="Eval RMSE (non-regime control)")
    ax.set_xlabel(f"Target prevalence of '{args.regime}' in training set")
    ax.set_ylabel("Held-out RMSE (bps)")
    ax.set_title(f"Prevalence sweep: baseline $\\ell={args.latent_dim}$, "
                 f"N_train={args.n_train}, {args.epochs} epochs")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig_path = os.path.join(out_dir, "sweep_rmse_vs_prevalence.png")
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"\nSaved plot: {fig_path}")
    print(f"Saved CSV : {csv_path}")
    print("Done.")


if __name__ == "__main__":
    main()

