# =============================================================================
# Experiment 2 — Regime-only training (isolation test)
#
# Question: CAN the baseline encoder--decoder represent negative-rate / inverted
# curves at all if there is nothing else competing for the latent space?
#
# Design:
#   For each regime in REGIMES, take ALL curves of that regime, split them
#   80/20 (fixed seed) into train/eval, train a fresh baseline FullModel on
#   the train split, and report both in-sample (train) and held-out (eval)
#   RMSE in basis points.
#
#   The "normal_positive" regime is included as a control: its RMSE sets the
#   benchmark for "what the architecture can achieve on a well-behaved
#   regime when trained only on that regime".
#
# Interpretation:
#   - regime-only RMSE for "any_negative" / "inverted" comparable to the
#     normal control -> the architecture is CAPABLE; the joint-training
#     failure observed in Chapter 8 is a data-composition problem.
#   - regime-only RMSE remains much worse than the normal control -> the
#     linear encoder is STRUCTURALLY limited for that regime, independent of
#     training-data mix.
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


REGIMES = ["normal_positive", "any_negative", "deeply_negative", "inverted"]


def parse_args():
    ap = argparse.ArgumentParser(description="Regime-only training experiment")
    ap.add_argument("--regimes", type=str, default=",".join(REGIMES),
                    help="Comma-separated regimes to evaluate. "
                         f"Choices: {REGIMES}")
    ap.add_argument("--latent-dim", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-frac", type=float, default=0.20,
                    help="Held-out fraction for evaluation.")
    ap.add_argument("--min-curves", type=int, default=30,
                    help="Skip a regime if it has fewer than this many curves.")
    ap.add_argument("--use", default="bbg", choices=["bbg", "test"])
    ap.add_argument("--model-type", default="baseline", choices=["baseline", "stable"],
                    help="Which model architecture to train: 'baseline' (FullModel) or "
                         "'stable' (FullModelStable with constrained K/H/R).")
    ap.add_argument("--tag", default="", help="Suffix added to output folder name.")
    return ap.parse_args()


def main():
    args = parse_args()
    regimes = [r.strip() for r in args.regimes.split(",") if r.strip()]

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Device: {device}")
    print(f"Args: {vars(args)}")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    meta, X_tensor, *_ = my_data(use=args.use)
    X_tensor = X_tensor.float()
    X_np = X_tensor.numpy()
    print(f"Loaded {X_np.shape[0]} curves with shape {X_np.shape}")

    flags = compute_regime_flags(X_np)

    # ------------------------------------------------------------------
    # Output dir
    # ------------------------------------------------------------------
    out_dir = os.path.join(
        REPO_ROOT, "Figures", "Experiments", "RegimeOnly",
        f"dim{args.latent_dim}_ep{args.epochs}_{args.model_type}{('_' + args.tag) if args.tag else ''}",
    )
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "run_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    csv_path = os.path.join(out_dir, "regime_only_results.csv")

    cols = ["regime", "n_total", "n_train", "n_eval",
            "rmse_bps_train", "rmse_bps_eval"]
    pd.DataFrame(columns=cols).to_csv(csv_path, index=False)
    print(f"Logging to {csv_path}")

    # ------------------------------------------------------------------
    # Train one model per regime
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
    for i, regime in enumerate(regimes):
        if regime not in flags:
            print(f"\n[skip] Unknown regime '{regime}'. Known: {list(flags)}")
            continue

        idx_all = np.where(flags[regime])[0]
        n_total = len(idx_all)
        if n_total < args.min_curves:
            print(f"\n[skip] Regime '{regime}' has only {n_total} curves (< {args.min_curves}).")
            continue

        rng = np.random.default_rng(args.seed + 1000 * (i + 1))
        idx_perm = rng.permutation(idx_all)
        n_eval = max(10, int(round(args.eval_frac * n_total)))
        eval_idx = idx_perm[:n_eval]
        train_idx = idx_perm[n_eval:]

        X_train = X_tensor[train_idx]
        X_eval = X_tensor[eval_idx]

        print(f"\n=== Regime '{regime}': n_total={n_total} "
              f"(train={len(train_idx)}, eval={len(eval_idx)}) ===")

        model, _ = train_with_retry(X_train, train_cfg, device=device, tag=regime)

        rmse_train = rmse_bps_overall(model, X_train, device)
        rmse_eval = rmse_bps_overall(model, X_eval, device)

        print(f"  RMSE (bps) — train={rmse_train:.2f}  eval={rmse_eval:.2f}")

        row = {
            "regime": regime,
            "n_total": int(n_total),
            "n_train": int(len(train_idx)),
            "n_eval": int(len(eval_idx)),
            "rmse_bps_train": rmse_train,
            "rmse_bps_eval": rmse_eval,
        }
        rows.append(row)
        pd.DataFrame([row], columns=cols).to_csv(csv_path, mode="a", header=False, index=False)

    if not rows:
        print("No regimes produced results; nothing to plot.")
        return

    df = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=160)
    x = np.arange(len(df))
    width = 0.38
    ax.bar(x - width / 2, df["rmse_bps_train"], width, label="In-sample (train)", color="0.5")
    ax.bar(x + width / 2, df["rmse_bps_eval"], width, label="Held-out (eval)", color="C0")
    ax.set_xticks(x)
    ax.set_xticklabels(df["regime"], rotation=15)
    ax.set_ylabel("RMSE (bps)")
    ax.set_title(f"Regime-only training — baseline $\\ell={args.latent_dim}$, "
                 f"{args.epochs} epochs")
    for xi, (t, e, n) in enumerate(zip(df["rmse_bps_train"], df["rmse_bps_eval"], df["n_total"])):
        ax.text(xi, max(t, e) * 1.02, f"n={n}", ha="center", fontsize=8, color="0.3")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig_path = os.path.join(out_dir, "regime_only_rmse.png")
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"\nSaved plot: {fig_path}")
    print(f"Saved CSV : {csv_path}")
    print("Done.")


if __name__ == "__main__":
    main()

