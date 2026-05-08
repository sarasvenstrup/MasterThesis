# ==================== Pricing Evaluation: All Stable Checkpoints ====================
"""
Evaluates one or more checkpoints on:
  1. Reconstruction quality  – RMSE in bps per currency (all data)
  2. Pricing quality         – model-implied ATM Bachelier vol vs. market vol
                               for every (date, expiry, tenor) in the EUR swaption data

Designed to work with both:
  • Raw state_dict checkpoints  (produced by Training.py / Training_stable.py)
  • Wrapped checkpoints          (produced by Training_joint.py)

Usage
-----
# Evaluate the latest joint checkpoint (auto-discovered):
    python Code/Pricing/eval_joint.py

# Evaluate a specific checkpoint:
    python Code/Pricing/eval_joint.py \\
        --checkpoint Figures/TrainingResults/dim4_stable/ep5000/checkpoint_dim4_ep5000.pt

# Evaluate ALL stable ep5000 checkpoints (dim 2, 3, 4):
    python Code/Pricing/eval_joint.py --all_stable

# Control MC paths and swaption sample size:
    python Code/Pricing/eval_joint.py --all_stable --n_paths 1000 --max_swaptions 200

Outputs (per checkpoint, inside <checkpoint_dir>/eval/):
    reconstruction_rmse.csv           — RMSE in bps per currency
    pricing_eval.csv                  — per-row model vol vs market vol + skip reasons
    pricing_eval_by_expiry_tenor.csv  — MAE/RMSE/bias grouped by (expiry × tenor)
    scatter_mkt_vs_mod.png            — 45° scatter plot
    hist_pricing_error.png            — error histogram
    heatmap_mae.png                   — MAE heatmap

Notes on simulation
-------------------
The model uses Euler-Maruyama simulation z0 → z_T.  For the stable variant
the drift eigenvalues are near zero (weak mean-reversion), which means
long-expiry swaptions may produce z_T far outside the training range and
cause NaN discount curves.  This is a model property, not a bug.  The script
reports skip reasons verbosely so you can see which (expiry, tenor) cells
succeed and which fail.  No clipping or masking is applied.
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# ───────────────────────────── path setup ─────────────────────────────────
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.getcwd()

REPO_ROOT    = os.path.abspath(os.path.join(_HERE, ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(REPO_ROOT, ".."))

for _p in (PROJECT_ROOT, REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("SKIP_VARIANT_CONFIRM", "1")

from Code import config
config.confirm_variant()

from Code.utils import helpers as H
from Code.load_swapdata import my_data
from Code.model.full_model_stable import FullModel as FullModelStable
from Code.model.full_model import FullModel as FullModelBaseline
from Code.Pricing.load_swapvol_ois import load_swaption_vol_data
from Code.Pricing.pricing import swap_rate_torch, implied_bachelier_vol
from Code.Simulation.simulate_model import simulate_to_expiry_differentiable

# ───────────────────────────── CLI ────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate checkpoint(s) on reconstruction + ATM swaption pricing")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to a single .pt checkpoint")
    p.add_argument("--all_stable", action="store_true",
                   help="Evaluate ALL stable ep5000 checkpoints (dim 2, 3, 4)")
    p.add_argument("--all_baseline", action="store_true",
                   help="Evaluate ALL baseline ep5000 checkpoints (dim 2, 3, 4)")
    p.add_argument("--baseline", action="store_true",
                   help="Force use of baseline (non-stable) FullModel for --checkpoint")
    p.add_argument("--n_paths", type=int, default=500,
                   help="MC paths per swaption (default 500)")
    p.add_argument("--dt", type=float, default=1/12,
                   help="Euler-Maruyama time-step in years (default 1/12 = monthly)")
    p.add_argument("--ccy", type=str, default="EUR",
                   help="Currency for pricing evaluation (default EUR)")
    p.add_argument("--max_swaptions", type=int, default=None,
                   help="Cap total swaption rows evaluated (for speed, e.g. 100)")
    p.add_argument("--use", type=str, default="bbg",
                   help="Data source for my_data (default bbg)")
    p.add_argument("--no_plots", action="store_true",
                   help="Skip saving figures (faster)")
    return p.parse_args()

# ─────────────────────────── checkpoint collection ────────────────────────

def collect_checkpoints(args) -> list[dict]:
    """Return list of {path, label, baseline} dicts to evaluate."""
    if args.all_stable:
        base = os.path.join(PROJECT_ROOT, "Figures", "TrainingResults")
        out = []
        for dim in (2, 3, 4):
            p = os.path.join(base, f"dim{dim}_stable", "ep5000",
                             f"checkpoint_dim{dim}_ep5000.pt")
            if os.path.isfile(p):
                out.append({"path": p, "label": f"dim{dim}_stable_ep5000", "baseline": False})
            else:
                print(f"  [MISSING] {p}")
        return out

    if args.all_baseline:
        base = os.path.join(PROJECT_ROOT, "Figures", "TrainingResults")
        out = []
        for dim in (2, 3, 4):
            p = os.path.join(base, f"dim{dim}_baseline", "ep5000",
                             f"checkpoint_dim{dim}_ep5000.pt")
            if os.path.isfile(p):
                out.append({"path": p, "label": f"dim{dim}_baseline_ep5000", "baseline": True})
            else:
                print(f"  [MISSING] {p}")
        return out

    if args.checkpoint:
        is_baseline = args.baseline or ("baseline" in args.checkpoint and "stable" not in args.checkpoint)
        return [{"path": args.checkpoint,
                 "label": os.path.basename(os.path.dirname(args.checkpoint)),
                 "baseline": is_baseline}]

    # Default: latest joint checkpoint
    search = os.path.join(PROJECT_ROOT, "Figures", "TrainingResults")
    hits = []
    for root, _, files in os.walk(search):
        if "joint" in root.lower():
            for f in files:
                if f.endswith(".pt"):
                    full = os.path.join(root, f)
                    hits.append((os.path.getmtime(full), full))
    if not hits:
        raise FileNotFoundError(
            f"No joint checkpoint found under {search}. "
            "Use --checkpoint or --all_stable.")
    path = sorted(hits, reverse=True)[0][1]
    return [{"path": path, "label": "latest_joint", "baseline": False}]


def load_checkpoint(path: str, device, baseline: bool = False):
    raw = torch.load(path, map_location=device)
    if "model_state_dict" in raw:
        state_dict = raw["model_state_dict"]
        latent_dim = raw.get("latent_dim",
                              raw.get("model_config", {}).get("latent_dim", 4))
    else:
        state_dict = raw
        w = raw.get("encoder.lin.weight")
        latent_dim = int(w.shape[0]) if w is not None else 4
    ModelClass = FullModelBaseline if baseline else FullModelStable
    model = ModelClass(latent_dim=latent_dim).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, latent_dim


def row_finite_mask(t: torch.Tensor) -> torch.Tensor:
    return torch.isfinite(t).all(dim=1)


@torch.no_grad()
def predict_all(model, X: torch.Tensor, device, batch_size: int = 256):
    outs = []
    for i in range(0, X.shape[0], batch_size):
        outs.append(model(X[i:i+batch_size].to(device)).detach().cpu())
    return torch.cat(outs, dim=0)

# ──────────────────────────── evaluation ──────────────────────────────────

def eval_reconstruction(model, X_tensor, meta, device):
    S_hat = predict_all(model, X_tensor, device)
    mask  = row_finite_mask(X_tensor) & row_finite_mask(S_hat)
    n_bad = int((~mask).sum())
    rmse  = H.rmse_bps_per_currency_paper(
        X_tensor[mask], S_hat[mask],
        meta.loc[mask.numpy()].reset_index(drop=True),
    )
    return rmse, n_bad


def eval_pricing(model, latent_dim, X_ccy, meta_ccy, df_vol, args, device):
    date_to_idx = {
        pd.Timestamp(r["as_of_date"]).normalize(): i
        for i, r in meta_ccy.iterrows()
    }
    records = []
    t0 = time.perf_counter()

    for row_i, row in df_vol.iterrows():
        date      = pd.Timestamp(row["as_of_date"]).normalize()
        expiry    = int(row["option_maturity"])
        tenor     = int(row["swap_tenor"])
        sigma_mkt = float(row["market_vol"])

        if date not in date_to_idx:
            continue

        idx = date_to_idx[date]
        xb  = X_ccy[idx:idx+1].to(device)

        with torch.no_grad():
            z0 = model.encoder(xb)

        dt_eff  = min(args.dt, expiry / 10.0)
        n_steps = max(1, int(round(expiry / dt_eff)))
        half    = args.n_paths // 2
        eps_h   = torch.randn(half, n_steps, latent_dim, device=device, dtype=xb.dtype)
        eps     = torch.cat([eps_h, -eps_h], dim=0)

        try:
            with torch.no_grad():
                z_T, D_T = simulate_to_expiry_differentiable(
                    model=model, z0=z0, n_steps=n_steps,
                    dt=dt_eff, n_paths=args.n_paths, eps=eps,
                )

            if not torch.isfinite(z_T).all():
                records.append(_fail(date, expiry, tenor, sigma_mkt, "nan_z_T"))
                continue

            with torch.no_grad():
                _, aux_T = model.decode_from_z(z_T, tau=None, return_aux=True)
                P_full_T = aux_T["P_full"]

            if not torch.isfinite(P_full_T).all():
                records.append(_fail(date, expiry, tenor, sigma_mkt, "nan_P_T"))
                continue

            F_T, A_T = swap_rate_torch(P_full_T, tenor=tenor)
            if not (torch.isfinite(F_T).all() and torch.isfinite(A_T).all()):
                records.append(_fail(date, expiry, tenor, sigma_mkt, "nan_swap"))
                continue

            with torch.no_grad():
                _, aux0 = model.decode_from_z(z0, tau=None, return_aux=True)
                P0 = aux0["P_full"]
            F_0, A_0 = swap_rate_torch(P0, tenor=tenor)
            F_0v, A_0v = float(F_0[0].item()), float(A_0[0].item())

            K      = F_0v
            payoff = A_T * torch.relu(F_T - K)
            V_MC   = float((D_T * payoff).mean().item())

            sigma_mod, fail_reason = implied_bachelier_vol(
                market_price=V_MC, forward=F_0v, strike=K,
                expiry=expiry, annuity=A_0v, payer=True,
                _return_failure_reason=True,
            )

            if sigma_mod is None or not np.isfinite(sigma_mod):
                records.append(_fail(date, expiry, tenor, sigma_mkt,
                                     fail_reason or "inv_vol_failed"))
                continue

            err_bp = (sigma_mod - sigma_mkt) * 1e4
            records.append({
                "date": date.date(), "expiry": expiry, "tenor": tenor,
                "mkt_vol_bp": round(sigma_mkt * 1e4, 4),
                "mod_vol_bp": round(sigma_mod * 1e4, 4),
                "err_bp":     round(err_bp, 4),
                "abs_err_bp": abs(err_bp),
                "status":     "ok",
            })

        except Exception as exc:
            records.append(_fail(date, expiry, tenor, sigma_mkt, str(exc)[:80]))

        if (row_i + 1) % 50 == 0 or row_i == len(df_vol) - 1:
            print(f"    {row_i+1:>5}/{len(df_vol)}  elapsed={time.perf_counter()-t0:.0f}s",
                  flush=True)

    return pd.DataFrame(records)


def _fail(date, expiry, tenor, sigma_mkt, reason):
    return {"date": date.date(), "expiry": expiry, "tenor": tenor,
            "mkt_vol_bp": sigma_mkt * 1e4,
            "mod_vol_bp": np.nan, "err_bp": np.nan, "abs_err_bp": np.nan,
            "status": reason}

# ──────────────────────────── plots ───────────────────────────────────────

def make_plots(ok: pd.DataFrame, out_dir: str):
    if len(ok) == 0:
        return

    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    ax.scatter(ok["mkt_vol_bp"], ok["mod_vol_bp"], alpha=0.4, s=8, color="steelblue")
    lo = min(ok["mkt_vol_bp"].min(), ok["mod_vol_bp"].min()) * 0.95
    hi = max(ok["mkt_vol_bp"].max(), ok["mod_vol_bp"].max()) * 1.05
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="45° line")
    ax.set_xlabel("Market vol (bp)"); ax.set_ylabel("Model vol (bp)")
    ax.set_title("ATM Swaption Vol: Market vs Model"); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "scatter_mkt_vs_mod.png"), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    ax.hist(ok["err_bp"].dropna(), bins=40, color="steelblue", edgecolor="white", lw=0.4)
    ax.axvline(0, color="black", lw=1.2, ls="--")
    ax.set_xlabel("Error = Model − Market (bp)"); ax.set_ylabel("Count")
    ax.set_title("Pricing Error Distribution"); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "hist_pricing_error.png"), dpi=200)
    plt.close(fig)

    try:
        pivot = ok.pivot_table(index="expiry", columns="tenor",
                               values="abs_err_bp", aggfunc="mean")
        fig, ax = plt.subplots(
            figsize=(max(4, len(pivot.columns)), max(3, len(pivot))), dpi=150)
        im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn_r")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{c}Y" for c in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{r}Y" for r in pivot.index])
        ax.set_xlabel("Swap tenor"); ax.set_ylabel("Option expiry")
        ax.set_title("Mean Abs Error by (Expiry × Tenor) [bp]")
        plt.colorbar(im, ax=ax, label="MAE (bp)")
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                v = pivot.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "heatmap_mae.png"), dpi=200)
        plt.close(fig)
    except Exception:
        pass

# ──────────────────────────── per-checkpoint run ──────────────────────────

def run_one(ckpt_info, X_tensor, meta, df_vol_raw, args, device):
    path, label = ckpt_info["path"], ckpt_info["label"]
    is_baseline = ckpt_info.get("baseline", False)
    print(f"\n{'='*70}")
    print(f"  {label}  ({'baseline' if is_baseline else 'stable'})")
    print(f"  {path}")
    print(f"{'='*70}")

    model, latent_dim = load_checkpoint(path, device, baseline=is_baseline)
    print(f"  Latent dim: {latent_dim}")

    out_dir = os.path.join(os.path.dirname(path), "eval")
    os.makedirs(out_dir, exist_ok=True)

    # 1. Reconstruction
    print("\n  Reconstruction quality:")
    rmse, n_bad = eval_reconstruction(model, X_tensor, meta, device)
    avg_rmse = float(rmse.mean())
    print(f"  {'Currency':<10} {'RMSE (bps)':>12}")
    print("  " + "-" * 24)
    for ccy, v in rmse.items():
        print(f"  {ccy:<10} {v:>12.4f}")
    print("  " + "-" * 24)
    print(f"  {'Average':<10} {avg_rmse:>12.4f}   non-finite rows: {n_bad}/{len(X_tensor)}")
    rmse.reset_index().rename(columns={"index": "currency", 0: "rmse_bps"}).to_csv(
        os.path.join(out_dir, "reconstruction_rmse.csv"), index=False)

    # 2. Pricing
    print(f"\n  Pricing quality ({args.ccy}, {args.n_paths} paths):")
    meta_ccy = meta[meta["ccy"] == args.ccy].reset_index(drop=True)
    X_ccy    = X_tensor[meta["ccy"] == args.ccy]
    dates_swap = set(pd.to_datetime(meta_ccy["as_of_date"]).dt.normalize())
    df_vol = df_vol_raw[df_vol_raw["as_of_date"].isin(dates_swap)].copy().reset_index(drop=True)

    if df_vol.empty:
        print("  No overlapping dates — pricing skipped.")
        return {"label": label, "avg_rmse_bps": avg_rmse, "n_bad": n_bad,
                "n_priced": 0, "n_total": 0,
                "mae_bp": np.nan, "rmse_bp": np.nan, "bias_bp": np.nan}

    if args.max_swaptions:
        df_vol = df_vol.sample(n=min(args.max_swaptions, len(df_vol)),
                               random_state=42).reset_index(drop=True)

    print(f"  {len(df_vol)} observations from {df_vol['as_of_date'].nunique()} dates")
    df_res = eval_pricing(model, latent_dim, X_ccy, meta_ccy, df_vol, args, device)
    df_res.to_csv(os.path.join(out_dir, "pricing_eval.csv"), index=False)

    ok      = df_res[df_res["status"] == "ok"]
    n_ok    = len(ok)
    n_total = len(df_res)

    fails = df_res[df_res["status"] != "ok"]
    if len(fails):
        print(f"\n  Skip reasons ({len(fails)} skipped):")
        for reason, cnt in fails["status"].value_counts().items():
            print(f"    {cnt:>5}×  {reason}")

    mae_bp = rmse_bp = bias_bp = np.nan
    print(f"\n  Priced: {n_ok}/{n_total}")
    if n_ok > 0:
        mae_bp  = float(ok["abs_err_bp"].mean())
        rmse_bp = float(math.sqrt((ok["err_bp"]**2).mean()))
        bias_bp = float(ok["err_bp"].mean())
        print(f"  MAE  (bp): {mae_bp:.2f}")
        print(f"  RMSE (bp): {rmse_bp:.2f}")
        print(f"  Bias (bp): {bias_bp:.2f}")
        print(f"  Max  (bp): {ok['abs_err_bp'].max():.2f}")

        grp = ok.groupby(["expiry", "tenor"])["err_bp"].agg(
            count="count",
            mae=lambda x: x.abs().mean(),
            rmse=lambda x: math.sqrt((x**2).mean()),
            bias="mean",
        ).reset_index()
        print(f"\n  Per (expiry × tenor):")
        print(f"  {'Exp':>4} {'Ten':>4} {'N':>5} {'MAE':>8} {'RMSE':>8} {'Bias':>8}")
        print("  " + "-" * 44)
        for _, r in grp.iterrows():
            print(f"  {int(r.expiry):>4} {int(r.tenor):>4} {int(r['count']):>5} "
                  f"{r.mae:>8.2f} {r.rmse:>8.2f} {r.bias:>8.2f}")
        grp.to_csv(os.path.join(out_dir, "pricing_eval_by_expiry_tenor.csv"), index=False)

        if not args.no_plots:
            make_plots(ok, out_dir)

    print(f"\n  -> {out_dir}")
    return {"label": label, "avg_rmse_bps": avg_rmse, "n_bad": n_bad,
            "n_priced": n_ok, "n_total": n_total,
            "mae_bp": mae_bp, "rmse_bp": rmse_bp, "bias_bp": bias_bp}

# ──────────────────────────── entry point ─────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Variant: {config.VARIANT}")

    ckpts = collect_checkpoints(args)
    if not ckpts:
        print("No checkpoints found."); return

    print(f"\nLoading swap data (use={args.use})…")
    meta, X_tensor, *_ = my_data(use=args.use)
    X_tensor = X_tensor.float()

    print(f"Loading swaption vol ({args.ccy})…")
    df_vol_raw = load_swaption_vol_data(currency=args.ccy)
    df_vol_raw["as_of_date"] = pd.to_datetime(df_vol_raw["as_of_date"]).dt.normalize()
    df_vol_raw["market_vol"] = df_vol_raw["vol"] / 1e4

    summary = [run_one(c, X_tensor, meta, df_vol_raw, args, device) for c in ckpts]

    if len(summary) > 1:
        print("\n" + "=" * 80)
        print("CROSS-CHECKPOINT SUMMARY")
        print("=" * 80)
        hdr = f"{'Label':<30} {'AvgRMSE':>9} {'NaN':>6} {'Priced':>8} {'MAE_bp':>8} {'RMSE_bp':>8} {'Bias_bp':>8}"
        print(hdr); print("-" * len(hdr))
        for r in summary:
            print(f"{r['label']:<30} {r['avg_rmse_bps']:>9.2f} "
                  f"{r.get('n_bad',0):>6} {r.get('n_priced',0):>8} "
                  f"{r['mae_bp']:>8.2f} {r['rmse_bp']:>8.2f} {r['bias_bp']:>8.2f}")


if __name__ == "__main__":
    main()

