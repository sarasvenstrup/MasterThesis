"""
CSV-driven swaption pricing workflow.

Expected CSV columns
--------------------
Required:
- expiry: expiry in years
- tenor: swap tenor in years
- strike: strike swap rate in decimal (e.g. 0.0325)
- option_type: payer / receiver (or call / put)
- one of:
    * norm_vol       (decimal, e.g. 0.005)
    * norm_vol_bp    (basis points, e.g. 50)

Optional:
- notional          (default 1.0)
- market_price      (decimal)
- market_price_bp   (basis points)
- label             (free text)

Workflow
--------
1. Load model and initial curve.
2. Simulate latent paths up to the maximum expiry in the quote file.
3. For each unique (expiry, tenor), extract model forward swap and annuity.
4. Price each quote from its normal volatility using Bachelier.
5. Optionally compute Monte Carlo model prices and implied normal vols.
6. Save results CSV and optional plots.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from Code.model.full_model import FullModel
from Code.price_options import (
    extract_forward_swap_curve_params,
    extract_market_params_at_expiry,
    implied_normal_vol,
    load_initial_curve,
    price_swaption,
    price_swaption_from_norm_vol,
    simulate_latent_paths,
)
def normalize_option_type(value: str) -> str:
    value = str(value).strip().lower()
    if value in {"payer", "call", "c", "payer_swaption"}:
        return "payer"
    if value in {"receiver", "put", "p", "receiver_swaption"}:
        return "receiver"
    raise ValueError(f"Unsupported option_type '{value}'. Use payer/receiver or call/put.")


def load_quotes(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required_base = {"expiry", "tenor", "strike", "option_type"}
    missing = required_base - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if "norm_vol" in df.columns:
        df["norm_vol"] = pd.to_numeric(df["norm_vol"], errors="coerce")
    elif "norm_vol_bp" in df.columns:
        df["norm_vol_bp"] = pd.to_numeric(df["norm_vol_bp"], errors="coerce")
        df["norm_vol"] = df["norm_vol_bp"] / 10000.0
    else:
        raise ValueError("CSV must contain either 'norm_vol' or 'norm_vol_bp'.")

    df["expiry"] = pd.to_numeric(df["expiry"], errors="coerce")
    df["tenor"] = pd.to_numeric(df["tenor"], errors="coerce").astype("Int64")
    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    df["option_type"] = df["option_type"].map(normalize_option_type)
    df["notional"] = pd.to_numeric(df.get("notional", 1.0), errors="coerce").fillna(1.0)
    df["label"] = df.get("label", "")

    if "market_price" in df.columns:
        df["market_price"] = pd.to_numeric(df["market_price"], errors="coerce")
    if "market_price_bp" in df.columns:
        df["market_price_bp"] = pd.to_numeric(df["market_price_bp"], errors="coerce")
        if "market_price" not in df.columns:
            df["market_price"] = df["market_price_bp"] / 10000.0

    if df[["expiry", "tenor", "strike", "norm_vol"]].isna().any().any():
        bad_rows = df[df[["expiry", "tenor", "strike", "norm_vol"]].isna().any(axis=1)]
        raise ValueError(f"Found invalid numeric values in rows:\n{bad_rows}")

    if (df["expiry"] <= 0).any() or (df["tenor"] <= 0).any() or (df["norm_vol"] < 0).any():
        raise ValueError("expiry and tenor must be positive and norm_vol must be non-negative.")

    return df


def load_model(latent_dim: int, epochs: int, use: str, device: torch.device) -> FullModel:
    checkpoint_path = REPO_ROOT / "checkpoints" / f"fullmodel_{use}_dim{latent_dim}_ep{epochs}.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = FullModel(latent_dim=checkpoint["latent_dim"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def create_plots(df: pd.DataFrame, plots_dir: Path, run_mc: bool) -> list[Path]:
    plots_dir.mkdir(parents=True, exist_ok=True)
    sns.set_style("whitegrid")
    saved_paths: list[Path] = []

    plot_df = df.copy()
    plot_df["quote_key"] = plot_df.apply(
        lambda r: r["label"] if isinstance(r["label"], str) and r["label"].strip() else f"{r['expiry']}Yx{int(r['tenor'])}Y {r['option_type']}",
        axis=1,
    )
    x = np.arange(len(plot_df))
    fig_width = float(max(10.0, len(plot_df) * 1.2))

    fig, ax = plt.subplots(figsize=(fig_width, 6))
    ax.bar(x, plot_df["norm_vol"] * 10000, color="#2E86AB", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["quote_key"], rotation=35, ha="right")
    ax.set_ylabel("Quoted normal vol (bp)")
    ax.set_title("Swaption normal volatility quotes")
    fig.tight_layout()
    path = plots_dir / "swaption_quote_vols.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(path)

    if "market_price" in plot_df.columns and plot_df["market_price"].notna().any():
        fig, ax = plt.subplots(figsize=(fig_width, 6))
        width = 0.38
        ax.bar(x - width / 2, plot_df["market_price"] * 10000, width, label="Market", color="#2E86AB", alpha=0.85)
        ax.bar(x + width / 2, plot_df["quoted_price"] * 10000, width, label="Model from quote vol", color="#F18F01", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(plot_df["quote_key"], rotation=35, ha="right")
        ax.set_ylabel("Price (bp)")
        ax.set_title("Market price vs model price from quoted normal vol")
        ax.legend()
        fig.tight_layout()
        path = plots_dir / "swaption_market_vs_quote_price.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(path)

    if run_mc and "mc_implied_norm_vol" in plot_df.columns and plot_df["mc_implied_norm_vol"].notna().any():
        fig, ax = plt.subplots(figsize=(fig_width, 6))
        width = 0.38
        ax.bar(x - width / 2, plot_df["norm_vol"] * 10000, width, label="Quoted", color="#2E86AB", alpha=0.85)
        ax.bar(x + width / 2, plot_df["mc_implied_norm_vol"] * 10000, width, label="MC implied", color="#A23B72", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(plot_df["quote_key"], rotation=35, ha="right")
        ax.set_ylabel("Normal vol (bp)")
        ax.set_title("Quoted normal vol vs model Monte Carlo implied normal vol")
        ax.legend()
        fig.tight_layout()
        path = plots_dir / "swaption_quote_vs_mc_implied_vol.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(path)

    return saved_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Price swaptions from a CSV of market quotes.")
    parser.add_argument("--quotes_csv", type=str, required=True, help="Path to swaption quote CSV")
    parser.add_argument("--output_csv", type=str, default=None, help="Optional path for output CSV")
    parser.add_argument("--plots_dir", type=str, default=None, help="Optional directory for plots")
    parser.add_argument("--latent_dim", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--use", type=str, default="bbg")
    parser.add_argument("--idx_choice", type=int, default=-1)
    parser.add_argument("--n_paths", type=int, default=1000)
    parser.add_argument("--dt", type=float, default=1.0 / 12.0)
    parser.add_argument("--n_steps", type=int, default=None, help="Defaults to cover the longest quoted expiry")
    parser.add_argument("--run_mc", action="store_true", help="Also price each quote by Monte Carlo and infer model normal vol")
    parser.add_argument("--simple_diffusion", action="store_true")
    parser.add_argument("--kappa", type=float, default=0.5)
    parser.add_argument("--theta", type=float, default=0.0)
    parser.add_argument("--sigma_simple", type=float, default=0.1)
    args = parser.parse_args()

    quotes_csv = Path(args.quotes_csv)
    if not quotes_csv.is_absolute():
        quotes_csv = (Path.cwd() / quotes_csv).resolve()
    if not quotes_csv.exists():
        raise FileNotFoundError(f"Quote file not found: {quotes_csv}")

    output_csv = Path(args.output_csv).resolve() if args.output_csv else quotes_csv.with_name(f"{quotes_csv.stem}_priced.csv")
    plots_dir = Path(args.plots_dir).resolve() if args.plots_dir else quotes_csv.with_name(f"{quotes_csv.stem}_plots")

    df = load_quotes(quotes_csv)
    max_expiry = float(df["expiry"].max())
    n_steps = args.n_steps if args.n_steps is not None else max(1, int(math.ceil(max_expiry / args.dt)) + 2)
    total_horizon = n_steps * args.dt
    if max_expiry > total_horizon + 1e-12:
        raise ValueError(f"n_steps*dt={total_horizon:.4f} is shorter than max expiry {max_expiry:.4f}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading quotes from: {quotes_csv}")
    print(f"Loaded {len(df)} swaption quote rows")

    model = load_model(args.latent_dim, args.epochs, args.use, device)
    print(f"Loaded model: fullmodel_{args.use}_dim{args.latent_dim}_ep{args.epochs}.pt")

    S0, meta_row, _, _ = load_initial_curve(args.use, args.idx_choice, device)
    with torch.no_grad():
        z0 = model.encoder(S0)

    z_paths = None
    r_paths = None
    curve_param_cache: dict[tuple[float, int], dict] = {}
    market_param_cache: dict[tuple[float, int], dict] = {}

    if args.run_mc:
        print(f"Simulating {args.n_paths} paths, {n_steps} steps, dt={args.dt:.6f}...")
        with torch.no_grad():
            z_paths, r_paths = simulate_latent_paths(
                model=model,
                z0=z0,
                n_paths=args.n_paths,
                n_steps=n_steps,
                dt=args.dt,
                device=device,
                simple_diffusion=args.simple_diffusion,
                kappa=args.kappa,
                theta=args.theta,
                sigma_simple=args.sigma_simple,
            )

    rows = []

    for _, row in df.iterrows():
        expiry = float(row["expiry"])
        tenor = int(row["tenor"])
        key = (expiry, tenor)
        if key not in curve_param_cache:
            curve_param_cache[key] = extract_forward_swap_curve_params(model, z0, expiry, tenor)

        params = curve_param_cache[key]
        is_call = row["option_type"] == "payer"
        quoted_price = price_swaption_from_norm_vol(
            forward=params["forward_swap"],
            strike=float(row["strike"]),
            norm_vol=float(row["norm_vol"]),
            expiry=expiry,
            annuity=params["annuity"],
            notional=float(row["notional"]),
            is_call=is_call,
        )

        out = {
            "label": row["label"],
            "expiry": expiry,
            "tenor": tenor,
            "option_type": row["option_type"],
            "strike": float(row["strike"]),
            "notional": float(row["notional"]),
            "norm_vol": float(row["norm_vol"]),
            "norm_vol_bp": float(row["norm_vol"]) * 10000.0,
            "forward_swap": params["forward_swap"],
            "annuity": params["annuity"],
            "quoted_price": quoted_price,
            "quoted_price_bp": quoted_price * 10000.0,
        }

        if "market_price" in row.index and pd.notna(row.get("market_price")):
            out["market_price"] = float(row["market_price"])
            out["market_price_bp"] = float(row["market_price"]) * 10000.0
            out["quote_minus_market_bp"] = (quoted_price - float(row["market_price"])) * 10000.0

        if args.run_mc:
            if key not in market_param_cache:
                market_param_cache[key] = extract_market_params_at_expiry(
                    z_paths=z_paths,
                    model=model,
                    device=device,
                    dt=args.dt,
                    expiry=expiry,
                    tenor=tenor,
                )
            mc_params = market_param_cache[key]
            mc_price = price_swaption(
                z_paths=z_paths,
                r_paths=r_paths,
                model=model,
                dt=args.dt,
                strike=float(row["strike"]),
                expiry=expiry,
                tenor=tenor,
                notional=float(row["notional"]),
            )
            out["mc_price"] = mc_price
            out["mc_price_bp"] = mc_price * 10000.0
            out["mc_minus_quote_bp"] = (mc_price - quoted_price) * 10000.0
            mc_implied = implied_normal_vol(
                market_price=mc_price,
                forward=mc_params["forward_swap"],
                strike=float(row["strike"]),
                expiry=expiry,
                annuity=mc_params["annuity"],
                notional=float(row["notional"]),
                is_call=is_call,
            )
            out["mc_implied_norm_vol"] = mc_implied
            out["mc_implied_norm_vol_bp"] = mc_implied * 10000.0 if np.isfinite(mc_implied) else np.nan
            out["mc_implied_minus_quote_bp"] = (
                (mc_implied - float(row["norm_vol"])) * 10000.0 if np.isfinite(mc_implied) else np.nan
            )

        rows.append(out)

    result_df = pd.DataFrame(rows)
    result_df.to_csv(output_csv, index=False)
    print(f"Saved pricing results to: {output_csv}")

    plot_paths = create_plots(result_df, plots_dir, args.run_mc)
    for path in plot_paths:
        print(f"Saved plot: {path}")

    display_cols = [
        "expiry", "tenor", "option_type", "strike", "norm_vol_bp",
        "forward_swap", "annuity", "quoted_price_bp"
    ]
    if args.run_mc:
        display_cols += ["mc_price_bp", "mc_implied_norm_vol_bp"]
    if "quote_minus_market_bp" in result_df.columns:
        display_cols += ["quote_minus_market_bp"]
    print("\nPricing summary:")
    print(result_df[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()


