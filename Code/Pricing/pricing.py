import os
import sys
import math
import random
import argparse
import warnings

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
CODE_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CODE_ROOT, ".."))
THESIS_ROOT = os.path.abspath(os.path.join(CODE_ROOT, "..", ".."))

if CODE_ROOT not in sys.path:
    sys.path.insert(0, CODE_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if THESIS_ROOT not in sys.path:
    sys.path.insert(0, THESIS_ROOT)

# ---------------------------------------------------------------------
# Config / model
# ---------------------------------------------------------------------
from Code import config
from Code.model.full_model import FullModel

print(f"Active model variant from config.py: {config.VARIANT}")

# ---------------------------------------------------------------------
# Import shared helpers from simulation script
# ---------------------------------------------------------------------
try:
    from simulate_model import (
        load_data_and_initial_curve,
        simulate_latent_paths,
        decode_from_latent_script,
        resolve_checkpoint_path,
        safe_load_state_dict,
        build_decoder_tau_grid,
        compute_discount_paths,
        get_grid_indices_for_values,
        parse_float_list,
        normalize_discretization_name,
    )
except ImportError:
    from Code.Pricing.simulate_model import (
        load_data_and_initial_curve,
        simulate_latent_paths,
        decode_from_latent_script,
        resolve_checkpoint_path,
        safe_load_state_dict,
        build_decoder_tau_grid,
        compute_discount_paths,
        get_grid_indices_for_values,
        parse_float_list,
        normalize_discretization_name,
    )


# ---------------------------------------------------------------------
# Basic utilities
# ---------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model(checkpoint_path: str, device: torch.device, latent_dim: int = 2) -> FullModel:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    raw = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" in raw:
        state_dict = raw["model_state_dict"]
        saved_variant = raw.get("variant", "unknown")
        if saved_variant != "unknown" and saved_variant != config.VARIANT:
            raise ValueError(
                f"Checkpoint variant '{saved_variant}' does not match active "
                f"config.VARIANT '{config.VARIANT}'."
            )
    else:
        state_dict = raw

    model = FullModel(latent_dim=latent_dim)
    safe_load_state_dict(model, state_dict)
    model = model.to(device).double()
    model.eval()

    print(f"Loaded model from {checkpoint_path}")
    print(f"  Active config variant: {config.VARIANT}")
    print(f"  Model dtype: {next(model.parameters()).dtype}")
    return model


def build_tau_grid_with_points(base_grid: torch.Tensor, extra_points, device, dtype):
    """
    Build a dense tau grid that contains:
      - 0
      - all points from base_grid up to max(extra_points)
      - all exact extra_points
    """
    extra_points = [float(x) for x in extra_points if float(x) > 0.0]
    if len(extra_points) == 0:
        raise ValueError("Need at least one positive tau point.")

    tau_max = max(extra_points)
    base_max = float(base_grid[-1].item())
    if tau_max > base_max + 1e-12:
        raise ValueError(
            f"Requested tau_max={tau_max:.6f} exceeds decoder grid max {base_max:.6f}. "
            f"Increase model.tau_max or reduce expiry/tenor."
        )

    extra = torch.tensor(extra_points, device=device, dtype=dtype)
    interior = base_grid[(base_grid > 0.0) & (base_grid < tau_max)]
    tau_grid = torch.cat(
        [
            torch.zeros(1, device=device, dtype=dtype),
            interior.to(device=device, dtype=dtype),
            extra,
        ]
    )
    tau_grid = torch.unique(tau_grid, sorted=True)
    return tau_grid


def expiry_index_from_time(expiry: float, dt: float, n_steps: int):
    if expiry < 0:
        raise ValueError(f"expiry must be non-negative, got {expiry}")

    idx = int(round(expiry / dt))
    idx = min(max(idx, 0), n_steps)
    actual_expiry = idx * dt

    if abs(actual_expiry - expiry) > 1e-10:
        warnings.warn(
            f"expiry={expiry:.6f} is not exactly on the simulation grid with dt={dt:.6f}; "
            f"using nearest grid time {actual_expiry:.6f} (index {idx}).",
            RuntimeWarning,
        )
    return idx, actual_expiry


def decode_discount_curve_batch_safe(model, z_batch, tau_grid, g0_floor=1e-5):
    """
    Robust batch decode:
      - try batch decode first
      - if it fails, fall back to pathwise decode
    Returns:
      P_full: (B, len(tau_grid)) with NaNs for failed paths
      valid:  (B,) bool
    """
    if z_batch.dim() != 2:
        raise ValueError(f"Expected z_batch shape (B,d), got {tuple(z_batch.shape)}")

    B = z_batch.shape[0]
    device = z_batch.device
    dtype = z_batch.dtype

    P_full_out = torch.full(
        (B, tau_grid.numel()),
        float("nan"),
        device=device,
        dtype=dtype,
    )
    valid = torch.zeros(B, device=device, dtype=torch.bool)

    try:
        P_full, _, _, _, _, _, _, _ = decode_from_latent_script(
            model,
            z_batch,
            tau_grid,
            G_floor=g0_floor,
            check_short_rate=False,
        )
        P_full_out[:] = P_full
        valid[:] = torch.isfinite(P_full).all(dim=1)
        return P_full_out, valid

    except RuntimeError as e:
        warnings.warn(
            f"Batch decode failed; falling back to pathwise decode.\nReason: {e}",
            RuntimeWarning,
        )

    for i in range(B):
        try:
            P_i, _, _, _, _, _, _, _ = decode_from_latent_script(
                model,
                z_batch[i : i + 1],
                tau_grid,
                G_floor=g0_floor,
                check_short_rate=False,
            )
            P_full_out[i] = P_i[0]
            valid[i] = True
        except RuntimeError:
            pass

    return P_full_out, valid


# ---------------------------------------------------------------------
# Swap helpers
# ---------------------------------------------------------------------
def spot_start_swap_rate_and_annuity_from_discount(P_payments: torch.Tensor, accrual: float = 1.0):
    """
    Spot-start swap at a given date t.

    Args:
        P_payments: (B, n_payments) with
            P(t,t+alpha), P(t,t+2alpha), ..., P(t,t+n*alpha)
        accrual: fixed-leg accrual fraction

    Returns:
        swap_rate: (B,)
        annuity:   (B,)
    """
    if P_payments.ndim != 2:
        raise ValueError(f"Expected P_payments shape (B,n), got {tuple(P_payments.shape)}")

    annuity = accrual * P_payments.sum(dim=1)
    terminal_df = P_payments[:, -1]
    swap_rate = (1.0 - terminal_df) / annuity
    return swap_rate, annuity


def forward_start_swap_rate_and_annuity_from_discount(
    P_start: torch.Tensor,
    P_end: torch.Tensor,
    P_payments: torch.Tensor,
    accrual: float = 1.0,
):
    annuity = accrual * P_payments.sum(dim=1)
    forward_swap = (P_start - P_end) / annuity
    return forward_swap, annuity


def time0_forward_swap_and_annuity_from_z(
    model,
    z0: torch.Tensor,
    decoder_tau_grid_base: torch.Tensor,
    expiry: float,
    tenor: int,
    g0_floor: float = 1e-5,
    accrual: float = 1.0,
):
    """
    Extract the time-0 forward swap rate and annuity for a swaption expiring
    at `expiry` into a `tenor`-year underlying swap.
    """
    if tenor <= 0:
        raise ValueError(f"tenor must be positive, got {tenor}")
    if expiry < 0:
        raise ValueError(f"expiry must be non-negative, got {expiry}")

    device = z0.device
    dtype = z0.dtype

    payment_times = [expiry + j for j in range(1, tenor + 1)]
    extra_points = payment_times.copy()
    if expiry > 0:
        extra_points = [expiry] + extra_points

    tau_grid = build_tau_grid_with_points(
        base_grid=decoder_tau_grid_base,
        extra_points=extra_points,
        device=device,
        dtype=dtype,
    )

    P_full, _, _, _, _, _, _, _ = decode_from_latent_script(
        model,
        z0,
        tau_grid,
        G_floor=g0_floor,
        check_short_rate=False,
    )

    payment_tensor = torch.tensor(payment_times, device=device, dtype=dtype)
    pay_idx = get_grid_indices_for_values(tau_grid, payment_tensor)
    P_payments = P_full[:, pay_idx]

    if expiry > 0:
        expiry_tensor = torch.tensor([expiry], device=device, dtype=dtype)
        start_idx = get_grid_indices_for_values(tau_grid, expiry_tensor)[0]
        P_start = P_full[:, start_idx]
    else:
        P_start = torch.ones(P_full.shape[0], device=device, dtype=dtype)

    P_end = P_payments[:, -1]
    fwd, ann = forward_start_swap_rate_and_annuity_from_discount(
        P_start=P_start,
        P_end=P_end,
        P_payments=P_payments,
        accrual=accrual,
    )

    return {
        "forward_swap": float(fwd[0].item()),
        "annuity": float(ann[0].item()),
        "forward_swaps": fwd.detach().cpu().numpy().tolist(),
        "annuities": ann.detach().cpu().numpy().tolist(),
    }


# ---------------------------------------------------------------------
# Swaption Monte Carlo
# ---------------------------------------------------------------------
def swaption_underlying_at_expiry(
    model,
    z_paths: torch.Tensor,
    r_paths: torch.Tensor,
    decoder_tau_grid_base: torch.Tensor,
    dt: float,
    expiry: float,
    tenor: int,
    g0_floor: float = 1e-5,
    accrual: float = 1.0,
    discount_paths: torch.Tensor = None,
):
    """
    Prepare pathwise quantities needed for a European swaption payoff:
      PV = D(0,T_exp) * A(T_exp) * max(S(T_exp) - K, 0)  [payer]
    """
    if tenor <= 0:
        raise ValueError(f"tenor must be positive, got {tenor}")
    if z_paths.ndim != 3:
        raise ValueError(f"Expected z_paths shape (n_paths,n_times,d), got {tuple(z_paths.shape)}")
    if r_paths.ndim != 2:
        raise ValueError(f"Expected r_paths shape (n_paths,n_times), got {tuple(r_paths.shape)}")

    n_paths, n_times, _ = z_paths.shape
    n_steps = n_times - 1
    expiry_idx, actual_expiry = expiry_index_from_time(expiry, dt, n_steps)

    device = z_paths.device
    dtype = z_paths.dtype

    z_at_expiry = z_paths[:, expiry_idx, :]

    rel_payment_times = [float(j) for j in range(1, tenor + 1)]
    tau_grid = build_tau_grid_with_points(
        base_grid=decoder_tau_grid_base,
        extra_points=rel_payment_times,
        device=device,
        dtype=dtype,
    )

    payment_tensor = torch.tensor(rel_payment_times, device=device, dtype=dtype)
    pay_idx = get_grid_indices_for_values(tau_grid, payment_tensor)

    P_full, valid = decode_discount_curve_batch_safe(
        model=model,
        z_batch=z_at_expiry,
        tau_grid=tau_grid,
        g0_floor=g0_floor,
    )

    n_valid = int(valid.sum().item())
    if n_valid == 0:
        raise RuntimeError("No valid decoded paths at swaption expiry.")

    P_payments = P_full[valid][:, pay_idx]
    swap_rate, annuity = spot_start_swap_rate_and_annuity_from_discount(
        P_payments,
        accrual=accrual,
    )

    if discount_paths is None:
        discount_paths = compute_discount_paths(r_paths, dt=dt, method="trapezoid")

    D_expiry = discount_paths[valid, expiry_idx]

    return {
        "swap_rate": swap_rate,
        "annuity": annuity,
        "discount_to_expiry": D_expiry,
        "valid_mask": valid,
        "n_valid_paths": n_valid,
        "frac_valid_paths": n_valid / n_paths,
        "expiry_idx": expiry_idx,
        "actual_expiry": actual_expiry,
        "mc_mean_swap_rate_at_expiry": float(swap_rate.mean().item()),
        "mc_mean_annuity_at_expiry": float(annuity.mean().item()),
    }


def price_swaption_from_underlying(
    underlying: dict,
    strike: float,
    notional: float = 1.0,
    payer: bool = True,
):
    swap_rate = underlying["swap_rate"]
    annuity = underlying["annuity"]
    D_expiry = underlying["discount_to_expiry"]

    if payer:
        intrinsic = torch.clamp(swap_rate - strike, min=0.0)
    else:
        intrinsic = torch.clamp(strike - swap_rate, min=0.0)

    pv_paths = notional * D_expiry * annuity * intrinsic
    pv_paths = pv_paths[torch.isfinite(pv_paths)]

    if pv_paths.numel() == 0:
        raise RuntimeError("All swaption PVs are non-finite after filtering.")

    price = float(pv_paths.mean().item())
    std = float(pv_paths.std(unbiased=False).item()) if pv_paths.numel() > 1 else 0.0
    stderr = std / math.sqrt(int(pv_paths.numel())) if pv_paths.numel() > 0 else float("nan")

    return {
        "price": price,
        "std": std,
        "stderr": stderr,
        "n_valid_pv": int(pv_paths.numel()),
    }


def price_swaption_mc(
    model,
    z_paths: torch.Tensor,
    r_paths: torch.Tensor,
    decoder_tau_grid_base: torch.Tensor,
    dt: float,
    strike: float,
    expiry: float,
    tenor: int,
    notional: float = 1.0,
    payer: bool = True,
    g0_floor: float = 1e-5,
    accrual: float = 1.0,
    discount_paths: torch.Tensor = None,
):
    underlying = swaption_underlying_at_expiry(
        model=model,
        z_paths=z_paths,
        r_paths=r_paths,
        decoder_tau_grid_base=decoder_tau_grid_base,
        dt=dt,
        expiry=expiry,
        tenor=tenor,
        g0_floor=g0_floor,
        accrual=accrual,
        discount_paths=discount_paths,
    )
    price_info = price_swaption_from_underlying(
        underlying=underlying,
        strike=strike,
        notional=notional,
        payer=payer,
    )
    return {**underlying, **price_info}


# ---------------------------------------------------------------------
# Bachelier / normal vol
# ---------------------------------------------------------------------
def bachelier_price(
    forward: float,
    strike: float,
    normal_vol: float,
    expiry: float,
    annuity: float,
    notional: float = 1.0,
    payer: bool = True,
):
    intrinsic = max(forward - strike, 0.0) if payer else max(strike - forward, 0.0)

    if expiry <= 0.0 or normal_vol <= 0.0:
        return notional * annuity * intrinsic

    vol_term = normal_vol * math.sqrt(expiry)
    if vol_term < 1e-16:
        return notional * annuity * intrinsic

    d = (forward - strike) / vol_term

    if payer:
        return notional * annuity * ((forward - strike) * norm.cdf(d) + vol_term * norm.pdf(d))
    else:
        return notional * annuity * ((strike - forward) * norm.cdf(-d) + vol_term * norm.pdf(d))


def implied_bachelier_vol(
    market_price: float,
    forward: float,
    strike: float,
    expiry: float,
    annuity: float,
    notional: float = 1.0,
    payer: bool = True,
    tol: float = 1e-12,
):
    intrinsic = notional * annuity * (max(forward - strike, 0.0) if payer else max(strike - forward, 0.0))

    if expiry <= 0.0 or annuity <= 0.0 or notional <= 0.0:
        return np.nan

    if market_price < intrinsic - tol:
        warnings.warn(
            f"Market/model price {market_price:.12f} is below intrinsic value {intrinsic:.12f}; "
            "cannot infer a Bachelier vol.",
            RuntimeWarning,
        )
        return np.nan

    if abs(market_price - intrinsic) <= tol:
        return 0.0

    def objective(sigma):
        return bachelier_price(
            forward=forward,
            strike=strike,
            normal_vol=sigma,
            expiry=expiry,
            annuity=annuity,
            notional=notional,
            payer=payer,
        ) - market_price

    lower = 1e-12
    upper = 1e-4

    price_upper = objective(upper)
    while price_upper < 0.0 and upper < 100.0:
        upper *= 2.0
        price_upper = objective(upper)

    if price_upper < 0.0:
        warnings.warn("Could not bracket implied normal vol; returning NaN.", RuntimeWarning)
        return np.nan

    try:
        return brentq(objective, lower, upper, xtol=1e-12, rtol=1e-10, maxiter=200)
    except ValueError:
        return np.nan


# ---------------------------------------------------------------------
# Vol surface grid
# ---------------------------------------------------------------------
def run_vol_surface_grid(
    model,
    z0: torch.Tensor,
    dt: float,
    notional: float,
    device,
    out_dir: str,
    strikes,
    expiries,
    tenors,
    n_paths: int,
    n_steps: int,
    discretization: str,
    decoder_tau_grid_base: torch.Tensor,
    g0_floor: float,
    payer: bool = True,
    accrual: float = 1.0,
    plot_dpi: int = 200,
    show_plots: bool = False,
):
    os.makedirs(out_dir, exist_ok=True)

    print("Simulating paths once for the full swaption surface...")
    with torch.no_grad():
        z_paths, r_paths, _, _ = simulate_latent_paths(
            model=model,
            z0=z0,
            n_paths=n_paths,
            n_steps=n_steps,
            dt=dt,
            device=device,
            discretization=discretization,
        )
    print("Simulation completed.")

    discount_paths = compute_discount_paths(r_paths, dt=dt, method="trapezoid")
    results = []

    for expiry in expiries:
        for tenor in tenors:
            row = {
                "expiry": float(expiry),
                "tenor": int(tenor),
            }

            try:
                underlying = swaption_underlying_at_expiry(
                    model=model,
                    z_paths=z_paths,
                    r_paths=r_paths,
                    decoder_tau_grid_base=decoder_tau_grid_base,
                    dt=dt,
                    expiry=float(expiry),
                    tenor=int(tenor),
                    g0_floor=g0_floor,
                    accrual=accrual,
                    discount_paths=discount_paths,
                )
                row["n_valid_paths"] = underlying["n_valid_paths"]
                row["frac_valid_paths"] = underlying["frac_valid_paths"]
                row["mc_mean_swap_rate_at_expiry"] = underlying["mc_mean_swap_rate_at_expiry"]
                row["mc_mean_annuity_at_expiry"] = underlying["mc_mean_annuity_at_expiry"]
            except Exception as e:
                warnings.warn(
                    f"Skipping expiry={expiry}, tenor={tenor} due to MC underlying failure: {e}",
                    RuntimeWarning,
                )
                continue

            try:
                t0_params = time0_forward_swap_and_annuity_from_z(
                    model=model,
                    z0=z0,
                    decoder_tau_grid_base=decoder_tau_grid_base,
                    expiry=float(expiry),
                    tenor=int(tenor),
                    g0_floor=g0_floor,
                    accrual=accrual,
                )
                row["forward_swap_t0"] = t0_params["forward_swap"]
                row["annuity_t0"] = t0_params["annuity"]
            except Exception as e:
                warnings.warn(
                    f"Could not compute time-0 forward/annuity for expiry={expiry}, tenor={tenor}: {e}",
                    RuntimeWarning,
                )
                row["forward_swap_t0"] = np.nan
                row["annuity_t0"] = np.nan

            for strike in strikes:
                price_info = price_swaption_from_underlying(
                    underlying=underlying,
                    strike=float(strike),
                    notional=notional,
                    payer=payer,
                )

                row[f"price_{strike}"] = price_info["price"]
                row[f"stderr_{strike}"] = price_info["stderr"]

                if np.isfinite(row["forward_swap_t0"]) and np.isfinite(row["annuity_t0"]):
                    vol = implied_bachelier_vol(
                        market_price=price_info["price"],
                        forward=row["forward_swap_t0"],
                        strike=float(strike),
                        expiry=float(expiry),
                        annuity=row["annuity_t0"],
                        notional=notional,
                        payer=payer,
                    )
                else:
                    vol = np.nan

                row[f"bachelier_vol_{strike}"] = vol

            results.append(row)

    df = pd.DataFrame(results)
    csv_path = os.path.join(out_dir, "swaption_bachelier_surface.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved swaption surface CSV to {csv_path}")

    for expiry in expiries:
        for tenor in tenors:
            sub = df[(df["expiry"] == float(expiry)) & (df["tenor"] == int(tenor))]
            if sub.empty:
                continue

            vols = [sub.iloc[0][f"bachelier_vol_{strike}"] for strike in strikes]

            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.plot(strikes, vols, marker="o")
            ax.set_title(f"Bachelier smile | expiry={expiry}y, tenor={tenor}y")
            ax.set_xlabel("strike")
            ax.set_ylabel("normal vol")
            ax.grid(True, alpha=0.3)

            plot_path = os.path.join(out_dir, f"bachelier_smile_exp{expiry}_ten{tenor}.png")
            fig.tight_layout()
            fig.savefig(plot_path, dpi=plot_dpi, bbox_inches="tight")
            print(f"Saved smile plot to {plot_path}")

            if show_plots:
                plt.show()
            else:
                plt.close(fig)

    return df


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Monte Carlo pricing of European swaptions + Bachelier vols")

    parser.add_argument("--latent_dim", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=2500)
    parser.add_argument("--use", type=str, default="bbg")
    parser.add_argument("--idx_choice", type=int, default=-1)

    parser.add_argument("--n_paths", type=int, default=2000)
    parser.add_argument("--n_steps", type=int, default=120)
    parser.add_argument("--dt", type=float, default=1 / 12)
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument(
        "--discretization",
        type=str,
        default="euler",
        choices=["euler", "milstein", "milstein_pc", "second_order_milstein"],
    )

    parser.add_argument("--strike", type=float, default=0.03)
    parser.add_argument("--strike_atm", action="store_true")
    parser.add_argument("--expiry", type=float, default=1.0)
    parser.add_argument("--tenor", type=int, default=5)
    parser.add_argument("--notional", type=float, default=1.0)
    parser.add_argument("--is_receiver", action="store_true")

    parser.add_argument(
        "--pricing_mode",
        type=str,
        default="monte_carlo",
        choices=["monte_carlo", "bachelier_quote"],
    )
    parser.add_argument(
        "--bachelier_vol",
        type=float,
        default=None,
        help="Input normal vol in decimal form, e.g. 0.005 = 50 bp",
    )
    parser.add_argument(
        "--output_bachelier_vol",
        action="store_true",
        help="After Monte Carlo pricing, invert to implied normal vol",
    )

    parser.add_argument("--g0_floor", type=float, default=1e-5)
    parser.add_argument("--tau_fine_step", type=float, default=1 / 52)
    parser.add_argument("--tau_fine_horizon", type=float, default=1.0)

    parser.add_argument("--run_surface", action="store_true")
    parser.add_argument("--strikes", type=str, default="0.01,0.02,0.03,0.04,0.05")
    parser.add_argument("--expiries", type=str, default="0.5,1.0,2.0,5.0")
    parser.add_argument("--tenors", type=str, default="1,2,5,10")
    parser.add_argument("--grid_n_paths", type=int, default=1000)
    parser.add_argument("--grid_n_steps", type=int, default=120)

    parser.add_argument("--plot_dpi", type=int, default=200)
    parser.add_argument("--show_plots", action="store_true")

    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(THESIS_ROOT, "Figures", "Pricing"),
    )

    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"Ignoring unknown args: {unknown}")

    if args.show_plots:
        plt.switch_backend("TkAgg")
    else:
        plt.switch_backend("Agg")

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Seed: {args.seed}")

    discretization = normalize_discretization_name(args.discretization)
    payer = not args.is_receiver

    checkpoint_path = resolve_checkpoint_path(CODE_ROOT, args.use, args.latent_dim, args.epochs)
    model = load_model(checkpoint_path, device=device, latent_dim=args.latent_dim)

    data = load_data_and_initial_curve(args.use, args.idx_choice, device)
    S0 = data["S0"]
    meta_row = data["meta_row"]
    print(f"Initial curve metadata row:\n{meta_row}")

    with torch.no_grad():
        z0 = model.encoder(S0)
    print(f"Initial latent state z0: {z0.detach().cpu().numpy().flatten()}")

    decoder_tau_grid_base = build_decoder_tau_grid(
        model=model,
        device=device,
        dtype=torch.float64,
        fine_step=args.tau_fine_step,
        fine_horizon=args.tau_fine_horizon,
    )
    print(
        "Decoder tau grid built with "
        f"{decoder_tau_grid_base.numel()} points; first positive tau = "
        f"{decoder_tau_grid_base[1].item():.6f}, "
        f"tau_max = {decoder_tau_grid_base[-1].item():.6f}"
    )

    # -------------------------------------------------------------
    # Time-0 quote inputs
    # -------------------------------------------------------------
    t0_quote = time0_forward_swap_and_annuity_from_z(
        model=model,
        z0=z0,
        decoder_tau_grid_base=decoder_tau_grid_base,
        expiry=args.expiry,
        tenor=args.tenor,
        g0_floor=args.g0_floor,
        accrual=1.0,
    )

    strike = args.strike
    if args.strike_atm:
        strike = t0_quote["forward_swap"]

    intrinsic_lb = args.notional * t0_quote["annuity"] * max(t0_quote["forward_swap"] - strike, 0.0) \
        if payer else args.notional * t0_quote["annuity"] * max(strike - t0_quote["forward_swap"], 0.0)

    print("\nTime-0 swaption quote inputs")
    print(f"  Forward swap rate F0 : {t0_quote['forward_swap']:.10f}")
    print(f"  Annuity A0           : {t0_quote['annuity']:.10f}")
    if args.strike_atm:
        print(f"  Strike K             : {strike:.10f} (ATM = F0)")
    else:
        print(f"  Strike K             : {strike:.10f}")
    print(f"  Intrinsic lower bound: {intrinsic_lb:.10f}")

    # -------------------------------------------------------------
    # Quote-style pricing from Bachelier vol
    # -------------------------------------------------------------
    if args.pricing_mode == "bachelier_quote":
        if args.bachelier_vol is None:
            raise ValueError("--bachelier_vol is required when pricing_mode=bachelier_quote")

        price = bachelier_price(
            forward=t0_quote["forward_swap"],
            strike=strike,
            normal_vol=args.bachelier_vol,
            expiry=args.expiry,
            annuity=t0_quote["annuity"],
            notional=args.notional,
            payer=payer,
        )

        swaption_type = "Receiver" if args.is_receiver else "Payer"
        print(f"\n{swaption_type} swaption price from Bachelier quote")
        print(f"  Price           : {price:.10f}")
        print(f"  Forward swap t0 : {t0_quote['forward_swap']:.10f}")
        print(f"  Annuity t0      : {t0_quote['annuity']:.10f}")
        print(f"  Strike          : {strike:.10f}")
        print(f"  Input norm vol  : {args.bachelier_vol:.10f} ({args.bachelier_vol * 10000:.2f} bp)")

    # -------------------------------------------------------------
    # Monte Carlo pricing from your simulated model
    # -------------------------------------------------------------
    else:
        print(
            f"Simulating {args.n_paths} paths with {args.n_steps} steps "
            f"(dt={args.dt}, scheme={discretization})..."
        )
        with torch.no_grad():
            z_paths, r_paths, _, _ = simulate_latent_paths(
                model=model,
                z0=z0,
                n_paths=args.n_paths,
                n_steps=args.n_steps,
                dt=args.dt,
                device=device,
                discretization=discretization,
            )
        print("Simulation completed.")

        discount_paths = compute_discount_paths(r_paths, dt=args.dt, method="trapezoid")

        mc = price_swaption_mc(
            model=model,
            z_paths=z_paths,
            r_paths=r_paths,
            decoder_tau_grid_base=decoder_tau_grid_base,
            dt=args.dt,
            strike=strike,
            expiry=args.expiry,
            tenor=args.tenor,
            notional=args.notional,
            payer=payer,
            g0_floor=args.g0_floor,
            accrual=1.0,
            discount_paths=discount_paths,
        )

        swaption_type = "Receiver" if args.is_receiver else "Payer"
        print(f"\n{swaption_type} swaption Monte Carlo result")
        print(f"  Price                 : {mc['price']:.10f}")
        print(f"  MC std. error         : {mc['stderr']:.10f}")
        print(f"  Valid decode fraction : {mc['frac_valid_paths']:.4f}")
        print(f"  Used expiry time      : {mc['actual_expiry']:.6f}")
        print(f"  Mean swap rate @ exp  : {mc['mc_mean_swap_rate_at_expiry']:.10f}")
        print(f"  Mean annuity @ exp    : {mc['mc_mean_annuity_at_expiry']:.10f}")
        print(f"  Strike used           : {strike:.10f}")

        if args.output_bachelier_vol:
            norm_vol = implied_bachelier_vol(
                market_price=mc["price"],
                forward=t0_quote["forward_swap"],
                strike=strike,
                expiry=args.expiry,
                annuity=t0_quote["annuity"],
                notional=args.notional,
                payer=payer,
            )

            if np.isfinite(norm_vol):
                print("\nImplied Bachelier normal vol from model price")
                print(f"  Forward swap t0 : {t0_quote['forward_swap']:.10f}")
                print(f"  Annuity t0      : {t0_quote['annuity']:.10f}")
                print(f"  Strike          : {strike:.10f}")
                print(f"  Normal vol      : {norm_vol:.10f} ({norm_vol * 10000:.2f} bp)")
            else:
                print("\nCould not infer Bachelier vol from model price.")

    w = mc["discount_to_expiry"] * mc["annuity"]
    mc_A0 = w.mean().item()
    mc_F0 = (w * mc["swap_rate"]).mean().item() / max(w.mean().item(), 1e-16)
    prob_itm = (mc["swap_rate"] > strike).double().mean().item()

    q_levels = torch.tensor([0.01, 0.05, 0.50, 0.95, 0.99], device=mc["swap_rate"].device, dtype=mc["swap_rate"].dtype)
    q_vals = torch.quantile(mc["swap_rate"], q_levels).detach().cpu().numpy()

    print("\nMonte Carlo consistency diagnostics")
    print(f"  A0 from time-0 curve     : {t0_quote['annuity']:.10f}")
    print(f"  E[D_T A_T] from MC       : {mc_A0:.10f}")
    print(f"  F0 from time-0 curve     : {t0_quote['forward_swap']:.10f}")
    print(f"  E[D_T A_T S_T]/E[D_T A_T]: {mc_F0:.10f}")
    print(f"  Prob(S_T > K)            : {prob_itm:.10f}")
    print("  Swap-rate quantiles @ expiry:")
    print(f"    1%  : {q_vals[0]:.10f}")
    print(f"    5%  : {q_vals[1]:.10f}")
    print(f"    50% : {q_vals[2]:.10f}")
    print(f"    95% : {q_vals[3]:.10f}")
    print(f"    99% : {q_vals[4]:.10f}")

    # -------------------------------------------------------------
    # Surface
    # -------------------------------------------------------------
    if args.run_surface:
        strikes = parse_float_list(args.strikes)
        expiries = parse_float_list(args.expiries)
        tenors = [int(round(x)) for x in parse_float_list(args.tenors)]

        print("\nRunning swaption surface...")
        run_vol_surface_grid(
            model=model,
            z0=z0,
            dt=args.dt,
            notional=args.notional,
            device=device,
            out_dir=args.output_dir,
            strikes=strikes,
            expiries=expiries,
            tenors=tenors,
            n_paths=args.grid_n_paths,
            n_steps=args.grid_n_steps,
            discretization=discretization,
            decoder_tau_grid_base=decoder_tau_grid_base,
            g0_floor=args.g0_floor,
            payer=payer,
            accrual=1.0,
            plot_dpi=args.plot_dpi,
            show_plots=args.show_plots,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()