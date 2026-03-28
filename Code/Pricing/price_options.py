import os
import sys
import argparse
import warnings

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize_scalar

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
CODE_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CODE_ROOT, ".."))

if CODE_ROOT not in sys.path:
    sys.path.insert(0, CODE_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------
# Force stable variant BEFORE importing FullModel / simulation helpers
# ---------------------------------------------------------------------
from Code import config
config.VARIANT = "stable"

from Code.load_swapdata import my_data
from Code.model.full_model import FullModel

# Import shared simulation / decode logic from your simulation file
try:
    from simulate_model import (
        load_initial_curve,
        simulate_latent_paths,
        decode_from_latent_script,
    )
except ImportError:
    from Code.Pricing.simulate_model import (
        load_initial_curve,
        simulate_latent_paths,
        decode_from_latent_script,
    )

SHOW_PLOTS = True


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def tenor_label(tenor_value):
    return f"{int(tenor_value)}Y"


def load_model(checkpoint_path: str, device: torch.device) -> FullModel:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_config" in checkpoint:
        model_config = checkpoint["model_config"]
        print(f"Loading model with saved configuration: {model_config}")
        model = FullModel(**model_config)
    else:
        print("[WARNING] Checkpoint missing 'model_config'. Using defaults.")
        model = FullModel(latent_dim=checkpoint["latent_dim"])

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device).double()
    model.eval()

    saved_variant = checkpoint.get("variant", "unknown")
    print(f"Loaded model from {checkpoint_path}")
    print(f"  Variant: {saved_variant}")
    print(f"  Model dtype: {next(model.parameters()).dtype}")
    return model


def discount_factors_from_short_rate_paths(r_paths: torch.Tensor, dt: float) -> torch.Tensor:
    """
    Pathwise discount factors D(t_i) = exp(-integral_0^{t_i} r(u) du)
    using trapezoidal integration on the simulated short-rate grid.

    Args:
        r_paths: (n_paths, n_steps+1)
        dt: time step

    Returns:
        D: (n_paths, n_steps+1), with D[:,0] = 1
    """
    n_paths, n_times = r_paths.shape
    D = torch.ones((n_paths, n_times), device=r_paths.device, dtype=r_paths.dtype)

    if n_times > 1:
        increments = 0.5 * dt * (r_paths[:, 1:] + r_paths[:, :-1])  # (n_paths, n_times-1)
        integral = torch.cumsum(increments, dim=1)
        D[:, 1:] = torch.exp(-integral)

    return D


def spot_swap_and_annuity_from_discount(P_mkt: torch.Tensor, tenor: int):
    """
    Spot-start par swap and annuity from discount factors observed at a given date.

    P_mkt[:,0] = P(t, t+1), ..., P_mkt[:, tenor-1] = P(t, t+tenor)
    """
    if tenor <= 0:
        raise ValueError(f"tenor must be positive, got {tenor}")
    if tenor > P_mkt.shape[1]:
        raise ValueError(f"tenor={tenor} exceeds available discount grid up to {P_mkt.shape[1]}Y")

    annuity = P_mkt[:, :tenor].sum(dim=1)
    terminal_df = P_mkt[:, tenor - 1]
    forward_swap = (1.0 - terminal_df) / annuity
    return forward_swap, annuity


def forward_start_swap_and_annuity_from_discount(P_mkt: torch.Tensor, start_idx: int, tenor: int):
    """
    Forward-starting swap and annuity from time-0 discount curve.

    Args:
        P_mkt: (B, tau_max) = [P(0,1), P(0,2), ..., P(0,tau_max)]
        start_idx: integer start in years (0 means spot-start)
        tenor: swap tenor in years
    """
    if start_idx < 0:
        raise ValueError(f"start_idx must be non-negative, got {start_idx}")
    if tenor <= 0:
        raise ValueError(f"tenor must be positive, got {tenor}")
    if start_idx + tenor > P_mkt.shape[1]:
        raise ValueError(
            f"start_idx + tenor = {start_idx + tenor} exceeds available grid up to {P_mkt.shape[1]}Y"
        )

    if start_idx == 0:
        P_start = torch.ones(P_mkt.shape[0], device=P_mkt.device, dtype=P_mkt.dtype)
    else:
        P_start = P_mkt[:, start_idx - 1]

    P_end = P_mkt[:, start_idx + tenor - 1]
    annuity = P_mkt[:, start_idx:start_idx + tenor].sum(dim=1)
    forward_swap = (P_start - P_end) / annuity
    return forward_swap, annuity


def extract_forward_swap_curve_params(model, z, expiry, tenor):
    """
    Extract time-0 forward-starting swap rate and annuity from today's decoded curve.

    This is the appropriate input for quote-style Bachelier pricing / implied vol,
    but only when expiry is on the model's annual grid.
    """
    expiry_int = int(round(expiry))
    if abs(expiry - expiry_int) > 1e-8:
        raise ValueError(
            f"expiry={expiry} is not on the model annual grid; time-0 forward extraction "
            f"currently requires integer-year expiry"
        )
    if expiry_int < 0:
        raise ValueError(f"expiry must be non-negative, got {expiry}")

    with torch.no_grad():
        P_mkt, _, _, _, _, _, _, _ = decode_from_latent_script(model, z)

    forward_swap, annuity = forward_start_swap_and_annuity_from_discount(P_mkt, expiry_int, tenor)

    return {
        "forward_swap": forward_swap.mean().item(),
        "annuity": annuity.mean().item(),
        "forward_swaps": forward_swap.detach().cpu().numpy().tolist(),
        "annuities": annuity.detach().cpu().numpy().tolist(),
    }


def extract_market_params_at_expiry(z_paths, model, dt, expiry, tenor):
    """
    Pathwise average spot-start swap rate / annuity at option expiry.
    This is a fallback for non-integer expiries when quote-style time-0 forward
    extraction is unavailable on the annual grid.
    """
    n_steps = z_paths.shape[1] - 1
    expiry_idx = min(int(round(expiry / dt)), n_steps)

    z_at_expiry = z_paths[:, expiry_idx, :]  # (n_paths, d)

    with torch.no_grad():
        P_mkt, _, _, _, _, _, _, _ = decode_from_latent_script(model, z_at_expiry)
        forward_swap, annuity = spot_swap_and_annuity_from_discount(P_mkt, tenor)

    return {
        "forward_swap": forward_swap.mean().item(),
        "annuity": annuity.mean().item(),
        "forward_swaps": forward_swap.detach().cpu().numpy().tolist(),
        "annuities": annuity.detach().cpu().numpy().tolist(),
    }


# ---------------------------------------------------------------------
# Product pricing
# ---------------------------------------------------------------------
def price_cap(r_paths: torch.Tensor, dt: float, strike: float, notional: float = 1.0) -> float:
    """
    Price a cap on the short rate using pathwise discounting.

    Payoff at each reset date t_i:
        max(r(t_i) - K, 0) * dt * notional
    """
    D = discount_factors_from_short_rate_paths(r_paths, dt)  # (n_paths, n_steps+1)

    payoffs = torch.clamp(r_paths[:, 1:] - strike, min=0.0) * dt * notional
    pv = (D[:, 1:] * payoffs).sum(dim=1)

    return pv.mean().item()


def price_swaption(
    z_paths: torch.Tensor,
    r_paths: torch.Tensor,
    model,
    dt: float,
    strike: float,
    expiry: float,
    tenor: int,
    notional: float = 1.0,
    is_call: bool = True,
) -> float:
    """
    Price a European payer/receiver swaption by Monte Carlo.

    At expiry:
      - decode the curve from z(expiry)
      - compute the spot-start underlying swap rate of tenor years
      - apply payer/receiver payoff
      - discount back using pathwise integrated short rate
    """
    n_steps = z_paths.shape[1] - 1
    total_time = n_steps * dt

    if expiry > total_time:
        warnings.warn(
            f"expiry={expiry} exceeds simulated horizon={total_time}; using last step instead",
            RuntimeWarning
        )
        expiry_idx = n_steps
    else:
        expiry_idx = min(int(round(expiry / dt)), n_steps)

    z_at_expiry = z_paths[:, expiry_idx, :]  # (n_paths, d)

    with torch.no_grad():
        P_mkt, _, _, _, _, _, _, _ = decode_from_latent_script(model, z_at_expiry)
        swap_rate, annuity = spot_swap_and_annuity_from_discount(P_mkt, tenor)

    if is_call:
        payoff = torch.clamp(swap_rate - strike, min=0.0)
    else:
        payoff = torch.clamp(strike - swap_rate, min=0.0)

    payoff = payoff * annuity * notional

    D = discount_factors_from_short_rate_paths(r_paths, dt)
    pv = D[:, expiry_idx] * payoff

    pv = pv[torch.isfinite(pv)]
    if pv.numel() == 0:
        warnings.warn("All swaption PVs are non-finite", RuntimeWarning)
        return 0.0

    return pv.mean().item()


# ---------------------------------------------------------------------
# Bachelier / normal vol
# ---------------------------------------------------------------------
def bachelier_price(forward, strike, sigma, expiry, annuity, notional, is_call=True):
    if sigma <= 0 or expiry <= 0:
        intrinsic = max(forward - strike, 0.0) if is_call else max(strike - forward, 0.0)
        return annuity * notional * intrinsic

    vol_term = sigma * np.sqrt(expiry)
    if vol_term < 1e-12:
        intrinsic = max(forward - strike, 0.0) if is_call else max(strike - forward, 0.0)
        return annuity * notional * intrinsic

    d = (forward - strike) / vol_term

    if is_call:
        return annuity * notional * ((forward - strike) * norm.cdf(d) + vol_term * norm.pdf(d))
    return annuity * notional * ((strike - forward) * norm.cdf(-d) + vol_term * norm.pdf(d))


def implied_normal_vol(market_price, forward, strike, expiry, annuity, notional, is_call=True):
    def objective(sigma):
        theoretical = bachelier_price(forward, strike, sigma, expiry, annuity, notional, is_call)
        return (theoretical - market_price) ** 2

    result = minimize_scalar(objective, bounds=(1e-8, 10.0), method="bounded")
    return result.x if result.success else np.nan


def price_swaption_from_norm_vol(forward, strike, norm_vol, expiry, annuity, notional=1.0, is_call=True):
    return bachelier_price(forward, strike, norm_vol, expiry, annuity, notional, is_call)


def choose_implied_vol_inputs(model, z0, z_paths, dt, expiry, tenor):
    """
    Use time-0 forward-start parameters when expiry is on the annual grid.
    Otherwise fall back to pathwise average parameters at expiry.
    """
    expiry_int = int(round(expiry))
    if abs(expiry - expiry_int) < 1e-8:
        params = extract_forward_swap_curve_params(model, z0, expiry, tenor)
        params["source"] = "time0_forward_curve"
        return params

    warnings.warn(
        f"expiry={expiry} is off the annual grid; using pathwise average expiry parameters "
        f"for implied vol inversion instead of time-0 forward-starting parameters",
        RuntimeWarning
    )
    params = extract_market_params_at_expiry(z_paths, model, dt, expiry, tenor)
    params["source"] = "pathwise_expiry_average"
    return params


# ---------------------------------------------------------------------
# Vol surface grid
# ---------------------------------------------------------------------
def run_vol_surface_grid(
    model,
    z0,
    dt,
    notional,
    device,
    out_dir,
    strikes,
    expiries,
    tenors,
    n_paths,
    n_steps,
    simple_diffusion=False,
    kappa=0.5,
    theta=0.0,
    sigma_simple=0.1,
    discretization="euler",
    is_call=True,
):
    os.makedirs(out_dir, exist_ok=True)

    print("Simulating paths once for the full vol grid...")
    with torch.no_grad():
        z_paths, r_paths, _, _ = simulate_latent_paths(
            model=model,
            z0=z0,
            n_paths=n_paths,
            n_steps=n_steps,
            dt=dt,
            device=device,
            simple_diffusion=simple_diffusion,
            kappa=kappa,
            theta=theta,
            sigma_simple=sigma_simple,
            discretization=discretization,
        )

    results = []

    for expiry in expiries:
        for tenor in tenors:
            row = {"expiry": expiry, "tenor": tenor}

            price_cache = {}
            for strike in strikes:
                price = price_swaption(
                    z_paths=z_paths,
                    r_paths=r_paths,
                    model=model,
                    dt=dt,
                    strike=strike,
                    expiry=expiry,
                    tenor=tenor,
                    notional=notional,
                    is_call=is_call,
                )
                price_cache[strike] = price
                row[f"price_{strike}"] = price

            vol_inputs = choose_implied_vol_inputs(model, z0, z_paths, dt, expiry, tenor)
            forward_swap = vol_inputs["forward_swap"]
            annuity = vol_inputs["annuity"]
            row["vol_input_source"] = vol_inputs["source"]
            row["forward_swap"] = forward_swap
            row["annuity"] = annuity

            for strike in strikes:
                norm_vol = implied_normal_vol(
                    market_price=price_cache[strike],
                    forward=forward_swap,
                    strike=strike,
                    expiry=expiry,
                    annuity=annuity,
                    notional=notional,
                    is_call=is_call,
                )
                row[f"vol_{strike}"] = norm_vol

            results.append(row)

    df = pd.DataFrame(results)
    csv_path = os.path.join(out_dir, "implied_vol_surface.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved implied vol surface CSV to {csv_path}")

    for expiry in expiries:
        for tenor in tenors:
            df_row = df[(df["expiry"] == expiry) & (df["tenor"] == tenor)]
            if df_row.empty:
                continue

            vols = [df_row.iloc[0][f"vol_{strike}"] for strike in strikes]

            plt.figure(figsize=(7, 4))
            plt.plot(strikes, vols, marker="o")
            plt.title(f"Implied Normal Vol Smile\nExpiry={expiry}y, Tenor={tenor}y")
            plt.xlabel("Strike")
            plt.ylabel("Implied Normal Volatility")
            plt.grid(True)

            plot_path = os.path.join(out_dir, f"vol_smile_exp{expiry}_ten{tenor}.png")
            plt.savefig(plot_path, dpi=200)
            print(f"Saved vol smile plot to {plot_path}")

            if SHOW_PLOTS:
                plt.show()
            plt.close()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Price options using simulated paths from FullModel")
    parser.add_argument("--latent_dim", type=int, default=2, help="Latent dimension")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--use", type=str, default="bbg", help="Data source")
    parser.add_argument("--n_paths", type=int, default=1000, help="Number of simulation paths")
    parser.add_argument("--n_steps", type=int, default=120, help="Number of time steps")
    parser.add_argument("--dt", type=float, default=1 / 12, help="Time step size")
    parser.add_argument("--idx_choice", type=int, default=-1, help="Index of initial curve")

    parser.add_argument("--option_type", type=str, choices=["cap", "swaption"], default="cap")
    parser.add_argument("--strike", type=float, default=0.03, help="Strike rate")
    parser.add_argument("--notional", type=float, default=1.0, help="Notional amount")

    parser.add_argument("--simple_diffusion", action="store_true", help="Use simple OU diffusion")
    parser.add_argument("--kappa", type=float, default=0.5, help="Mean reversion for simple diffusion")
    parser.add_argument("--theta", type=float, default=0.0, help="Long-run mean for simple diffusion")
    parser.add_argument("--sigma_simple", type=float, default=0.1, help="Volatility for simple diffusion")
    parser.add_argument(
        "--discretization",
        type=str,
        default="euler",
        choices=["euler", "milstein", "second_order_milstein"],
        help="Discretization scheme for latent SDE",
    )

    parser.add_argument("--expiry", type=float, default=1.0, help="Expiry for swaption")
    parser.add_argument("--tenor", type=int, default=5, help="Tenor for swaption")
    parser.add_argument("--output_norm_vol", action="store_true", help="Output implied normal volatility")
    parser.add_argument(
        "--norm_vol",
        type=float,
        default=None,
        help="Input normal volatility for direct Bachelier pricing (decimal, e.g. 0.01)",
    )
    parser.add_argument(
        "--pricing_mode",
        type=str,
        choices=["monte_carlo", "norm_vol_quote"],
        default="monte_carlo",
        help="Pricing mode",
    )
    parser.add_argument("--is_receiver", action="store_true", help="Receiver swaption instead of payer")
    parser.add_argument("--run_vol_surface_grid", action="store_true", help="Run grid of swaption pricings")

    parser.add_argument("--strikes", type=str, default="0.01,0.02,0.03,0.04,0.05")
    parser.add_argument("--expiries", type=str, default="0.5,1.0,1.5,2.0")
    parser.add_argument("--tenors", type=str, default="1,2,3,4,5")
    parser.add_argument("--grid_n_paths", type=int, default=100)
    parser.add_argument("--grid_n_steps", type=int, default=60)
    parser.add_argument("--output_dir", type=str, default=".")

    args = parser.parse_args()

    LATENT_DIM = args.latent_dim
    EPOCHS = args.epochs
    USE = args.use
    N_PATHS = args.n_paths
    N_STEPS = args.n_steps
    DT = args.dt
    IDX_CHOICE = args.idx_choice
    OPTION_TYPE = args.option_type
    STRIKE = args.strike
    NOTIONAL = args.notional
    SIMPLE_DIFFUSION = args.simple_diffusion
    KAPPA = args.kappa
    THETA = args.theta
    SIGMA_SIMPLE = args.sigma_simple
    DISCRETIZATION = args.discretization
    EXPIRY = args.expiry
    TENOR = args.tenor
    OUTPUT_NORM_VOL = args.output_norm_vol
    PRICING_MODE = args.pricing_mode
    NORM_VOL_INPUT = args.norm_vol
    IS_CALL = not args.is_receiver

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Keep this if you want access to tenors / scaling metadata
    meta, X_tensor, meta_full, X_tensor_full, tenors, df_wide, df_wide_all, SCALE_IS_PERCENT = my_data(use=USE)

    checkpoint_path = os.path.join(PROJECT_ROOT, "checkpoints", f"fullmodel_{USE}_dim{LATENT_DIM}_ep{EPOCHS}.pt")
    model = load_model(checkpoint_path, device)

    S0, meta_row, X_tensor, meta = load_initial_curve(USE, IDX_CHOICE, device)
    with torch.no_grad():
        z0 = model.encoder(S0)
    print(f"Initial latent state z0: {z0.detach().cpu().numpy().flatten()}")

    if OPTION_TYPE == "cap":
        print(f"Simulating {N_PATHS} paths with {N_STEPS} steps (dt={DT}, scheme={DISCRETIZATION})...")
        with torch.no_grad():
            z_paths, r_paths, _, _ = simulate_latent_paths(
                model=model,
                z0=z0,
                n_paths=N_PATHS,
                n_steps=N_STEPS,
                dt=DT,
                device=device,
                simple_diffusion=SIMPLE_DIFFUSION,
                kappa=KAPPA,
                theta=THETA,
                sigma_simple=SIGMA_SIMPLE,
                discretization=DISCRETIZATION,
            )
        print("Simulation completed.")

        price = price_cap(r_paths, DT, STRIKE, NOTIONAL)
        print(f"Cap price: {price:.6f} (strike={STRIKE}, notional={NOTIONAL})")

    elif OPTION_TYPE == "swaption":
        if PRICING_MODE == "norm_vol_quote":
            if NORM_VOL_INPUT is None:
                raise ValueError("--norm_vol is required when pricing_mode=norm_vol_quote")

            print("Extracting time-0 forward swap and annuity from current curve...")
            market_params = extract_forward_swap_curve_params(model, z0, EXPIRY, TENOR)
            forward_swap = market_params["forward_swap"]
            annuity = market_params["annuity"]

            print(f"  Forward swap rate: {forward_swap:.6f}")
            print(f"  Annuity factor:    {annuity:.6f}")
            print(f"  Input norm vol:    {NORM_VOL_INPUT:.6f} ({NORM_VOL_INPUT * 10000:.2f} bp)")

            price = price_swaption_from_norm_vol(
                forward=forward_swap,
                strike=STRIKE,
                norm_vol=NORM_VOL_INPUT,
                expiry=EXPIRY,
                annuity=annuity,
                notional=NOTIONAL,
                is_call=IS_CALL,
            )
            swaption_type = "Payer" if IS_CALL else "Receiver"
            print(f"{swaption_type} swaption price (from norm vol): {price:.6f}")
            print(f"  (strike={STRIKE}, expiry={EXPIRY}, tenor={TENOR}, notional={NOTIONAL})")

        else:
            print(f"Simulating {N_PATHS} paths with {N_STEPS} steps (dt={DT}, scheme={DISCRETIZATION})...")
            with torch.no_grad():
                z_paths, r_paths, _, _ = simulate_latent_paths(
                    model=model,
                    z0=z0,
                    n_paths=N_PATHS,
                    n_steps=N_STEPS,
                    dt=DT,
                    device=device,
                    simple_diffusion=SIMPLE_DIFFUSION,
                    kappa=KAPPA,
                    theta=THETA,
                    sigma_simple=SIGMA_SIMPLE,
                    discretization=DISCRETIZATION,
                )
            print("Simulation completed.")

            price = price_swaption(
                z_paths=z_paths,
                r_paths=r_paths,
                model=model,
                dt=DT,
                strike=STRIKE,
                expiry=EXPIRY,
                tenor=TENOR,
                notional=NOTIONAL,
                is_call=IS_CALL,
            )

            swaption_type = "Payer" if IS_CALL else "Receiver"
            print(
                f"{swaption_type} swaption price (Monte Carlo): {price:.6f} "
                f"(strike={STRIKE}, expiry={EXPIRY}, tenor={TENOR}, notional={NOTIONAL})"
            )

            if OUTPUT_NORM_VOL or NORM_VOL_INPUT is not None:
                vol_inputs = choose_implied_vol_inputs(model, z0, z_paths, DT, EXPIRY, TENOR)
                forward_swap = vol_inputs["forward_swap"]
                annuity = vol_inputs["annuity"]

                norm_vol = implied_normal_vol(
                    market_price=price,
                    forward=forward_swap,
                    strike=STRIKE,
                    expiry=EXPIRY,
                    annuity=annuity,
                    notional=NOTIONAL,
                    is_call=IS_CALL,
                )

                if np.isfinite(norm_vol):
                    print(f"Implied normal volatility: {norm_vol:.6f} ({norm_vol * 10000:.2f} bp)")
                    print(f"Implied vol inputs source: {vol_inputs['source']}")

                    if NORM_VOL_INPUT is not None:
                        diff = abs(norm_vol - NORM_VOL_INPUT)
                        print(f"Input normal volatility: {NORM_VOL_INPUT:.6f} ({NORM_VOL_INPUT * 10000:.2f} bp)")
                        print(f"Difference: {diff:.6f} ({diff * 10000:.2f} bp)")
                else:
                    print("Could not compute implied normal volatility (optimization failed)")

    if args.run_vol_surface_grid:
        strikes = [float(s) for s in args.strikes.split(",")]
        expiries = [float(e) for e in args.expiries.split(",")]
        tenors_grid = [int(t) for t in args.tenors.split(",")]
        output_dir = args.output_dir

        print("Running vol surface grid pricing...")
        run_vol_surface_grid(
            model=model,
            z0=z0,
            dt=DT,
            notional=NOTIONAL,
            device=device,
            out_dir=output_dir,
            strikes=strikes,
            expiries=expiries,
            tenors=tenors_grid,
            n_paths=args.grid_n_paths,
            n_steps=args.grid_n_steps,
            simple_diffusion=SIMPLE_DIFFUSION,
            kappa=KAPPA,
            theta=THETA,
            sigma_simple=SIGMA_SIMPLE,
            discretization=DISCRETIZATION,
            is_call=IS_CALL,
        )

    print("Pricing completed.")


if __name__ == "__main__":
    main()

