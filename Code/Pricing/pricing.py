import os
import sys
import math
import random
import warnings

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq

# ---------------------------------------------------------------------
# User setup
# ---------------------------------------------------------------------
USE_PRICING_CHECKPOINT = True
PRICING_RUN_NAME = "pricing_ep200"

USE = "bbg"
LATENT_DIM = 2
EPOCHS = 200
IDX_CHOICE = 0
CCY_FILTER = "EUR"
SEED = 1234

# Simulation defaults
N_PATHS = 2000
N_STEPS = 120
DT = 1 / 12
DISCRETIZATION = "euler"
SIM_MODE = "full"
DIFFUSION_SCALE = 1.0

# Swaption defaults
EXPIRY = 1.0
TENOR = 5
STRIKE = 0.03
STRIKE_ATM = True
NOTIONAL = 1.0
PAYER = True
G0_FLOOR = 1e-5
ACCRUAL = 1.0

# What to run when the file is executed directly
RUN_T0_QUOTE = True
RUN_MC_PRICE = True
RUN_SURFACE = False

# Surface settings
SURFACE_STRIKES = "0.01,0.02,0.03,0.04,0.05"
SURFACE_EXPIRIES = "0.5,1.0,2.0,5.0"
SURFACE_TENORS = "1,2,5,10"
SURFACE_OUT_DIR = None
PLOT_DPI = 200
SHOW_PLOTS = False

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
try:
    CODE_ROOT = os.path.dirname(os.path.abspath(__file__))
except NameError:
    CODE_ROOT = os.getcwd()

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

# ---------------------------------------------------------------------
# Import helpers strictly from simulate_model_naive
# ---------------------------------------------------------------------
from Code.Pricing.simulate_model_naive import (
    load_data,
    simulate_latent_paths,
    decode_from_latent_script,
    build_decoder_tau_grid,
    compute_discount_paths,
    get_grid_indices_for_values,
)


def normalize_discretization_name(name: str) -> str:
    name = str(name).strip().lower()
    if name not in {"euler", "milstein"}:
        raise ValueError("discretization must be 'euler' or 'milstein'")
    return name


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


def resolve_checkpoint_path_current(
    thesis_root: str,
    use: str,
    latent_dim: int,
    epochs: int,
    use_pricing_checkpoint: bool = False,
    pricing_run_name: str = "pricing_ep200",
    explicit_checkpoint_path: str = None,
) -> str:
    if explicit_checkpoint_path:
        if os.path.exists(explicit_checkpoint_path):
            return os.path.abspath(explicit_checkpoint_path)
        raise FileNotFoundError(f"Explicit checkpoint path does not exist: {explicit_checkpoint_path}")

    variant = config.VARIANT
    base_dir = os.path.join(
        thesis_root,
        "Figures",
        "TrainingResults",
        f"dim{latent_dim}_{variant}",
    )

    if use_pricing_checkpoint:
        candidates = [
            os.path.join(base_dir, pricing_run_name, "full_checkpoint.pt"),
            os.path.join(base_dir, pricing_run_name, f"checkpoint_dim{latent_dim}_{pricing_run_name}.pt"),
            os.path.join(base_dir, pricing_run_name, f"best_checkpoint_dim{latent_dim}.pt"),
        ]
    else:
        candidates = [
            os.path.join(base_dir, f"ep{epochs}", f"checkpoint_dim{latent_dim}_ep{epochs}.pt"),
            os.path.join(base_dir, f"ep{epochs}", f"best_checkpoint_dim{latent_dim}.pt"),
            os.path.join(thesis_root, "..", "checkpoints", f"fullmodel_{use}_dim{latent_dim}_ep{epochs}.pt"),
        ]

    for path in candidates:
        if os.path.exists(path):
            return os.path.abspath(path)

    searched = "\n".join(f"  - {os.path.abspath(p)}" for p in candidates)
    raise FileNotFoundError(f"Checkpoint not found. Searched:\n{searched}")


def maybe_upgrade_old_stable_k_state_dict(state_dict, model):
    """
    Convert old stable-K checkpoint:
        mu(z) = M z + N
    into new OU-style stable-K:
        mu(z) = M (z - theta)

    using:
        -M theta = N  =>  theta = solve(M, -N)

    and set kappa = 1, i.e. softplus(raw_kappa) = 1.
    """
    if config.VARIANT != "stable":
        return state_dict

    has_old = ("K.N" in state_dict) and ("K.B" in state_dict) and ("K.L" in state_dict)
    has_new_model = hasattr(model.K, "theta") and hasattr(model.K, "raw_kappa")

    if not (has_old and has_new_model):
        return state_dict

    print("[compat] Converting old stable-K checkpoint to new OU-style K...")

    new_state = dict(state_dict)

    B = new_state["K.B"]
    L = new_state["K.L"]
    N = new_state["K.N"]

    d = B.shape[0]
    device = B.device
    dtype = B.dtype

    eps = float(getattr(model.K, "epsilon", 1e-3))
    I = torch.eye(d, device=device, dtype=dtype)

    S = B - B.t()
    A = L @ L.t() + eps * I
    M = S - A

    theta = torch.linalg.solve(M, (-N).unsqueeze(-1)).squeeze(-1)

    one = torch.tensor(1.0, device=device, dtype=dtype)
    raw_kappa = torch.log(torch.expm1(one))

    new_state["K.theta"] = theta
    new_state["K.raw_kappa"] = raw_kappa
    del new_state["K.N"]

    print(f"[compat] Created K.theta = {theta.detach().cpu().numpy()}")
    print("[compat] Set K.raw_kappa so that softplus(raw_kappa) = 1")

    return new_state


def safe_load_state_dict_compat(model, state_dict):
    incompat = model.load_state_dict(state_dict, strict=False)

    missing = list(incompat.missing_keys)
    unexpected = list(incompat.unexpected_keys)

    allowed_missing = set()
    allowed_unexpected = set()

    real_missing = [k for k in missing if k not in allowed_missing]
    real_unexpected = [k for k in unexpected if k not in allowed_unexpected]

    if missing:
        print(f"[load_state_dict] Missing keys: {missing}")
    if unexpected:
        print(f"[load_state_dict] Unexpected keys: {unexpected}")

    if real_missing or real_unexpected:
        raise RuntimeError(
            "Non-benign checkpoint/model mismatch detected.\n"
            f"Real missing keys: {real_missing}\n"
            f"Real unexpected keys: {real_unexpected}"
        )


def build_model_init_kwargs(raw_checkpoint, latent_dim: int):
    model_kwargs = {"latent_dim": int(latent_dim)}

    if isinstance(raw_checkpoint, dict) and "model_config" in raw_checkpoint:
        cfg = raw_checkpoint["model_config"]
        model_kwargs["latent_dim"] = int(cfg.get("latent_dim", model_kwargs["latent_dim"]))

        if cfg.get("sigma_init", None) is not None:
            model_kwargs["sigma_init"] = float(cfg["sigma_init"])

        if cfg.get("k_drift_scale_init", None) is not None:
            model_kwargs["k_drift_scale_init"] = float(cfg["k_drift_scale_init"])

        if cfg.get("k_z_center_init", None) is not None:
            model_kwargs["k_z_center_init"] = np.asarray(cfg["k_z_center_init"], dtype=np.float32)

        if cfg.get("k_learn_center", None) is not None:
            model_kwargs["k_learn_center"] = bool(cfg["k_learn_center"])

    return model_kwargs


def load_model(
    checkpoint_path: str,
    device: torch.device,
    latent_dim: int = 2,
    use_pricing_checkpoint: bool = False,
    pricing_run_name: str = "pricing_ep200",
) -> FullModel:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    raw = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(raw, dict) and "model_state_dict" in raw:
        state_dict = raw["model_state_dict"]
        saved_variant = raw.get("variant", "unknown")
        if saved_variant != "unknown" and saved_variant != config.VARIANT:
            raise ValueError(
                f"Checkpoint variant '{saved_variant}' does not match active "
                f"config.VARIANT '{config.VARIANT}'."
            )
    else:
        state_dict = raw

    model_kwargs = build_model_init_kwargs(raw, latent_dim=latent_dim)
    print("Model init kwargs:")
    print(model_kwargs)

    model = FullModel(**model_kwargs)
    state_dict = maybe_upgrade_old_stable_k_state_dict(state_dict, model)
    safe_load_state_dict_compat(model, state_dict)
    model = model.to(device).double()
    model.eval()

    print(f"Loaded model from {checkpoint_path}")
    print(f"  Active config variant: {config.VARIANT}")
    print(f"  Using pricing checkpoint: {use_pricing_checkpoint}")
    if use_pricing_checkpoint:
        print(f"  Pricing run name: {pricing_run_name}")
    print(f"  Model dtype: {next(model.parameters()).dtype}")
    return model


def parse_float_list(text: str):
    if text is None or str(text).strip() == "":
        return []
    vals = []
    for chunk in str(text).split(","):
        s = chunk.strip()
        if s == "":
            continue
        vals.append(float(s))
    return vals


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

    batch_size = z_batch.shape[0]
    device = z_batch.device
    dtype = z_batch.dtype

    P_full_out = torch.full(
        (batch_size, tau_grid.numel()),
        float("nan"),
        device=device,
        dtype=dtype,
    )
    valid = torch.zeros(batch_size, device=device, dtype=torch.bool)

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

    for i in range(batch_size):
        try:
            P_i, _, _, _, _, _, _, _ = decode_from_latent_script(
                model,
                z_batch[i: i + 1],
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
# Convenience runners
# ---------------------------------------------------------------------
def prepare_pricing_context(
    use="bbg",
    latent_dim=2,
    epochs=200,
    idx_choice=0,
    ccy_filter="EUR",
    seed=1234,
    tau_fine_step=1 / 52,
    tau_fine_horizon=1.0,
    device=None,
    use_pricing_checkpoint=False,
    pricing_run_name="pricing_ep200",
    explicit_checkpoint_path=None,
):
    set_seed(seed)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Repo root: {THESIS_ROOT}")
    print(f"Code root: {CODE_ROOT}")
    print(f"Active model variant from config.py: {config.VARIANT}")
    print(f"Using device: {device}")
    print(f"Seed: {seed}")

    checkpoint_path = resolve_checkpoint_path_current(
        thesis_root=THESIS_ROOT,
        use=use,
        latent_dim=latent_dim,
        epochs=epochs,
        use_pricing_checkpoint=use_pricing_checkpoint,
        pricing_run_name=pricing_run_name,
        explicit_checkpoint_path=explicit_checkpoint_path,
    )
    model = load_model(
        checkpoint_path,
        device=device,
        latent_dim=latent_dim,
        use_pricing_checkpoint=use_pricing_checkpoint,
        pricing_run_name=pricing_run_name,
    )

    data = load_data(use=use, ccy_filter=ccy_filter, idx_choice=idx_choice, device=device)
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
        fine_step=tau_fine_step,
        fine_horizon=tau_fine_horizon,
    )
    print(
        "Decoder tau grid built with "
        f"{decoder_tau_grid_base.numel()} points; first positive tau = "
        f"{decoder_tau_grid_base[1].item():.6f}, "
        f"tau_max = {decoder_tau_grid_base[-1].item():.6f}"
    )

    return {
        "model": model,
        "device": device,
        "data": data,
        "S0": S0,
        "z0": z0,
        "meta_row": meta_row,
        "decoder_tau_grid_base": decoder_tau_grid_base,
        "checkpoint_path": checkpoint_path,
        "use": use,
        "latent_dim": latent_dim,
        "epochs": epochs,
        "seed": seed,
        "tau_fine_step": tau_fine_step,
        "tau_fine_horizon": tau_fine_horizon,
        "use_pricing_checkpoint": use_pricing_checkpoint,
        "pricing_run_name": pricing_run_name,
    }


def quote_swaption_time0(
    ctx: dict,
    expiry: float,
    tenor: int,
    strike: float = 0.03,
    strike_atm: bool = False,
    notional: float = 1.0,
    payer: bool = True,
    g0_floor: float = 1e-5,
    accrual: float = 1.0,
):
    t0_quote = time0_forward_swap_and_annuity_from_z(
        model=ctx["model"],
        z0=ctx["z0"],
        decoder_tau_grid_base=ctx["decoder_tau_grid_base"],
        expiry=expiry,
        tenor=tenor,
        g0_floor=g0_floor,
        accrual=accrual,
    )

    if strike_atm:
        strike = t0_quote["forward_swap"]

    intrinsic_lb = notional * t0_quote["annuity"] * (
        max(t0_quote["forward_swap"] - strike, 0.0)
        if payer else max(strike - t0_quote["forward_swap"], 0.0)
    )

    out = {
        "forward_swap_t0": t0_quote["forward_swap"],
        "annuity_t0": t0_quote["annuity"],
        "strike": strike,
        "intrinsic_lower_bound": intrinsic_lb,
        "expiry": expiry,
        "tenor": tenor,
        "payer": payer,
    }

    print("\nTime-0 swaption quote inputs")
    print(f"  Forward swap rate F0 : {out['forward_swap_t0']:.10f}")
    print(f"  Annuity A0           : {out['annuity_t0']:.10f}")
    print(f"  Strike K             : {out['strike']:.10f}")
    print(f"  Intrinsic lower bound: {out['intrinsic_lower_bound']:.10f}")

    return out


def price_from_bachelier_quote(
    quote: dict,
    normal_vol: float,
    notional: float = 1.0,
):
    price = bachelier_price(
        forward=quote["forward_swap_t0"],
        strike=quote["strike"],
        normal_vol=normal_vol,
        expiry=quote["expiry"],
        annuity=quote["annuity_t0"],
        notional=notional,
        payer=quote["payer"],
    )

    print("\nSwaption price from Bachelier quote")
    print(f"  Price           : {price:.10f}")
    print(f"  Input norm vol  : {normal_vol:.10f} ({normal_vol * 10000:.2f} bp)")

    return {
        "price": price,
        "normal_vol": normal_vol,
    }


def simulate_for_pricing(
    ctx: dict,
    n_paths: int = 2000,
    n_steps: int = 120,
    dt: float = 1 / 12,
    discretization: str = "euler",
    sim_mode: str = "full",
    diffusion_scale: float = 1.0,
):
    discretization = normalize_discretization_name(discretization)

    print(
        f"Simulating {n_paths} paths with {n_steps} steps "
        f"(dt={dt}, scheme={discretization}, sim_mode={sim_mode}, diffusion_scale={diffusion_scale})..."
    )
    with torch.no_grad():
        z_paths, r_paths, mu_paths, L_paths = simulate_latent_paths(
            model=ctx["model"],
            z0=ctx["z0"],
            n_paths=n_paths,
            n_steps=n_steps,
            dt=dt,
            device=ctx["device"],
            discretization=discretization,
            sim_mode=sim_mode,
            diffusion_scale=diffusion_scale,
        )
    print("Simulation completed.")

    discount_paths = compute_discount_paths(r_paths, dt=dt, method="trapezoid")

    return {
        "z_paths": z_paths,
        "r_paths": r_paths,
        "mu_paths": mu_paths,
        "L_paths": L_paths,
        "discount_paths": discount_paths,
        "dt": dt,
        "n_paths": n_paths,
        "n_steps": n_steps,
        "discretization": discretization,
        "sim_mode": sim_mode,
        "diffusion_scale": diffusion_scale,
    }


def price_swaption_from_simulation(
    ctx: dict,
    sim: dict,
    expiry: float,
    tenor: int,
    strike: float,
    notional: float = 1.0,
    payer: bool = True,
    g0_floor: float = 1e-5,
    accrual: float = 1.0,
    output_bachelier_vol: bool = True,
):
    mc = price_swaption_mc(
        model=ctx["model"],
        z_paths=sim["z_paths"],
        r_paths=sim["r_paths"],
        decoder_tau_grid_base=ctx["decoder_tau_grid_base"],
        dt=sim["dt"],
        strike=strike,
        expiry=expiry,
        tenor=tenor,
        notional=notional,
        payer=payer,
        g0_floor=g0_floor,
        accrual=accrual,
        discount_paths=sim["discount_paths"],
    )

    t0_quote = time0_forward_swap_and_annuity_from_z(
        model=ctx["model"],
        z0=ctx["z0"],
        decoder_tau_grid_base=ctx["decoder_tau_grid_base"],
        expiry=expiry,
        tenor=tenor,
        g0_floor=g0_floor,
        accrual=accrual,
    )

    print("\nSwaption Monte Carlo result")
    print(f"  Price                 : {mc['price']:.10f}")
    print(f"  MC std. error         : {mc['stderr']:.10f}")
    print(f"  Valid decode fraction : {mc['frac_valid_paths']:.4f}")
    print(f"  Used expiry time      : {mc['actual_expiry']:.6f}")
    print(f"  Mean swap rate @ exp  : {mc['mc_mean_swap_rate_at_expiry']:.10f}")
    print(f"  Mean annuity @ exp    : {mc['mc_mean_annuity_at_expiry']:.10f}")
    print(f"  Strike used           : {strike:.10f}")

    implied_vol = np.nan
    if output_bachelier_vol:
        implied_vol = implied_bachelier_vol(
            market_price=mc["price"],
            forward=t0_quote["forward_swap"],
            strike=strike,
            expiry=expiry,
            annuity=t0_quote["annuity"],
            notional=notional,
            payer=payer,
        )
        if np.isfinite(implied_vol):
            print("\nImplied Bachelier normal vol from model price")
            print(f"  Normal vol      : {implied_vol:.10f} ({implied_vol * 10000:.2f} bp)")
        else:
            print("\nCould not infer Bachelier vol from model price.")

    w = mc["discount_to_expiry"] * mc["annuity"]
    mc_A0 = w.mean().item()
    mc_F0 = (w * mc["swap_rate"]).mean().item() / max(w.mean().item(), 1e-16)
    prob_itm = (mc["swap_rate"] > strike).double().mean().item() if payer else (mc["swap_rate"] < strike).double().mean().item()

    q_levels = torch.tensor([0.01, 0.05, 0.50, 0.95, 0.99], device=mc["swap_rate"].device, dtype=mc["swap_rate"].dtype)
    q_vals = torch.quantile(mc["swap_rate"], q_levels).detach().cpu().numpy()

    print("\nMonte Carlo consistency diagnostics")
    print(f"  A0 from time-0 curve     : {t0_quote['annuity']:.10f}")
    print(f"  E[D_T A_T] from MC       : {mc_A0:.10f}")
    print(f"  F0 from time-0 curve     : {t0_quote['forward_swap']:.10f}")
    print(f"  E[D_T A_T S_T]/E[D_T A_T]: {mc_F0:.10f}")
    print(f"  Prob(option ITM)         : {prob_itm:.10f}")
    print("  Swap-rate quantiles @ expiry:")
    print(f"    1%  : {q_vals[0]:.10f}")
    print(f"    5%  : {q_vals[1]:.10f}")
    print(f"    50% : {q_vals[2]:.10f}")
    print(f"    95% : {q_vals[3]:.10f}")
    print(f"    99% : {q_vals[4]:.10f}")

    return {
        "mc": mc,
        "time0_quote": t0_quote,
        "implied_bachelier_vol": implied_vol,
        "mc_A0": mc_A0,
        "mc_F0": mc_F0,
        "prob_itm": prob_itm,
        "swap_rate_quantiles": q_vals,
    }


def run_vol_surface_grid(
    ctx: dict,
    dt: float,
    notional: float,
    out_dir: str,
    strikes,
    expiries,
    tenors,
    n_paths: int,
    n_steps: int,
    discretization: str,
    g0_floor: float,
    payer: bool = True,
    accrual: float = 1.0,
    sim_mode: str = "full",
    diffusion_scale: float = 1.0,
    plot_dpi: int = 200,
    show_plots: bool = False,
):
    os.makedirs(out_dir, exist_ok=True)

    sim = simulate_for_pricing(
        ctx=ctx,
        n_paths=n_paths,
        n_steps=n_steps,
        dt=dt,
        discretization=discretization,
        sim_mode=sim_mode,
        diffusion_scale=diffusion_scale,
    )

    results = []

    for expiry in expiries:
        for tenor in tenors:
            row = {
                "expiry": float(expiry),
                "tenor": int(tenor),
            }

            try:
                underlying = swaption_underlying_at_expiry(
                    model=ctx["model"],
                    z_paths=sim["z_paths"],
                    r_paths=sim["r_paths"],
                    decoder_tau_grid_base=ctx["decoder_tau_grid_base"],
                    dt=dt,
                    expiry=float(expiry),
                    tenor=int(tenor),
                    g0_floor=g0_floor,
                    accrual=accrual,
                    discount_paths=sim["discount_paths"],
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
                    model=ctx["model"],
                    z0=ctx["z0"],
                    decoder_tau_grid_base=ctx["decoder_tau_grid_base"],
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

            vols = [sub.iloc[0].get(f"bachelier_vol_{strike}", np.nan) for strike in strikes]

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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    surface_out_dir = SURFACE_OUT_DIR
    if surface_out_dir is None:
        surface_out_dir = os.path.join(THESIS_ROOT, "Figures", "Pricing", "pricing_outputs")

    ctx = prepare_pricing_context(
        use=USE,
        latent_dim=LATENT_DIM,
        epochs=EPOCHS,
        idx_choice=IDX_CHOICE,
        ccy_filter=CCY_FILTER,
        seed=SEED,
        device=device,
        use_pricing_checkpoint=USE_PRICING_CHECKPOINT,
        pricing_run_name=PRICING_RUN_NAME,
    )

    quote = None
    if RUN_T0_QUOTE or RUN_MC_PRICE:
        quote = quote_swaption_time0(
            ctx=ctx,
            expiry=EXPIRY,
            tenor=TENOR,
            strike=STRIKE,
            strike_atm=STRIKE_ATM,
            notional=NOTIONAL,
            payer=PAYER,
            g0_floor=G0_FLOOR,
            accrual=ACCRUAL,
        )

    if RUN_MC_PRICE:
        sim = simulate_for_pricing(
            ctx=ctx,
            n_paths=N_PATHS,
            n_steps=N_STEPS,
            dt=DT,
            discretization=DISCRETIZATION,
            sim_mode=SIM_MODE,
            diffusion_scale=DIFFUSION_SCALE,
        )

        _ = price_swaption_from_simulation(
            ctx=ctx,
            sim=sim,
            expiry=EXPIRY,
            tenor=TENOR,
            strike=quote["strike"],
            notional=NOTIONAL,
            payer=PAYER,
            g0_floor=G0_FLOOR,
            accrual=ACCRUAL,
            output_bachelier_vol=True,
        )

    if RUN_SURFACE:
        _ = run_vol_surface_grid(
            ctx=ctx,
            dt=DT,
            notional=NOTIONAL,
            out_dir=surface_out_dir,
            strikes=parse_float_list(SURFACE_STRIKES),
            expiries=parse_float_list(SURFACE_EXPIRIES),
            tenors=[int(round(x)) for x in parse_float_list(SURFACE_TENORS)],
            n_paths=N_PATHS,
            n_steps=N_STEPS,
            discretization=DISCRETIZATION,
            g0_floor=G0_FLOOR,
            payer=PAYER,
            accrual=ACCRUAL,
            sim_mode=SIM_MODE,
            diffusion_scale=DIFFUSION_SCALE,
            plot_dpi=PLOT_DPI,
            show_plots=SHOW_PLOTS,
        )


if __name__ == "__main__":
    main()