import os
import sys
import time
import math
import random
import warnings
import importlib.util
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch


THIS_DIR = Path(__file__).resolve().parent
BASE_SCRIPT = THIS_DIR / "simulate_model.py"


def _load_base_module():
    if not BASE_SCRIPT.exists():
        raise FileNotFoundError(
            f"Could not find base script at {BASE_SCRIPT}. "
            "Place this file in the same folder as simulate_model.py."
        )

    spec = importlib.util.spec_from_file_location("_base_simulate_model", str(BASE_SCRIPT))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module spec from {BASE_SCRIPT}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _load_base_module()


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
    if getattr(base.config, "VARIANT", None) != "stable":
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

    # Old M = S - A
    S = B - B.t()
    A = L @ L.t() + eps * I
    M = S - A

    # Solve M theta = -N
    theta = torch.linalg.solve(M, (-N).unsqueeze(-1)).squeeze(-1)

    # Set kappa = 1  => raw_kappa = softplus^{-1}(1) = log(expm1(1))
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


def load_and_setup_model_compat(device, use, latent_dim, epochs):
    if latent_dim != 2:
        raise ValueError("This script currently supports only the 2-factor model (latent_dim=2).")

    checkpoint_path = base.resolve_checkpoint_path(str(THIS_DIR), use, latent_dim, epochs)
    raw = torch.load(checkpoint_path, map_location=device)

    from Code.model.full_model import FullModel

    if "model_state_dict" in raw:
        state_dict = raw["model_state_dict"]
        saved_variant = raw.get("variant", "unknown")
        if saved_variant != "unknown" and saved_variant != base.config.VARIANT:
            raise ValueError(
                f"Checkpoint variant '{saved_variant}' does not match active "
                f"config.VARIANT '{base.config.VARIANT}'. Update Code/config.py."
            )
    else:
        state_dict = raw

    model = FullModel(latent_dim=latent_dim)

    # Backward compatibility: old stable-K checkpoint -> new OU-style K
    state_dict = maybe_upgrade_old_stable_k_state_dict(state_dict, model)

    safe_load_state_dict_compat(model, state_dict)

    model.to(device).double()
    model.eval()

    print(f"Loaded model from {checkpoint_path}")
    print(f"  Active config variant: {base.config.VARIANT}")
    print(f"  Model dtype: {next(model.parameters()).dtype}")

    return model


def parse_int_list(text: str):
    if text is None or str(text).strip() == "":
        return []
    vals = []
    for chunk in str(text).split(","):
        s = chunk.strip()
        if s == "":
            continue
        vals.append(int(s))
    return vals


def resolve_start_indices(idx_choices_text: str, fallback_idx: int, n_obs: int):
    if str(idx_choices_text).strip():
        raw = parse_int_list(idx_choices_text)
    else:
        raw = [fallback_idx]

    out = []
    seen = set()
    for idx in raw:
        if idx < 0:
            idx = n_obs + idx
        if idx < 0 or idx >= n_obs:
            raise IndexError(f"Requested start index {idx} out of bounds for dataset of length {n_obs}")
        if idx not in seen:
            out.append(idx)
            seen.add(idx)
    return out


def _meta_value(meta_row, key, default=""):
    if meta_row is None:
        return default
    try:
        val = meta_row[key]
        if pd.isna(val):
            return default
        return val
    except Exception:
        return default


def filter_dataset_by_currency(meta, X_tensor, ccy_filter: str):
    if not ccy_filter or str(ccy_filter).strip() == "":
        return meta.reset_index(drop=True), X_tensor

    ccy_filter = str(ccy_filter).strip().upper()
    mask = meta["ccy"].astype(str).str.upper() == ccy_filter
    n_keep = int(mask.sum())

    if n_keep == 0:
        available = sorted(meta["ccy"].astype(str).str.upper().unique().tolist())
        raise ValueError(
            f"No rows found for ccy_filter='{ccy_filter}'. "
            f"Available currencies: {available}"
        )

    meta_f = meta.loc[mask].reset_index(drop=True)
    X_tensor_f = X_tensor[mask.to_numpy()]

    print(f"Filtered dataset to currency {ccy_filter}: kept {n_keep} rows")
    return meta_f, X_tensor_f


def compute_mahal_matrix(z_paths: torch.Tensor, z_train_mean: torch.Tensor, z_train_cov: torch.Tensor):
    eps_reg = 1e-8
    I_reg = torch.eye(z_train_cov.shape[0], device=z_train_cov.device, dtype=z_train_cov.dtype)
    z_cov_inv = torch.linalg.inv(z_train_cov + eps_reg * I_reg)

    centered = z_paths - z_train_mean.view(1, 1, -1)
    quad = torch.einsum("ntd,dd,ntd->nt", centered, z_cov_inv, centered)
    return torch.sqrt(torch.clamp(quad, min=0.0))


def compute_g0_summary(model, z_paths: torch.Tensor):
    device = z_paths.device
    dtype = z_paths.dtype
    tau0 = torch.zeros(1, device=device, dtype=dtype)

    min_abs_g0_over_time = float("inf")
    min_abs_g0_final = float("inf")

    with torch.no_grad():
        for t in range(z_paths.shape[1]):
            z_t = z_paths[:, t, :]
            G0_t = model.G(z_t, tau0)
            if G0_t.ndim == 2:
                G0_t = G0_t[:, 0]
            abs_g0_t = G0_t.abs()

            this_min = float(abs_g0_t.min().item())
            min_abs_g0_over_time = min(min_abs_g0_over_time, this_min)

            if t == z_paths.shape[1] - 1:
                min_abs_g0_final = this_min

    return {
        "min_absG0_over_time": min_abs_g0_over_time,
        "min_absG0_final": min_abs_g0_final,
    }


def decode_final_swaps_summary(
    model,
    z_final,
    decoder_tau_grid,
    annual_indices,
    tenors,
    selected_tenors,
    g0_floor,
):
    n_paths = z_final.shape[0]
    valid_decode = np.zeros(n_paths, dtype=bool)
    S_sim_np = np.full((n_paths, len(tenors)), np.nan, dtype=float)
    batch_error = ""

    try:
        P_full, _, _, _, _, _, _, _ = base.decode_from_latent_script(
            model,
            z_final,
            decoder_tau_grid,
            G_floor=g0_floor,
            check_short_rate=False,
        )
        P_annual = P_full[:, annual_indices]
        S_sim = base.par_swap_from_discount(P_annual, tenors)
        S_sim_np = S_sim.detach().cpu().numpy()
        valid_decode[:] = True

    except RuntimeError as e:
        batch_error = str(e)

        for p in range(n_paths):
            try:
                P_full_p, _, _, _, _, _, _, _ = base.decode_from_latent_script(
                    model,
                    z_final[p : p + 1],
                    decoder_tau_grid,
                    G_floor=g0_floor,
                    check_short_rate=False,
                )
                P_annual_p = P_full_p[:, annual_indices]
                S_sim_p = base.par_swap_from_discount(P_annual_p, tenors)
                S_sim_np[p, :] = S_sim_p[0].detach().cpu().numpy()
                valid_decode[p] = True
            except RuntimeError:
                pass

    out = {
        "final_valid_decode_count": int(valid_decode.sum()),
        "final_valid_decode_frac": float(valid_decode.mean()),
        "final_decode_batch_error": batch_error,
    }

    tenor_arr = np.asarray(tenors, dtype=float)
    valid_rows = S_sim_np[valid_decode]

    for ten in selected_tenors:
        key = base.tenor_label(ten)
        idx_matches = np.where(np.isclose(tenor_arr, float(ten)))[0]

        if len(idx_matches) == 0 or valid_rows.shape[0] == 0:
            out[f"final_swap_{key}_median"] = np.nan
            out[f"final_swap_{key}_mean"] = np.nan
            continue

        j = int(idx_matches[0])
        vals = valid_rows[:, j]
        vals = vals[np.isfinite(vals)]

        if len(vals) == 0:
            out[f"final_swap_{key}_median"] = np.nan
            out[f"final_swap_{key}_mean"] = np.nan
        else:
            out[f"final_swap_{key}_median"] = float(np.nanmedian(vals))
            out[f"final_swap_{key}_mean"] = float(np.nanmean(vals))

    return out


def _existing_option_strings(parser):
    out = set()
    for action in parser._actions:
        for opt in getattr(action, "option_strings", []):
            out.add(opt)
    return out


def _apply_drift_step(model, z, shock, dt):
    """
    Supports both old and new stable K variants.

    Old:
        mu(z) = M z + N

    New:
        mu(z) = M (z - theta)
    """
    d = z.shape[1]
    I = torch.eye(d, device=z.device, dtype=z.dtype)

    # New OU-style stable K
    if hasattr(model.K, "drift_matrix") and hasattr(model.K, "theta"):
        M = model.K.drift_matrix().to(device=z.device, dtype=z.dtype)
        theta = model.K.theta.to(device=z.device, dtype=z.dtype)

        A = I - dt * M
        rhs = z + shock - dt * (theta.unsqueeze(0) @ M.t())

        A_batch = A.unsqueeze(0).expand(rhs.shape[0], -1, -1)
        return torch.linalg.solve(A_batch, rhs.unsqueeze(-1)).squeeze(-1)

    # Old stable K
    if hasattr(model.K, "stable_matrix"):
        M = model.K.stable_matrix().to(device=z.device, dtype=z.dtype)
        N = getattr(model.K, "N", None)

        A = I - dt * M
        rhs = z + shock
        if N is not None:
            rhs = rhs + dt * N.to(device=z.device, dtype=z.dtype).unsqueeze(0)

        A_batch = A.unsqueeze(0).expand(rhs.shape[0], -1, -1)
        return torch.linalg.solve(A_batch, rhs.unsqueeze(-1)).squeeze(-1)

    # Fallback: explicit Euler
    return z + base.get_mu(model, z) * dt + shock


def simulate_latent_paths_diagnostic(
    model,
    z0,
    n_paths,
    n_steps,
    dt,
    device,
    discretization="euler",
    sim_mode="full",
    diffusion_scale=1.0,
):
    sim_mode = str(sim_mode).strip().lower()
    if sim_mode not in {"full", "drift_only", "diffusion_only"}:
        raise ValueError("sim_mode must be one of: full, drift_only, diffusion_only")

    if diffusion_scale < 0:
        raise ValueError("diffusion_scale must be non-negative")

    if z0.dim() != 2 or z0.shape[0] != 1:
        raise ValueError(f"Expected z0 shape (1,d), got {tuple(z0.shape)}")

    disc = discretization
    if hasattr(base, "normalize_discretization_name"):
        disc = base.normalize_discretization_name(discretization)

    if disc != "euler":
        warnings.warn(
            "Diagnostic sim_mode currently uses Euler stepping only. "
            f"Overriding discretization='{disc}' to 'euler'.",
            RuntimeWarning,
            stacklevel=2,
        )

    d = z0.shape[1]
    sqrt_dt = math.sqrt(dt)

    z = z0.repeat(n_paths, 1).to(device)

    z_paths = torch.empty((n_paths, n_steps + 1, d), device=device, dtype=z.dtype)
    r_paths = torch.empty((n_paths, n_steps + 1), device=device, dtype=z.dtype)
    mu_paths = torch.empty((n_paths, n_steps + 1, d), device=device, dtype=z.dtype)
    L_paths = torch.empty((n_paths, n_steps + 1, d, d), device=device, dtype=z.dtype)

    z_paths[:, 0, :] = z
    r_paths[:, 0] = base.get_r(model, z)
    mu_paths[:, 0, :] = base.get_mu(model, z)
    L_paths[:, 0, :, :] = base.get_L(model, z)

    for t in range(n_steps):
        B = base.get_L(model, z)
        dW = torch.randn(n_paths, B.shape[-1], device=device, dtype=z.dtype) * sqrt_dt
        shock = torch.bmm(B, dW.unsqueeze(-1)).squeeze(-1)
        shock = diffusion_scale * shock

        if sim_mode == "full":
            z = _apply_drift_step(model, z, shock, dt)
        elif sim_mode == "drift_only":
            z = _apply_drift_step(model, z, torch.zeros_like(z), dt)
        else:  # diffusion_only
            z = z + shock

        if not torch.isfinite(z).all():
            raise RuntimeError(f"Non-finite latent state encountered at step {t + 1}")

        z_paths[:, t + 1, :] = z
        r_paths[:, t + 1] = base.get_r(model, z)
        mu_paths[:, t + 1, :] = base.get_mu(model, z)
        L_paths[:, t + 1, :, :] = base.get_L(model, z)

    return z_paths, r_paths, mu_paths, L_paths


def build_parser():
    parser = base.build_parser()
    existing = _existing_option_strings(parser)

    if "--idx_choices" not in existing:
        parser.add_argument(
            "--idx_choices",
            type=str,
            default="",
            help=(
                "Comma-separated list of initial curve indices for a simple multi-start diagnostic. "
                "If provided, overrides --idx_choice and runs a compact summary across starts."
            ),
        )

    if "--summary_tenors" not in existing:
        parser.add_argument(
            "--summary_tenors",
            type=str,
            default="1,5,10,30",
            help="Comma-separated tenors to summarize at the final simulation time in multi-start mode.",
        )

    if "--ccy_filter" not in existing:
        parser.add_argument(
            "--ccy_filter",
            type=str,
            default="",
            help="Optional currency filter for multi-start diagnostics, e.g. EUR or USD.",
        )

    if "--sim_mode" not in existing:
        parser.add_argument(
            "--sim_mode",
            type=str,
            default="full",
            choices=["full", "drift_only", "diffusion_only"],
            help="Diagnostic simulation mode: full dynamics, drift only, or diffusion only.",
        )

    if "--diffusion_scale" not in existing:
        parser.add_argument(
            "--diffusion_scale",
            type=float,
            default=1.0,
            help="Scale factor applied to the diffusion shock in diagnostic simulation.",
        )

    return parser


def _resolve_use(args):
    if hasattr(args, "use"):
        return args.use
    if hasattr(base, "config") and hasattr(base.config, "USE"):
        return base.config.USE
    raise AttributeError("Could not resolve model variant. Expected args.use or config.USE.")


def _resolve_latent_dim(args):
    if hasattr(args, "latent_dim"):
        return int(args.latent_dim)
    if hasattr(base, "config") and hasattr(base.config, "LATENT_DIM"):
        return int(base.config.LATENT_DIM)
    raise AttributeError("Could not resolve latent dimension. Expected args.latent_dim or config.LATENT_DIM.")


def _resolve_epochs(args):
    if hasattr(args, "epochs"):
        return int(args.epochs)
    if hasattr(base, "config") and hasattr(base.config, "EPOCHS"):
        return int(base.config.EPOCHS)
    raise AttributeError("Could not resolve epochs. Expected args.epochs or config.EPOCHS.")


def _resolve_device(args):
    requested = getattr(args, "device", None)

    if isinstance(requested, torch.device):
        return requested

    if isinstance(requested, str) and requested.strip():
        if requested.lower() == "cuda" and not torch.cuda.is_available():
            print("Requested CUDA but it is not available. Falling back to CPU.")
            return torch.device("cpu")
        return torch.device(requested)

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_out_dir(args, use, latent_dim, epochs):
    if hasattr(args, "out_dir") and getattr(args, "out_dir"):
        out_dir = str(getattr(args, "out_dir"))
    else:
        out_dir = str(THIS_DIR / "outputs_multistart" / f"dim{latent_dim}_{use}" / f"ep{epochs}")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def run_multi_start_diagnostic(
    model,
    X_tensor,
    meta,
    tenors,
    device,
    args,
    out_dir,
    latent_dim,
    scale_is_percent,
):
    start_indices = resolve_start_indices(args.idx_choices, args.idx_choice, X_tensor.shape[0])
    selected_tenors = base.parse_float_list(args.summary_tenors)
    g0_floor = getattr(args, "g0_floor", 1e-5)
    sim_mode = getattr(args, "sim_mode", "full")
    diffusion_scale = float(getattr(args, "diffusion_scale", 1.0))

    print(f"SCALE_IS_PERCENT from my_data(): {scale_is_percent}")
    print(f"Running simple multi-start diagnostic for {len(start_indices)} starts: {start_indices}")
    print(f"Simulation mode: {sim_mode} | diffusion_scale: {diffusion_scale:g}")

    decoder_tau_grid = base.build_decoder_tau_grid(
        model,
        device=device,
        dtype=torch.float64,
        fine_step=args.tau_fine_step,
        fine_horizon=args.tau_fine_horizon,
    )
    annual_tau = torch.arange(1.0, float(model.tau_max) + 1.0, 1.0, device=device, dtype=torch.float64)
    annual_indices = base.get_grid_indices_for_values(decoder_tau_grid, annual_tau)

    z_train_mean, z_train_cov, z_train_std = base.compute_latent_statistics(model, X_tensor, device, latent_dim)

    if hasattr(base, "diagnose_G0_on_training_cloud"):
        base.diagnose_G0_on_training_cloud(model, X_tensor, device)

    rows = []
    t0_multi = time.time()

    for k, idx in enumerate(start_indices, start=1):
        print("\n" + "-" * 60)
        print(f"[{k}/{len(start_indices)}] start idx = {idx}")

        S0 = X_tensor[idx : idx + 1].to(device)
        meta_row = meta.iloc[idx] if hasattr(meta, "iloc") else None

        as_of_date = _meta_value(meta_row, "as_of_date", "")
        ccy = _meta_value(meta_row, "ccy", "")

        if meta_row is not None:
            print(f"  as_of_date={as_of_date} | ccy={ccy}")

        with torch.no_grad():
            z0 = model.encoder(S0)

        print(f"  z0 = {z0.detach().cpu().numpy().flatten()}")

        z_paths, r_paths, mu_paths, L_paths = simulate_latent_paths_diagnostic(
            model=model,
            z0=z0,
            n_paths=args.n_paths,
            n_steps=args.n_steps,
            dt=args.dt,
            device=device,
            discretization=args.discretization,
            sim_mode=sim_mode,
            diffusion_scale=diffusion_scale,
        )

        discount_paths = base.compute_discount_paths(r_paths, dt=args.dt, method="trapezoid")
        mahal = compute_mahal_matrix(z_paths, z_train_mean, z_train_cov)
        g0_stats = compute_g0_summary(model, z_paths)

        final_decode = decode_final_swaps_summary(
            model=model,
            z_final=z_paths[:, -1, :],
            decoder_tau_grid=decoder_tau_grid,
            annual_indices=annual_indices,
            tenors=tenors,
            selected_tenors=selected_tenors,
            g0_floor=g0_floor,
        )

        row = {
            "idx_choice": int(idx),
            "as_of_date": as_of_date,
            "ccy": ccy,
            "sim_mode": sim_mode,
            "diffusion_scale": diffusion_scale,
            "mean_r_0": float(r_paths[:, 0].mean().item()),
            "mean_r_final": float(r_paths[:, -1].mean().item()),
            "std_r_final": float(r_paths[:, -1].std(unbiased=False).item()),
            "mean_discount_final": float(discount_paths[:, -1].mean().item()),
            "max_mahal_over_time": float(mahal.max().item()),
            "mean_mahal_final": float(mahal[:, -1].mean().item()),
            "min_absG0_over_time": float(g0_stats["min_absG0_over_time"]),
            "min_absG0_final": float(g0_stats["min_absG0_final"]),
        }

        for d in range(latent_dim):
            row[f"z0_{d+1}"] = float(z0[0, d].item())
            row[f"mean_z{d+1}_final"] = float(z_paths[:, -1, d].mean().item())
            row[f"std_z{d+1}_final"] = float(z_paths[:, -1, d].std(unbiased=False).item())

        row.update(final_decode)
        rows.append(row)

        print(
            f"  max Mahalanobis={row['max_mahal_over_time']:.3f} | "
            f"min |G0| over time={row['min_absG0_over_time']:.3e} | "
            f"final valid decode frac={row['final_valid_decode_frac']:.3f}"
        )

    summary_df = pd.DataFrame(rows)

    disc = getattr(args, "discretization", "euler")
    if hasattr(base, "normalize_discretization_name"):
        disc = base.normalize_discretization_name(disc)

    diff_tag = f"{diffusion_scale:g}".replace(".", "p")
    suffix = (
        f"{getattr(args, 'use', 'bbg')}"
        f"_dim{latent_dim}"
        f"_ep{getattr(args, 'epochs', 'NA')}"
        f"_paths{args.n_paths}"
        f"_steps{args.n_steps}"
        f"_seed{getattr(args, 'seed', 1234)}"
        f"_{disc}"
        f"_{sim_mode}"
        f"_diff{diff_tag}"
    )

    summary_csv = os.path.join(
        out_dir,
        f"multi_start_summary_{suffix}_nstarts{len(start_indices)}.csv",
    )
    summary_df.to_csv(summary_csv, index=False)

    elapsed = time.time() - t0_multi
    print("\n" + "=" * 60)
    print(f"Saved simple multi-start summary to {summary_csv}")
    print(f"Multi-start diagnostic finished in {elapsed:.2f}s")
    print("=" * 60)

    cols_to_show = [
        "idx_choice",
        "as_of_date",
        "ccy",
        "sim_mode",
        "diffusion_scale",
        "max_mahal_over_time",
        "min_absG0_over_time",
        "final_valid_decode_frac",
    ]
    for ten in selected_tenors:
        cols_to_show.append(f"final_swap_{base.tenor_label(ten)}_median")

    cols_to_show = [c for c in cols_to_show if c in summary_df.columns]
    if cols_to_show:
        print(summary_df[cols_to_show].to_string(index=False))

    return summary_csv


def main():
    parser = build_parser()
    args, unknown = parser.parse_known_args()

    if unknown:
        print(f"Ignoring unknown args: {unknown}")

    if not str(getattr(args, "idx_choices", "")).strip():
        return base.main()

    warnings.filterwarnings("ignore", category=UserWarning)

    use = _resolve_use(args)
    latent_dim = _resolve_latent_dim(args)
    epochs = _resolve_epochs(args)
    device = _resolve_device(args)
    out_dir = _resolve_out_dir(args, use, latent_dim, epochs)

    if hasattr(args, "seed"):
        _set_seed(int(args.seed))
        print(f"Seed: {args.seed}")

    print(f"Using device: {device}")

    meta, X_tensor, meta_full, X_tensor_full, tenors, df_wide, df_wide_all, scale_is_percent = base.my_data(use=use)
    X_tensor = X_tensor.double()

    meta, X_tensor = filter_dataset_by_currency(
        meta=meta,
        X_tensor=X_tensor,
        ccy_filter=getattr(args, "ccy_filter", ""),
    )

    model = load_and_setup_model_compat(device, use, latent_dim, epochs)

    run_multi_start_diagnostic(
        model=model,
        X_tensor=X_tensor,
        meta=meta,
        tenors=tenors,
        device=device,
        args=args,
        out_dir=out_dir,
        latent_dim=latent_dim,
        scale_is_percent=scale_is_percent,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
