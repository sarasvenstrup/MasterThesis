import math
import os
import sys
import time
import random
import warnings

import numpy as np
import torch

try:
    REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
except NameError:
    REPO_ROOT = os.getcwd()

PROJECT_ROOT = os.path.abspath(os.path.join(REPO_ROOT, ".."))
THESIS_ROOT = os.path.abspath(os.path.join(REPO_ROOT, "..", ".."))

if THESIS_ROOT not in sys.path:
    sys.path.insert(0, THESIS_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Code import config
from Code.load_swapdata import my_data
from Code.model.sigma_matrix import L_from_sigmas_rhos

print(f"Repo root: {REPO_ROOT}")
print(f"Active model variant from config.py: {config.VARIANT}")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_checkpoint_path(repo_root: str, use: str, latent_dim: int, epochs: int) -> str:
    variant = config.VARIANT

    new_filename = f"checkpoint_dim{latent_dim}_ep{epochs}.pt"
    new_path = os.path.join(
        THESIS_ROOT,
        "Figures",
        "TrainingResults",
        f"dim{latent_dim}_{variant}",
        f"ep{epochs}",
        new_filename,
    )

    old_filename = f"fullmodel_{use}_dim{latent_dim}_ep{epochs}.pt"
    candidates = [
        new_path,
        os.path.join(repo_root, "..", "checkpoints", old_filename),
        os.path.join(repo_root, "checkpoints", old_filename),
    ]

    for path in candidates:
        if os.path.exists(path):
            return os.path.abspath(path)

    searched = "\n".join(f"  - {os.path.abspath(p)}" for p in candidates)
    raise FileNotFoundError(f"Checkpoint not found. Searched:\n{searched}")


def safe_load_state_dict(model, state_dict):
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


def load_and_setup_model(device, use, latent_dim, epochs):
    if latent_dim != 2:
        raise ValueError("This script currently supports only the 2-factor model (latent_dim=2).")

    checkpoint_path = resolve_checkpoint_path(REPO_ROOT, use, latent_dim, epochs)
    raw = torch.load(checkpoint_path, map_location=device)

    from Code.model.full_model import FullModel

    if "model_state_dict" in raw:
        state_dict = raw["model_state_dict"]
        saved_variant = raw.get("variant", "unknown")
        if saved_variant != "unknown" and saved_variant != config.VARIANT:
            raise ValueError(
                f"Checkpoint variant '{saved_variant}' does not match active "
                f"config.VARIANT '{config.VARIANT}'. Update Code/config.py."
            )
    else:
        state_dict = raw

    model = FullModel(latent_dim=latent_dim)
    safe_load_state_dict(model, state_dict)

    model.to(device).double()
    model.eval()

    print(f"Loaded model from {checkpoint_path}")
    print(f"  Active config variant: {config.VARIANT}")
    print(f"  Model dtype: {next(model.parameters()).dtype}")

    return model, checkpoint_path


@torch.no_grad()
def get_mu(model, z):
    return model.K(z)


@torch.no_grad()
def get_L(model, z):
    H_out = model.H(z)

    if isinstance(H_out, tuple) and len(H_out) == 2:
        sigmas, rhos = H_out
        return L_from_sigmas_rhos(sigmas, rhos)

    if torch.is_tensor(H_out) and H_out.ndim == 3:
        return H_out

    raise TypeError(
        "Unsupported model.H(z) output. Expected either "
        "(sigmas, rhos) or a tensor L of shape (B,d,d)."
    )


@torch.no_grad()
def get_r(model, z):
    r = model.R(z)
    if r.ndim == 2 and r.shape[-1] == 1:
        r = r.squeeze(-1)
    return r


def _finite_diff_diffusion_jacobian(model, z, eps=1e-4):
    """Compute Jacobian of diffusion matrix L via finite differences."""
    B0 = get_L(model, z)
    n, d, m = B0.shape
    jac_B = torch.empty((n, d, m, d), device=z.device, dtype=z.dtype)

    for k in range(d):
        perturb = torch.zeros_like(z)
        step = eps * torch.maximum(torch.ones_like(z[:, k]), z[:, k].abs())
        perturb[:, k] = step

        B_plus = get_L(model, z + perturb)
        B_minus = get_L(model, z - perturb)

        denom = (2.0 * step).view(-1, 1, 1)
        jac_B[:, :, :, k] = (B_plus - B_minus) / denom

    return B0, jac_B


def _milstein_correction(B, jac_B, dW, dt):
    """Compute first-order Milstein correction term."""
    directional_deriv = torch.einsum("nkj,nijk->nij", B, jac_B)
    return 0.5 * torch.sum(directional_deriv * ((dW**2 - dt).unsqueeze(1)), dim=2)


def make_experiment_suffix(use, latent_dim, epochs, n_paths, n_steps, seed, discretization, sim_mode="full", diffusion_scale=1.0):
    disc = discretization.lower()
    diff_tag = f"{diffusion_scale:g}".replace(".", "p")
    return (
        f"{use}_dim{latent_dim}_ep{epochs}_paths{n_paths}_steps{n_steps}_seed{seed}_{disc}"
        f"_{sim_mode}_diff{diff_tag}"
    )


def get_simulation_out_dir():
    out_dir = os.path.join(THESIS_ROOT, "Figures", "Pricing", "simulations")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


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


def load_data(use="bbg", ccy_filter="", idx_choice=1390, device=None):
    meta, X_tensor, meta_full, X_tensor_full, tenors, df_wide, df_wide_all, scale_is_percent = my_data(use=use)
    X_tensor = X_tensor.double()

    meta, X_tensor = filter_dataset_by_currency(meta, X_tensor, ccy_filter)

    if idx_choice < 0:
        idx_choice = X_tensor.shape[0] + idx_choice

    if idx_choice < 0 or idx_choice >= X_tensor.shape[0]:
        raise IndexError(f"idx_choice={idx_choice} out of bounds for X_tensor of length {X_tensor.shape[0]}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    S0 = X_tensor[idx_choice : idx_choice + 1].to(device)
    meta_row = meta.iloc[idx_choice] if hasattr(meta, "iloc") else None

    return {
        "meta": meta,
        "X_tensor": X_tensor,
        "meta_full": meta_full,
        "X_tensor_full": X_tensor_full,
        "tenors": tenors,
        "df_wide": df_wide,
        "df_wide_all": df_wide_all,
        "SCALE_IS_PERCENT": scale_is_percent,
        "idx_choice": idx_choice,
        "S0": S0,
        "meta_row": meta_row,
    }


def build_decoder_tau_grid(model, device, dtype, fine_step=1 / 52, fine_horizon=1.0):
    if fine_step <= 0:
        raise ValueError("tau_fine_step must be positive")
    if fine_horizon < 0:
        raise ValueError("tau_fine_horizon must be non-negative")

    tau_max = float(model.tau_max)
    fine_horizon = min(float(fine_horizon), tau_max)

    fine_tau = torch.arange(
        0.0,
        fine_horizon + 0.5 * fine_step,
        fine_step,
        device=device,
        dtype=dtype,
    )
    annual_tau = torch.arange(1.0, tau_max + 1.0, 1.0, device=device, dtype=dtype)
    tau_grid = torch.unique(torch.cat([fine_tau, annual_tau]), sorted=True)

    if tau_grid[0].item() != 0.0:
        tau_grid = torch.cat([torch.zeros(1, device=device, dtype=dtype), tau_grid])
        tau_grid = torch.unique(tau_grid, sorted=True)

    return tau_grid


def get_grid_indices_for_values(grid: torch.Tensor, values: torch.Tensor, tol: float = 1e-10):
    idx_list = []
    for v in values:
        diffs = torch.abs(grid - v)
        idx = torch.argmin(diffs)
        if diffs[idx].item() > tol:
            raise RuntimeError(
                f"Requested tau={float(v):.12f} not found on decoder grid within tolerance {tol:.1e}."
            )
        idx_list.append(int(idx.item()))
    return idx_list


def compute_latent_statistics(model, X_tensor, device, latent_dim):
    z_train_list = []
    with torch.no_grad():
        for i in range(0, X_tensor.shape[0], 256):
            batch = X_tensor[i : min(i + 256, X_tensor.shape[0])].to(device)
            z_batch = model.encoder(batch)
            z_train_list.append(z_batch)

    z_train = torch.cat(z_train_list, dim=0)
    z_train_mean = z_train.mean(dim=0).detach()
    z_train_cov = torch.cov(z_train.t()).detach()
    z_train_std = z_train.std(dim=0).detach()

    print("Training latent cloud mean:", z_train_mean.cpu().numpy())
    print("Training latent cloud std: ", z_train_std.cpu().numpy())
    for d in range(latent_dim):
        print(f"  z[{d}] range = [{z_train[:, d].min().item():.6f}, {z_train[:, d].max().item():.6f}]")

    return z_train_mean, z_train_cov, z_train_std




def simulate_latent_paths(
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
    if z0.dim() != 2 or z0.shape[0] != 1:
        raise ValueError(f"Expected z0 shape (1,d), got {tuple(z0.shape)}")

    discretization = discretization.lower()
    valid_discretizations = {"euler", "milstein"}
    if discretization not in valid_discretizations:
        raise ValueError(f"Unknown discretization='{discretization}'. Choose from {sorted(valid_discretizations)}")

    sim_mode = str(sim_mode).strip().lower()
    if sim_mode not in {"full", "drift_only", "diffusion_only"}:
        raise ValueError("sim_mode must be one of: full, drift_only, diffusion_only")

    if diffusion_scale < 0:
        raise ValueError("diffusion_scale must be non-negative")

    d = z0.shape[1]
    sqrt_dt = math.sqrt(dt)

    if discretization == "milstein" and d > 1:
        warnings.warn(
            "Milstein scheme uses a commutative-noise approximation and ignores Lévy-area terms "
            "for multidimensional latent diffusion.",
            RuntimeWarning,
            stacklevel=2,
        )

    z = z0.repeat(n_paths, 1).to(device)

    z_paths = torch.empty((n_paths, n_steps + 1, d), device=device, dtype=z.dtype)
    r_paths = torch.empty((n_paths, n_steps + 1), device=device, dtype=z.dtype)
    mu_paths = torch.empty((n_paths, n_steps + 1, d), device=device, dtype=z.dtype)
    L_paths = torch.empty((n_paths, n_steps + 1, d, d), device=device, dtype=z.dtype)

    z_paths[:, 0, :] = z
    r_paths[:, 0] = get_r(model, z)
    mu_paths[:, 0, :] = get_mu(model, z)
    L_paths[:, 0, :, :] = get_L(model, z)

    for t in range(n_steps):
        if discretization == "euler":
            B = get_L(model, z)
            dW = torch.randn(n_paths, B.shape[-1], device=device, dtype=z.dtype) * sqrt_dt
            shock = diffusion_scale * torch.bmm(B, dW.unsqueeze(-1)).squeeze(-1)
            drift = get_mu(model, z) * dt

            if sim_mode == "full":
                z = z + drift + shock
            elif sim_mode == "drift_only":
                z = z + drift
            else:  # diffusion_only
                z = z + shock

        else:  # milstein
            B, jac_B = _finite_diff_diffusion_jacobian(model, z)
            dW = torch.randn(n_paths, B.shape[-1], device=device, dtype=z.dtype) * sqrt_dt
            shock = diffusion_scale * torch.bmm(B, dW.unsqueeze(-1)).squeeze(-1)
            corr = diffusion_scale * _milstein_correction(B, jac_B, dW, dt)
            drift = get_mu(model, z) * dt

            if sim_mode == "full":
                z = z + drift + shock + corr
            elif sim_mode == "drift_only":
                z = z + drift
            else:  # diffusion_only
                z = z + shock + corr


        if not torch.isfinite(z).all():
            raise RuntimeError(f"Non-finite latent state encountered at step {t + 1}")

        z_paths[:, t + 1, :] = z
        r_paths[:, t + 1] = get_r(model, z)
        mu_paths[:, t + 1, :] = get_mu(model, z)
        L_paths[:, t + 1, :, :] = get_L(model, z)

    return z_paths, r_paths, mu_paths, L_paths


def compute_discount_paths(r_paths: torch.Tensor, dt: float, method: str = "trapezoid") -> torch.Tensor:
    if dt <= 0:
        raise ValueError("dt must be positive")
    if r_paths.ndim != 2:
        raise ValueError(f"Expected r_paths to have shape (n_paths, n_steps+1), got {tuple(r_paths.shape)}")

    n_paths, n_times = r_paths.shape
    if n_times < 2:
        return torch.ones_like(r_paths)

    if method == "left":
        increments = r_paths[:, :-1] * dt
    elif method == "trapezoid":
        increments = 0.5 * (r_paths[:, :-1] + r_paths[:, 1:]) * dt
    else:
        raise ValueError("method must be 'left' or 'trapezoid'")

    int_r = torch.cumsum(increments, dim=1)
    disc = torch.ones((n_paths, n_times), device=r_paths.device, dtype=r_paths.dtype)
    disc[:, 1:] = torch.exp(-int_r)
    return disc


def save_simulation_bundle(
    out_dir,
    suffix,
    z_paths,
    r_paths,
    mu_paths,
    L_paths,
    discount_paths,
    times,
    z_train_mean,
    z_train_cov,
    tenors,
    decoder_tau_grid,
    annual_indices,
    metadata: dict,
):
    bundle = {
        "z_paths": z_paths.detach().cpu(),
        "r_paths": r_paths.detach().cpu(),
        "mu_paths": mu_paths.detach().cpu(),
        "L_paths": L_paths.detach().cpu(),
        "discount_paths": discount_paths.detach().cpu(),
        "times": np.asarray(times),
        "z_train_mean": z_train_mean.detach().cpu(),
        "z_train_cov": z_train_cov.detach().cpu(),
        "tenors": np.asarray(tenors),
        "decoder_tau_grid": decoder_tau_grid.detach().cpu(),
        "annual_indices": list(annual_indices),
        "metadata": metadata,
    }
    bundle_path = os.path.join(out_dir, f"simulation_bundle_{suffix}.pt")
    torch.save(bundle, bundle_path)
    print(f"Saved simulation bundle to {bundle_path}")
    return bundle_path


def run_simulation(
    use="bbg",
    latent_dim=2,
    epochs=3500,
    n_paths=100,
    n_steps=24,
    dt=1/12,
    idx_choice=1390,
    ccy_filter="",
    discretization="euler",
    sim_mode="full",
    diffusion_scale=1.0,
    seed=1234,
    tau_fine_step=1/52,
    tau_fine_horizon=1.0,
    device=None,
    save_bundle=True,
):
    set_seed(seed)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Seed: {seed}")

    model, checkpoint_path = load_and_setup_model(device, use, latent_dim, epochs)

    data = load_data(use=use, ccy_filter=ccy_filter, idx_choice=idx_choice, device=device)
    X_tensor = data["X_tensor"]
    tenors = data["tenors"]
    S0 = data["S0"]
    meta_row = data["meta_row"]
    scale_is_percent = data["SCALE_IS_PERCENT"]
    idx_choice = data["idx_choice"]

    print(f"SCALE_IS_PERCENT from my_data(): {scale_is_percent}")
    if meta_row is not None:
        print(f"Initial curve metadata row:\n{meta_row}")

    decoder_tau_grid = build_decoder_tau_grid(
        model,
        device=device,
        dtype=torch.float64,
        fine_step=tau_fine_step,
        fine_horizon=tau_fine_horizon,
    )
    annual_tau = torch.arange(1.0, float(model.tau_max) + 1.0, 1.0, device=device, dtype=torch.float64)
    annual_indices = get_grid_indices_for_values(decoder_tau_grid, annual_tau)

    z_train_mean, z_train_cov, z_train_std = compute_latent_statistics(model, X_tensor, device, latent_dim)

    with torch.no_grad():
        z0 = model.encoder(S0)
    print(f"Initial latent state z0: {z0.detach().cpu().numpy().flatten()}")

    print(
        f"Simulating {n_paths} paths with {n_steps} steps "
        f"(dt={dt}, scheme={discretization}, sim_mode={sim_mode}, diffusion_scale={diffusion_scale})..."
    )
    t0 = time.time()
    z_paths, r_paths, mu_paths, L_paths = simulate_latent_paths(
        model=model,
        z0=z0,
        n_paths=n_paths,
        n_steps=n_steps,
        dt=dt,
        device=device,
        discretization=discretization,
        sim_mode=sim_mode,
        diffusion_scale=diffusion_scale,
    )
    print(f"Simulation completed in {time.time() - t0:.2f}s.")

    discount_paths = compute_discount_paths(r_paths, dt=dt, method="trapezoid")
    print(
        f"Built path discount factors: D_t range = "
        f"[{discount_paths.min().item():.6f}, {discount_paths.max().item():.6f}]"
    )

    times = np.arange(n_steps + 1) * dt
    suffix = make_experiment_suffix(
        use=use,
        latent_dim=latent_dim,
        epochs=epochs,
        n_paths=n_paths,
        n_steps=n_steps,
        seed=seed,
        discretization=discretization,
        sim_mode=sim_mode,
        diffusion_scale=diffusion_scale,
    )

    bundle_path = None
    if save_bundle:
        out_dir = get_simulation_out_dir()
        metadata = {
            "use": use,
            "latent_dim": latent_dim,
            "epochs": epochs,
            "n_paths": n_paths,
            "n_steps": n_steps,
            "dt": dt,
            "idx_choice": idx_choice,
            "ccy_filter": ccy_filter,
            "discretization": discretization,
            "sim_mode": sim_mode,
            "diffusion_scale": diffusion_scale,
            "seed": seed,
            "tau_fine_step": tau_fine_step,
            "tau_fine_horizon": tau_fine_horizon,
            "scale_is_percent": bool(scale_is_percent),
            "variant": config.VARIANT,
            "checkpoint_path": checkpoint_path,
            "as_of_date": str(meta_row["as_of_date"]) if meta_row is not None and "as_of_date" in meta_row else "",
            "ccy": str(meta_row["ccy"]) if meta_row is not None and "ccy" in meta_row else "",
        }
        bundle_path = save_simulation_bundle(
            out_dir=out_dir,
            suffix=suffix,
            z_paths=z_paths,
            r_paths=r_paths,
            mu_paths=mu_paths,
            L_paths=L_paths,
            discount_paths=discount_paths,
            times=times,
            z_train_mean=z_train_mean,
            z_train_cov=z_train_cov,
            tenors=tenors,
            decoder_tau_grid=decoder_tau_grid,
            annual_indices=annual_indices,
            metadata=metadata,
        )

    return {
        "model": model,
        "data": data,
        "z0": z0,
        "z_paths": z_paths,
        "r_paths": r_paths,
        "mu_paths": mu_paths,
        "L_paths": L_paths,
        "discount_paths": discount_paths,
        "times": times,
        "z_train_mean": z_train_mean,
        "z_train_cov": z_train_cov,
        "z_train_std": z_train_std,
        "tenors": tenors,
        "decoder_tau_grid": decoder_tau_grid,
        "annual_indices": annual_indices,
        "suffix": suffix,
        "bundle_path": bundle_path,
    }


# Example lines to run one by one
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sim_out = run_simulation(
     use="bbg",
     latent_dim=2,
     epochs=3500,
     n_paths=500,
     n_steps=24,
     dt=1/12,
     idx_choice=0,
     ccy_filter="EUR",
     discretization="euler",
     sim_mode="full",
     diffusion_scale=1.0,
     seed=1234,
     device=device,
 )
bundle_path = sim_out["bundle_path"]
