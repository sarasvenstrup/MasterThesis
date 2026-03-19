# ==========================================================
# SECTION 1: Imports and environment
# ==========================================================
import os
import sys
import math
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

torch.set_num_threads(4)
torch.set_num_interop_threads(2)

try:
    REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
except NameError:
    REPO_ROOT = os.getcwd()

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Code.load_swapdata import my_data
from Code.model.full_model import FullModel
from Code.utils.sigma_matrix import L_from_sigmas_rhos
from Code.utils.ode import (
    d_tau_autograd_nodewise,
    grad_and_trace_cov_hess_G,
    paper_alpha_beta_gamma_trace,
    solve_AB
)

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("MPS available:", hasattr(torch.backends, "mps") and torch.backends.mps.is_available())

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Using device:", device)

torch.backends.mkldnn.enabled = True

# ==========================================================
# SECTION 2: User settings
# ==========================================================
USE = "bbg"
LATENT_DIM = 3
EPOCHS = 100

CHECKPOINT_PATH = os.path.join(
    REPO_ROOT,
    "checkpoints",
    f"fullmodel_{USE}_dim{LATENT_DIM}_ep{EPOCHS}.pt"
)

OUT_DIR = os.path.join(
    REPO_ROOT,
    "Figures",
    "pricing_debug",
    f"{USE}_dim{LATENT_DIM}_ep{EPOCHS}"
)
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 1234
IDX_CHOICE = -1

N_PATHS = 200
DT = 1.0 / 12.0
N_YEARS = 10
N_STEPS = int(round(N_YEARS / DT))

N_PLOT_PATHS = 30
N_PLOT_YIELD_PATHS = 20

USE_DRIFT = True
USE_DIFFUSION = True

YEARS_TO_PLOT = [0, 1, 3, 5, 10]
SAMPLE_CURVE_YEAR = 5.0
CHECK_FINITE_YEAR = 5.0


# ==========================================================
# SECTION 3: Helpers
# ==========================================================
def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def load_trained_model(checkpoint_path: str, latent_dim: int, device: torch.device) -> FullModel:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = FullModel(latent_dim=latent_dim).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict)
    model.eval()
    return model


@torch.no_grad()
def load_initial_curve(use: str, idx_choice: int, device: torch.device):
    meta, X_tensor, meta_full, X_tensor_full, tenors, df_wide, df_wide_all, SCALE_IS_PERCENT = my_data(use=use)
    X_tensor = X_tensor.float()

    if idx_choice < 0:
        idx_choice = X_tensor.shape[0] + idx_choice

    if idx_choice < 0 or idx_choice >= X_tensor.shape[0]:
        raise IndexError(f"idx_choice={idx_choice} out of bounds")

    S0 = X_tensor[idx_choice:idx_choice + 1].to(device)
    meta_row = meta.iloc[idx_choice] if hasattr(meta, "iloc") else None
    return S0, meta_row, X_tensor, meta


@torch.no_grad()
def encode_initial_state(model: FullModel, S0: torch.Tensor) -> torch.Tensor:
    return model.encoder(S0)


@torch.no_grad()
def encode_training_set(model: FullModel, X_tensor: torch.Tensor, device: torch.device, batch_size: int = 4096):
    outs = []
    n = X_tensor.shape[0]
    for i in range(0, n, batch_size):
        xb = X_tensor[i:i + batch_size].to(device)
        outs.append(model.encoder(xb))
    return torch.cat(outs, dim=0)


@torch.no_grad()
def get_mu(model: FullModel, z: torch.Tensor) -> torch.Tensor:
    return model.K(z)


@torch.no_grad()
def get_L(model: FullModel, z: torch.Tensor) -> torch.Tensor:
    sigmas, rhos = model.H(z)
    return L_from_sigmas_rhos(sigmas, rhos)


@torch.no_grad()
def get_r(model: FullModel, z: torch.Tensor) -> torch.Tensor:
    r = model.R(z)
    if r.ndim == 2 and r.shape[1] == 1:
        r = r.squeeze(1)
    return r


@torch.no_grad()
def inspect_initial_drift_and_vol(model: FullModel, z0: torch.Tensor):
    mu0 = get_mu(model, z0)
    L0 = get_L(model, z0)
    r0 = get_r(model, z0)

    print("\n================ INITIAL DRIFT / VOL CHECK ================")
    print("z0:", z0.detach().cpu().numpy())
    print("mu(z0):", mu0.detach().cpu().numpy())
    print("||mu(z0)||:", torch.norm(mu0, dim=1).detach().cpu().numpy())
    print("L(z0):", L0.detach().cpu().numpy())
    print("||L(z0)||_F:", torch.linalg.matrix_norm(L0, dim=(1, 2)).detach().cpu().numpy())
    print("r(z0):", r0.detach().cpu().numpy())

@torch.no_grad()
def inspect_K_matrix(model):
    V = model.K.V.detach().cpu()
    M = model.K.stable_matrix().detach().cpu()
    N = model.K.N.detach().cpu() if model.K.N is not None else None

    print("\n================ K MATRIX CHECK ================")
    print("V =")
    print(V.numpy())

    print("M = -(V^T V + eps I) =")
    print(M.numpy())

    eigvals = torch.linalg.eigvals(M).cpu()
    print("Eigenvalues of M:")
    print(eigvals.numpy())

    real_parts = eigvals.real
    print("Real parts:")
    print(real_parts.numpy())
    print("Max real part:", real_parts.max().item())

    if N is not None:
        print("N =")
        print(N.numpy())
        try:
            z_star = -torch.linalg.solve(M, N)
            print("Implied fixed point z* = -M^{-1}N:")
            print(z_star.numpy())
        except RuntimeError:
            print("Could not compute fixed point z*; M may be singular.")

@torch.no_grad()
def simulate_latent_paths(
    model: FullModel,
    z0: torch.Tensor,
    n_paths: int,
    n_steps: int,
    dt: float,
    device: torch.device,
    use_drift: bool = True,
    use_diffusion: bool = True,
):
    if z0.dim() != 2 or z0.shape[0] != 1:
        raise ValueError(f"Expected z0 shape (1,d), got {tuple(z0.shape)}")

    d = z0.shape[1]
    sqrt_dt = math.sqrt(dt)

    z = z0.repeat(n_paths, 1).to(device)

    z_paths = torch.empty((n_paths, n_steps + 1, d), device=device, dtype=z.dtype)
    r_paths = torch.empty((n_paths, n_steps + 1), device=device, dtype=z.dtype)

    z_paths[:, 0, :] = z
    r_paths[:, 0] = get_r(model, z)

    for t in range(n_steps):
        mu = get_mu(model, z)
        L = get_L(model, z)

        drift = mu * dt if use_drift else torch.zeros_like(z)

        if use_diffusion:
            eps = torch.randn(n_paths, d, device=device, dtype=z.dtype)
            shock = torch.bmm(L, eps.unsqueeze(-1)).squeeze(-1) * sqrt_dt
        else:
            shock = torch.zeros_like(z)

        z = z + drift + shock

        if not torch.isfinite(z).all():
            raise RuntimeError(f"Non-finite z encountered at step {t + 1}")

        z_paths[:, t + 1, :] = z
        r_paths[:, t + 1] = get_r(model, z)

    return z_paths, r_paths


@torch.no_grad()
def inspect_increment_statistics(z_paths: torch.Tensor):
    dz = z_paths[:, 1:, :] - z_paths[:, :-1, :]
    dz_flat = dz.reshape(-1, dz.shape[-1])

    mean_inc = dz_flat.mean(dim=0)
    std_inc = dz_flat.std(dim=0)
    absmax_inc = dz_flat.abs().max(dim=0).values

    print("\n================ INCREMENT STATISTICS ================")
    for j in range(dz.shape[-1]):
        print(
            f"dim {j+1}: "
            f"mean(dz)={mean_inc[j].item():.6f}   "
            f"std(dz)={std_inc[j].item():.6f}   "
            f"max|dz|={absmax_inc[j].item():.6f}"
        )


@torch.no_grad()
def compare_training_vs_simulated_latent_ranges(Z_train: torch.Tensor, z_paths: torch.Tensor):
    z_sim = z_paths.reshape(-1, z_paths.shape[-1])

    train_min = Z_train.min(dim=0).values
    train_max = Z_train.max(dim=0).values
    sim_min = z_sim.min(dim=0).values
    sim_max = z_sim.max(dim=0).values

    print("\n================ TRAINING VS SIMULATED LATENT RANGE ================")
    for j in range(Z_train.shape[1]):
        print(
            f"dim {j + 1}: "
            f"train[min,max]=({train_min[j].item():.6f}, {train_max[j].item():.6f})   "
            f"sim[min,max]=({sim_min[j].item():.6f}, {sim_max[j].item():.6f})"
        )


def plot_latent_paths(z_paths: torch.Tensor, out_path: str, n_plot_paths: int = 30):
    z_cpu = z_paths.detach().cpu().numpy()
    n_paths, n_steps1, d = z_cpu.shape
    n_show = min(n_plot_paths, n_paths)

    fig, axes = plt.subplots(d, 1, figsize=(8, 2.8 * d), dpi=160, sharex=True)
    if d == 1:
        axes = [axes]

    tgrid = np.arange(n_steps1)

    for j in range(d):
        for i in range(n_show):
            axes[j].plot(tgrid, z_cpu[i, :, j], linewidth=0.8, alpha=0.75)
        axes[j].set_ylabel(f"$z_{{{j + 1}}}$")
        axes[j].grid(True, alpha=0.25)

    axes[-1].set_xlabel("Simulation step")
    fig.suptitle("Simulated latent paths", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_increment_histograms(z_paths: torch.Tensor, out_path: str):
    dz = (z_paths[:, 1:, :] - z_paths[:, :-1, :]).detach().cpu().numpy()
    dz_flat = dz.reshape(-1, dz.shape[-1])
    d = dz_flat.shape[1]

    fig, axes = plt.subplots(d, 1, figsize=(8, 2.8 * d), dpi=160)
    if d == 1:
        axes = [axes]

    for j in range(d):
        axes[j].hist(dz_flat[:, j], bins=50, alpha=0.8)
        axes[j].set_xlabel(f"$\\Delta z_{{{j + 1}}}$")
        axes[j].set_ylabel("Count")
        axes[j].grid(True, alpha=0.25)

    fig.suptitle("One-step increment histograms", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_latent_range_vs_training(Z_train: torch.Tensor, z_paths: torch.Tensor, out_path: str):
    Z_train_cpu = Z_train.detach().cpu().numpy()
    z_sim_cpu = z_paths.reshape(-1, z_paths.shape[-1]).detach().cpu().numpy()

    d = Z_train_cpu.shape[1]
    fig, axes = plt.subplots(d, 1, figsize=(8, 2.8 * d), dpi=160)
    if d == 1:
        axes = [axes]

    for j in range(d):
        train_min = Z_train_cpu[:, j].min()
        train_max = Z_train_cpu[:, j].max()
        sim_min = z_sim_cpu[:, j].min()
        sim_max = z_sim_cpu[:, j].max()

        axes[j].axvspan(train_min, train_max, alpha=0.3, label="training range")
        axes[j].axvline(sim_min, linestyle="--", linewidth=1.2, label="sim min")
        axes[j].axvline(sim_max, linestyle="--", linewidth=1.2, label="sim max")
        axes[j].set_ylabel(f"$z_{{{j + 1}}}$")
        axes[j].grid(True, alpha=0.25)

    axes[0].legend()
    axes[-1].set_xlabel("Latent value")
    fig.suptitle("Training latent range vs simulated latent range", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def print_summary(z0: torch.Tensor, meta_row, z_paths: torch.Tensor, r_paths: torch.Tensor):
    print("\n================ INITIAL STATE ================")
    if meta_row is not None:
        try:
            print(meta_row)
        except Exception:
            pass

    print("z0 shape:", tuple(z0.shape))
    print("z0:", z0.detach().cpu().numpy())

    print("\n================ SIMULATION SHAPES ================")
    print("z_paths:", tuple(z_paths.shape))
    print("r_paths:", tuple(r_paths.shape))

    z_last = z_paths[:, -1, :].detach().cpu()
    r_last = r_paths[:, -1].detach().cpu()

    print("\n================ TERMINAL SUMMARY ================")
    for j in range(z_last.shape[1]):
        print(
            f"z[{j}]  mean={z_last[:, j].mean():.6f}  "
            f"std={z_last[:, j].std():.6f}  "
            f"min={z_last[:, j].min():.6f}  "
            f"max={z_last[:, j].max():.6f}"
        )

    print(
        f"r(T)   mean={r_last.mean():.6f}  "
        f"std={r_last.std():.6f}  "
        f"min={r_last.min():.6f}  "
        f"max={r_last.max():.6f}"
    )


# ==========================================================
# SECTION 5B: Decode discount curves directly in this script
# ==========================================================
@torch.no_grad()
def decode_from_latent_script(model: FullModel, z: torch.Tensor):
    squeeze_back = False

    if z.dim() == 1:
        z = z.unsqueeze(0)
        squeeze_back = True

    device = z.device
    dtype = z.dtype
    tau = torch.arange(0, model.tau_max + 1, device=device, dtype=dtype)

    G_vals = model.G(z, tau)
    if G_vals.dim() == 1:
        G_vals = G_vals.unsqueeze(0)

    mu = model.K(z)
    sigmas, rhos = model.H(z)
    r_tilde = model.R(z)
    sigma = L_from_sigmas_rhos(sigmas, rhos)

    def G_single(z_single: torch.Tensor) -> torch.Tensor:
        return model.G(z_single.unsqueeze(0), tau).squeeze(0)

    dG_dtau = d_tau_autograd_nodewise(model.G, z, tau)
    grad_z_G, trace_cov_hess = grad_and_trace_cov_hess_G(G_single, z, sigma)

    alpha, beta, gamma = paper_alpha_beta_gamma_trace(
        G=G_vals,
        dG_dtau=dG_dtau,
        grad_z_G=grad_z_G,
        trace_cov_hess=trace_cov_hess,
        mu=mu,
        sigma=sigma,
        r_tilde=r_tilde,
    )

    A_vals, B_vals = solve_AB(tau, alpha, beta, gamma)

    r = r_tilde if r_tilde.ndim == 2 else r_tilde.unsqueeze(1)
    r = r.expand(-1, G_vals.shape[1])

    gTmu = (grad_z_G * mu.unsqueeze(1)).sum(dim=2)
    bracket = dG_dtau - gTmu - 0.5 * trace_cov_hess

    dB_dtau = alpha * B_vals + beta
    dA_dtau = gamma * (B_vals ** 2)

    R_tau = (
        -r
        - dA_dtau
        + G_vals * dB_dtau
        + B_vals * bracket
        + (B_vals ** 2) * gamma
    )

    sigma_bar = 0.006
    tau_safe = torch.clamp(tau.unsqueeze(0), min=1e-8)
    SR_tau = R_tau / (tau_safe * sigma_bar)

    arb = {
        "R_tau": R_tau[:, 1:],
        "SR_tau": SR_tau[:, 1:],
        "tau_grid": tau[1:],
        "max_abs_R": R_tau[:, 1:].abs().max(dim=1).values,
        "max_abs_SR_1to30": SR_tau[:, 1:].abs().max(dim=1).values,
    }

    expo = A_vals - B_vals * G_vals
    P_full = torch.exp(expo)
    P_mkt = P_full[:, 1:]

    if squeeze_back:
        P_mkt = P_mkt.squeeze(0)
        A_vals = A_vals.squeeze(0)
        B_vals = B_vals.squeeze(0)
        G_vals = G_vals.squeeze(0)
        mu = mu.squeeze(0)
        sigma = sigma.squeeze(0)
        if isinstance(r_tilde, torch.Tensor):
            r_tilde = r_tilde.squeeze(0)
        arb = {
            "R_tau": arb["R_tau"].squeeze(0),
            "SR_tau": arb["SR_tau"].squeeze(0),
            "tau_grid": arb["tau_grid"],
            "max_abs_R": arb["max_abs_R"].squeeze(0),
            "max_abs_SR_1to30": arb["max_abs_SR_1to30"].squeeze(0),
        }

    return P_mkt, A_vals, B_vals, G_vals, mu, sigma, r_tilde, arb


@torch.no_grad()
def discount_to_spot_yields(P_tau: torch.Tensor) -> torch.Tensor:
    if P_tau.dim() == 1:
        tau = torch.arange(1, P_tau.shape[0] + 1, device=P_tau.device, dtype=P_tau.dtype)
        return -torch.log(torch.clamp(P_tau, min=1e-12)) / tau
    elif P_tau.dim() == 2:
        tau = torch.arange(1, P_tau.shape[1] + 1, device=P_tau.device, dtype=P_tau.dtype).unsqueeze(0)
        return -torch.log(torch.clamp(P_tau, min=1e-12)) / tau
    else:
        raise ValueError(f"Expected P_tau dim 1 or 2, got {P_tau.dim()}")


@torch.no_grad()
def decode_curve_at_time_index(model: FullModel, z_paths: torch.Tensor, time_index: int):
    z_t = z_paths[:, time_index, :]
    P_tau, _, _, _, _, _, _, arb = decode_from_latent_script(model, z_t)
    y_tau = discount_to_spot_yields(P_tau)
    return P_tau, y_tau, arb


@torch.no_grad()
def inspect_curve_finiteness_and_monotonicity(model: FullModel, z_paths: torch.Tensor, time_index: int):
    P_tau, y_tau, arb = decode_curve_at_time_index(model, z_paths, time_index)

    frac_finite_P = torch.isfinite(P_tau).float().mean().item()
    frac_finite_y = torch.isfinite(y_tau).float().mean().item()
    finite_y_by_maturity = torch.isfinite(y_tau).float().mean(dim=0)
    increasing_df_frac = (P_tau[:, 1:] > P_tau[:, :-1]).float().mean().item()

    print("\n================ CURVE FINITENESS / MONOTONICITY CHECK ================")
    print(f"time_index = {time_index}")
    print(f"Fraction finite P_tau overall: {frac_finite_P:.6f}")
    print(f"Fraction finite y_tau overall: {frac_finite_y:.6f}")
    print(f"Fraction discount-curve increases with maturity: {increasing_df_frac:.6f}")
    print("Finite yield fraction by maturity:")
    print(finite_y_by_maturity.detach().cpu().numpy())

    print("Arbitrage diagnostic summary:")
    print(f"max |R_tau| mean  = {arb['max_abs_R'].mean().item():.6f}")
    print(f"max |R_tau| max   = {arb['max_abs_R'].max().item():.6f}")
    print(f"max |SR_tau| mean = {arb['max_abs_SR_1to30'].mean().item():.6f}")
    print(f"max |SR_tau| max  = {arb['max_abs_SR_1to30'].max().item():.6f}")


def plot_mean_yield_curves_over_time(
    model: FullModel,
    z_paths: torch.Tensor,
    dt: float,
    years_to_plot,
    out_path: str
):
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)

    for yr in years_to_plot:
        idx = int(round(yr / dt))
        if idx >= z_paths.shape[1]:
            raise ValueError(f"Requested year {yr} exceeds simulated horizon")

        _, y_tau, _ = decode_curve_at_time_index(model, z_paths, idx)
        y_mean = y_tau.mean(dim=0).detach().cpu().numpy()

        maturities = np.arange(1, len(y_mean) + 1)
        ax.plot(maturities, y_mean, linewidth=1.5, label=f"t={yr:g}Y")

    ax.set_xlabel("Maturity (years)")
    ax.set_ylabel("Spot yield")
    ax.set_title("Average simulated spot-yield curves")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_sample_yield_curves_at_time(
    model: FullModel,
    z_paths: torch.Tensor,
    dt: float,
    year_to_plot: float,
    n_sample_paths: int,
    out_path: str
):
    idx = int(round(year_to_plot / dt))
    if idx >= z_paths.shape[1]:
        raise ValueError(f"Requested year {year_to_plot} exceeds simulated horizon")

    _, y_tau, _ = decode_curve_at_time_index(model, z_paths, idx)
    y_cpu = y_tau.detach().cpu().numpy()

    n_show = min(n_sample_paths, y_cpu.shape[0])
    maturities = np.arange(1, y_cpu.shape[1] + 1)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    for i in range(n_show):
        ax.plot(maturities, y_cpu[i], linewidth=0.9, alpha=0.75)

    ax.set_xlabel("Maturity (years)")
    ax.set_ylabel("Spot yield")
    ax.set_title(f"Sample simulated spot-yield curves at t={year_to_plot:g}Y")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# ==========================================================
# SECTION 6: Run everything
# ==========================================================
set_seed(SEED)

print("\nLoading model...")
model = load_trained_model(CHECKPOINT_PATH, latent_dim=LATENT_DIM, device=device)
inspect_K_matrix(model)

print("Loading initial curve...")
S0, meta_row, X_tensor, meta = load_initial_curve(USE, IDX_CHOICE, device=device)

with torch.no_grad():
    z0 = encode_initial_state(model, S0)

print("Encoding training set for latent-range comparison...")
with torch.no_grad():
    Z_train = encode_training_set(model, X_tensor, device=device)

inspect_initial_drift_and_vol(model, z0)

print("\nSimulating latent paths...")
with torch.no_grad():
    z_paths, r_paths = simulate_latent_paths(
        model=model,
        z0=z0,
        n_paths=N_PATHS,
        n_steps=N_STEPS,
        dt=DT,
        device=device,
        use_drift=USE_DRIFT,
        use_diffusion=USE_DIFFUSION,
    )

print_summary(z0, meta_row, z_paths, r_paths)
inspect_increment_statistics(z_paths)
compare_training_vs_simulated_latent_ranges(Z_train, z_paths)

check_idx = int(round(CHECK_FINITE_YEAR / DT))
if check_idx < z_paths.shape[1]:
    inspect_curve_finiteness_and_monotonicity(model, z_paths, check_idx)

mode_tag = f"drift{int(USE_DRIFT)}_diff{int(USE_DIFFUSION)}"

plot_path = os.path.join(
    OUT_DIR,
    f"latent_paths_{USE}_dim{LATENT_DIM}_ep{EPOCHS}_{mode_tag}.png"
)
plot_latent_paths(z_paths, plot_path, n_plot_paths=N_PLOT_PATHS)
print("\nSaved plot to:", plot_path)

increment_hist_path = os.path.join(
    OUT_DIR,
    f"increment_histograms_{USE}_dim{LATENT_DIM}_ep{EPOCHS}_{mode_tag}.png"
)
plot_increment_histograms(z_paths, increment_hist_path)
print("Saved increment histogram plot to:", increment_hist_path)

range_plot_path = os.path.join(
    OUT_DIR,
    f"latent_range_vs_training_{USE}_dim{LATENT_DIM}_ep{EPOCHS}_{mode_tag}.png"
)
plot_latent_range_vs_training(Z_train, z_paths, range_plot_path)
print("Saved latent-range comparison plot to:", range_plot_path)

print("\nDecoding simulated curves...")

mean_curve_plot_path = os.path.join(
    OUT_DIR,
    f"mean_simulated_yield_curves_{USE}_dim{LATENT_DIM}_ep{EPOCHS}_{mode_tag}.png"
)
plot_mean_yield_curves_over_time(
    model=model,
    z_paths=z_paths,
    dt=DT,
    years_to_plot=YEARS_TO_PLOT,
    out_path=mean_curve_plot_path
)
print("Saved mean yield-curve plot to:", mean_curve_plot_path)

sample_curve_plot_path = os.path.join(
    OUT_DIR,
    f"sample_yield_curves_t{int(SAMPLE_CURVE_YEAR)}Y_{USE}_dim{LATENT_DIM}_ep{EPOCHS}_{mode_tag}.png"
)
plot_sample_yield_curves_at_time(
    model=model,
    z_paths=z_paths,
    dt=DT,
    year_to_plot=SAMPLE_CURVE_YEAR,
    n_sample_paths=N_PLOT_YIELD_PATHS,
    out_path=sample_curve_plot_path
)
print("Saved sample yield-curve plot to:", sample_curve_plot_path)
