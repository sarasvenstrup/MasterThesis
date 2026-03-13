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
LATENT_DIM = 1
EPOCHS = 5000   # must match saved model

CHECKPOINT_PATH = os.path.join(
    REPO_ROOT,
    "checkpoints",
    f"fullmodel_{USE}_dim{LATENT_DIM}_ep{EPOCHS}.pt"
)

OUT_DIR = os.path.join(REPO_ROOT, "Figures", "pricing_debug")
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 1234

# choose initial row from your dataset
IDX_CHOICE = -1   # last row

# simulation controls
N_PATHS = 200
DT = 1.0 / 12.0      # monthly step
N_YEARS = 10
N_STEPS = int(round(N_YEARS / DT))

# how many paths to show in the plot
N_PLOT_PATHS = 30

# ==========================================================
# SECTION 3: Small helpers
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
    meta, X_tensor, meta_full, X_tensor_full, tenors, df_wide, SCALE_IS_PERCENT = my_data(use=use)
    X_tensor = X_tensor.float()

    if idx_choice < 0:
        idx_choice = X_tensor.shape[0] + idx_choice

    if idx_choice < 0 or idx_choice >= X_tensor.shape[0]:
        raise IndexError(f"idx_choice={idx_choice} out of bounds")

    S0 = X_tensor[idx_choice:idx_choice+1].to(device)
    meta_row = meta.iloc[idx_choice] if hasattr(meta, "iloc") else None
    return S0, meta_row, X_tensor, meta

@torch.no_grad()
def encode_initial_state(model: FullModel, S0: torch.Tensor) -> torch.Tensor:
    return model.encoder(S0)

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

# ==========================================================
# SECTION 4: Simulate latent paths
# ==========================================================
@torch.no_grad()
def simulate_latent_paths(
    model: FullModel,
    z0: torch.Tensor,
    n_paths: int,
    n_steps: int,
    dt: float,
    device: torch.device
):
    """
    Euler simulation:
        z_{n+1} = z_n + mu(z_n) dt + L(z_n) sqrt(dt) eps_n

    Returns
    -------
    z_paths : (n_paths, n_steps+1, d)
    r_paths : (n_paths, n_steps+1)
    """
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
        mu = get_mu(model, z)               # (M,d)
        L = get_L(model, z)                 # (M,d,d)

        eps = torch.randn(n_paths, d, device=device, dtype=z.dtype)
        shock = torch.bmm(L, eps.unsqueeze(-1)).squeeze(-1)

        z = z + mu * dt + shock * sqrt_dt

        if not torch.isfinite(z).all():
            raise RuntimeError(f"Non-finite z encountered at step {t+1}")

        z_paths[:, t + 1, :] = z
        r_paths[:, t + 1] = get_r(model, z)

    return z_paths, r_paths

# ==========================================================
# SECTION 5: Plot helpers
# ==========================================================
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
        axes[j].set_ylabel(f"$z_{{{j+1}}}$")
        axes[j].grid(True, alpha=0.25)

    axes[-1].set_xlabel("Simulation step")
    fig.suptitle("Simulated latent paths", y=0.995)
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
# SECTION 6: Run everything
# ==========================================================
set_seed(SEED)

print("\nLoading model...")
model = load_trained_model(CHECKPOINT_PATH, latent_dim=LATENT_DIM, device=device)

print("Loading initial curve...")
S0, meta_row, X_tensor, meta = load_initial_curve(USE, IDX_CHOICE, device=device)

with torch.no_grad():
    z0 = encode_initial_state(model, S0)

print("Simulating latent paths...")
with torch.no_grad():
    z_paths, r_paths = simulate_latent_paths(
        model=model,
        z0=z0,
        n_paths=N_PATHS,
        n_steps=N_STEPS,
        dt=DT,
        device=device
    )

print_summary(z0, meta_row, z_paths, r_paths)

plot_path = os.path.join(OUT_DIR, "latent_paths_only.png")
plot_latent_paths(z_paths, plot_path, n_plot_paths=N_PLOT_PATHS)
print("\nSaved plot to:", plot_path)