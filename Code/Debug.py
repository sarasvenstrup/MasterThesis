# debug_shapes_3factor.py
# ------------------------------------------------------------
# Run ONE forward pass through FullModel with latent_dim=3
# and print key tensor shapes + a PSD sanity check on sigma.
# ------------------------------------------------------------

# ============================= Import Packages ===============================
import os
import sys
import torch
import torch.nn as nn

# --- Torch thread settings MUST be first Torch-related thing ---
torch.set_num_threads(4)
torch.set_num_interop_threads(2)

import pandas as pd  # noqa: F401
import matplotlib.pyplot as plt  # noqa: F401

# ============================= Environment Setup & Imports ===============================
try:
    REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
except NameError:
    REPO_ROOT = os.getcwd()

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Code.load_swapdata import my_data, custom_palette, TARGET_TENORS  # noqa: F401
from Code.model.full_model import FullModel

# ============================= Device ===============================
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("MPS available:", hasattr(torch.backends, "mps") and torch.backends.mps.is_available())

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Using device:", device)

torch.backends.mkldnn.enabled = True
print("CPU threads:", torch.get_num_threads(), "interop:", torch.get_num_interop_threads())

# If you want value printing to be readable:
torch.set_printoptions(precision=10, sci_mode=True)

# ============================= Load Data ===============================
ccy_order = ["AUD", "CAD", "DKK", "EUR", "JPY", "NOK", "SEK", "GBP", "USD"]
currency_color_map = {ccy: custom_palette[i % len(custom_palette)] for i, ccy in enumerate(ccy_order)}  # noqa: F401

USE = "bbg"  # or "test"
meta, X_tensor, tenors, df_wide, SCALE_IS_PERCENT = my_data(use=USE)  # noqa: F841
X_tensor = X_tensor.float()

# ============================= One-batch loader ===============================
from torch.utils.data import TensorDataset, DataLoader

dataset = TensorDataset(X_tensor)
loader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=False)

# ============================= Build Model (3-factor) ===============================
torch.manual_seed(0)

LATENT_DIM = 3
model = FullModel(latent_dim=LATENT_DIM).to(device)
model.train()  # IMPORTANT: keep grads enabled (do NOT use torch.no_grad)

print("\n--- Debug run (one batch) ---")
xb = next(iter(loader))[0].to(device)[:100]  # take 2 samples
print("xb:", tuple(xb.shape), xb.dtype, xb.device)
print("model.latent_dim:", model.latent_dim)

# ============================= Forward pass ===============================
# This will print shapes from inside FullModel.forward() (DEBUG_SHAPES=True there).
out = model(xb)

# ============================= Quick external shape checks ===============================
S_hat = out[0]
z = out[1]
P_mkt = out[2]
A_vals = out[3]
B_vals = out[4]
G_vals = out[5]
mu = out[6]
sigma = out[7]
r_tilde = out[8]
arb = out[9]

print("\n--- Returned tuple shapes ---")
print("S_hat:", tuple(S_hat.shape))
print("z:", tuple(z.shape))
print("P_mkt:", tuple(P_mkt.shape))
print("A_vals:", tuple(A_vals.shape))
print("B_vals:", tuple(B_vals.shape))
print("G_vals:", tuple(G_vals.shape))
print("mu:", tuple(mu.shape))
print("sigma:", tuple(sigma.shape))
print("r_tilde:", tuple(r_tilde.shape))
print("arb['SR_tau']:", tuple(arb["SR_tau"].shape))

# ============================= PSD sanity check for sigma ===============================
# sigma is Cholesky-like (lower-triangular). cov = sigma @ sigma^T must be PSD.
with torch.no_grad():
    cov = sigma @ sigma.transpose(-1, -2)
    eig_min = torch.linalg.eigvalsh(cov).min(dim=-1).values  # (B,)
    print("\n--- Sigma PSD check ---")
    print("min_eig(cov):", eig_min.detach().cpu())
    print("any NaN/Inf in sigma?:", (not torch.isfinite(sigma).all().item()))

with torch.no_grad():
    print("min|G|:", G_vals.abs().min(dim=1).values)


with torch.no_grad():
    print("max|R_tau|:", arb["max_abs_R"])

print("\nDone.")