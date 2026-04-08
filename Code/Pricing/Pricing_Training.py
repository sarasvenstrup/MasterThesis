# =============================================================================
# pricing_training_simple.py  (v2)
# =============================================================================
# Builds on v1 (theta fix + transition NLL).
#
# What changed vs v1:
#   - Added sigma regularisation toward SIGMA_TARGET.
#     v1 fitted sigma to historical transitions -> sigma ~0.012, vols ~30-39 bp.
#     Market 5Y/10Y vols are ~18-27 bp, requiring lower sigma (~0.008).
#     The NLL pushes back if data truly needs more vol, so the balance
#     point is determined by both losses together.
#
# What stays the same:
#   - theta = EUR latent mean, frozen
#   - encoder, G, R frozen
#   - no cloud penalty, no kappa reinit, no G updates
# =============================================================================

import os
import sys
import time
import math

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()

CODE_ROOT   = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
THESIS_ROOT = os.path.abspath(os.path.join(CODE_ROOT,  ".."))
for p in [THESIS_ROOT, CODE_ROOT, SCRIPT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from Code import config
config.confirm_variant()
from Code.load_swapdata import my_data
from Code.model.full_model import FullModel
from Code.model.sigma_matrix import L_from_sigmas_rhos

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Torch   : {torch.__version__}")
print(f"Device  : {device}")
print(f"Variant : {config.VARIANT}")

# =============================================================================
# SETTINGS
# =============================================================================

CHECKPOINT_PATH = (
    r"C:\Users\Bruger\PycharmProjects\MasterThesis\Figures\TrainingResults"
    r"\dim2_stable\ep300\checkpoint_dim2_ep300.pt"
)

OUT_DIR = os.path.join(
    THESIS_ROOT, "Figures", "TrainingResults",
    f"dim2_{config.VARIANT}", "pricing_simple_v2"
)
os.makedirs(OUT_DIR, exist_ok=True)

CCY        = "EUR"
EPOCHS     = 200
BATCH_SIZE = 32
LR         = 1e-4
VAL_FRAC   = 0.15

# Sigma regularisation — disabled (LAMBDA_SIGMA=0 turns it off).
# The sigma penalty did not move sigma meaningfully vs the NLL.
# The good vol results from v2 come from the correct theta, not sigma reg.
# Kept here in case you want to experiment — set LAMBDA_SIGMA > 0 to activate.
SIGMA_TARGET = 0.008
LAMBDA_SIGMA = 0.0

JITTER = 1e-8

# =============================================================================
# DATA
# =============================================================================

def build_transition_pairs(df_wide, tenors, scale_is_percent, ccy):
    df = df_wide.copy()
    df["as_of_date"] = pd.to_datetime(df["as_of_date"])
    df = df[df["ccy"].astype(str).str.upper() == ccy.upper()].copy()
    df = df.sort_values("as_of_date").reset_index(drop=True)

    tenor_cols = [int(t) for t in tenors]
    X_arr = df[tenor_cols].to_numpy(dtype=np.float64)
    if scale_is_percent:
        X_arr /= 100.0

    dates = df["as_of_date"].to_numpy()
    X_t_list, X_tp1_list, dt_list = [], [], []
    for i in range(len(df) - 1):
        dt_days = (dates[i + 1] - dates[i]) / np.timedelta64(1, "D")
        if dt_days <= 0:
            continue
        X_t_list.append(X_arr[i])
        X_tp1_list.append(X_arr[i + 1])
        dt_list.append(float(dt_days) / 365.25)

    X_t  = torch.tensor(np.array(X_t_list),  dtype=torch.float64)
    Xtp1 = torch.tensor(np.array(X_tp1_list), dtype=torch.float64)
    dt   = torch.tensor(np.array(dt_list),    dtype=torch.float64).unsqueeze(1)
    print(f"  Transition pairs: {len(X_t_list)}  "
          f"dt mean={dt.mean().item():.4f}y  max={dt.max().item():.4f}y")
    return TensorDataset(X_t, Xtp1, dt)


def split_timewise(dataset, val_frac):
    n     = len(dataset)
    n_val = max(1, int(math.ceil(val_frac * n)))
    return (torch.utils.data.Subset(dataset, range(n - n_val)),
            torch.utils.data.Subset(dataset, range(n - n_val, n)))


# =============================================================================
# LOSSES
# =============================================================================

def transition_nll(model, z_t, z_tp1, dt):
    """Euler-Gaussian NLL for z_{t+1} | z_t."""
    mu   = model.K(z_t)
    sigmas, rhos = model.H(z_t)
    L    = L_from_sigmas_rhos(sigmas, rhos)
    B, d = z_t.shape
    I    = torch.eye(d, device=z_t.device, dtype=z_t.dtype).unsqueeze(0)
    Sigma = (L @ L.transpose(1, 2)) * dt.view(-1, 1, 1) + JITTER * I
    resid = (z_tp1 - z_t - mu * dt).unsqueeze(-1)
    quad  = resid.transpose(1, 2).bmm(
                torch.linalg.solve(Sigma, resid)
            ).squeeze()
    return (0.5 * (quad + torch.logdet(Sigma))).mean()


def sigma_reg(model, z_t):
    """Pull sigma mean toward SIGMA_TARGET."""
    sigmas, _ = model.H(z_t)
    return (sigmas.mean() - SIGMA_TARGET).pow(2)


# =============================================================================
# LOAD DATA
# =============================================================================
print("\n── Loading data ──")
(meta, X_tensor, meta_full, X_tensor_full,
 tenors, df_wide, df_wide_all, SCALE_IS_PERCENT) = my_data(use="bbg")

X_tensor = X_tensor.double()
mask_ccy = meta["ccy"].astype(str).str.upper() == CCY.upper()
X_ccy    = X_tensor[mask_ccy.to_numpy()]
print(f"  EUR curves : {X_ccy.shape[0]}")

print("\n── Building transition pairs ──")
full_ds = build_transition_pairs(df_wide, tenors, SCALE_IS_PERCENT, CCY)
train_set, val_set = split_timewise(full_ds, VAL_FRAC)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  drop_last=False)
val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
print(f"  Train: {len(train_set)}   Val: {len(val_set)}")

# =============================================================================
# LOAD MODEL
# =============================================================================
print(f"\n── Loading checkpoint ──")
raw_ckpt   = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
state_dict = (raw_ckpt["model_state_dict"]
              if isinstance(raw_ckpt, dict) and "model_state_dict" in raw_ckpt
              else raw_ckpt)

model = FullModel(latent_dim=2).to(device).double()
model.load_state_dict(state_dict, strict=False)

for p in model.encoder.parameters():
    p.requires_grad = False
for p in model.G.parameters():
    p.requires_grad = False
for p in model.R.parameters():
    p.requires_grad = False

# =============================================================================
# SET AND FREEZE THETA
# =============================================================================
print(f"\n── Setting theta to EUR latent mean ──")
with torch.no_grad():
    zs = []
    for i in range(0, X_ccy.shape[0], 256):
        zs.append(model.encoder(X_ccy[i:i+256].to(device)).detach().cpu())
    z_eur_mean = torch.cat(zs, dim=0).mean(dim=0)

print(f"  EUR latent mean : {z_eur_mean.numpy()}")
print(f"  theta before    : {model.K.theta.detach().cpu().numpy()}")
with torch.no_grad():
    model.K.theta.copy_(z_eur_mean.to(device=device, dtype=model.K.theta.dtype))
model.K.theta.requires_grad = False
print(f"  theta after     : {model.K.theta.detach().cpu().numpy()}  [FROZEN]")

# =============================================================================
# OPTIMIZER
# =============================================================================
trainable = [p for p in model.parameters() if p.requires_grad]
print(f"\n── Trainable parameters: {sum(p.numel() for p in trainable):,}  (K excl. theta, H)")
print(f"   SIGMA_TARGET={SIGMA_TARGET}  LAMBDA_SIGMA={LAMBDA_SIGMA}")

optim     = torch.optim.Adam(trainable, lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=EPOCHS, eta_min=1e-6)

# =============================================================================
# HELPERS
# =============================================================================
def curve_rmse(model, X_ccy):
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, X_ccy.shape[0], 256):
            outs.append(model(X_ccy[i:i+256].to(device)).detach().cpu())
    S_hat = torch.cat(outs)
    rmse  = float(((X_ccy.cpu() - S_hat)**2).mean().sqrt().item() * 10000)
    model.train()
    return rmse

def kappa_hl(model):
    k = F.softplus(model.K.raw_kappa).item()
    e = torch.linalg.eigvals(model.K.drift_matrix()).real.detach().cpu().numpy()
    hl = math.log(2) / float(np.abs(e).min()) if np.abs(e).min() > 0 else float("inf")
    return k, hl

rmse_init = curve_rmse(model, X_ccy)
k0, hl0   = kappa_hl(model)
with torch.no_grad():
    z_s = model.encoder(X_ccy[:32].to(device))
    s0, _ = model.H(z_s)
print(f"\n  Initial: kappa={k0:.4f} hl={hl0:.2f}y  "
      f"sigma={s0.mean().item():.6f}  rmse={rmse_init:.2f}bp")

# =============================================================================
# TRAINING LOOP
# =============================================================================
print(f"\n── Training ({EPOCHS} epochs) ──")

csv_path = os.path.join(OUT_DIR, "metrics.csv")
cols = ["epoch", "train_nll", "train_sigma_loss", "train_total",
        "val_nll", "val_sigma_loss", "val_total",
        "curve_rmse_bp", "kappa", "hl_y", "sigma_mean", "lr", "time_sec"]
pd.DataFrame(columns=cols).to_csv(csv_path, index=False)

best_val  = float("inf")
best_path = os.path.join(OUT_DIR, "best_checkpoint.pt")
t0        = time.perf_counter()

for epoch in range(EPOCHS):
    model.train()
    sum_nll, sum_sig, sum_tot, n_train, n_nan = 0., 0., 0., 0, 0

    for X_t, X_tp1, dt in train_loader:
        X_t, X_tp1, dt = X_t.to(device), X_tp1.to(device), dt.to(device)
        optim.zero_grad()

        with torch.no_grad():
            z_t   = model.encoder(X_t)
            z_tp1 = model.encoder(X_tp1)

        l_nll = transition_nll(model, z_t, z_tp1, dt)
        l_sig = sigma_reg(model, z_t)
        loss  = l_nll + LAMBDA_SIGMA * l_sig

        if not torch.isfinite(loss):
            n_nan += 1
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optim.step()

        bs = X_t.shape[0]
        sum_nll += l_nll.item() * bs
        sum_sig += l_sig.item() * bs
        sum_tot += loss.item()  * bs
        n_train += bs

    scheduler.step()

    # Validation
    model.eval()
    v_nll, v_sig, v_tot, n_val = 0., 0., 0., 0
    with torch.no_grad():
        for X_t, X_tp1, dt in val_loader:
            X_t, X_tp1, dt = X_t.to(device), X_tp1.to(device), dt.to(device)
            z_t   = model.encoder(X_t)
            z_tp1 = model.encoder(X_tp1)
            ln = transition_nll(model, z_t, z_tp1, dt)
            ls = sigma_reg(model, z_t)
            lt = ln + LAMBDA_SIGMA * ls
            if torch.isfinite(lt):
                v_nll += ln.item() * X_t.shape[0]
                v_sig += ls.item() * X_t.shape[0]
                v_tot += lt.item() * X_t.shape[0]
                n_val += X_t.shape[0]

    n  = max(n_train, 1);  nv = max(n_val, 1)
    tr_nll = sum_nll / n;  tr_sig = sum_sig / n;  tr_tot = sum_tot / n
    vl_nll = v_nll / nv;   vl_sig = v_sig / nv;   vl_tot = v_tot / nv

    if vl_tot < best_val:
        best_val = vl_tot
        torch.save(model.state_dict(), best_path)

    k, hl = kappa_hl(model)
    with torch.no_grad():
        z_s = model.encoder(X_ccy[:32].to(device))
        sig, _ = model.H(z_s)
        sigma_now = float(sig.mean().item())
    rmse_bp = curve_rmse(model, X_ccy)
    lr_now  = optim.param_groups[0]["lr"]
    elapsed = time.perf_counter() - t0

    row = {
        "epoch": epoch,
        "train_nll": tr_nll, "train_sigma_loss": tr_sig, "train_total": tr_tot,
        "val_nll": vl_nll,   "val_sigma_loss": vl_sig,   "val_total": vl_tot,
        "curve_rmse_bp": rmse_bp, "kappa": k, "hl_y": hl,
        "sigma_mean": sigma_now, "lr": lr_now, "time_sec": elapsed,
    }
    pd.DataFrame([row], columns=cols).to_csv(csv_path, mode="a", header=False, index=False)

    print(f"ep {epoch:3d} | "
          f"nll={tr_nll:+.3f}  sig={tr_sig:.6f}  tot={tr_tot:.3f} | "
          f"val={vl_tot:.3f} | "
          f"rmse={rmse_bp:.1f}bp | "
          f"kappa={k:.4f} hl={hl:.2f}y | "
          f"sigma={sigma_now:.5f}→{SIGMA_TARGET} | "
          f"lr={lr_now:.1e}  nan={n_nan}  t={elapsed/60:.1f}min")

# =============================================================================
# SAVE & SUMMARY
# =============================================================================
final_path = os.path.join(OUT_DIR, "final_checkpoint.pt")
torch.save(model.state_dict(), final_path)

k_f, hl_f = kappa_hl(model)
with torch.no_grad():
    z_s = model.encoder(X_ccy[:32].to(device))
    sf, _ = model.H(z_s)

print(f"\nDone.")
print(f"  kappa={k_f:.4f}  hl={hl_f:.2f}y")
print(f"  sigma_mean={sf.mean().item():.6f}  (target={SIGMA_TARGET})")
print(f"  curve RMSE : {curve_rmse(model, X_ccy):.2f} bp  (was {rmse_init:.2f} bp)")
print(f"  theta      : {model.K.theta.detach().cpu().numpy()}  [frozen]")
print(f"  Saved final : {final_path}")
print(f"  Saved best  : {best_path}")