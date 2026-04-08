# =============================================================================
# Pricing Continuation Training  (v4)
# =============================================================================
# Changes vs v3:
#   Fix A — K.theta is frozen after being set to the EUR latent mean.
#            In v3, theta drifted to [-0.007, 0.001] during training despite
#            being initialised to the EUR mean [-0.026, -0.003]. This pulled
#            paths toward the wrong center, causing z[0] to escape the training
#            box. Freezing theta prevents this and ensures mean-reversion always
#            pulls back to the correct EUR latent center.
#
#   Fix B — LAMBDA_SIGMA raised from 500 to 2000.
#            Sigma flattened at 0.0106 in v3 (implied/training std = 1.58).
#            Target is below 1.2. Stronger penalty pushes the balance point
#            lower — the NLL will push back if the data truly requires larger
#            sigma, so this is safe to increase.
#
# All other changes from v3 are retained:
#   - Cloud stats from ALL currencies (not EUR-only)
#   - CosineAnnealingLR with higher base LR
#   - G unfrozen at 100x smaller LR
#   - Encoder frozen throughout
# =============================================================================

import os
import sys
import time
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

torch.set_num_threads(4)
torch.set_num_interop_threads(2)

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
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
from Code.utils import helpers as H

print(f"Torch        : {torch.__version__}")
print(f"CUDA         : {torch.cuda.is_available()}")
print(f"Variant      : {config.VARIANT}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device       : {device}")

# =============================================================================
# ── SETTINGS ─────────────────────────────────────────────────────────────────
# =============================================================================

CHECKPOINT_PATH = (
    r"C:\Users\Bruger\PycharmProjects\MasterThesis\Figures\TrainingResults"
    r"\dim2_stable\ep200\checkpoint_dim2_ep200.pt"
)

OUT_DIR = os.path.join(
    THESIS_ROOT, "Figures", "TrainingResults",
    f"dim2_{config.VARIANT}", "pricing_continuation"
)
os.makedirs(OUT_DIR, exist_ok=True)

# --- data ---
CCY = "EUR"
USE = "bbg"

# --- training ---
EPOCHS     = 300
BATCH_SIZE = 32

# --- learning rates ---
MAX_LR_KHR = 2e-4   # K, H, R  (theta is frozen so only B, L, raw_kappa, H, R update)
MAX_LR_G   = 2e-6   # G — 100x smaller, fine-tune only

# --- kappa reinitialisation ---
KAPPA_REINIT = 0.5  # softplus(raw_kappa) = 0.5 → half-life ~1.4y starting point

# --- loss weights ---
CURVE_SCALE  = 1e6
LAMBDA_CURVE = 2.0
LAMBDA_TRANS = 150.0
LAMBDA_CLOUD = 30.0

# --- sigma regularisation (Fix B: raised from 500 to 2000) ---
# v3 sigma flattened at 0.0106 (implied/training std = 1.58).
# With LAMBDA_SIGMA=2000, the balance point will be lower.
# The NLL pushes back proportionally, so this is safe.
SIGMA_TARGET = 0.010
LAMBDA_SIGMA = 2000.0

# --- cloud / rollout ---
ROLLOUT_STEPS   = 12
CLOUD_THRESHOLD = 2.5
CLOUD_JITTER    = 1e-8

VAL_FRAC = 0.15

# =============================================================================
# ── HELPERS ──────────────────────────────────────────────────────────────────
# =============================================================================

def freeze(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False


def get_L(model, z):
    sigmas, rhos = model.H(z)
    return L_from_sigmas_rhos(sigmas, rhos)


def reinit_kappa(model, kappa_value: float):
    """Set raw_kappa so softplus(raw_kappa) = kappa_value. Call AFTER load_state_dict."""
    if not hasattr(model.K, "raw_kappa"):
        print("  [WARN] model.K has no raw_kappa — skipping")
        return
    raw_val = math.log(math.exp(kappa_value) - 1.0)
    with torch.no_grad():
        model.K.raw_kappa.fill_(raw_val)
    kappa_check = F.softplus(model.K.raw_kappa).item()
    M    = model.K.drift_matrix()
    eigs = torch.linalg.eigvals(M).real.detach().cpu().numpy()
    hl   = math.log(2) / float(np.abs(eigs).min()) if np.abs(eigs).min() > 0 else float("inf")
    print(f"  raw_kappa → {raw_val:.4f}   kappa = {kappa_check:.4f}")
    print(f"  M eigenvalues : {np.round(eigs, 4)}")
    print(f"  Half-life     : {hl:.2f}y")


def print_dynamics(model, label=""):
    if not hasattr(model.K, "raw_kappa"):
        return
    kappa = F.softplus(model.K.raw_kappa).item()
    M     = model.K.drift_matrix()
    eigs  = torch.linalg.eigvals(M).real.detach().cpu().numpy()
    theta = model.K.theta.detach().cpu().numpy() if hasattr(model.K, "theta") else None
    hl    = math.log(2) / float(np.abs(eigs).min()) if np.abs(eigs).min() > 0 else float("inf")
    tag   = f"[{label}] " if label else ""
    print(f"  {tag}kappa={kappa:.4f}  eigs={np.round(eigs,4)}  "
          f"hl={hl:.2f}y  theta={theta}")


def print_sigma_diag(model, X_sample, label=""):
    with torch.no_grad():
        z = model.encoder(X_sample[:32].to(device))
        sigmas, _ = model.H(z)
        s = sigmas.detach().cpu().numpy()
    tag = f"[{label}] " if label else ""
    print(f"  {tag}sigma mean={s.mean(axis=0).round(6)}  "
          f"min={s.min(axis=0).round(6)}  max={s.max(axis=0).round(6)}")


# =============================================================================
# ── DATA ─────────────────────────────────────────────────────────────────────
# =============================================================================

def build_transition_pairs(df_wide, tenors, scale_is_percent, ccy):
    """Build (X_t, X_{t+1}, dt) pairs. dt from actual calendar dates."""
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

    X_t   = torch.tensor(np.array(X_t_list),   dtype=torch.float64)
    X_tp1 = torch.tensor(np.array(X_tp1_list), dtype=torch.float64)
    dt    = torch.tensor(np.array(dt_list),     dtype=torch.float64).unsqueeze(1)

    print(f"  Pairs for {ccy}: {len(dt_list)}")
    print(f"  dt  min={dt.min().item():.4f}y  "
          f"max={dt.max().item():.4f}y  "
          f"mean={dt.mean().item():.4f}y")

    return TensorDataset(X_t, X_tp1, dt)


def split_timewise(dataset, val_frac):
    n     = len(dataset)
    n_val = max(1, int(math.ceil(val_frac * n)))
    return (torch.utils.data.Subset(dataset, range(n - n_val)),
            torch.utils.data.Subset(dataset, range(n - n_val, n)))


# =============================================================================
# ── LATENT CLOUD (all currencies) ────────────────────────────────────────────
# =============================================================================

@torch.no_grad()
def latent_cloud_stats(model, X_all, batch_size=256):
    """
    Compute cloud stats from ALL training curves (all currencies).
    Using EUR-only gives noisy covariance and inflates exceed fraction.
    """
    model.eval()
    zs = []
    for i in range(0, X_all.shape[0], batch_size):
        zs.append(model.encoder(X_all[i:i + batch_size].to(device)).detach().cpu())
    z_all     = torch.cat(zs, dim=0).double()
    z_mean    = z_all.mean(dim=0)
    z_cov     = torch.cov(z_all.T)
    z_cov_inv = torch.linalg.inv(
        z_cov + CLOUD_JITTER * torch.eye(z_cov.shape[0], dtype=z_cov.dtype)
    )
    print(f"  Cloud mean (all ccy) : {z_mean.numpy()}")
    print(f"  Cloud std  (all ccy) : {z_all.std(dim=0).numpy()}")
    print(f"  N curves used        : {z_all.shape[0]}")
    return z_mean.to(device), z_cov_inv.to(device)


def mahal_sq(z, z_mean, z_cov_inv):
    dz = z - z_mean.unsqueeze(0)
    return (dz @ z_cov_inv * dz).sum(dim=1)


# =============================================================================
# ── LOSS FUNCTIONS ───────────────────────────────────────────────────────────
# =============================================================================

def transition_nll(model, z_t, z_tp1, dt):
    """
    Euler-Gaussian NLL:
        z_{t+1} | z_t ~ N( z_t + K(z_t)*dt,  L(z_t)L^T(z_t)*dt )
    Calibrates K (excluding frozen theta) and H to actual monthly transitions.
    dt is the TRUE calendar spacing from df_wide dates.
    """
    mu    = model.K(z_t)
    L     = get_L(model, z_t)
    B, d  = z_t.shape
    I     = torch.eye(d, device=z_t.device, dtype=z_t.dtype).unsqueeze(0)
    Sigma = (L @ L.transpose(1, 2)) * dt.view(-1, 1, 1) + CLOUD_JITTER * I
    resid = (z_tp1 - z_t - mu * dt).unsqueeze(-1)
    quad  = resid.transpose(1, 2).bmm(
                torch.linalg.solve(Sigma, resid)
            ).squeeze(-1).squeeze(-1)
    return (0.5 * (quad + torch.logdet(Sigma))).mean()


def sigma_regularisation(model, z_t):
    """
    Penalise sigma directly toward SIGMA_TARGET.

    The NLL alone cannot reliably shrink sigma because it sits in a flat
    region. This penalty pulls sigma down from above — the NLL pushes back
    if the data truly requires larger sigma, so the balance point is
    determined by both losses together.

    Fix B: LAMBDA_SIGMA raised from 500 to 2000 to push the balance point
    lower (v3 flattened at 0.0106; target is 0.008-0.010).
    """
    sigmas, _ = model.H(z_t)
    sigma_mean = sigmas.mean()
    return (sigma_mean - SIGMA_TARGET).pow(2)


def cloud_penalty(model, z_start, dt, z_mean, z_cov_inv):
    """Short Euler rollout penalising paths escaping the latent cloud."""
    z   = z_start
    pen = z.new_zeros(1)
    for _ in range(ROLLOUT_STEPS):
        mu    = model.K(z)
        L     = get_L(model, z)
        shock = L.bmm(torch.randn_like(z).unsqueeze(-1)).squeeze(-1) * dt.sqrt()
        z     = z + mu * dt + shock
        d_mah = mahal_sq(z, z_mean, z_cov_inv).clamp(min=0.0).sqrt()
        pen   = pen + F.relu(d_mah - CLOUD_THRESHOLD).pow(2).mean()
    return pen / ROLLOUT_STEPS


# =============================================================================
# ── EVALUATION ───────────────────────────────────────────────────────────────
# =============================================================================

@torch.no_grad()
def evaluate(model, loader, z_mean, z_cov_inv):
    model.eval()
    tot = dict(curve=0., trans=0., cloud=0., sigma=0., total=0., exceed=0., n=0)

    for X_t, X_tp1, dt in loader:
        X_t, X_tp1, dt = X_t.to(device), X_tp1.to(device), dt.to(device)
        lc  = F.mse_loss(model(X_t), X_t) * CURVE_SCALE
        z_t = model.encoder(X_t)
        lt  = transition_nll(model, z_t, model.encoder(X_tp1), dt)
        lk  = cloud_penalty(model, z_t, dt, z_mean, z_cov_inv)
        ls  = sigma_regularisation(model, z_t)
        tl  = LAMBDA_CURVE * lc + LAMBDA_TRANS * lt + LAMBDA_CLOUD * lk + LAMBDA_SIGMA * ls
        exc = (mahal_sq(z_t, z_mean, z_cov_inv).sqrt() > CLOUD_THRESHOLD).float().mean()
        bs  = X_t.shape[0]
        for k, v in zip(["curve","trans","cloud","sigma","total","exceed"],
                         [lc, lt, lk, ls, tl, exc]):
            tot[k] += v.item() * bs
        tot["n"] += bs

    n = max(tot["n"], 1)
    model.train()
    return {k: tot[k] / n for k in ["curve","trans","cloud","sigma","total","exceed"]}


@torch.no_grad()
def eval_curve_rmse(model, X_ccy, meta_ccy):
    model.eval()
    outs = []
    for i in range(0, X_ccy.shape[0], 256):
        outs.append(model(X_ccy[i:i+256].to(device)).detach().cpu())
    S_hat = torch.cat(outs, dim=0)
    mask  = torch.isfinite(X_ccy).all(dim=1) & torch.isfinite(S_hat).all(dim=1)
    rmse  = H.rmse_bps_per_currency_paper(
        X_ccy[mask], S_hat[mask],
        meta_ccy.loc[mask.numpy()].reset_index(drop=True)
    )
    model.train()
    return float(rmse.mean())


# =============================================================================
# ── LOAD DATA ─────────────────────────────────────────────────────────────────
# =============================================================================
print("\n── Loading data ──────────────────────────────────────────────────────")

(meta, X_tensor, meta_full, X_tensor_full,
 tenors, df_wide, df_wide_all, SCALE_IS_PERCENT) = my_data(use=USE)

X_tensor = X_tensor.double()
mask_ccy = meta["ccy"].astype(str).str.upper() == CCY.upper()
X_ccy    = X_tensor[mask_ccy.to_numpy()]
meta_ccy = meta.loc[mask_ccy].reset_index(drop=True)
print(f"  EUR curves : {X_ccy.shape[0]}")
print(f"  All curves : {X_tensor.shape[0]}  (used for cloud stats)")

print("\n── Building transition pairs ─────────────────────────────────────────")
full_ds      = build_transition_pairs(df_wide, tenors, SCALE_IS_PERCENT, CCY)
train_set, val_set = split_timewise(full_ds, VAL_FRAC)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  drop_last=False)
val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
print(f"  Train: {len(train_set)}   Val: {len(val_set)}")

# =============================================================================
# ── LOAD MODEL ────────────────────────────────────────────────────────────────
# =============================================================================
print(f"\n── Loading checkpoint ────────────────────────────────────────────────")
print(f"  {CHECKPOINT_PATH}")

raw_ckpt   = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
state_dict = (raw_ckpt["model_state_dict"]
              if isinstance(raw_ckpt, dict) and "model_state_dict" in raw_ckpt
              else raw_ckpt)

model = FullModel(latent_dim=2).to(device).double()
missing, unexpected = model.load_state_dict(state_dict, strict=False)
if missing or unexpected:
    print(f"  [WARN] missing={missing}  unexpected={unexpected}")

# Freeze encoder only — K (excl. theta), H, R, G are trainable
freeze(model.encoder)

# =============================================================================
# ── REINITIALISE KAPPA ────────────────────────────────────────────────────────
# =============================================================================
print(f"\n── Kappa reinitialisation ────────────────────────────────────────────")
print_dynamics(model, label="checkpoint")
reinit_kappa(model, KAPPA_REINIT)
print_dynamics(model, label="after reinit")

# =============================================================================
# ── SET AND FREEZE THETA (Fix A) ──────────────────────────────────────────────
# =============================================================================
print(f"\n── Setting and freezing theta ────────────────────────────────────────")

# Compute EUR-specific latent mean to use as the mean-reversion center
with torch.no_grad():
    zs_eur = []
    for i in range(0, X_ccy.shape[0], 256):
        zs_eur.append(model.encoder(X_ccy[i:i+256].to(device)).detach().cpu())
    z_eur_mean = torch.cat(zs_eur, dim=0).mean(dim=0)

with torch.no_grad():
    model.K.theta.copy_(z_eur_mean.to(device=device, dtype=model.K.theta.dtype))

# Fix A: freeze theta so it cannot drift during training
# In v3, theta drifted from [-0.026, -0.003] to [-0.007, 0.001],
# pulling mean-reversion toward the wrong center.
model.K.theta.requires_grad = False
print(f"  K.theta frozen at: {model.K.theta.detach().cpu().numpy()}")

# Build trainable param groups AFTER freezing theta
# K params: B, L, raw_kappa (theta excluded)
# G params: all G weights
trainable_khr = [p for name, p in model.named_parameters()
                 if p.requires_grad and not name.startswith("G.")]
trainable_g   = [p for name, p in model.named_parameters()
                 if p.requires_grad and name.startswith("G.")]
trainable_all = trainable_khr + trainable_g

print(f"  Trainable K/H/R : {sum(p.numel() for p in trainable_khr):,}  "
      f"(theta frozen, so excludes it)")
print(f"  Trainable G     : {sum(p.numel() for p in trainable_g):,}")
print(f"  Total           : {sum(p.numel() for p in trainable_all):,}")

# =============================================================================
# ── CLOUD STATS FROM ALL CURRENCIES ───────────────────────────────────────────
# =============================================================================
print(f"\n── Computing latent cloud stats (all currencies) ─────────────────────")
z_mean, z_cov_inv = latent_cloud_stats(model, X_tensor)

# Sigma before training
print_sigma_diag(model, X_ccy, label="before training")
rmse_init = eval_curve_rmse(model, X_ccy, meta_ccy)
print(f"  Initial EUR curve RMSE: {rmse_init:.2f} bp  (baseline to protect)")

# =============================================================================
# ── OPTIMIZER ─────────────────────────────────────────────────────────────────
# =============================================================================
optim = torch.optim.Adam([
    {"params": trainable_khr, "lr": MAX_LR_KHR},
    {"params": trainable_g,   "lr": MAX_LR_G},
])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optim, T_max=EPOCHS, eta_min=1e-6
)

# =============================================================================
# ── TRAINING LOOP ─────────────────────────────────────────────────────────────
# =============================================================================
print(f"\n── Training ({EPOCHS} epochs) ────────────────────────────────────────")
print(f"  LR K/H/R={MAX_LR_KHR:.1e}  LR G={MAX_LR_G:.1e}  (CosineAnnealing → 1e-6)")
print(f"  λ_curve={LAMBDA_CURVE}  λ_trans={LAMBDA_TRANS}  "
      f"λ_cloud={LAMBDA_CLOUD}  λ_sigma={LAMBDA_SIGMA}")
print(f"  SIGMA_TARGET={SIGMA_TARGET}  KAPPA_REINIT={KAPPA_REINIT}  "
      f"ROLLOUT_STEPS={ROLLOUT_STEPS}")
print(f"  theta FROZEN at EUR latent mean  (Fix A)")
print(f"  LAMBDA_SIGMA={LAMBDA_SIGMA}  (Fix B, was 500)")

csv_path = os.path.join(OUT_DIR, "metrics.csv")
cols = ["epoch",
        "train_curve", "train_trans", "train_cloud", "train_sigma", "train_total",
        "val_curve",   "val_trans",   "val_cloud",   "val_sigma",   "val_total",
        "val_exceed",  "curve_rmse_bp", "sigma_mean",
        "kappa",       "half_life_y", "theta_0", "theta_1", "lr_khr", "time_sec"]
pd.DataFrame(columns=cols).to_csv(csv_path, index=False)

best_val  = float("inf")
best_path = os.path.join(OUT_DIR, "best_checkpoint.pt")
history   = []
t0        = time.perf_counter()

for epoch in range(EPOCHS):
    model.train()
    sums = dict(curve=0., trans=0., cloud=0., sigma=0., total=0., n=0, nan=0)

    for X_t, X_tp1, dt in train_loader:
        X_t, X_tp1, dt = X_t.to(device), X_tp1.to(device), dt.to(device)
        optim.zero_grad()

        # 1) curve loss
        l_curve = F.mse_loss(model(X_t), X_t) * CURVE_SCALE

        # 2) transition NLL — encoder frozen, no grad needed through it
        with torch.no_grad():
            z_t   = model.encoder(X_t)
            z_tp1 = model.encoder(X_tp1)

        l_trans = transition_nll(model, z_t, z_tp1, dt)

        # 3) cloud penalty
        l_cloud = cloud_penalty(model, z_t.detach(), dt, z_mean, z_cov_inv)

        # 4) sigma regularisation
        l_sigma = sigma_regularisation(model, z_t)

        loss = (LAMBDA_CURVE * l_curve
                + LAMBDA_TRANS * l_trans
                + LAMBDA_CLOUD * l_cloud
                + LAMBDA_SIGMA * l_sigma)

        if not torch.isfinite(loss):
            sums["nan"] += 1
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_all, max_norm=1.0)
        optim.step()

        bs = X_t.shape[0]
        sums["curve"] += l_curve.item() * bs
        sums["trans"] += l_trans.item() * bs
        sums["cloud"] += l_cloud.item() * bs
        sums["sigma"] += l_sigma.item() * bs
        sums["total"] += loss.item()    * bs
        sums["n"]     += bs

    scheduler.step()

    n   = max(sums["n"], 1)
    tr  = {k: sums[k] / n for k in ["curve","trans","cloud","sigma","total"]}
    val = evaluate(model, val_loader, z_mean, z_cov_inv)

    # current sigma
    with torch.no_grad():
        z_sample = model.encoder(X_ccy[:64].to(device))
        sig_now, _ = model.H(z_sample)
        sigma_mean_now = float(sig_now.mean().item())

    # kappa / half-life
    kappa_now = F.softplus(model.K.raw_kappa).item() if hasattr(model.K, "raw_kappa") else float("nan")
    eigs_now  = torch.linalg.eigvals(model.K.drift_matrix()).real.detach().cpu().numpy()
    hl_now    = (math.log(2) / float(np.abs(eigs_now).min())
                 if np.abs(eigs_now).min() > 0 else float("inf"))

    # theta — should stay constant (frozen); log it to confirm
    theta_now = model.K.theta.detach().cpu().numpy()

    lr_now  = optim.param_groups[0]["lr"]
    rmse_bp = eval_curve_rmse(model, X_ccy, meta_ccy)
    elapsed = time.perf_counter() - t0

    if val["total"] < best_val:
        best_val = val["total"]
        torch.save(model.state_dict(), best_path)

    row = {
        "epoch":          epoch,
        "train_curve":    tr["curve"],    "train_trans":   tr["trans"],
        "train_cloud":    tr["cloud"],    "train_sigma":   tr["sigma"],
        "train_total":    tr["total"],
        "val_curve":      val["curve"],   "val_trans":     val["trans"],
        "val_cloud":      val["cloud"],   "val_sigma":     val["sigma"],
        "val_total":      val["total"],   "val_exceed":    val["exceed"],
        "curve_rmse_bp":  rmse_bp,        "sigma_mean":    sigma_mean_now,
        "kappa":          kappa_now,      "half_life_y":   hl_now,
        "theta_0":        float(theta_now[0]), "theta_1": float(theta_now[1]),
        "lr_khr":         lr_now,         "time_sec":      elapsed,
    }
    pd.DataFrame([row], columns=cols).to_csv(csv_path, mode="a", header=False, index=False)
    history.append(row)

    # sigma loss displayed with more decimals to confirm it's active
    print(
        f"ep {epoch:3d} | "
        f"train={tr['total']:.2f} "
        f"(c={tr['curve']:.2f} t={tr['trans']:.2f} "
        f"k={tr['cloud']:.3f} s={tr['sigma']:.6f}) | "
        f"val={val['total']:.2f}  exceed={val['exceed']:.1%} | "
        f"rmse={rmse_bp:.1f}bp  sigma={sigma_mean_now:.5f} | "
        f"kappa={kappa_now:.3f}  hl={hl_now:.1f}y  "
        f"θ=[{theta_now[0]:.4f},{theta_now[1]:.4f}] | "
        f"lr={lr_now:.2e}  nan={sums['nan']}  t={elapsed/60:.1f}min"
    )

print("\nTraining done.")
print_dynamics(model, label="final")
print_sigma_diag(model, X_ccy, label="final")
final_rmse = eval_curve_rmse(model, X_ccy, meta_ccy)
print(f"Final EUR curve RMSE  : {final_rmse:.2f} bp")
print(f"Initial EUR curve RMSE: {rmse_init:.2f} bp")

# Verify theta did not move (sanity check for Fix A)
theta_final = model.K.theta.detach().cpu().numpy()
theta_init  = z_eur_mean.numpy()
theta_drift = np.abs(theta_final - theta_init).max()
print(f"Theta drift (max abs) : {theta_drift:.2e}  (should be ~0 — frozen)")

# =============================================================================
# ── SAVE CHECKPOINTS ──────────────────────────────────────────────────────────
# =============================================================================
final_path = os.path.join(OUT_DIR, "final_checkpoint.pt")
torch.save(model.state_dict(), final_path)
print(f"Saved final : {final_path}")
print(f"Saved best  : {best_path}")

# =============================================================================
# ── PLOTS ─────────────────────────────────────────────────────────────────────
# =============================================================================
df_hist = pd.DataFrame(history)

fig, axes = plt.subplots(2, 3, figsize=(18, 8))
axes = axes.flatten()

# 1) Total loss
axes[0].plot(df_hist["epoch"], df_hist["train_total"], label="train")
axes[0].plot(df_hist["epoch"], df_hist["val_total"],   label="val")
axes[0].set_title("Total loss")
axes[0].set_xlabel("Epoch")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 2) Transition + cloud losses
axes[1].plot(df_hist["epoch"], df_hist["train_trans"], label="trans")
axes[1].plot(df_hist["epoch"], df_hist["train_cloud"], label="cloud")
axes[1].set_title("Transition & cloud losses")
axes[1].set_xlabel("Epoch")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# 3) Sigma evolution — key diagnostic for 1Y vol fix
axes[2].plot(df_hist["epoch"], df_hist["sigma_mean"])
axes[2].axhline(SIGMA_TARGET, color="red", linestyle="--", linewidth=1,
                label=f"target ({SIGMA_TARGET})")
axes[2].set_title("Sigma — target drives 1Y vol fix")
axes[2].set_xlabel("Epoch")
axes[2].set_ylabel("Mean sigma")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

# 4) Curve RMSE
axes[3].plot(df_hist["epoch"], df_hist["curve_rmse_bp"])
axes[3].axhline(rmse_init, color="red", linestyle="--", linewidth=1,
                label=f"initial ({rmse_init:.1f} bp)")
axes[3].set_title(f"{CCY} curve RMSE (bp) — G fit monitor")
axes[3].set_xlabel("Epoch")
axes[3].legend()
axes[3].grid(True, alpha=0.3)

# 5) Kappa / half-life
axes[4].plot(df_hist["epoch"], df_hist["kappa"],       label="kappa")
axes[4].plot(df_hist["epoch"], df_hist["half_life_y"], label="half-life (y)", linestyle="--")
axes[4].axhline(10, color="red", linestyle=":", linewidth=1, label="10y reference")
axes[4].set_title("Mean-reversion strength")
axes[4].set_xlabel("Epoch")
axes[4].legend()
axes[4].grid(True, alpha=0.3)

# 6) Theta stability — should be flat lines (Fix A verification)
axes[5].plot(df_hist["epoch"], df_hist["theta_0"], label="θ[0]")
axes[5].plot(df_hist["epoch"], df_hist["theta_1"], label="θ[1]")
axes[5].set_title("Theta (frozen — should be flat)")
axes[5].set_xlabel("Epoch")
axes[5].legend()
axes[5].grid(True, alpha=0.3)

fig.tight_layout()
plot_path = os.path.join(OUT_DIR, "training_curves.png")
fig.savefig(plot_path, dpi=150)
plt.show()
print(f"Saved plot  : {plot_path}")
