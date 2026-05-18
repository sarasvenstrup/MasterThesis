# ==================== Constant MPR — Regime-Window OOS ====================
"""
Constant MPR with three regime-window out-of-sample tests.

For each of three named regimes, train a fresh Constant MPR on data BEFORE
the regime start and evaluate on the regime itself.  Each regime is a clean
out-of-sample test against a different historical period:

  1. negative_rates     train < 2014-01  test 2014-01 .. 2019-12
  2. covid              train < 2020-01  test 2020-01 .. 2021-12
  3. rate_normalisation train < 2022-01  test 2022-01 .. end-of-data

The training cutoff defines an expanding window — each regime uses ALL data
up to its start.

This is the headline out-of-sample evaluation for the yield-curve-only model.

Output: Figures/TrainingResults/dim4_constant_mpr_regimes/{regime_name}/ep{EPOCHS}/
        Figures/pricing/eval_constant_mpr_regimes/per_cell_final.csv
"""

import os, sys

_N_TORCH   = 4
_N_INTEROP = 2
os.environ.setdefault("OMP_NUM_THREADS",        str(_N_TORCH))
os.environ.setdefault("MKL_NUM_THREADS",        str(_N_TORCH))
os.environ.setdefault("OPENBLAS_NUM_THREADS",   str(_N_TORCH))
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(_N_TORCH))
os.environ.setdefault("NUMEXPR_NUM_THREADS",    str(_N_TORCH))

import time, math, json
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

torch.set_num_threads(_N_TORCH)
torch.set_num_interop_threads(_N_INTEROP)

try:
    import psutil
    psutil.Process().cpu_affinity(list(range(os.cpu_count()))[:_N_TORCH * 2])
except Exception:
    pass

# ── paths ──────────────────────────────────────────────────────────────────────
try:
    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    REPO_ROOT = os.getcwd()
PROJECT_ROOT = os.path.abspath(os.path.join(REPO_ROOT, ".."))
for p in [PROJECT_ROOT, REPO_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from Code import config
config.confirm_variant()

from Code.utils import helpers as H
from Code.load_swapdata import my_data
from Code.model.full_model_price import FullModelPrice as FullModel
from Code.Pricing.load_swapvol_ois import load_swaption_vol_data
from Code.Pricing.pricing import swap_rate_torch, forward_swap_rate_torch
from Code.model.sigma_matrix import L_from_sigmas_rhos
from Code.Simulation.simulate_model import simulate_to_expiry_differentiable

print("Torch:", torch.__version__, "| CUDA:", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device, "| Variant:", config.VARIANT)

# ── regimes ────────────────────────────────────────────────────────────────────
REGIMES = [
    {
        "name":       "negative_rates",
        "label":      "Negative rates era (2014--2019)",
        "train_end":  pd.Timestamp("2013-12-31"),
        "test_start": pd.Timestamp("2014-01-01"),
        "test_end":   pd.Timestamp("2019-12-31"),
    },
    {
        "name":       "covid",
        "label":      "COVID shock (2020--2021)",
        "train_end":  pd.Timestamp("2019-12-31"),
        "test_start": pd.Timestamp("2020-01-01"),
        "test_end":   pd.Timestamp("2021-12-31"),
    },
    {
        "name":       "rate_normalisation",
        "label":      "Rate-normalisation shock (2022--)",
        "train_end":  pd.Timestamp("2021-12-31"),
        "test_start": pd.Timestamp("2022-01-01"),
        "test_end":   pd.Timestamp("2030-12-31"),  # end of data, sentinel
    },
]

# ── settings ───────────────────────────────────────────────────────────────────
LATENT_DIM  = 4
EXPIRY_VALS = [1, 5, 10]
TENOR_VALS  = [1, 5, 10]

EPOCHS                = 1000
EVAL_EVERY            = 100
HEADER_EVERY          = 20
SAVE_EVERY            = 200
N_STEPS_PER_EPOCH     = 4
N_SWAPTIONS_PER_BATCH = 8
N_PATHS_PRICING       = 512
DT_PRICING            = 1 / 6

PRETRAIN_CKPT = os.path.join(PROJECT_ROOT, "Figures", "TrainingResults",
                              "dim4_stable", "ep5000", "checkpoint_dim4_ep5000.pt")

LAMBDA_VOL    = 1.0
LAMBDA_BIAS   = 0.5
LAMBDA_L2     = 1e-3

LR            = 5e-4   # match Training_constant_mpr.py (was 2e-4, too small for 8-param model)
LR_SIG_MULT   = 10.0
LR_WARMUP     = 50     # match Training_constant_mpr.py (was 30)

MIN_FINITE_PATHS_ABS  = 16
MIN_FINITE_PATHS_FRAC = 0.10
LOSS_SKIP_THRESH      = 1e4

# Eval settings
N_PATHS_EVAL = 512
RATE_CLIP    = 0.50
ANNUITY_MAX  = 50.0

USE        = "bbg"
CCY_FILTER = "EUR"

BASE_FIGURES_DIR = os.path.join(PROJECT_ROOT, "Figures", "TrainingResults",
                                 f"dim{LATENT_DIM}_constant_mpr_regimes")
EVAL_DIR         = os.path.join(PROJECT_ROOT, "Figures", "pricing", "eval_constant_mpr_regimes")
os.makedirs(BASE_FIGURES_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)

# ── module ─────────────────────────────────────────────────────────────────────

class ConstantMPRAdjustment(nn.Module):
    """K*(z) = K(z) + L(z) · lambda_0 — constant drift correction, frozen base."""

    def __init__(self, kp_module, h_module, latent_dim):
        super().__init__()
        self.kp = kp_module
        self.h  = h_module
        self.latent_dim = latent_dim
        self.lambda_0      = nn.Parameter(torch.zeros(latent_dim))
        self.log_sigma_vec = nn.Parameter(torch.full((latent_dim,), -1.8))

    def forward(self, z):
        k_base       = self.kp(z)
        sigmas, rhos = self.h(z)
        L            = L_from_sigmas_rhos(sigmas, rhos, validate=False)
        lam0 = self.lambda_0.unsqueeze(0).expand(z.shape[0], -1)
        return k_base + torch.einsum('bij,bj->bi', L, lam0)

    @property
    def sigma_vec(self):
        return self.log_sigma_vec.exp()


# ── helpers ────────────────────────────────────────────────────────────────────

def row_finite_mask(t):
    return torch.isfinite(t).all(dim=1)

@torch.no_grad()
def predict_S_hat(model, X, batch_size=256):
    was_train = model.training; model.eval()
    outs = [model(X[i:i+batch_size].to(device)).detach().cpu()
            for i in range(0, X.shape[0], batch_size)]
    if was_train: model.train()
    return torch.cat(outs, 0)

def eval_rmse_bps(model, X_full, meta_full):
    S_hat = predict_S_hat(model, X_full)
    mask  = row_finite_mask(X_full) & row_finite_mask(S_hat)
    rmse  = H.rmse_bps_per_currency_paper(
        X_full[mask], S_hat[mask],
        meta_full.loc[mask.numpy()].reset_index(drop=True))
    return rmse, float(rmse.mean())


# ── pricing loss (training) ────────────────────────────────────────────────────

def compute_pricing_loss(model, lm, X_batch, df_vol_window, date_to_idx,
                         n_swaptions, n_paths, dt, device, dtype):
    if len(df_vol_window) == 0:
        zero = torch.tensor(0.0, device=device, dtype=dtype)
        return zero, zero, 0, 0, 0.0

    sample      = df_vol_window.sample(n=min(n_swaptions, len(df_vol_window)))
    total_vol   = torch.zeros(1, device=device, dtype=dtype)
    total_bias  = torch.zeros(1, device=device, dtype=dtype)
    n_valid     = 0; n_attempted = 0
    path_fracs  = []
    min_paths   = max(MIN_FINITE_PATHS_ABS, int(n_paths * MIN_FINITE_PATHS_FRAC))

    for _, row in sample.iterrows():
        date      = pd.Timestamp(row["as_of_date"]).normalize()
        expiry    = int(row["option_maturity"])
        tenor     = int(row["swap_tenor"])
        sigma_mkt = float(row["market_vol"])
        sigma_mkt_bp = sigma_mkt * 1e4

        if date not in date_to_idx:
            continue

        n_attempted += 1
        idx = date_to_idx[date]
        xb  = X_batch[idx:idx + 1].to(device)

        with torch.no_grad():
            z0 = model.encoder(xb)
            _, aux0 = model.decode_from_z(z0, tau=None, return_aux=True)
            P0 = aux0["P_full"][0]

        if expiry + tenor > P0.shape[0] - 1:
            continue
        F_0, A_0 = forward_swap_rate_torch(P0, expiry, tenor)
        if not (math.isfinite(F_0) and math.isfinite(A_0) and A_0 > 1e-6):
            continue

        dt_eff  = min(dt, expiry / 10.0)
        n_steps = max(12, int(round(expiry / dt_eff)))
        half    = n_paths // 2

        with torch.no_grad():
            eps_z = torch.randn(half, n_steps, LATENT_DIM, device=device, dtype=dtype)

        try:
            z_T, D_T = simulate_to_expiry_differentiable(
                model, z0, n_steps=n_steps, dt=dt_eff,
                n_paths=2*half, eps=eps_z,
                k_override=lm,
                sigma_scale=lm.sigma_vec,
                antithetic=True, freeze_H=True,
            )
            z_ok = torch.isfinite(z_T).all(1) & torch.isfinite(D_T)
            if int(z_ok.sum()) < min_paths: continue

            with torch.no_grad():
                _, aux_T = model.decode_from_z(z_T, tau=None, return_aux=True,
                                               k_override=lm, sigma_scale=lm.sigma_vec)
                p_ok = torch.isfinite(aux_T["P_full"]).all(1)

            mask = z_ok & p_ok
            if int(mask.sum()) < min_paths: continue
            path_fracs.append(float(mask.float().mean()))

            _, aux_k = model.decode_from_z(z_T[mask], tau=None, return_aux=True,
                                           k_override=lm, sigma_scale=lm.sigma_vec)
            F_T, A_T = swap_rate_torch(aux_k["P_full"], tenor=tenor)
            D_keep   = D_T[mask]

            fa_ok = (torch.isfinite(F_T) & torch.isfinite(A_T)
                     & (F_T > -0.5) & (F_T < 0.5)
                     & (A_T > 1e-6) & (A_T < 50.0))
            if int(fa_ok.sum()) < min_paths: continue
            F_T, A_T, D_keep = F_T[fa_ok], A_T[fa_ok], D_keep[fa_ok]

            V_pay = (D_keep * A_T * torch.relu(F_T - F_0)).mean()
            V_rec = (D_keep * A_T * torch.relu(F_0 - F_T)).mean()
            if not (torch.isfinite(V_pay) and torch.isfinite(V_rec)
                    and float(V_pay.detach()) >= 0 and float(V_rec.detach()) >= 0):
                continue

            sqrt_2pi     = math.sqrt(2 * math.pi)
            sigma_str_bp = (V_pay + V_rec) * 0.5 * sqrt_2pi / (A_0 * math.sqrt(expiry)) * 1e4
            loss_vol_ij  = ((sigma_str_bp - sigma_mkt_bp) / 100.0).pow(2)
            if not torch.isfinite(loss_vol_ij) or float(loss_vol_ij.detach()) > LOSS_SKIP_THRESH:
                continue

            fwd_bias_bp  = (V_pay - V_rec) / A_0 * 1e4
            loss_bias_ij = (fwd_bias_bp / 100.0).pow(2)
            if not torch.isfinite(loss_bias_ij): continue

            total_vol  = total_vol  + loss_vol_ij
            total_bias = total_bias + loss_bias_ij
            n_valid   += 1
        except Exception:
            continue

    mean_pfrac = float(np.mean(path_fracs)) if path_fracs else 0.0
    zero = torch.tensor(0.0, device=device, dtype=dtype)
    if n_valid > 0:
        return total_vol/n_valid, total_bias/n_valid, n_attempted, n_valid, mean_pfrac
    return zero, zero, n_attempted, 0, mean_pfrac


# ── eval one cell ──────────────────────────────────────────────────────────────
sqrt_2pi = math.sqrt(2 * math.pi)

def price_cell_eval(model, lm, X_eur, date_to_idx, date, expiry, tenor):
    if date not in date_to_idx:
        return None
    idx = date_to_idx[date]
    xb  = X_eur[idx:idx+1].to(device)
    with torch.no_grad():
        z0 = model.encoder(xb)
        _, aux0 = model.decode_from_z(z0, tau=None, return_aux=True)
        P0 = aux0["P_full"][0]
    if expiry + tenor > P0.shape[0] - 1: return None
    F0, A0 = forward_swap_rate_torch(P0, expiry, tenor)
    if not (math.isfinite(F0) and math.isfinite(A0) and A0 > 1e-6): return None

    dt_eff  = min(1.0/12.0, expiry/10.0)
    n_steps = max(12, int(round(expiry/dt_eff)))
    half    = N_PATHS_EVAL // 2

    with torch.no_grad():
        eps_z = torch.randn(half, n_steps, LATENT_DIM, device=device)
        z_T, D_T = simulate_to_expiry_differentiable(
            model, z0, n_steps=n_steps, dt=dt_eff,
            n_paths=N_PATHS_EVAL, eps=eps_z,
            k_override=lm, sigma_scale=lm.sigma_vec,
            antithetic=True, freeze_H=True)
    ok = torch.isfinite(z_T).all(1) & torch.isfinite(D_T)
    if ok.sum() < 16: return None
    z_k, D_k = z_T[ok], D_T[ok]

    with torch.no_grad():
        _, aux_T = model.decode_from_z(z_k, tau=None, return_aux=True,
                                       k_override=lm, sigma_scale=lm.sigma_vec)
    P_T = aux_T["P_full"]
    dok = torch.isfinite(P_T).all(1)
    if dok.sum() < 16: return None

    F_T, A_T = swap_rate_torch(P_T[dok], tenor=tenor)
    D_k = D_k[dok]
    sane = (torch.isfinite(F_T) & torch.isfinite(A_T)
            & (F_T > -RATE_CLIP) & (F_T < RATE_CLIP)
            & (A_T > 1e-6) & (A_T < ANNUITY_MAX))
    if sane.sum() < 16: return None
    F_T, A_T, D_k = F_T[sane], A_T[sane], D_k[sane]

    V_pay = float((D_k * A_T * torch.relu(F_T - F0)).mean())
    V_rec = float((D_k * A_T * torch.relu(F0 - F_T)).mean())
    if not (math.isfinite(V_pay) and math.isfinite(V_rec)): return None
    return {
        "sigma_str_bp":    (V_pay+V_rec)*0.5*sqrt_2pi/(A0*math.sqrt(expiry))*1e4,
        "forward_bias_bp": (V_pay-V_rec)/A0*1e4,
        "path_frac":       float(ok.float().mean()),
    }


# ── load data once ─────────────────────────────────────────────────────────────
SEED = 0
torch.manual_seed(SEED); np.random.seed(SEED)

meta, X_tensor, meta_full, X_tensor_full, *_ = my_data(use=USE)
X_tensor = X_tensor.float()
meta_ccy     = meta[meta["ccy"] == CCY_FILTER].reset_index(drop=True)
X_tensor_ccy = X_tensor[meta["ccy"] == CCY_FILTER]

# Load model once — frozen base, reused across all regimes
model = FullModel(latent_dim=LATENT_DIM).to(device)
raw   = torch.load(PRETRAIN_CKPT, map_location=device, weights_only=False)
model.load_state_dict(raw.get("model_state_dict", raw))
for p in model.parameters(): p.requires_grad_(False)
print("Base model loaded and frozen.")

# Load swaption data once
df_vol = load_swaption_vol_data(currency=CCY_FILTER)
df_vol["as_of_date"] = pd.to_datetime(df_vol["as_of_date"]).dt.normalize()
df_vol["market_vol"] = df_vol["vol"] / 10_000.0
df_vol = df_vol[df_vol["option_maturity"].isin(EXPIRY_VALS)
                & df_vol["swap_tenor"].isin(TENOR_VALS)].copy()

dates_swap  = set(pd.to_datetime(meta_ccy["as_of_date"]).dt.normalize())
df_vol      = df_vol[df_vol["as_of_date"].isin(dates_swap)].copy()
date_to_idx = {pd.Timestamp(r["as_of_date"]).normalize(): i for i, r in meta_ccy.iterrows()}

if df_vol.empty: raise RuntimeError("No swaption vol data.")
print(f"Loaded {len(df_vol)} total vol targets from {df_vol['as_of_date'].nunique()} dates")
print(f"Date range: {df_vol['as_of_date'].min().date()} -> {df_vol['as_of_date'].max().date()}")

# ── regime loop ────────────────────────────────────────────────────────────────
all_test_rows = []   # accumulated across regimes

for regime_idx, regime in enumerate(REGIMES):
    name      = regime["name"]
    label     = regime["label"]
    train_end = regime["train_end"]
    test_start= regime["test_start"]
    test_end  = regime["test_end"]

    # Filter
    df_train = df_vol[df_vol["as_of_date"] <= train_end].copy()
    df_test  = df_vol[(df_vol["as_of_date"] >= test_start)
                       & (df_vol["as_of_date"] <= test_end)].copy()

    n_train_obs = len(df_train); n_test_obs = len(df_test)
    n_train_dt  = df_train["as_of_date"].nunique(); n_test_dt = df_test["as_of_date"].nunique()

    print("\n" + "="*100)
    print(f"REGIME {regime_idx+1}/3:  {name}  ({label})")
    print(f"  Train: <= {train_end.date()}   n_obs={n_train_obs}  n_dates={n_train_dt}")
    print(f"  Test:  {test_start.date()} -> {test_end.date()}   n_obs={n_test_obs}  n_dates={n_test_dt}")
    print("="*100)

    if n_train_obs == 0 or n_test_obs == 0:
        print(f"  SKIP — insufficient data")
        continue

    regime_dir = os.path.join(BASE_FIGURES_DIR, name, f"ep{EPOCHS}")
    os.makedirs(regime_dir, exist_ok=True)

    # Fresh model for each regime — pure OOS
    torch.manual_seed(SEED + regime_idx)
    np.random.seed(SEED + regime_idx)
    lm = ConstantMPRAdjustment(model.K, model.H, LATENT_DIM).to(device)
    model.train()

    optim = torch.optim.Adam([
        {'params': [lm.lambda_0],      'lr': LR,               'name': 'lambda'},
        {'params': [lm.log_sigma_vec], 'lr': LR * LR_SIG_MULT, 'name': 'sigma'},
    ], lr=LR)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optim,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(optim, 1e-3, 1.0, LR_WARMUP),
            torch.optim.lr_scheduler.CosineAnnealingLR(optim, max(EPOCHS-LR_WARMUP,1), 1e-7),
        ],
        milestones=[LR_WARMUP]
    )

    # CSV logger
    csv_path = os.path.join(regime_dir, f"train_log_{name}_ep{EPOCHS}.csv")
    pd.DataFrame(columns=["epoch","time_total_sec","loss_vol","loss_bias",
                          "lambda_0_norm","sigma_mean",
                          "swaption_priced_frac","path_finite_frac","lr"]
                 ).to_csv(csv_path, index=False)
    with open(os.path.join(regime_dir, "run_config.json"), "w") as f:
        json.dump({"regime": name, "label": label,
                   "train_end":  train_end.date().isoformat(),
                   "test_start": test_start.date().isoformat(),
                   "test_end":   test_end.date().isoformat(),
                   "n_train_obs": n_train_obs, "n_test_obs": n_test_obs,
                   "epochs": EPOCHS}, f, indent=2)

    # Training loop
    t0 = time.perf_counter(); t_last = t0
    for epoch in range(EPOCHS):
        model.train(); lm.train()
        r_vol=r_bias=0.0; n_bat=0
        ep_att=ep_pri=0; ep_pf=[]

        for step in range(N_STEPS_PER_EPOCH):
            print(f"\r  ep {epoch}  step {step+1}/{N_STEPS_PER_EPOCH} ...", end="", flush=True)
            optim.zero_grad(set_to_none=True)
            lv, lb, na, np_, pf = compute_pricing_loss(
                model, lm, X_tensor_ccy, df_train, date_to_idx,
                N_SWAPTIONS_PER_BATCH, N_PATHS_PRICING, DT_PRICING,
                device, torch.float32)
            ep_att+=na; ep_pri+=np_
            if pf>0: ep_pf.append(pf)
            l2  = LAMBDA_L2 * lm.lambda_0.pow(2).sum()
            loss = LAMBDA_VOL*lv + LAMBDA_BIAS*lb + l2
            if not torch.isfinite(loss): continue
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in lm.parameters()):
                optim.zero_grad(set_to_none=True); continue
            for pg in optim.param_groups:
                torch.nn.utils.clip_grad_norm_(pg['params'], max_norm=2.0)
            optim.step()
            r_vol+=float(lv.detach()); r_bias+=float(lb.detach())
            n_bat+=1
        print("\r"+" "*40+"\r", end="", flush=True)
        scheduler.step()

        n_bat = max(n_bat, 1)
        ep_vol=r_vol/n_bat; ep_bias=r_bias/n_bat
        swp=ep_pri/max(ep_att,1); pth=float(np.mean(ep_pf)) if ep_pf else 0.0

        with torch.no_grad():
            lam0_norm  = float(lm.lambda_0.norm())
            sigma_mean = float(lm.sigma_vec.mean())

        t_now = time.perf_counter(); dt_ep=t_now-t_last; t_last=t_now
        lr_now = optim.param_groups[0]["lr"]

        pd.DataFrame([{"epoch":epoch,"time_total_sec":round(t_now-t0,1),
                       "loss_vol":ep_vol,"loss_bias":ep_bias,
                       "lambda_0_norm":lam0_norm,"sigma_mean":sigma_mean,
                       "swaption_priced_frac":swp,"path_finite_frac":pth,
                       "lr":lr_now}]
                     ).to_csv(csv_path, mode="a", header=False, index=False)

        if epoch % HEADER_EVERY == 0:
            print(f"\n  {'ep':>5} {'vol':>10} {'bias':>10} {'|λ0|':>7} {'σ_mean':>7} {'swp%':>5} {'pth%':>5} {'t/ep':>5}")
            print("  " + "-"*70)
        print(f"  {epoch:>5d} {ep_vol:>10.4e} {ep_bias:>10.4e} "
              f"{lam0_norm:>7.4f} {sigma_mean:>7.4f} "
              f"{swp*100:>4.0f}% {pth*100:>4.0f}% {dt_ep:>5.1f}s")

        if (epoch+1) % SAVE_EVERY == 0 or epoch == EPOCHS-1:
            ckpt_path = os.path.join(regime_dir, f"checkpoint_constant_mpr_{name}_ep{epoch+1}.pt")
            torch.save({"lm_state_dict": lm.state_dict(),
                        "regime": name, "epoch": epoch+1,
                        "train_end": train_end.date().isoformat()}, ckpt_path)

    print(f"  Training complete: λ_0 = {lm.lambda_0.detach().cpu().numpy().round(4)}")
    print(f"                     σ_vec = {lm.sigma_vec.detach().cpu().numpy().round(4)}")

    # ── eval on this regime's test period ──────────────────────────────────────
    print(f"  Evaluating on test period ({n_test_obs} obs, {n_test_dt} dates) ...")
    lm.eval(); model.eval()
    torch.manual_seed(42)  # different seed for eval

    t_eval = time.time()
    combos = df_test[["as_of_date","option_maturity","swap_tenor","market_vol"]].drop_duplicates()
    rows = []
    for counter, (_, row) in enumerate(combos.iterrows()):
        date   = pd.Timestamp(row["as_of_date"]).normalize()
        expiry = int(row["option_maturity"]); tenor = int(row["swap_tenor"])
        mkt_bp = float(row["market_vol"]) * 1e4
        if expiry not in EXPIRY_VALS or tenor not in TENOR_VALS: continue
        result = price_cell_eval(model, lm, X_tensor_ccy, date_to_idx, date, expiry, tenor)
        if counter % 50 == 0:
            print(f"    {counter}/{len(combos)}  ({date.date()} {expiry}Yx{tenor}Y)  {time.time()-t_eval:.0f}s")
        if result is None: continue
        rows.append({"regime": name, "date": date, "expiry": expiry, "tenor": tenor,
                     "mkt_bp": mkt_bp, "sigma_str_bp": result["sigma_str_bp"],
                     "vol_error_bp": result["sigma_str_bp"] - mkt_bp,
                     "forward_bias_bp": result["forward_bias_bp"],
                     "path_frac": result["path_frac"]})
    df_eval = pd.DataFrame(rows)
    df_eval.to_csv(os.path.join(regime_dir, f"per_cell_test_{name}.csv"), index=False)
    print(f"  Eval done: {len(df_eval)} priced  |  test MAE = {df_eval['vol_error_bp'].abs().mean():.1f} bp")

    all_test_rows.extend(rows)

# ── aggregate across regimes ──────────────────────────────────────────────────
df_all = pd.DataFrame(all_test_rows)
df_all.to_csv(os.path.join(EVAL_DIR, "per_cell_final.csv"), index=False)
print("\n" + "="*100)
print("AGGREGATED RESULTS")
print("="*100)

# Per-regime summary
summary_rows = []
for regime in REGIMES:
    name = regime["name"]
    sub = df_all[df_all["regime"] == name]
    if len(sub) == 0: continue
    print(f"\n-- {name} ({regime['label']}) --")
    print(f"  Overall: MAE = {sub['vol_error_bp'].abs().mean():.1f} bp  "
          f"RMSE = {np.sqrt((sub['vol_error_bp']**2).mean()):.1f} bp  "
          f"bias = {sub['vol_error_bp'].mean():+.1f} bp  N = {len(sub)}")
    for e in EXPIRY_VALS:
        for t in TENOR_VALS:
            c = sub[(sub['expiry']==e) & (sub['tenor']==t)]
            if len(c):
                print(f"    {e}Yx{t}Y: MAE = {c['vol_error_bp'].abs().mean():>6.1f}  "
                      f"bias = {c['vol_error_bp'].mean():>+6.1f}  N = {len(c)}")
    summary_rows.append({
        "regime": name, "label": regime["label"],
        "n_obs": len(sub),
        "mae_bp":  sub["vol_error_bp"].abs().mean(),
        "rmse_bp": float(np.sqrt((sub["vol_error_bp"]**2).mean())),
        "bias_bp": sub["vol_error_bp"].mean(),
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(EVAL_DIR, "regime_summary.csv"), index=False)
print(f"\nOverall (all regimes pooled): MAE = {df_all['vol_error_bp'].abs().mean():.1f} bp  N = {len(df_all)}")

# LaTeX summary table
lines = [r"\begin{table}[H]", r"\centering",
         r"\caption{Constant MPR: out-of-sample MAE across three EUR regimes.}",
         r"\label{tab:constant_mpr_regimes}",
         r"\small",
         r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
         r"\textbf{Regime} & \textbf{N obs} & \textbf{MAE (bp)} & \textbf{RMSE (bp)} & \textbf{Bias (bp)} \\",
         r"\midrule"]
for s in summary_rows:
    lines.append(f"  {s['label']} & {s['n_obs']} & {s['mae_bp']:.1f} & {s['rmse_bp']:.1f} & {s['bias_bp']:+.1f} \\\\")
lines += [r"\midrule",
          f"  \\textbf{{Average}} & {len(df_all)} & {df_all['vol_error_bp'].abs().mean():.1f} & "
          f"{np.sqrt((df_all['vol_error_bp']**2).mean()):.1f} & "
          f"{df_all['vol_error_bp'].mean():+.1f} \\\\",
          r"\bottomrule", r"\end{tabular}", r"\end{table}"]
with open(os.path.join(EVAL_DIR, "tab_constant_mpr_regimes.tex"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

# Per-cell summary heatmap (each regime)
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, regime in zip(axes, REGIMES):
    name = regime["name"]
    sub = df_all[df_all["regime"] == name]
    if len(sub) == 0: ax.set_visible(False); continue
    mae_grid = np.full((3, 3), np.nan)
    for i, e in enumerate(EXPIRY_VALS):
        for j, t in enumerate(TENOR_VALS):
            c = sub[(sub['expiry']==e) & (sub['tenor']==t)]
            if len(c): mae_grid[i, j] = c["vol_error_bp"].abs().mean()
    vmax = np.nanmax([np.nanmax(mae_grid) if not np.all(np.isnan(mae_grid)) else 0,
                       1])
    im = ax.imshow(mae_grid, cmap="YlOrRd", aspect="auto", vmin=0, vmax=vmax)
    plt.colorbar(im, ax=ax, label="MAE (bp)")
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels([f"{t}Y" for t in TENOR_VALS])
    ax.set_yticklabels([f"{e}Y" for e in EXPIRY_VALS])
    for ii in range(3):
        for jj in range(3):
            if not np.isnan(mae_grid[ii,jj]):
                ax.text(jj, ii, f"{mae_grid[ii,jj]:.0f}", ha="center", va="center",
                        fontsize=10, fontweight="bold",
                        color="white" if mae_grid[ii,jj] > vmax*0.6 else "black")
    ax.set_xlabel("Tenor"); ax.set_ylabel("Expiry")
    ax.set_title(regime["label"], fontsize=9)
fig.suptitle("Constant MPR — out-of-sample MAE by regime", fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(EVAL_DIR, "fig_regime_heatmaps.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"\nAll outputs saved to: {EVAL_DIR}")
print(f"Per-regime checkpoints under: {BASE_FIGURES_DIR}")
print("="*100)
