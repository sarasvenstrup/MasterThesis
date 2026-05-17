# ==================== Surface Vol MPR — Regime-Window OOS ====================
"""
Surface Vol MPR with three regime-window out-of-sample tests.

Identical regime structure to Training_constant_mpr_regimes.py, but using the
Surface Vol model that consumes the full vol-surface cross-section via three
leave-one-cell-out features:

    log σ_{e,n,k}(z_0, v_t^e, v_t^n, v_t^g) =
          a_k + b_{e,k} + c_{n,k}
        + δ_k · tanh(W · z_0)
        + d^expiry_{e,k} · v_t^e
        + d^tenor_{n,k}  · v_t^n
        + d^global_k     · v_t^global

LOO features are computed independently per date, so they use only that date's
observed cross-section (no future leakage).

Regimes:
  1. negative_rates     train < 2014-01  test 2014-01 .. 2019-12
  2. covid              train < 2020-01  test 2020-01 .. 2021-12
  3. rate_normalisation train < 2022-01  test 2022-01 .. end-of-data

Each regime trains from a fresh ETV warm-start.  This is the headline OOS
evaluation for the full-information (yield curve + observed vols) model.

Output: Figures/TrainingResults/dim4_surface_vol_mpr_regimes/{regime}/ep{EPOCHS}/
        Figures/pricing/eval_surface_vol_mpr_regimes/per_cell_final.csv
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
        "test_end":   pd.Timestamp("2030-12-31"),
    },
]

# ── settings ───────────────────────────────────────────────────────────────────
LATENT_DIM  = 4
EXPIRY_VALS = [1, 5, 10]
TENOR_VALS  = [1, 5, 10]

EPOCHS                = 500    # less than single-split (1000) to keep total ≤24h
EVAL_EVERY            = 100
HEADER_EVERY          = 20
SAVE_EVERY            = 100
N_STEPS_PER_EPOCH     = 4
N_SWAPTIONS_PER_BATCH = 8
N_PATHS_PRICING       = 512
DT_PRICING            = 1 / 6

PRETRAIN_CKPT = os.path.join(PROJECT_ROOT, "Figures", "TrainingResults",
                              "dim4_stable", "ep5000", "checkpoint_dim4_ep5000.pt")
ETV_CKPT      = os.path.join(PROJECT_ROOT, "Figures", "TrainingResults",
                              "dim4_expiry_tenor_vol_mpr", "ep1000",
                              "checkpoint_expiry_tenor_vol_mpr_ep1000.pt")

LAMBDA_VOL      = 1.0
LAMBDA_BIAS     = 0.5
LAMBDA_L2_LAM   = 1e-3
LAMBDA_L2_VOL   = 1e-3
LAMBDA_L2_REG   = 1e-3
LAMBDA_L2_VFEAT = 1e-2   # 10× the static reg — features have magnitude 1

LR             = 2e-4
LR_SCALE_MULT  = 10.0
LR_OFFSET_MULT = 5.0
LR_REG_MULT    = 5.0
LR_VFEAT_MULT  = 1.0
LR_WARMUP      = 30

V_FEATURE_SCALE = 100.0  # rescale bp features into [0.5, 2.0] range

MIN_FINITE_PATHS_ABS  = 16
MIN_FINITE_PATHS_FRAC = 0.10
LOSS_SKIP_THRESH      = 1e4

# Eval
N_PATHS_EVAL = 512
RATE_CLIP    = 0.50
ANNUITY_MAX  = 50.0

USE        = "bbg"
CCY_FILTER = "EUR"

BASE_FIGURES_DIR = os.path.join(PROJECT_ROOT, "Figures", "TrainingResults",
                                 f"dim{LATENT_DIM}_surface_vol_mpr_regimes")
EVAL_DIR         = os.path.join(PROJECT_ROOT, "Figures", "pricing", "eval_surface_vol_mpr_regimes")
os.makedirs(BASE_FIGURES_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)

# ── module ─────────────────────────────────────────────────────────────────────

class SurfaceVolMPR(nn.Module):
    def __init__(self, kp_module, h_module, latent_dim, expiry_vals, tenor_vals):
        super().__init__()
        self.kp = kp_module; self.h = h_module
        self.latent_dim = latent_dim
        self.expiry_vals = expiry_vals; self.tenor_vals = tenor_vals
        self.expiry_to_idx = {e: i for i, e in enumerate(expiry_vals)}
        self.tenor_to_idx  = {t: i for i, t in enumerate(tenor_vals)}
        n_exp = len(expiry_vals); n_ten = len(tenor_vals)

        self.lambda_expiry    = nn.Parameter(torch.zeros(n_exp, latent_dim))
        self.log_sigma_base   = nn.Parameter(torch.full((latent_dim,), -1.8))
        self.log_sigma_expiry = nn.Parameter(torch.zeros(n_exp, latent_dim))
        self.log_sigma_tenor  = nn.Parameter(torch.zeros(n_ten, latent_dim))
        self.W     = nn.Parameter(torch.randn(latent_dim, latent_dim) * 0.01)
        self.delta = nn.Parameter(torch.zeros(latent_dim))
        self.d_expiry = nn.Parameter(torch.zeros(n_exp, latent_dim))
        self.d_tenor  = nn.Parameter(torch.zeros(n_ten, latent_dim))
        self.d_global = nn.Parameter(torch.zeros(latent_dim))

    def get_sigma_eff(self, expiry, tenor, z0, v_e, v_n, v_g):
        e = self.expiry_to_idx[expiry]; n = self.tenor_to_idx[tenor]
        z = z0.squeeze(0) if z0.dim() == 2 else z0
        regime = self.delta * torch.tanh(self.W @ z)
        return (self.log_sigma_base
                + self.log_sigma_expiry[e]
                + self.log_sigma_tenor[n]
                + regime
                + self.d_expiry[e] * v_e
                + self.d_tenor[n]  * v_n
                + self.d_global    * v_g).exp()

    def drift(self, z_t, expiry):
        k_base = self.kp(z_t); sigmas, rhos = self.h(z_t)
        L = L_from_sigmas_rhos(sigmas, rhos, validate=False)
        lam = self.lambda_expiry[self.expiry_to_idx[expiry]].unsqueeze(0).expand(z_t.shape[0], -1)
        return k_base + torch.einsum('bij,bj->bi', L, lam)

    def forward(self, z_t):
        k_base = self.kp(z_t); sigmas, rhos = self.h(z_t)
        L = L_from_sigmas_rhos(sigmas, rhos, validate=False)
        lam = self.lambda_expiry.mean(0).unsqueeze(0).expand(z_t.shape[0], -1)
        return k_base + torch.einsum('bij,bj->bi', L, lam)

    @property
    def sigma_vec(self): return self.log_sigma_base.exp()


class SurfaceVolDriftWrapper(nn.Module):
    def __init__(self, model, expiry, tenor, z0, v_e, v_n, v_g):
        super().__init__()
        self.model = model; self.expiry = expiry; self.tenor = tenor
        self._z0 = z0; self._v_e = v_e; self._v_n = v_n; self._v_g = v_g

    def forward(self, z_t): return self.model.drift(z_t, self.expiry)

    @property
    def sigma_vec(self):
        return self.model.get_sigma_eff(self.expiry, self.tenor, self._z0,
                                        self._v_e, self._v_n, self._v_g)


# ── LOO features ───────────────────────────────────────────────────────────────

def build_loo_features(df_vol):
    """For each (date, e, n): (v_t^e, v_t^n, v_t^global) in bp / V_FEATURE_SCALE."""
    out = {}
    for date, grp in df_vol.groupby("as_of_date"):
        date_ts = pd.Timestamp(date).normalize()
        for _, r in grp.iterrows():
            e = int(r["option_maturity"]); t = int(r["swap_tenor"])
            same_e_other_t = grp[(grp["option_maturity"] == e) & (grp["swap_tenor"] != t)]
            same_t_other_e = grp[(grp["swap_tenor"]  == t) & (grp["option_maturity"] != e)]
            other_both    = grp[~((grp["option_maturity"] == e) & (grp["swap_tenor"] == t))]
            if len(same_e_other_t) == 0 or len(same_t_other_e) == 0 or len(other_both) == 0:
                continue
            v_e = float(same_e_other_t["market_vol"].mean()) * 1e4 / V_FEATURE_SCALE
            v_n = float(same_t_other_e["market_vol"].mean()) * 1e4 / V_FEATURE_SCALE
            v_g = float(other_both["market_vol"].mean()) * 1e4 / V_FEATURE_SCALE
            out[(date_ts, e, t)] = (v_e, v_n, v_g)
    return out


# ── pricing loss ───────────────────────────────────────────────────────────────

def compute_pricing_loss(model, lm, X_batch, df_vol_window, date_to_idx, loo_features,
                         n_swaptions, n_paths, dt, device, dtype):
    if len(df_vol_window) == 0:
        zero = torch.tensor(0.0, device=device, dtype=dtype)
        return zero, zero, 0, 0, 0.0

    sample      = df_vol_window.sample(n=min(n_swaptions, len(df_vol_window)))
    total_vol   = torch.zeros(1, device=device, dtype=dtype)
    total_bias  = torch.zeros(1, device=device, dtype=dtype)
    n_valid = 0; n_attempted = 0; path_fracs = []
    min_paths = max(MIN_FINITE_PATHS_ABS, int(n_paths * MIN_FINITE_PATHS_FRAC))

    for _, row in sample.iterrows():
        date   = pd.Timestamp(row["as_of_date"]).normalize()
        expiry = int(row["option_maturity"]); tenor = int(row["swap_tenor"])
        sigma_mkt_bp = float(row["market_vol"]) * 1e4

        if (date not in date_to_idx
                or expiry not in lm.expiry_to_idx
                or tenor  not in lm.tenor_to_idx):
            continue
        loo = loo_features.get((date, expiry, tenor), None)
        if loo is None: continue
        v_e, v_n, v_g = loo

        n_attempted += 1
        idx = date_to_idx[date]
        xb  = X_batch[idx:idx+1].to(device)
        with torch.no_grad():
            z0 = model.encoder(xb)
            _, aux0 = model.decode_from_z(z0, tau=None, return_aux=True)
            P0 = aux0["P_full"][0]
        if expiry + tenor > P0.shape[0] - 1: continue
        F_0, A_0 = forward_swap_rate_torch(P0, expiry, tenor)
        if not (math.isfinite(F_0) and math.isfinite(A_0) and A_0 > 1e-6): continue

        dt_eff  = min(dt, expiry / 10.0)
        n_steps = max(12, int(round(expiry / dt_eff)))
        half    = n_paths // 2
        with torch.no_grad():
            eps_z = torch.randn(half, n_steps, LATENT_DIM, device=device, dtype=dtype)
        wrapper = SurfaceVolDriftWrapper(lm, expiry, tenor, z0, v_e, v_n, v_g)

        try:
            z_T, D_T = simulate_to_expiry_differentiable(
                model, z0, n_steps=n_steps, dt=dt_eff,
                n_paths=2*half, eps=eps_z,
                k_override=wrapper, sigma_scale=wrapper.sigma_vec,
                antithetic=True, freeze_H=True)
            z_ok = torch.isfinite(z_T).all(1) & torch.isfinite(D_T)
            if int(z_ok.sum()) < min_paths: continue

            with torch.no_grad():
                _, aux_T = model.decode_from_z(z_T, tau=None, return_aux=True,
                                               k_override=wrapper, sigma_scale=wrapper.sigma_vec)
                p_ok = torch.isfinite(aux_T["P_full"]).all(1)
            mask = z_ok & p_ok
            if int(mask.sum()) < min_paths: continue
            path_fracs.append(float(mask.float().mean()))

            _, aux_k = model.decode_from_z(z_T[mask], tau=None, return_aux=True,
                                           k_override=wrapper, sigma_scale=wrapper.sigma_vec)
            F_T, A_T = swap_rate_torch(aux_k["P_full"], tenor=tenor)
            D_keep = D_T[mask]
            fa_ok = (torch.isfinite(F_T) & torch.isfinite(A_T)
                     & (F_T > -0.5) & (F_T < 0.5) & (A_T > 1e-6) & (A_T < 50.0))
            if int(fa_ok.sum()) < min_paths: continue
            F_T, A_T, D_keep = F_T[fa_ok], A_T[fa_ok], D_keep[fa_ok]

            V_pay = (D_keep * A_T * torch.relu(F_T - F_0)).mean()
            V_rec = (D_keep * A_T * torch.relu(F_0 - F_T)).mean()
            if not (torch.isfinite(V_pay) and torch.isfinite(V_rec)
                    and float(V_pay.detach()) >= 0 and float(V_rec.detach()) >= 0): continue

            sqrt_2pi = math.sqrt(2 * math.pi)
            sigma_str_bp = (V_pay + V_rec) * 0.5 * sqrt_2pi / (A_0 * math.sqrt(expiry)) * 1e4
            loss_vol_ij  = ((sigma_str_bp - sigma_mkt_bp) / 100.0).pow(2)
            if not torch.isfinite(loss_vol_ij) or float(loss_vol_ij.detach()) > LOSS_SKIP_THRESH:
                continue
            fwd_bias_bp = (V_pay - V_rec) / A_0 * 1e4
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

def price_cell_eval(model, lm, X_eur, date_to_idx, loo_features, date, expiry, tenor):
    if date not in date_to_idx: return None
    loo = loo_features.get((date, expiry, tenor), None)
    if loo is None: return None
    v_e, v_n, v_g = loo
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
    wrapper = SurfaceVolDriftWrapper(lm, expiry, tenor, z0, v_e, v_n, v_g)

    with torch.no_grad():
        eps_z = torch.randn(half, n_steps, LATENT_DIM, device=device)
        z_T, D_T = simulate_to_expiry_differentiable(
            model, z0, n_steps=n_steps, dt=dt_eff,
            n_paths=N_PATHS_EVAL, eps=eps_z,
            k_override=wrapper, sigma_scale=wrapper.sigma_vec,
            antithetic=True, freeze_H=True)
    ok = torch.isfinite(z_T).all(1) & torch.isfinite(D_T)
    if ok.sum() < 16: return None
    z_k, D_k = z_T[ok], D_T[ok]

    with torch.no_grad():
        _, aux_T = model.decode_from_z(z_k, tau=None, return_aux=True,
                                       k_override=wrapper, sigma_scale=wrapper.sigma_vec)
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

model = FullModel(latent_dim=LATENT_DIM).to(device)
raw   = torch.load(PRETRAIN_CKPT, map_location=device, weights_only=False)
model.load_state_dict(raw.get("model_state_dict", raw))
for p in model.parameters(): p.requires_grad_(False)
print("Base model loaded and frozen.")

df_vol = load_swaption_vol_data(currency=CCY_FILTER)
df_vol["as_of_date"] = pd.to_datetime(df_vol["as_of_date"]).dt.normalize()
df_vol["market_vol"] = df_vol["vol"] / 10_000.0
df_vol = df_vol[df_vol["option_maturity"].isin(EXPIRY_VALS)
                & df_vol["swap_tenor"].isin(TENOR_VALS)].copy()
dates_swap  = set(pd.to_datetime(meta_ccy["as_of_date"]).dt.normalize())
df_vol      = df_vol[df_vol["as_of_date"].isin(dates_swap)].copy()
date_to_idx = {pd.Timestamp(r["as_of_date"]).normalize(): i for i, r in meta_ccy.iterrows()}
if df_vol.empty: raise RuntimeError("No swaption vol data.")

print("Precomputing LOO features ...")
loo_features = build_loo_features(df_vol)
print(f"  {len(loo_features)} cell-date entries")

print(f"Loaded {len(df_vol)} vol targets from {df_vol['as_of_date'].nunique()} dates")
print(f"Date range: {df_vol['as_of_date'].min().date()} -> {df_vol['as_of_date'].max().date()}")

# ── regime loop ────────────────────────────────────────────────────────────────
all_test_rows = []

for regime_idx, regime in enumerate(REGIMES):
    name      = regime["name"]; label = regime["label"]
    train_end = regime["train_end"]
    test_start= regime["test_start"]; test_end = regime["test_end"]

    df_train = df_vol[df_vol["as_of_date"] <= train_end].copy()
    df_test  = df_vol[(df_vol["as_of_date"] >= test_start)
                       & (df_vol["as_of_date"] <= test_end)].copy()

    n_train = len(df_train); n_test = len(df_test)
    n_tr_dt = df_train["as_of_date"].nunique(); n_te_dt = df_test["as_of_date"].nunique()

    print("\n" + "="*100)
    print(f"REGIME {regime_idx+1}/3:  {name}  ({label})")
    print(f"  Train: <= {train_end.date()}   n={n_train}  dates={n_tr_dt}")
    print(f"  Test:  {test_start.date()} -> {test_end.date()}   n={n_test}  dates={n_te_dt}")
    print("="*100)

    if n_train == 0 or n_test == 0:
        print("  SKIP — insufficient data"); continue

    regime_dir = os.path.join(BASE_FIGURES_DIR, name, f"ep{EPOCHS}")
    os.makedirs(regime_dir, exist_ok=True)

    # Fresh model, ETV warm-start
    torch.manual_seed(SEED + regime_idx); np.random.seed(SEED + regime_idx)
    lm = SurfaceVolMPR(model.K, model.H, LATENT_DIM, EXPIRY_VALS, TENOR_VALS).to(device)
    if os.path.exists(ETV_CKPT):
        raw_e = torch.load(ETV_CKPT, map_location=device, weights_only=False)
        es    = raw_e.get("lm_state_dict", raw_e)
        with torch.no_grad():
            for key in ["lambda_expiry","log_sigma_base","log_sigma_expiry","log_sigma_tenor"]:
                if key in es: getattr(lm, key).copy_(es[key].to(device))
        print(f"  Warm-started static params from ETV MPR")

    model.train()
    optim = torch.optim.Adam([
        {'params': [lm.lambda_expiry],    'lr': LR},
        {'params': [lm.log_sigma_base],   'lr': LR * LR_SCALE_MULT},
        {'params': [lm.log_sigma_expiry], 'lr': LR * LR_OFFSET_MULT},
        {'params': [lm.log_sigma_tenor],  'lr': LR * LR_OFFSET_MULT},
        {'params': [lm.W, lm.delta],      'lr': LR * LR_REG_MULT},
        {'params': [lm.d_expiry, lm.d_tenor, lm.d_global], 'lr': LR * LR_VFEAT_MULT},
    ], lr=LR)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optim,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(optim, 1e-3, 1.0, LR_WARMUP),
            torch.optim.lr_scheduler.CosineAnnealingLR(optim, max(EPOCHS-LR_WARMUP,1), 1e-7),
        ],
        milestones=[LR_WARMUP])

    csv_path = os.path.join(regime_dir, f"train_log_{name}_ep{EPOCHS}.csv")
    pd.DataFrame(columns=["epoch","time_total_sec","loss_vol","loss_bias",
                          "lam_mean_norm","delta_norm","W_norm",
                          "d_expiry_norm","d_tenor_norm","d_global_norm",
                          "swaption_priced_frac","path_finite_frac","lr"]
                 ).to_csv(csv_path, index=False)
    with open(os.path.join(regime_dir, "run_config.json"), "w") as f:
        json.dump({"regime": name, "label": label,
                   "train_end":  train_end.date().isoformat(),
                   "test_start": test_start.date().isoformat(),
                   "test_end":   test_end.date().isoformat(),
                   "n_train_obs": n_train, "n_test_obs": n_test,
                   "epochs": EPOCHS, "v_feature_scale": V_FEATURE_SCALE}, f, indent=2)

    t0 = time.perf_counter(); t_last = t0
    for epoch in range(EPOCHS):
        model.train(); lm.train()
        r_vol = r_bias = 0.0; n_bat = 0
        ep_att = ep_pri = 0; ep_pf = []

        for step in range(N_STEPS_PER_EPOCH):
            print(f"\r  ep {epoch}  step {step+1}/{N_STEPS_PER_EPOCH} ...", end="", flush=True)
            optim.zero_grad(set_to_none=True)
            lv, lb, na, np_, pf = compute_pricing_loss(
                model, lm, X_tensor_ccy, df_train, date_to_idx, loo_features,
                N_SWAPTIONS_PER_BATCH, N_PATHS_PRICING, DT_PRICING,
                device, torch.float32)
            ep_att+=na; ep_pri+=np_
            if pf>0: ep_pf.append(pf)
            l2_lam  = LAMBDA_L2_LAM * lm.lambda_expiry.pow(2).sum()
            l2_vol  = LAMBDA_L2_VOL * (lm.log_sigma_expiry.pow(2).sum()
                                        + lm.log_sigma_tenor.pow(2).sum())
            l2_reg  = LAMBDA_L2_REG * (lm.delta.pow(2).sum() + lm.W.pow(2).sum())
            l2_feat = LAMBDA_L2_VFEAT * (lm.d_expiry.pow(2).sum()
                                          + lm.d_tenor.pow(2).sum()
                                          + lm.d_global.pow(2).sum())
            loss = LAMBDA_VOL*lv + LAMBDA_BIAS*lb + l2_lam + l2_vol + l2_reg + l2_feat
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
        ep_vol = r_vol/n_bat; ep_bias = r_bias/n_bat
        swp = ep_pri/max(ep_att, 1); pth = float(np.mean(ep_pf)) if ep_pf else 0.0

        with torch.no_grad():
            lam_mean   = float(lm.lambda_expiry.mean(0).norm())
            delta_norm = float(lm.delta.norm())
            W_norm     = float(lm.W.norm())
            de_norm    = float(lm.d_expiry.norm())
            dn_norm    = float(lm.d_tenor.norm())
            dg_norm    = float(lm.d_global.norm())

        t_now = time.perf_counter(); dt_ep = t_now-t_last; t_last = t_now
        lr_now = optim.param_groups[0]["lr"]

        pd.DataFrame([{"epoch":epoch,"time_total_sec":round(t_now-t0,1),
                       "loss_vol":ep_vol,"loss_bias":ep_bias,
                       "lam_mean_norm":lam_mean,"delta_norm":delta_norm,"W_norm":W_norm,
                       "d_expiry_norm":de_norm,"d_tenor_norm":dn_norm,"d_global_norm":dg_norm,
                       "swaption_priced_frac":swp,"path_finite_frac":pth,"lr":lr_now}]
                     ).to_csv(csv_path, mode="a", header=False, index=False)

        if epoch % HEADER_EVERY == 0:
            print(f"\n  {'ep':>5} {'vol':>10} {'bias':>9} {'|λ|':>6} {'|δ|':>6} {'|W|':>6} "
                  f"{'|d_e|':>6} {'|d_n|':>6} {'|d_g|':>6} {'swp%':>4} {'t/ep':>5}")
            print("  " + "-"*100)
        print(f"  {epoch:>5d} {ep_vol:>10.4e} {ep_bias:>9.3e} "
              f"{lam_mean:>6.4f} {delta_norm:>6.4f} {W_norm:>6.4f} "
              f"{de_norm:>6.4f} {dn_norm:>6.4f} {dg_norm:>6.4f} "
              f"{swp*100:>3.0f}% {dt_ep:>5.1f}s")

        if (epoch+1) % SAVE_EVERY == 0 or epoch == EPOCHS-1:
            torch.save({"lm_state_dict": lm.state_dict(),
                        "regime": name, "epoch": epoch+1,
                        "train_end": train_end.date().isoformat()},
                       os.path.join(regime_dir, f"checkpoint_surface_vol_{name}_ep{epoch+1}.pt"))

    print(f"  Final: |λ|={lam_mean:.4f}  |δ|={delta_norm:.4f}  |W|={W_norm:.4f}")
    print(f"         |d_e|={de_norm:.4f}  |d_n|={dn_norm:.4f}  |d_g|={dg_norm:.4f}")

    # ── eval ───────────────────────────────────────────────────────────────────
    print(f"  Evaluating on {n_test} test obs ...")
    lm.eval(); model.eval()
    torch.manual_seed(42)
    t_eval = time.time()
    combos = df_test[["as_of_date","option_maturity","swap_tenor","market_vol"]].drop_duplicates()
    rows = []
    for counter, (_, row) in enumerate(combos.iterrows()):
        date   = pd.Timestamp(row["as_of_date"]).normalize()
        expiry = int(row["option_maturity"]); tenor = int(row["swap_tenor"])
        mkt_bp = float(row["market_vol"]) * 1e4
        if expiry not in EXPIRY_VALS or tenor not in TENOR_VALS: continue
        result = price_cell_eval(model, lm, X_tensor_ccy, date_to_idx, loo_features,
                                  date, expiry, tenor)
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
    print(f"  Done: {len(df_eval)} priced  |  MAE = {df_eval['vol_error_bp'].abs().mean():.1f} bp")
    all_test_rows.extend(rows)

# ── aggregate ──────────────────────────────────────────────────────────────────
df_all = pd.DataFrame(all_test_rows)
df_all.to_csv(os.path.join(EVAL_DIR, "per_cell_final.csv"), index=False)
print("\n" + "="*100)
print("AGGREGATED RESULTS — Surface Vol MPR")
print("="*100)

summary_rows = []
for regime in REGIMES:
    name = regime["name"]
    sub = df_all[df_all["regime"] == name]
    if len(sub) == 0: continue
    print(f"\n-- {name} ({regime['label']}) --")
    print(f"  Overall: MAE = {sub['vol_error_bp'].abs().mean():.1f} bp  "
          f"RMSE = {np.sqrt((sub['vol_error_bp']**2).mean()):.1f} bp  N = {len(sub)}")
    for e in EXPIRY_VALS:
        for t in TENOR_VALS:
            c = sub[(sub['expiry']==e) & (sub['tenor']==t)]
            if len(c):
                print(f"    {e}Yx{t}Y: MAE = {c['vol_error_bp'].abs().mean():>6.1f}  N = {len(c)}")
    summary_rows.append({"regime": name, "label": regime["label"], "n_obs": len(sub),
                         "mae_bp": sub["vol_error_bp"].abs().mean(),
                         "rmse_bp": float(np.sqrt((sub["vol_error_bp"]**2).mean())),
                         "bias_bp": sub["vol_error_bp"].mean()})

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(EVAL_DIR, "regime_summary.csv"), index=False)
print(f"\nOverall pooled: MAE = {df_all['vol_error_bp'].abs().mean():.1f} bp  N = {len(df_all)}")

# LaTeX
lines = [r"\begin{table}[H]", r"\centering",
         r"\caption{Surface Vol MPR: out-of-sample MAE across three EUR regimes (full yield-curve + observed vol-surface information).}",
         r"\label{tab:surface_vol_mpr_regimes}", r"\small",
         r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
         r"\textbf{Regime} & \textbf{N obs} & \textbf{MAE (bp)} & \textbf{RMSE (bp)} & \textbf{Bias (bp)} \\",
         r"\midrule"]
for s in summary_rows:
    lines.append(f"  {s['label']} & {s['n_obs']} & {s['mae_bp']:.1f} & {s['rmse_bp']:.1f} & {s['bias_bp']:+.1f} \\\\")
lines += [r"\midrule",
          f"  \\textbf{{Average}} & {len(df_all)} & {df_all['vol_error_bp'].abs().mean():.1f} & "
          f"{np.sqrt((df_all['vol_error_bp']**2).mean()):.1f} & {df_all['vol_error_bp'].mean():+.1f} \\\\",
          r"\bottomrule", r"\end{tabular}", r"\end{table}"]
with open(os.path.join(EVAL_DIR, "tab_surface_vol_mpr_regimes.tex"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

# Heatmaps
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
    vmax = np.nanmax([np.nanmax(mae_grid) if not np.all(np.isnan(mae_grid)) else 0, 1])
    im = ax.imshow(mae_grid, cmap="YlOrRd", aspect="auto", vmin=0, vmax=vmax)
    plt.colorbar(im, ax=ax, label="MAE (bp)")
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels([f"{t}Y" for t in TENOR_VALS])
    ax.set_yticklabels([f"{e}Y" for e in EXPIRY_VALS])
    for ii in range(3):
        for jj in range(3):
            if not np.isnan(mae_grid[ii, jj]):
                ax.text(jj, ii, f"{mae_grid[ii, jj]:.0f}", ha="center", va="center",
                        fontsize=10, fontweight="bold",
                        color="white" if mae_grid[ii, jj] > vmax*0.6 else "black")
    ax.set_xlabel("Tenor"); ax.set_ylabel("Expiry")
    ax.set_title(regime["label"], fontsize=9)
fig.suptitle("Surface Vol MPR — out-of-sample MAE by regime", fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(EVAL_DIR, "fig_regime_heatmaps.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"\nAll outputs saved to: {EVAL_DIR}")
print(f"Per-regime checkpoints under: {BASE_FIGURES_DIR}")
print("="*100)
