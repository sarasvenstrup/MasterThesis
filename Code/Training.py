# train_and_arbitrage.py
# ------------------------------------------------------------
# Full training script + arbitrage diagnostics (annual tau grid)
# + robust handling of non-finite batches
# + nan-safe arbitrage report
# + optional SR diagnostics using your sharpe_ratio.py
# ------------------------------------------------------------

import os
import sys
import math
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

# Thread settings (keep first Torch-related thing)
torch.set_num_threads(4)
torch.set_num_interop_threads(2)

# -----------------------------
# Repo path
# -----------------------------
try:
    REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
except NameError:
    REPO_ROOT = os.getcwd()

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from torch.utils.data import TensorDataset, DataLoader

from Code.load_swapdata import build_all_dataframes, TARGET_TENORS
from Code.model.full_model import FullModel

from Code.utils.helpers import check_monotonicity, instantaneous_forward, finite_minmax

# SR diagnostics (your file)
from Code.utils.sharpe_ratio import SR_andreasen_reference


# ============================================================
# 0) Device
# ============================================================
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Using device:", device)


# ============================================================
# 1) Load data
# ============================================================
USE = "bbg"  # "test" or "bbg"
data = build_all_dataframes()

if USE == "test":
    df_wide_full = data["df_wide_test_full"].copy()
else:
    df_wide_full = data["df_wide_bbg_full"].copy()

df_wide = df_wide_full[["as_of_date", "ccy"] + list(TARGET_TENORS)].copy()
df_wide["as_of_date"] = pd.to_datetime(df_wide["as_of_date"])
df_wide = df_wide[df_wide["as_of_date"] >= "2010-01-01"].copy()

meta = df_wide[["as_of_date", "ccy"]].reset_index(drop=True)

X = df_wide[list(TARGET_TENORS)].to_numpy(dtype=np.float32)
print("Wide shape:", X.shape)

# Auto-detect scaling (percent vs decimal)
median_abs = float(np.nanmedian(np.abs(X)))
SCALE_IS_PERCENT = median_abs > 0.5
print("Median |swap|:", median_abs, "=> SCALE_IS_PERCENT =", SCALE_IS_PERCENT)
if SCALE_IS_PERCENT:
    X = X / 100.0

X_tensor = torch.from_numpy(X)  # (N,8), CPU
print("X_tensor:", tuple(X_tensor.shape))


# ============================================================
# 2) DataLoader
# ============================================================
BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 100

dataset = TensorDataset(X_tensor)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)


# ============================================================
# 3) Model
# ============================================================
torch.manual_seed(0)

model = FullModel(latent_dim=2, ab_solver="chen").to(device)

# IMPORTANT:
# Your FullModel currently clamps Xexp only if self.training.
# That makes eval potentially blow up. Recommended: clamp ALWAYS in FullModel.
# If you didn't change FullModel, we keep model in train() even for diagnostics that call model,
# and explicitly set do_arb_checks=False except a tiny debug call at the end.

model.train()

optim = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

GRAD_CLIP_NORM = 1.0   # helps stability a lot
PRINT_BAD_BATCH_DIAGS = True


# ============================================================
# 4) Train
# ============================================================
train_losses = []
nan_batches_total = 0

for epoch in range(EPOCHS):
    running = 0.0
    n_obs = 0
    nan_batches = 0

    for (xb_cpu,) in loader:
        xb = xb_cpu.to(device)

        optim.zero_grad(set_to_none=True)

        # Forward
        out = model(xb, do_arb_checks=False)
        S_hat, z, P, A, B, G, mu, L, r = out

        loss = loss_fn(S_hat, xb)

        if not torch.isfinite(loss):
            nan_batches += 1

            if PRINT_BAD_BATCH_DIAGS:
                print("\n[WARN] Non-finite loss detected. Skipping batch.")
                print("finite S_hat:", torch.isfinite(S_hat).all().item(),
                      "finite P:", torch.isfinite(P).all().item(),
                      "finite A:", torch.isfinite(A).all().item(),
                      "finite B:", torch.isfinite(B).all().item(),
                      "finite G:", torch.isfinite(G).all().item(),
                      "finite mu:", torch.isfinite(mu).all().item(),
                      "finite L:", torch.isfinite(L).all().item(),
                      "finite r:", torch.isfinite(r).all().item())

                pmin, pmax = finite_minmax(P)
                amin, amax = finite_minmax(A)
                bmin, bmax = finite_minmax(B)
                gmin, gmax = finite_minmax(G)
                print("P finite min/max:", pmin, pmax)
                print("A finite min/max:", amin, amax)
                print("B finite min/max:", bmin, bmax)
                print("G finite min/max:", gmin, gmax)

            # skip this batch
            continue

        # Backward
        loss.backward()

        # Gradient clipping
        if GRAD_CLIP_NORM is not None and GRAD_CLIP_NORM > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)

        optim.step()

        running += float(loss.detach().cpu()) * xb.shape[0]
        n_obs += xb.shape[0]

    nan_batches_total += nan_batches
    epoch_loss = running / max(n_obs, 1)
    train_losses.append(epoch_loss)

    if epoch % 10 == 0 or epoch == EPOCHS - 1:
        print(
            f"epoch={epoch:4d} loss={epoch_loss:.6e} "
            f"used_obs={n_obs} nan_batches={nan_batches} total_nan_batches={nan_batches_total}"
        )

print("\nTraining done.")


# ============================================================
# 5) Arbitrage diagnostics utilities (nan-safe)
# ============================================================
@torch.no_grad()
def arbitrage_report_from_P(P: torch.Tensor, tau: torch.Tensor, tag: str = ""):
    """
    P: (B,T) discount factors on tau grid (includes tau=0)
    tau: (T,)
    """
    # Make safe versions for stats (don't change the raw P for future use)
    P_safe = torch.nan_to_num(P, nan=1.0, posinf=1.0, neginf=1.0)

    # P(0) check
    p0_min = float(P_safe[:, 0].min().cpu())
    p0_max = float(P_safe[:, 0].max().cpu())
    p_min  = float(P_safe.min().cpu())

    mono_viol = int(check_monotonicity(P_safe))

    f = instantaneous_forward(P_safe, tau)
    f_safe = torch.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
    neg_forwards = int((f_safe < -1e-8).sum().item())
    f_min = float(f_safe.min().cpu())

    print("\n==============================")
    print("ARBITRAGE REPORT", f"({tag})" if tag else "")
    print("==============================")
    print(f"P(0) min/max: {p0_min:.6g} / {p0_max:.6g}")
    print(f"Min P:        {p_min:.6g}")
    print(f"Monotonicity violations (P up): {mono_viol}")
    print(f"Negative forward count:         {neg_forwards}")
    print(f"Min forward:                    {f_min:.6g}")


@torch.no_grad()
def run_arbitrage_on_batch(model: nn.Module, xb_cpu: torch.Tensor, tau_max: int, tag: str):
    xb = xb_cpu.to(next(model.parameters()).device)

    # Keep model in train() if your FullModel only clamps in training.
    # If you updated FullModel to clamp always, you can safely call model.eval().
    was_training = model.training
    model.train()

    out = model(xb, do_arb_checks=False)
    P = out[2]  # (B,T=tau_max+1)

    tau = torch.arange(0, tau_max + 1, device=P.device, dtype=P.dtype)
    arbitrage_report_from_P(P, tau, tag=tag)

    if not was_training:
        model.eval()


@torch.no_grad()
def pick_one_curve_per_currency_on_date(meta_df: pd.DataFrame, date_pick):
    m = meta_df.copy()
    m["as_of_date"] = pd.to_datetime(m["as_of_date"])
    date_pick = pd.to_datetime(date_pick)
    sel = m[m["as_of_date"] == date_pick].copy()
    if sel.empty:
        raise ValueError(f"No rows in meta_df for date {date_pick.date()}")
    sel = sel.sort_values(["ccy", "as_of_date"]).drop_duplicates(subset=["ccy"], keep="last")
    return sel.index.to_numpy(), sel


@torch.no_grad()
def run_arbitrage_one_per_ccy(
    model: nn.Module,
    X_tensor_cpu: torch.Tensor,
    meta_df: pd.DataFrame,
    date_pick,
    tau_max: int
):
    idxs, sel = pick_one_curve_per_currency_on_date(meta_df, date_pick)
    xb = X_tensor_cpu[idxs].to(next(model.parameters()).device)

    was_training = model.training
    model.train()

    out = model(xb, do_arb_checks=False)
    P = out[2]
    tau = torch.arange(0, tau_max + 1, device=P.device, dtype=P.dtype)

    arbitrage_report_from_P(P, tau, tag=f"one-per-ccy on {pd.to_datetime(date_pick).date()}")
    print("Currencies included:", list(sel["ccy"].values))

    if not was_training:
        model.eval()


# ============================================================
# 6) Run arbitrage diagnostics
# ============================================================
# A) random batch
rand_idx = torch.randperm(X_tensor.shape[0])[:128]
run_arbitrage_on_batch(model, X_tensor[rand_idx], tau_max=model.tau_max, tag="random 128 curves")

# B) paper date (if available), else first date
paper_date = pd.to_datetime("2016-08-30")
date_pick = paper_date if (meta["as_of_date"] == paper_date).any() else meta["as_of_date"].iloc[0]
run_arbitrage_one_per_ccy(model, X_tensor, meta, date_pick=date_pick, tau_max=model.tau_max)


# ============================================================
# 7) Optional: run your internal checks once (prints inside forward)
# ============================================================
xb_small = X_tensor[:8].to(device)
_ = model(xb_small, do_arb_checks=True)


# ============================================================
# 8) Optional: Andreasen SR diagnostic (your reference)
# ============================================================
# Note: SR_andreasen_reference calls model(S_in.requires_grad_(True)) and uses annual FD in tau.
# This is consistent with your annual tau grid (0..tau_max).
#
# If you didn't change FullModel to clamp in eval, keep model in train() so Xexp clamp remains active.

model.train()
xb1 = X_tensor[:1].to(device)

N1, LN1, SR1, tau1 = SR_andreasen_reference(
    model,
    xb1,
    tau_max=model.tau_max,
    sigma_bar=0.006,
    verbose=False
)

print("\n==============================")
print("SR DIAGNOSTIC (Andreasen ref)")
print("==============================")
print("N:  finite =", bool(torch.isfinite(N1).all().item()),
      "min/max =", float(torch.nan_to_num(N1, nan=0.0).min().cpu()),
      float(torch.nan_to_num(N1, nan=0.0).max().cpu()))
print("LN: finite =", bool(torch.isfinite(LN1).all().item()),
      "min/max =", float(torch.nan_to_num(LN1, nan=0.0).min().cpu()),
      float(torch.nan_to_num(LN1, nan=0.0).max().cpu()))
print("SR: finite =", bool(torch.isfinite(SR1).all().item()),
      "min/max =", float(torch.nan_to_num(SR1, nan=0.0).min().cpu()),
      float(torch.nan_to_num(SR1, nan=0.0).max().cpu()))

print("Max |LN|:", float(torch.nan_to_num(LN1, nan=0.0).abs().max().cpu()))
print("Mean |LN|:", float(torch.nan_to_num(LN1, nan=0.0).abs().mean().cpu()))