import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# =============================================================================
# Paths
# =============================================================================
try:
    CODE_ROOT = os.path.dirname(os.path.abspath(__file__))
except NameError:
    CODE_ROOT = os.getcwd()

PROJECT_ROOT = os.path.abspath(os.path.join(CODE_ROOT, ".."))
THESIS_ROOT  = os.path.abspath(os.path.join(CODE_ROOT, "..", ".."))

for p in [CODE_ROOT, PROJECT_ROOT, THESIS_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from Code.model.full_model_stable import FullModel
from Code.load_swapdata import my_data
from Code.Pricing.load_swapvol_ois import load_swaption_vol_data
from Code.Pricing.pricing import bachelier_price_torch, swap_rate_torch
from Code.Simulation.simulate_model import simulate_to_expiry_differentiable


# =============================================================================
# 0.  SWAPTION PRICE LOSS FOR ONE (expiry, tenor) PAIR
# =============================================================================

def swaption_price_loss_single(
    model,
    z0          : torch.Tensor,    # (1, d)
    expiry      : int,
    tenor       : int,
    sigma_market: float,           # market Bachelier normal vol (absolute)
    n_paths     : int,
    dt          : float,
    device      : torch.device,
    dtype       : torch.dtype,
    freeze_K    : bool = True,
    dt_max      : float = 1/12,    # maximum step size (years)
) -> tuple[torch.Tensor, float, float, dict]:
    """
    Compute the Bachelier price loss for one (expiry, tenor) swaption.

    Loss
    ----
        V_MC     = mean( D_T · A_T · relu(F_T − K) )         [ATM MC price]
        V_market = bachelier_price_torch(F_0, K, σ_mkt, T, A_0)  [fixed target]
        loss     = ( (V_MC − V_market) · 10_000 )²

    Antithetic variates: eps_half and −eps_half are paired to halve MC
    variance and ensure roughly half the paths are always in-the-money.

    Gradient path
    -------------
        H  → L(z_t) → z_T → P_full(z_T) → F_T, A_T → V_MC → loss
        G  → P_full(z_T) → F_T, A_T → V_MC → loss       (if train_G)
        K.N → drift → z_T → …                            (if not freeze_K)

    Returns
    -------
    loss         : scalar tensor with grad
    V_market_bp  : market vol in bp (logging only)
    sigma_mod_bp : MC-implied model vol in bp (logging only)
    stats        : dict with diagnostics (z_max, valid_fraction, etc.)
    """
    # Adaptive step sizing for long expiries
    dt_eff = min(dt, dt_max, expiry / 10.0)  # at least 10 steps
    n_steps = max(1, int(round(expiry / dt_eff)))
    d       = z0.shape[1]
    half    = n_paths // 2

    # ── Antithetic noise ──────────────────────────────────────────────────────
    eps_half = torch.randn(half, n_steps, d, device=device, dtype=dtype)
    eps      = torch.cat([eps_half, -eps_half], dim=0)    # (n_paths, n_steps, d)

    # ── Time-0 F_0, A_0 — fixed market reference, no grad ────────────────────
    with torch.no_grad():
        _, aux0  = model.decode_from_z(z0, tau=None, return_aux=True)
        P_full_0 = aux0["P_full"]                          # (1, tau_max+1)
        F_0_t, A_0_t = swap_rate_torch(P_full_0, tenor=tenor)
        F_0 = float(F_0_t[0].item())
        A_0 = float(A_0_t[0].item())

    K = F_0    # ATM strike (fixed scalar)

    # ── Fixed market price target (pure Python → float, no grad) ─────────────
    V_market_float = bachelier_price_torch(F_0, K, sigma_market, expiry, A_0)
    V_market = torch.tensor(V_market_float, device=device, dtype=dtype)

    # ── Differentiable simulation ─────────────────────────────────────────────
    z_T, D_T = simulate_to_expiry_differentiable(
        model=model, z0=z0, n_steps=n_steps, dt=dt_eff,
        n_paths=n_paths, eps=eps, freeze_K=freeze_K,
    )    # z_T: (n_paths, d) WITH grad;  D_T: (n_paths,) detached

    # ── Check for NaN in z_T immediately ──────────────────────────────────────
    if torch.isnan(z_T).any():
        return torch.tensor(1e6, device=device, dtype=dtype), \
               sigma_market * 10_000, 0.0, {
            "z_max": float("nan"),
            "valid_frac": 0.0,
            "n_steps": n_steps,
            "dt_eff": dt_eff,
            "error": "NaN in z_T"
        }

    # ── Decode bond prices at expiry (grad flows through G and z_T) ──────────
    _, aux_T = model.decode_from_z(z_T, tau=None, return_aux=True)
    P_full_T = aux_T["P_full"]                             # (n_paths, tau_max+1)

    # ── Swap rates at expiry ──────────────────────────────────────────────────
    F_T, A_T = swap_rate_torch(P_full_T, tenor=tenor)      # (n_paths,)

    # ── Monte Carlo ATM payer price ───────────────────────────────────────────
    payoff = A_T * torch.relu(F_T - K)                     # (n_paths,)
    V_MC   = (D_T * payoff).mean()                         # scalar WITH grad

    # ── Loss in scaled price units ────────────────────────────────────────────
    loss = ((V_MC - V_market) * 10_000) ** 2

    # ── Logging: invert MC price → implied Bachelier vol ─────────────────────
    #   V_ATM = A_0 · σ · √T · φ(0)   =>   σ = V_MC / (A_0 · √T · φ(0))
    phi0         = 1.0 / math.sqrt(2.0 * math.pi)
    denom        = A_0 * math.sqrt(expiry) * phi0 + 1e-8
    sigma_mod_bp = float(V_MC.detach()) / max(denom, 1e-12) * 10_000
    V_market_bp  = sigma_market * 10_000

    return loss, V_market_bp, sigma_mod_bp


# =============================================================================
# 1.  HELPER — gradient L2-norm across a parameter group
# =============================================================================

def _grad_norm(params) -> float:
    sq = sum(
        float(p.grad.detach().norm() ** 2)
        for p in params if p.grad is not None
    )
    return math.sqrt(sq)


# =============================================================================
# 2.  SECOND-STAGE CALIBRATION LOOP
# =============================================================================

def calibrate_second_stage(
    checkpoint_path : str,
    ccy             : str   = "EUR",
    n_paths         : int   = 512,
    dt              : float = 1 / 12,
    lr              : float = 1e-4,
    n_epochs        : int   = 500,
    batch_size      : int   = 3,
    seed            : int   = 42,
    device                  = None,
    save_path       : str   = None,
    log_every       : int   = 25,
    # ── extended parameter groups ──────────────────────────────────────────
    train_G         : bool  = True,
    train_K_N       : bool  = True,
    lambda_G        : float = 1.0,
    lambda_K        : float = 0.01,
    # ── temporal train/test split ──────────────────────────────────────────
    split_date      : str   = None,   # ISO date; only rows ≤ split_date are used for training
    train_ratio     : float = None,   # alternative: fraction of dates chronologically
) -> tuple:
    """
    Full second-stage pricing calibration.

    Trainable parameter groups
    --------------------------
        H   (always)      — diffusion network; drives vol surface shape
        G   (if train_G)  — bond-pricing decoder; 10× slower lr + λ_G reg
        K.N (if train_K_N)— mean-reversion level; 10× slower lr + λ_K reg

    K.V is ALWAYS frozen → negative-definiteness / mean-reversion preserved.
    R   is ALWAYS frozen → avoids confounding rate and vol calibration.
    Encoder is ALWAYS frozen.

    Temporal train/test split
    -------------------------
    Pass ``split_date`` (ISO string) to restrict the calibration to dates
    ≤ split_date.  Dates after split_date are excluded from training but
    their z0 values are still precomputed so that a subsequent out-of-sample
    evaluation can reuse the same checkpoint.

    Alternatively pass ``train_ratio`` ∈ (0,1) to keep the first fraction
    of unique dates (sorted chronologically) as training data.

    Regularization
    --------------
        λ_G · Σ_p ‖G_p − G_frozen_p‖²   keeps G close to the stage-1 solution
        λ_K · ‖K.N − K.N_frozen‖²        allows equilibrium shift, not a jump

    Returns
    -------
    model             : calibrated FullModel
    loss_history      : list[float]  — total loss per epoch
    grad_norm_history : dict         — {"H": [...], "G": [...], "K_N": [...]}
    df_vol            : market vol DataFrame (training rows only)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    # ── Load model ────────────────────────────────────────────────────────────
    ckpt   = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state  = ckpt.get("model_state_dict", ckpt)   # backwards compatible

    # Infer latent_dim from the checkpoint so the pipeline is dimension-agnostic
    latent_dim = None
    for key, tensor in state.items():
        if key in ("K.V", "K.V_param") and tensor.ndim == 2:
            latent_dim = tensor.shape[0]
            break
    if latent_dim is None:
        raise ValueError(
            f"Cannot infer latent_dim from {checkpoint_path}. "
            "Check that the checkpoint contains 'K.V' or 'K.V_param'."
        )
    print(f"  [load] inferred latent_dim={latent_dim} from checkpoint")

    model  = FullModel(latent_dim=latent_dim).to(device)
    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys:
        print(f"  [load] dropped old params: {result.unexpected_keys}")
    model = model.double()
    dtype = next(model.parameters()).dtype
    print(f"Loaded : {os.path.basename(checkpoint_path)}")

    # ── Freeze everything, then selectively unfreeze ──────────────────────────
    for param in model.parameters():
        param.requires_grad = False

    for param in model.H.parameters():          # always train H
        param.requires_grad = True

    if train_G:
        for param in model.G.parameters():
            param.requires_grad = True

    freeze_K = True
    if train_K_N and (model.K.N is not None):
        model.K.N.requires_grad = True
        freeze_K = False

    # ── Snapshot frozen values for regularization ─────────────────────────────
    G_frozen   = [p.detach().clone() for p in model.G.parameters()]
    K_N_frozen = model.K.N.detach().clone() if (model.K.N is not None) else None

    # ── Parameter summary ─────────────────────────────────────────────────────
    n_H     = sum(p.numel() for p in model.H.parameters() if p.requires_grad)
    n_G     = sum(p.numel() for p in model.G.parameters() if p.requires_grad) \
              if train_G else 0
    n_KN    = model.K.N.numel() if (train_K_N and model.K.N is not None) else 0
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable — H: {n_H}  G: {n_G}  K.N: {n_KN}  (of {n_total} total)")
    print(f"Regularization — lambda_G={lambda_G}  lambda_K={lambda_K}")

    # ── Optimizer with per-group learning rates ───────────────────────────────
    # G and K.N use 10× lower lr to stay close to the stage-1 solution
    param_groups = [{"params": list(model.H.parameters()), "lr": lr, "name": "H"}]
    if train_G:
        param_groups.append({
            "params": list(model.G.parameters()), "lr": lr * 0.1, "name": "G"
        })
    if train_K_N and (model.K.N is not None):
        param_groups.append({
            "params": [model.K.N], "lr": lr * 0.1, "name": "K.N"
        })
    optimizer = torch.optim.Adam(param_groups)

    # ── Load swap data and precompute z0 ──────────────────────────────────────
    meta, X_tensor, *_ = my_data(ccy_filter=ccy)
    X_tensor = X_tensor.to(device=device, dtype=dtype)

    model.eval()
    with torch.no_grad():
        z0_list = [model.encoder(X_tensor[i:i+1]) for i in range(X_tensor.shape[0])]

    dates   = [pd.Timestamp(meta.iloc[i]["as_of_date"]).normalize()
               for i in range(len(z0_list))]
    z0_dict = dict(zip(dates, z0_list))
    print(f"Precomputed z0 for {len(z0_dict)} dates")

    # ── Load and filter market vols ───────────────────────────────────────────
    df_vol = load_swaption_vol_data(currency=ccy)
    df_vol["as_of_date"] = pd.to_datetime(df_vol["as_of_date"]).dt.normalize()
    df_vol = df_vol[df_vol["as_of_date"].isin(set(z0_dict.keys()))].copy()
    df_vol["market_vol"] = df_vol["vol"] / 10_000.0   # bp → absolute

    if df_vol.empty:
        raise ValueError(
            "No overlapping dates between market vols and swap data. "
            "Check load_swaption_vol_data()."
        )
    print(f"Vol targets : {len(df_vol)} triplets, "
          f"{df_vol['as_of_date'].nunique()} dates")

    # ── Temporal train/test split ─────────────────────────────────────────
    all_dates = sorted(df_vol["as_of_date"].unique())

    if split_date is not None:
        split_ts   = pd.Timestamp(split_date).normalize()
        train_dates = [d for d in all_dates if d <= split_ts]
        test_dates  = [d for d in all_dates if d > split_ts]
        df_vol_train = df_vol[df_vol["as_of_date"].isin(train_dates)].copy()
        print(f"Train/test split at {split_date}:  "
              f"{len(train_dates)} train dates, {len(test_dates)} test dates  "
              f"({len(df_vol_train)} training triplets)")
    elif train_ratio is not None:
        n_train = max(1, int(round(train_ratio * len(all_dates))))
        train_dates  = all_dates[:n_train]
        test_dates   = all_dates[n_train:]
        df_vol_train = df_vol[df_vol["as_of_date"].isin(train_dates)].copy()
        print(f"Train ratio={train_ratio:.2f}: first {n_train} of {len(all_dates)} dates → "
              f"{len(df_vol_train)} training triplets")
    else:
        df_vol_train = df_vol.copy()
        test_dates   = []
        print("No train/test split — calibrating on all dates.")

    if df_vol_train.empty:
        raise ValueError("Training split is empty. Check split_date / train_ratio.")

    # ── Training loop ─────────────────────────────────────────────────────────
    loss_history      = []
    grad_norm_history = {"H": [], "G": [], "K_N": []}

    print(f"\nStarting calibration: {n_epochs} epochs | batch={batch_size} | "
          f"n_paths={n_paths} | lr={lr}")
    print("Extras: antithetic variates | Bachelier price loss | "
          "pathwise D_T discounting")
    print("=" * 70)

    torch.autograd.set_detect_anomaly(True)

    for epoch in range(n_epochs):
        model.train()

        batch = df_vol_train.sample(n=min(batch_size, len(df_vol_train)))
        optimizer.zero_grad()

        total_loss = torch.zeros(1, device=device, dtype=dtype)
        batch_log  = []

        for _, row in batch.iterrows():
            date      = pd.Timestamp(row["as_of_date"]).normalize()
            expiry    = int(row["option_maturity"])
            tenor     = int(row["swap_tenor"])
            sigma_mkt = float(row["market_vol"])

            z0 = z0_dict[date].detach()    # (1, d) — encoder frozen

            loss_ij, V_mkt_bp, sigma_mod_bp, stats = swaption_price_loss_single(
                model=model, z0=z0,
                expiry=expiry, tenor=tenor,
                sigma_market=sigma_mkt,
                n_paths=n_paths, dt=dt,
                device=device, dtype=dtype,
                freeze_K=freeze_K,
            )

            total_loss = total_loss + loss_ij
            batch_log.append({
                "date"   : date.date(),
                "exp"    : expiry,
                "ten"    : tenor,
                "mkt_bp" : round(sigma_mkt * 10_000, 2),
                "mod_bp" : round(sigma_mod_bp, 2),
                "err_bp" : round(sigma_mod_bp - sigma_mkt * 10_000, 2),
                "z_max"  : round(stats["z_max"], 2),
                "valid%" : round(stats["valid_frac"] * 100, 1),
            })

        total_loss = total_loss / len(batch)

        # ── Regularization penalties ──────────────────────────────────────────
        if train_G:
            reg_G = sum(
                ((p - p0) ** 2).sum()
                for p, p0 in zip(model.G.parameters(), G_frozen)
            )
            total_loss = total_loss + lambda_G * reg_G

        if train_K_N and (K_N_frozen is not None):
            reg_K      = ((model.K.N - K_N_frozen) ** 2).sum()
            total_loss = total_loss + lambda_K * reg_K

        # It's crucial to check for NaN/inf loss *before* backpropagation
        if not torch.isfinite(total_loss):
            print(f"WARNING: Non-finite loss detected in epoch {epoch}. Skipping backward pass and optimizer step.")
            print("Problematic batch:")
            print(pd.DataFrame(batch_log).to_string(index=False))
            # Detach and record the non-finite loss for history, then skip to next epoch
            loss_val = total_loss.detach().cpu().item()
            loss_history.append(loss_val)
            grad_norm_history["H"].append(0.0)
            grad_norm_history["G"].append(0.0)
            grad_norm_history["K_N"].append(0.0)
            continue

        total_loss.backward()

        # ── Gradient clipping per group (AFTER backward, BEFORE step) ────────
        # This is a critical step to prevent exploding gradients from destabilizing training.
        nn.utils.clip_grad_norm_(model.H.parameters(), max_norm=1.0)
        if train_G:
            nn.utils.clip_grad_norm_(model.G.parameters(), max_norm=0.5)
        if train_K_N and (model.K.N is not None):
            nn.utils.clip_grad_norm_([model.K.N], max_norm=0.5)

        # ── Record gradient norms (after clipping) ────────────────────────────
        gn_H  = _grad_norm(model.H.parameters())
        gn_G  = _grad_norm(model.G.parameters()) if train_G else 0.0
        gn_KN = _grad_norm([model.K.N]) \
                if (train_K_N and model.K.N is not None) else 0.0
        grad_norm_history["H"].append(gn_H)
        grad_norm_history["G"].append(gn_G)
        grad_norm_history["K_N"].append(gn_KN)

        optimizer.step()

        loss_val = float(total_loss.detach())
        loss_history.append(loss_val)

        if epoch % log_every == 0 or epoch == n_epochs - 1:
            print(f"\nEpoch {epoch:4d}  loss={loss_val:10.4f}  "
                  f"||gH||={gn_H:.4f}  "
                  f"||gG||={gn_G:.4f}  "
                  f"||gKN||={gn_KN:.4f}")
            print(pd.DataFrame(batch_log).to_string(index=False))

    print("\n" + "=" * 70)
    print(f"Final loss : {loss_history[-1]:.4f}")

    # ── Save calibrated checkpoint ────────────────────────────────────────────
    if save_path is None:
        base      = os.path.dirname(checkpoint_path)
        save_path = os.path.join(base, "checkpoint_stage2.pt")

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    torch.save(
        {
            "model_state_dict" : model.state_dict(),
            "model_config"     : {"latent_dim": latent_dim},
            "latent_dim"       : latent_dim,
            "stage"            : 2,
            "n_epochs"         : n_epochs,
            "lr"               : lr,
            "batch_size"       : batch_size,
            "n_paths"          : n_paths,
            "dt"               : dt,
            "ccy"              : ccy,
            "train_G"          : train_G,
            "train_K_N"        : train_K_N,
            "lambda_G"         : lambda_G,
            "lambda_K"         : lambda_K,
            "split_date"       : split_date,
            "loss_history"     : loss_history,
            "grad_norm_history": grad_norm_history,
            "stage1_ckpt"      : checkpoint_path,
        },
        save_path,
    )
    print(f"Saved calibrated checkpoint → {save_path}")

    return model, loss_history, grad_norm_history, df_vol_train


# =============================================================================
# 3.  SCRIPT ENTRY POINT
#     Run directly:  python Code/Pricing/Pricing_Training.py
#     Edit the settings block below before running.
# =============================================================================

if __name__ == "__main__":
    import time as _time

    # =========================================================================
    # SETTINGS  — edit here
    # =========================================================================

    LATENT_DIM = 4       # match the stage-1 checkpoint you are using
    STAGE1_EP  = 5000    # epoch tag of the stage-1 checkpoint
    EPOCHS     = 500     # stage-2 calibration epochs
    CCY        = "EUR"
    N_PATHS    = 1024    # MC paths during training (antithetic → effective 2×)
    DT         = 1 / 12
    LR         = 1e-4
    BATCH_SIZE = 4
    LOG_EVERY  = 50
    TRAIN_G    = True
    TRAIN_K_N  = True
    LAMBDA_G   = 1.0
    LAMBDA_K   = 0.01
    SEED       = 42

    # Temporal train/test split (ISO date string or None for full sample)
    SPLIT_DATE = "2018-12-31"

    # =========================================================================
    # Derived paths  — nothing to edit below this line
    # =========================================================================

    _DIM_TAG = f"dim{LATENT_DIM}"

    STAGE1_CKPT = os.path.join(
        THESIS_ROOT, "Figures", "TrainingResults",
        f"{_DIM_TAG}_stable", f"ep{STAGE1_EP}",
        f"checkpoint_{_DIM_TAG}_ep{STAGE1_EP}.pt"
    )

    OUT_DIR   = os.path.join(THESIS_ROOT, "Figures", "Pricing", "stage2_checkpoints")
    CKPT_NAME = f"checkpoint_stage2_{_DIM_TAG}_ep{EPOCHS}.pt"
    CKPT_PATH = os.path.join(OUT_DIR, CKPT_NAME)
    CSV_NAME  = f"training_log_stage2_{_DIM_TAG}_ep{EPOCHS}.csv"
    CSV_PATH  = os.path.join(OUT_DIR, CSV_NAME)

    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 70)
    print("STAGE-2 SWAPTION-VOL CALIBRATION")
    print("=" * 70)
    print(f"  Stage-1 checkpoint : {STAGE1_CKPT}")
    print(f"  Output dir         : {OUT_DIR}")
    print(f"  Checkpoint         : {CKPT_NAME}")
    print(f"  Latent dim / CCY   : {LATENT_DIM} / {CCY}")
    print(f"  Epochs / split     : {EPOCHS} / {SPLIT_DATE}")
    print(f"  n_paths / dt       : {N_PATHS} / {DT:.4f}")
    print(f"  lr / batch         : {LR} / {BATCH_SIZE}")
    print(f"  train_G={TRAIN_G}  train_K_N={TRAIN_K_N}")
    print(f"  lambda_G={LAMBDA_G}  lambda_K={LAMBDA_K}")
    print("=" * 70)

    # ── Calibration ───────────────────────────────────────────────────────────
    t0 = _time.perf_counter()

    _, loss_history, grad_norm_history, _ = calibrate_second_stage(
        checkpoint_path = STAGE1_CKPT,
        ccy             = CCY,
        n_paths         = N_PATHS,
        dt              = DT,
        lr              = LR,
        n_epochs        = EPOCHS,
        batch_size      = BATCH_SIZE,
        seed            = SEED,
        save_path       = CKPT_PATH,
        log_every       = LOG_EVERY,
        train_G         = TRAIN_G,
        train_K_N       = TRAIN_K_N,
        lambda_G        = LAMBDA_G,
        lambda_K        = LAMBDA_K,
        split_date      = SPLIT_DATE,
    )

    elapsed = _time.perf_counter() - t0
    print(f"\nCalibration finished in {elapsed / 60:.1f} min.")

    # ── CSV training log ──────────────────────────────────────────────────────
    n = len(loss_history)
    df_log = pd.DataFrame({
        "epoch"        : list(range(n)),
        "loss"         : loss_history,
        "grad_norm_H"  : grad_norm_history.get("H",   [float("nan")] * n),
        "grad_norm_G"  : grad_norm_history.get("G",   [float("nan")] * n),
        "grad_norm_K_N": grad_norm_history.get("K_N", [float("nan")] * n),
    })
    df_log.to_csv(CSV_PATH, index=False)
    print(f"  Training log → {CSV_PATH}")

    # ── Diagnostic plots ──────────────────────────────────────────────────────
    colours = {"H": "steelblue", "G": "darkorange", "K_N": "green"}

    # Loss curve
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    ax.plot(loss_history, linewidth=1.2, color="steelblue")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (scaled price²)")
    ax.set_title(f"Stage-2 Calibration Loss  ({_DIM_TAG}, {EPOCHS} epochs)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    loss_fig = os.path.join(OUT_DIR, f"calibration_loss_stage2_{_DIM_TAG}_ep{EPOCHS}.png")
    fig.savefig(loss_fig, dpi=150)
    plt.close(fig)
    print(f"  Loss plot  → {loss_fig}")

    # Gradient norms
    has_any_grad = any(
        len(v) > 0 and any(x > 0 for x in v)
        for v in grad_norm_history.values()
    )
    if has_any_grad:
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
        for group, values in grad_norm_history.items():
            if values and any(x > 0 for x in values):
                ax.plot(values, label=f"‖g_{group}‖", linewidth=1.2,
                        color=colours.get(group))
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Gradient norm (post-clip)")
        ax.set_title(f"Stage-2 Gradient Norms  ({_DIM_TAG}, {EPOCHS} epochs)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        grad_fig = os.path.join(OUT_DIR, f"gradient_norms_stage2_{_DIM_TAG}_ep{EPOCHS}.png")
        fig.savefig(grad_fig, dpi=150)
        plt.close(fig)
        print(f"  Grad plot  → {grad_fig}")

    # ── Final summary ─────────────────────────────────────────────────────────
    best_ep = int(np.argmin(loss_history))
    print(f"\n  Initial loss : {loss_history[0]:.4f}")
    print(f"  Final   loss : {loss_history[-1]:.4f}")
    print(f"  Best    loss : {min(loss_history):.4f}  (epoch {best_ep})")
    print(f"\n  Checkpoint → {CKPT_PATH}")
    print(f"  CSV log    → {CSV_PATH}")
    print("=" * 70)
    print("DONE")
    print("=" * 70)

