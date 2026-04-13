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

from Code.model.full_model import FullModel
from Code.model.sigma_matrix import L_from_sigmas_rhos
from Code.load_swapdata import my_data
from Code.Pricing.load_swapvol_ois import load_swaption_vol_data
from Code.Pricing.pricing import bachelier_price_torch, swap_rate_torch
from Code.Pricing.simulate_model import simulate_to_expiry_differentiable


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
) -> tuple[torch.Tensor, float, float]:
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
    """
    n_steps = max(1, int(round(expiry / dt)))
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
        model=model, z0=z0, n_steps=n_steps, dt=dt,
        n_paths=n_paths, eps=eps, freeze_K=freeze_K,
    )    # z_T: (n_paths, d) WITH grad;  D_T: (n_paths,) detached

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
    denom        = A_0 * math.sqrt(expiry) * phi0
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

    Regularization
    --------------
        λ_G · Σ_p ‖G_p − G_frozen_p‖²   keeps G close to the stage-1 solution
        λ_K · ‖K.N − K.N_frozen‖²        allows equilibrium shift, not a jump

    Improvements over the original calibrate_H
    -------------------------------------------
        ✓ Proper Bachelier price loss (not a vol proxy)
        ✓ Antithetic variates for halved MC variance
        ✓ Pathwise money-market discounting (D_T via trapezoid on r)
        ✓ Optional G and K.N fine-tuning with regularisation snapshots
        ✓ Per-group gradient norm logging every log_every epochs

    Returns
    -------
    model             : calibrated FullModel
    loss_history      : list[float]  — total loss per epoch
    grad_norm_history : dict         — {"H": [...], "G": [...], "K_N": [...]}
    df_vol            : market vol DataFrame
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    # ── Load model ────────────────────────────────────────────────────────────
    model  = FullModel(latent_dim=2).to(device)
    state  = torch.load(checkpoint_path, map_location=device, weights_only=False)
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

    # ── Training loop ─────────────────────────────────────────────────────────
    loss_history      = []
    grad_norm_history = {"H": [], "G": [], "K_N": []}

    print(f"\nStarting calibration: {n_epochs} epochs | batch={batch_size} | "
          f"n_paths={n_paths} | lr={lr}")
    print("Extras: antithetic variates | Bachelier price loss | "
          "pathwise D_T discounting")
    print("=" * 70)

    for epoch in range(n_epochs):
        model.train()

        batch = df_vol.sample(n=min(batch_size, len(df_vol)))
        optimizer.zero_grad()

        total_loss = torch.zeros(1, device=device, dtype=dtype)
        batch_log  = []

        for _, row in batch.iterrows():
            date      = pd.Timestamp(row["as_of_date"]).normalize()
            expiry    = int(row["option_maturity"])
            tenor     = int(row["swap_tenor"])
            sigma_mkt = float(row["market_vol"])

            z0 = z0_dict[date].detach()    # (1, d) — encoder frozen

            loss_ij, V_mkt_bp, sigma_mod_bp = swaption_price_loss_single(
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

        total_loss.backward()

        # ── Gradient clipping per group ───────────────────────────────────────
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

    torch.save(model.state_dict(), save_path)
    print(f"Saved calibrated model → {save_path}")

    return model, loss_history, grad_norm_history, df_vol


# =============================================================================
# 3.  BACKWARD-COMPAT WRAPPER
# =============================================================================

def calibrate_H(*args, **kwargs):
    """
    Thin backward-compatibility wrapper around calibrate_second_stage.
    Runs with train_G=False and train_K_N=False (H only, original behaviour).
    """
    kwargs.setdefault("train_G",   False)
    kwargs.setdefault("train_K_N", False)
    model, loss_history, _, df_vol = calibrate_second_stage(*args, **kwargs)
    return model, loss_history, df_vol


# =============================================================================
# 4.  DT PROFILER
# =============================================================================

def profile_dt(
    checkpoint_path : str,
    dt_values       : tuple = (1 / 12, 1 / 4, 1.0),
    n_epochs        : int   = 100,
    save_dir        : str   = None,
    **kwargs,
) -> dict:
    """
    Run calibrate_second_stage for each dt and compare convergence.

    Addresses Further Consideration 2: path-length vs. gradient-variance
    trade-off.  Produces a side-by-side plot of loss curves and H gradient
    norm curves across dt values.

    Parameters
    ----------
    dt_values : iterable of floats, e.g. (1/12, 1/4, 1.0)
    n_epochs  : epochs per run (keep small — this is diagnostic, not full training)
    save_dir  : directory for checkpoints + plot (defaults to checkpoint dir)
    **kwargs  : forwarded to calibrate_second_stage
                (e.g. ccy, n_paths, lr, batch_size, train_G, train_K_N, …)

    Returns
    -------
    dict : { dt_value : {"loss": list[float], "grads": dict} }
    """
    out_dir = save_dir or os.path.dirname(checkpoint_path)
    os.makedirs(out_dir, exist_ok=True)

    # Remove save_path from kwargs so we can set it per-dt below
    kwargs.pop("save_path", None)

    results = {}

    for dt in dt_values:
        label = f"dt{dt:.4f}"
        print(f"\n{'='*70}")
        print(f"PROFILING  dt={dt:.4f}  "
              f"(~{int(round(expiry / dt)) if (expiry := kwargs.get('expiry', 1)) else int(round(1/dt))} "
              f"steps to 1Y expiry)")
        print(f"{'='*70}")

        sp = os.path.join(out_dir, f"checkpoint_stage2_{label}.pt")
        _, loss_history, grad_history, _ = calibrate_second_stage(
            checkpoint_path=checkpoint_path,
            dt=dt,
            n_epochs=n_epochs,
            save_path=sp,
            log_every=max(1, n_epochs // 5),
            **kwargs,
        )
        results[dt] = {"loss": loss_history, "grads": grad_history}

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for dt, res in results.items():
        steps_per_yr = int(round(1 / dt))
        ax.plot(res["loss"], label=f"dt={dt:.3f}  ({steps_per_yr} steps/yr)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (scaled price²)")
    ax.set_title("Stage-2 Calibration Loss vs. dt")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for dt, res in results.items():
        ax.plot(res["grads"]["H"], label=f"dt={dt:.3f}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("||∇H|| (post-clip)")
    ax.set_title("H Gradient Norm vs. dt")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(out_dir, "dt_profile.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"\ndt profile plot saved → {fig_path}")

    return results


# =============================================================================
# 5.  ENTRY POINT
# =============================================================================

def main():
    CHECKPOINT_PATH = (
        r"C:\Users\Bruger\PycharmProjects\MasterThesis"
        r"\Figures\TrainingResults\dim2_stable\ep3500\checkpoint_dim2_ep3500.pt"
    )

    model, loss_history, grad_history, df_vol = calibrate_second_stage(
        checkpoint_path = CHECKPOINT_PATH,
        ccy             = "EUR",
        n_paths         = 512,
        dt              = 1 / 12,
        lr              = 1e-4,
        n_epochs        = 500,
        batch_size      = 3,
        log_every       = 25,
        train_G         = True,
        train_K_N       = True,
        lambda_G        = 1.0,
        lambda_K        = 0.01,
    )

    # ── Diagnostics plot ──────────────────────────────────────────────────────
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].plot(loss_history)
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss (scaled price²)")
        axes[0].set_title("Stage-2 Calibration Loss")
        axes[0].grid(True, alpha=0.3)

        for group, values in grad_history.items():
            if any(v > 0 for v in values):
                axes[1].plot(values, label=f"||g_{group}||")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Gradient norm (post-clip)")
        axes[1].set_title("Gradient Norms per Parameter Group")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        out_dir  = os.path.dirname(CHECKPOINT_PATH)
        fig_path = os.path.join(out_dir, "calibration_stage2.png")
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"Plot saved → {fig_path}")
    except Exception as e:
        print(f"Could not save plot: {e}")


if __name__ == "__main__":
    main()








