import math
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# =============================================================================
# Paths (adjust to your project structure)
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


# =============================================================================
# 1.  DIFFERENTIABLE SIMULATION TO EXPIRY
# =============================================================================

def simulate_to_expiry_differentiable(
    model,
    z0    : torch.Tensor,   # (1, d)  — initial latent state, detached
    n_steps: int,           # Euler steps to reach expiry T
    dt    : float,
    n_paths: int,
    eps   : torch.Tensor,   # (n_paths, n_steps, d) — PRE-DRAWN, fixed noise
) -> torch.Tensor:          # (n_paths, d)  — z at expiry, WITH grad w.r.t. H
    """
    Differentiable Euler-Maruyama.

    The noise eps is drawn BEFORE this function and passed in as a fixed
    tensor (no grad). This is the reparameterization trick of Mohamed et al.:
    the randomness is decoupled from the parameters, so autograd flows cleanly
    through L(z_t; H) at every step.

    K is called with torch.no_grad() because we freeze it — this avoids
    building an unnecessary computation graph for K's parameters.
    """
    sqrt_dt = math.sqrt(dt)

    z = z0.expand(n_paths, -1).clone()   # (n_paths, d)

    for t in range(n_steps):
        # --- Volatility: gradients flow through H --------------------------
        sigmas, rhos = model.H(z)
        L = L_from_sigmas_rhos(sigmas, rhos)         # (n_paths, d, d)

        # --- Drift: detach since K is frozen --------------------------------
        with torch.no_grad():
            mu = model.K(z)                          # (n_paths, d)

        # --- Euler step with fixed noise ------------------------------------
        dW    = eps[:, t, :] * sqrt_dt              # (n_paths, d)
        shock = torch.bmm(L, dW.unsqueeze(-1)).squeeze(-1)  # (n_paths, d)

        z = z + mu.detach() * dt + shock            # grad flows through shock
        # Note: z itself now carries grad w.r.t. H, which feeds into
        # model.H(z) at the NEXT step — the gradient propagates in time.

    return z   # (n_paths, d)


# =============================================================================
# 2.  DIFFERENTIABLE SWAP RATE FROM BOND PRICE CURVE
# =============================================================================

def swap_rate_torch(
    P_full : torch.Tensor,   # (n_paths, tau_max+1)  — P(z_T, 0), P(z_T,1), ...
    tenor  : int,
    accrual: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute swap rate F and annuity A from the discount curve.

    P_full[:, j] = P(z_T, j) for j = 0, 1, ..., tau_max (integer year grid).
    Payment dates are accrual, 2*accrual, ..., tenor*accrual years from expiry.

    Returns
    -------
    F : (n_paths,)   swap rates, WITH grad w.r.t. H via P_full
    A : (n_paths,)   annuities
    """
    pay_idx     = [int(round(accrual * j)) for j in range(1, tenor + 1)]
    payment_dfs = P_full[:, pay_idx]                 # (n_paths, tenor)
    A           = accrual * payment_dfs.sum(dim=1)   # (n_paths,)
    P_end       = payment_dfs[:, -1]                 # (n_paths,)
    F           = (1.0 - P_end) / A.clamp(min=1e-8) # (n_paths,)
    return F, A


# =============================================================================
# 3.  SWAPTION VOL LOSS FOR ONE (expiry, tenor) PAIR
# =============================================================================

def swaption_vol_loss_single(
    model        ,
    z0           : torch.Tensor,   # (1, d)
    expiry       : int,
    tenor        : int,
    sigma_market : float,          # market Bachelier normal vol (absolute)
    n_paths      : int,
    dt           : float,
    device       : torch.device,
    dtype        : torch.dtype,
) -> tuple[torch.Tensor, float]:
    """
    Compute squared vol error for one swaption via the pathwise estimator.

    sigma_model  = std(F_T) / sqrt(T)      [Bachelier vol proxy]
    loss         = (sigma_model - sigma_market)^2  in bp^2

    Gradient path:
        H -> L(z_t) -> z_T -> P(z_T,tau) -> F_T -> std(F_T) -> loss
    """
    n_steps = max(1, int(round(expiry / dt)))

    # Draw noise outside the graph — this is the reparameterization trick.
    # eps is fixed; only L(z;H) carries the parameter dependence.
    eps = torch.randn(n_paths, n_steps, z0.shape[1],
                      device=device, dtype=dtype)    # no grad

    # --- Differentiable simulation to expiry ---------------------------------
    z_T = simulate_to_expiry_differentiable(
        model=model, z0=z0, n_steps=n_steps,
        dt=dt, n_paths=n_paths, eps=eps,
    )   # (n_paths, d) — grad w.r.t. H

    # --- Decode bond prices (NO torch.no_grad — gradient flows through ODE) -
    # model.decode_from_z is differentiable: it was used in original training.
    _, aux = model.decode_from_z(z_T, tau=None, do_arb_checks=False, return_aux=True)
    P_full = aux["P_full"]   # (n_paths, tau_max+1) — grad w.r.t. H via z_T

    # --- Swap rates ----------------------------------------------------------
    F_T, _ = swap_rate_torch(P_full, tenor=tenor)   # (n_paths,)

    # --- Bachelier vol proxy -------------------------------------------------
    sigma_model = F_T.std() / math.sqrt(expiry)     # scalar tensor

    # --- Loss in bp^2 (scaling prevents vanishingly small gradients) ---------
    loss = ((sigma_model - sigma_market) * 10_000) ** 2

    return loss, float(sigma_model.detach()) * 10_000   # (loss, model_vol_bp)


# =============================================================================
# 4.  MAIN CALIBRATION LOOP
# =============================================================================

def calibrate_H(
    checkpoint_path : str,
    ccy             : str   = "EUR",
    n_paths         : int   = 512,
    dt              : float = 1 / 12,
    lr              : float = 1e-4,
    n_epochs        : int   = 500,
    batch_size      : int   = 3,      # (date, expiry, tenor) triplets per step
    seed            : int   = 42,
    device                  = None,
    save_path       : str   = None,
    log_every       : int   = 25,
) -> tuple:
    """
    Fine-tune network H to match market swaption normal vols.

    Parameters
    ----------
    checkpoint_path : path to the ep3500 checkpoint
    ccy             : currency filter for swap and vol data
    n_paths         : MC paths per gradient step (512-1000 recommended)
    dt              : simulation time step (1/12 = monthly)
    lr              : Adam learning rate for H
    n_epochs        : calibration epochs
    batch_size      : number of (date, expiry, tenor) triplets per step
    seed            : random seed for reproducibility
    save_path       : where to save calibrated checkpoint (None = auto)
    log_every       : print diagnostics every N epochs

    Returns
    -------
    model        : calibrated FullModel
    loss_history : list of loss values per epoch
    df_vol       : market vol DataFrame used for calibration
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    # -------------------------------------------------------------------------
    # Load model from checkpoint
    # -------------------------------------------------------------------------
    model = FullModel(latent_dim=2).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys:
        print(f"  [load] dropped old params: {result.unexpected_keys}")
    model = model.double()
    dtype = next(model.parameters()).dtype
    print(f"Loaded checkpoint: {os.path.basename(checkpoint_path)}")

    # -------------------------------------------------------------------------
    # Freeze everything except H
    # -------------------------------------------------------------------------
    for param in model.parameters():
        param.requires_grad = False
    for param in model.H.parameters():
        param.requires_grad = True

    n_trainable = sum(p.numel() for p in model.H.parameters())
    n_total     = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_trainable} trainable (H) / {n_total} total")

    # -------------------------------------------------------------------------
    # Load swap data and precompute z0 for every date
    # -------------------------------------------------------------------------
    meta, X_tensor, *_ = my_data(ccy_filter=ccy)
    X_tensor = X_tensor.to(device=device, dtype=dtype)

    model.eval()
    with torch.no_grad():
        z0_list = [model.encoder(X_tensor[i:i+1]) for i in range(X_tensor.shape[0])]

    dates = [pd.Timestamp(meta.iloc[i]["as_of_date"]).normalize()
             for i in range(len(z0_list))]
    z0_dict = dict(zip(dates, z0_list))
    print(f"Precomputed z0 for {len(z0_dict)} dates")

    # -------------------------------------------------------------------------
    # Load and filter market vols
    # -------------------------------------------------------------------------
    df_vol = load_swaption_vol_data(currency=ccy)
    df_vol["as_of_date"] = pd.to_datetime(df_vol["as_of_date"]).dt.normalize()
    df_vol = df_vol[df_vol["as_of_date"].isin(set(z0_dict.keys()))].copy()
    df_vol["market_vol"] = df_vol["vol"] / 10_000.0   # bp -> absolute

    print(f"Vol targets: {len(df_vol)} triplets across "
          f"{df_vol['as_of_date'].nunique()} dates")

    if df_vol.empty:
        raise ValueError(
            "No overlapping dates between market vol data and swap data. "
            "Check that load_swaption_vol_data() returns dates matching your swap data."
        )

    # -------------------------------------------------------------------------
    # Optimizer (H only)
    # -------------------------------------------------------------------------
    optimizer = torch.optim.Adam(model.H.parameters(), lr=lr)

    # -------------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------------
    loss_history = []

    print(f"\nStarting calibration: {n_epochs} epochs, "
          f"batch_size={batch_size}, n_paths={n_paths}, lr={lr}")
    print("=" * 65)

    for epoch in range(n_epochs):
        model.train()

        # Sample a random mini-batch of (date, expiry, tenor) triplets
        batch = df_vol.sample(n=min(batch_size, len(df_vol)))

        optimizer.zero_grad()

        total_loss   = torch.zeros(1, device=device, dtype=dtype)
        batch_log    = []

        for _, row in batch.iterrows():
            date      = pd.Timestamp(row["as_of_date"]).normalize()
            expiry    = int(row["option_maturity"])
            tenor     = int(row["swap_tenor"])
            sigma_mkt = float(row["market_vol"])

            # Detach z0 from encoder graph — encoder is frozen
            z0 = z0_dict[date].detach()   # (1, 2)

            loss_ij, sigma_mod_bp = swaption_vol_loss_single(
                model=model,
                z0=z0,
                expiry=expiry,
                tenor=tenor,
                sigma_market=sigma_mkt,
                n_paths=n_paths,
                dt=dt,
                device=device,
                dtype=dtype,
            )

            total_loss = total_loss + loss_ij

            batch_log.append({
                "date"         : date.date(),
                "expiry"       : expiry,
                "tenor"        : tenor,
                "mkt_vol_bp"   : round(sigma_mkt * 10_000, 2),
                "mod_vol_bp"   : round(sigma_mod_bp, 2),
                "error_bp"     : round(sigma_mod_bp - sigma_mkt * 10_000, 2),
            })

        total_loss = total_loss / len(batch)
        total_loss.backward()

        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(model.H.parameters(), max_norm=1.0)

        optimizer.step()

        loss_val = float(total_loss.detach())
        loss_history.append(loss_val)

        if epoch % log_every == 0 or epoch == n_epochs - 1:
            print(f"\nEpoch {epoch:4d}  loss = {loss_val:8.2f} bp²")
            df_log = pd.DataFrame(batch_log)
            print(df_log.to_string(index=False))

    print("\n" + "=" * 65)
    print(f"Final loss: {loss_history[-1]:.2f} bp²")

    # -------------------------------------------------------------------------
    # Save calibrated checkpoint
    # -------------------------------------------------------------------------
    if save_path is None:
        base      = os.path.dirname(checkpoint_path)
        save_path = os.path.join(base, "checkpoint_H_calibrated.pt")

    torch.save(model.state_dict(), save_path)
    print(f"Saved calibrated model → {save_path}")

    return model, loss_history, df_vol


# =============================================================================
# 5.  ENTRY POINT
# =============================================================================

def main():
    CHECKPOINT_PATH = (
        r"C:\Users\Bruger\PycharmProjects\MasterThesis"
        r"\Figures\TrainingResults\dim2_stable\ep3500\checkpoint_dim2_ep3500.pt"
    )

    model, loss_history, df_vol = calibrate_H(
        checkpoint_path = CHECKPOINT_PATH,
        ccy             = "EUR",
        n_paths         = 512,
        dt              = 1 / 12,
        lr              = 1e-4,
        n_epochs        = 500,
        batch_size      = 3,
        log_every       = 25,
    )

    # Quick loss plot (optional)
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 4))
        plt.plot(loss_history)
        plt.xlabel("Epoch")
        plt.ylabel("Loss (bp²)")
        plt.title("Swaption calibration loss — H network")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out_dir = os.path.dirname(CHECKPOINT_PATH)
        plt.savefig(os.path.join(out_dir, "calibration_loss.png"), dpi=150)
        print(f"Loss plot saved to {out_dir}")
    except Exception as e:
        print(f"Could not save plot: {e}")


if __name__ == "__main__":
    main()