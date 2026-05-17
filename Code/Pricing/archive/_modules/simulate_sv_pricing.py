"""
Stochastic-vol Euler-Maruyama simulation for swaption pricing.

Extends `simulate_to_expiry_differentiable` (no SV) by adding a CIR variance
state v_t that scales the diffusion of z:

    dz_t = K*(z_t) dt + sqrt(v_t / theta) * diag(sigma_eff) * L(z_t) * dW^z_t
    dv_t = kappa*(theta - v_t) dt + sigma_v * sqrt(v_t) * dW^v_t

The factor sqrt(v_t/theta) normalises the SV scaling so that at the CIR
stationary mean (v_t = theta), the diffusion magnitude equals the static
sigma_eff baseline — i.e. when SV is "off" (v=theta), this reduces exactly
to the State-Cond / ETV MPR simulation.

Per-path v_t is tracked through the trajectory.  Discount factor is
trapezoidally accumulated using r(z) only (the SV does not affect r).
"""

import math
import torch
import torch.nn as nn
from typing import Optional

from Code.model.sigma_matrix import L_from_sigmas_rhos


def simulate_sv_to_expiry_differentiable(
    model,                            # FullModelPrice (encoder/decoder/K/H/R, frozen during pricing)
    cir,                              # CIRVolPricing module
    z0:           torch.Tensor,       # (1, d) initial latent state
    n_steps:      int,
    dt:           float,
    n_paths:      int,
    eps_z:        torch.Tensor,       # (n_paths//2, n_steps, d)  pre-drawn N(0,1) for z (antithetic)
    eps_v:        torch.Tensor,       # (n_paths//2, n_steps)     pre-drawn N(0,1) for v (antithetic-paired)
    k_override                = None, # callable z -> drift K*(z); falls back to model.K
    sigma_scale:  Optional[torch.Tensor] = None,  # (d,) static σ_eff (z_0-conditioned), grad-carrying
    v0:           Optional[torch.Tensor] = None,  # scalar tensor; if None uses cir.initial_v(z0)
    antithetic:   bool = True,
    freeze_H:     bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns
    -------
    z_T : (n_paths, d)   terminal latent state WITH gradients
    v_T : (n_paths,)     terminal variance WITH gradients (through cir params)
    D_T : (n_paths,)     pathwise discount factor (no gradient)

    Notes
    -----
    • antithetic=True: eps_z is mirrored to -eps_z to give n_paths total.
        eps_v is paired so that both antithetic z-paths share the same v-path.
        This means eps_v has shape (n_paths//2, n_steps) and is duplicated.
    • sigma_scale is the static (z_0-dependent) per-factor scaling.  v_t modulates
        this on top:  effective shock = sqrt(v_t/theta) * sigma_scale * L * dW.
    """
    half     = n_paths // 2
    sqrt_dt  = math.sqrt(dt)
    device   = z0.device
    dtype    = z0.dtype
    theta    = cir.theta

    # ── antithetic z-noise ─────────────────────────────────────────────────
    if antithetic:
        eps_z_full = torch.cat([eps_z, -eps_z], dim=0)   # (n_paths, n_steps, d)
        eps_v_full = torch.cat([eps_v,  eps_v], dim=0)   # (n_paths, n_steps)  same v-path
    else:
        eps_z_full = eps_z
        eps_v_full = eps_v

    # ── initial state ──────────────────────────────────────────────────────
    z = z0.expand(n_paths, -1).clone()                   # (n_paths, d)
    if v0 is None:
        v0 = cir.initial_v(z0)                            # scalar
    v = v0.expand(n_paths).clone()                        # (n_paths,)

    # ── discount-factor accumulator ────────────────────────────────────────
    with torch.no_grad():
        r_prev = model.R(z).squeeze(-1)                   # (n_paths,)
    log_D = torch.zeros(n_paths, device=device, dtype=dtype)

    # ── Euler loop ─────────────────────────────────────────────────────────
    for t in range(n_steps):
        # 1) Step v (CIR).  Antithetic pairs share the same v-path → use half size.
        if antithetic:
            v_half = v[:half]
            v_half = cir.step(v_half, eps_v_full[:half, t], dt=dt)
            v      = torch.cat([v_half, v_half], dim=0)
        else:
            v = cir.step(v, eps_v_full[:, t], dt=dt)

        # 2) Step z (z dynamics with sv-scaled shock)
        if freeze_H:
            with torch.no_grad():
                sigmas, rhos = model.H(z.detach())
                L = L_from_sigmas_rhos(sigmas, rhos, validate=False)
        else:
            sigmas, rhos = model.H(z)
            L = L_from_sigmas_rhos(sigmas, rhos, validate=False)

        dW = eps_z_full[:, t, :] * sqrt_dt                # (n_paths, d)

        # SV scaling: sqrt(v_t/theta) per path × static σ_eff per factor
        # → per-path, per-factor scale of shape (n_paths, d)
        sv_factor = (v / theta).sqrt().unsqueeze(-1)       # (n_paths, 1)
        if sigma_scale is not None:
            # sigma_scale: (d,) → broadcast to (1, d) then × (n_paths, 1) → (n_paths, d)
            row_scale = sigma_scale.unsqueeze(0) * sv_factor  # (n_paths, d)
        else:
            row_scale = sv_factor                              # (n_paths, 1) broadcasts

        # shock = row_scale * (L @ dW): row_scale scales each factor (row) of L
        L_dW   = torch.bmm(L, dW.unsqueeze(-1)).squeeze(-1)   # (n_paths, d)
        shock  = row_scale * L_dW                              # (n_paths, d)

        # drift
        drift = (k_override(z) if k_override is not None else model.K(z)) * dt

        z = z + drift + shock

        # 3) Discount accumulator (R is frozen)
        with torch.no_grad():
            r_next = model.R(z.detach()).squeeze(-1)
            log_D  = log_D - 0.5 * (r_prev + r_next) * dt
            r_prev = r_next

    D_T = log_D.clamp(min=-30.0, max=30.0).exp().detach()
    return z, v, D_T
