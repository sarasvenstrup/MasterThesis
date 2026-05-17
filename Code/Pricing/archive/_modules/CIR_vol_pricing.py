"""
CIR variance process module for stochastic-vol pricing.

The CIR process governs a latent variance state v_t that scales the
diffusion of the latent rate factors z_t during pricing simulations:

    dv_t = kappa (theta - v_t) dt + sigma_v sqrt(v_t) dW_v
    v_0  = theta * exp( alpha_v * tanh( w_v . z_0 ) )    (state-conditioned)

Trainable parameters (log-parameterised to keep kappa, theta, sigma_v positive):
    log_kappa   scalar  — mean-reversion speed
    log_theta   scalar  — long-run variance
    log_sigma_v scalar  — vol-of-vol
    w_v         (d,)    — projection for state-conditioned v_0
    alpha_v     scalar  — scale of v_0 state conditioning

At init (alpha_v = 0): v_0 = theta for every date (matches no-SV "v_ref = theta" baseline).
w_v is initialised small-random so ∂L/∂alpha_v = tanh(w_v . z_0) is nonzero
(breaks the (w_v=0, alpha_v=0) saddle point we hit in the State-Cond Vol MPR).

Feller condition for strict positivity:
    2 * kappa * theta > sigma_v^2
We do NOT enforce this hard; instead we clamp v_t > 1e-10 in simulation.
"""

import math
import torch
import torch.nn as nn


class CIRVolPricing(nn.Module):
    """
    CIR stochastic variance with state-conditioned initial level.

    Use cases
    ---------
    - Sample v_0 from current latent state z_0
    - Take one Euler step of the CIR process
    - Expose kappa, theta, sigma_v for the ODE / regularisers
    """

    def __init__(
        self,
        latent_dim: int,
        log_kappa_init:   float = 0.0,    # kappa   = 1.0   (1-year mean-reversion)
        log_theta_init:   float = -4.02,  # theta   ≈ 0.018 (sqrt(theta) ≈ 0.134)
        log_sigma_v_init: float = -3.0,   # sigma_v ≈ 0.05
        w_v_init_scale:   float = 0.01,   # small random — breaks saddle at alpha_v=0
        alpha_v_init:     float = 0.0,    # v_0 starts at theta for every date
    ):
        super().__init__()
        self.latent_dim = latent_dim

        # CIR scalars (log-parameterised → strictly positive)
        self.log_kappa   = nn.Parameter(torch.tensor(float(log_kappa_init)))
        self.log_theta   = nn.Parameter(torch.tensor(float(log_theta_init)))
        self.log_sigma_v = nn.Parameter(torch.tensor(float(log_sigma_v_init)))

        # State-conditioned v_0:  v_0(z_0) = theta * exp(alpha_v * tanh(w_v . z_0))
        self.w_v     = nn.Parameter(torch.randn(latent_dim) * w_v_init_scale)
        self.alpha_v = nn.Parameter(torch.tensor(float(alpha_v_init)))

    # ── parameter properties (always positive, grad-friendly) ──────────────────
    @property
    def kappa(self) -> torch.Tensor:
        return self.log_kappa.exp()

    @property
    def theta(self) -> torch.Tensor:
        return self.log_theta.exp()

    @property
    def sigma_v(self) -> torch.Tensor:
        return self.log_sigma_v.exp()

    @property
    def feller(self) -> torch.Tensor:
        """2*kappa*theta - sigma_v^2.  Positive ⇒ Feller condition satisfied."""
        return 2.0 * self.kappa * self.theta - self.sigma_v.pow(2)

    # ── initial v from current latent state ───────────────────────────────────
    def initial_v(self, z0: torch.Tensor) -> torch.Tensor:
        """
        State-conditioned initial variance.

        z0 : (1, d) or (d,)  — current latent state from the encoder
        Returns: scalar tensor with grad through (w_v, alpha_v, log_theta).
        """
        z = z0.squeeze(0) if z0.dim() == 2 else z0          # (d,)
        return self.theta * torch.exp(self.alpha_v * torch.tanh(self.w_v @ z))

    # ── single Euler step on v (clamped to keep CIR positive) ──────────────────
    def step(
        self,
        v_prev: torch.Tensor,    # (n_paths,)
        eps_v:  torch.Tensor,    # (n_paths,)  pre-drawn N(0,1)
        dt:     float,
        v_min:  float = 1e-10,
    ) -> torch.Tensor:
        """
        Euler-Maruyama step:
            v_t+dt = v_t + kappa*(theta - v_t)*dt + sigma_v*sqrt(v_t)*sqrt(dt)*eps_v
        Clamped to ≥ v_min so sqrt(v) never blows up.
        """
        kappa   = self.kappa
        theta   = self.theta
        sigma_v = self.sigma_v
        sqrt_dt = math.sqrt(dt)

        v_safe = v_prev.clamp(min=v_min)
        dv = (kappa * (theta - v_safe) * dt
              + sigma_v * v_safe.sqrt() * sqrt_dt * eps_v)
        return (v_safe + dv).clamp(min=v_min)

    # ── diagnostic ────────────────────────────────────────────────────────────
    def summary(self) -> dict:
        with torch.no_grad():
            return {
                "kappa":      float(self.kappa),
                "theta":      float(self.theta),
                "sigma_v":    float(self.sigma_v),
                "sqrt_theta": float(self.theta.sqrt()),
                "feller":     float(self.feller),
                "alpha_v":    float(self.alpha_v),
                "w_v_norm":   float(self.w_v.norm()),
            }
