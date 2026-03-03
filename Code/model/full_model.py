# full_model.py  (FINAL – no positive-G wrapper)
# ------------------------------------------------------------
# Notes:
# - Default solver is "rk38" (Poulsen RK4 3/8 forward, plain autograd).
# - No extra positivity/offset transform on G (as requested).
# - Hard shape asserts instead of silent expand() fallbacks.
# ------------------------------------------------------------

import torch
import torch.nn as nn

from Code.model.Encoder import Encoder
from Code.model.DecoderG import DecoderG
from Code.model.K_mu import KMu
from Code.model.H_sigma import HSigma
from Code.model.R_short import RShort

from Code.utils.rates import par_swap_from_discount
from Code.utils.sigma_matrix import L_from_sigmas_rhos
from Code.utils.ode import (
    d_tau_autograd_nodewise,
    grad_and_trace_cov_hess_G,
    paper_alpha_beta_gamma_trace,
    solve_AB
)

class FullModel(nn.Module):
    def __init__(
        self,
        input_dim=8,
        latent_dim=2,
        tau_max=30,
        tenors=None,
        g_hidden=10,
        h_hidden=4,
        r_hidden=4,
        g_bias=False,
        hr_bias=False,
        do_arb_checks = False
    ):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.tau_max = tau_max  # build discount curve 0..tau_max
        self.tenors = tenors if tenors is not None else [1, 2, 3, 5, 10, 15, 20, 30]

        # Networks
        self.encoder = Encoder(input_dim, latent_dim)
        self.G = DecoderG(latent_dim, g_hidden, g_bias)
        self.K = KMu(latent_dim, bias=True)
        self.H = HSigma(latent_dim, h_hidden, hr_bias)
        self.R = RShort(latent_dim, r_hidden, hr_bias)

    def forward(self, S_in: torch.Tensor, do_arb_checks: bool = False):

        # ensure batch
        squeeze_back = False

        if S_in.dim() == 1:
            S_in = S_in.unsqueeze(0)
            squeeze_back = True

        device = S_in.device
        dtype = S_in.dtype

        # 1) Encode: (B,8) -> (B,d)
        z = self.encoder(S_in)

        # 2) maturity grid 0..tau_max inclusive
        tau = torch.arange(0, self.tau_max + 1, device=device, dtype=dtype)  # (30,)

        # 3) Evaluate G(z,tau) on the grid -> (B,N)
        G_vals = self.G(z, tau)

        if G_vals.dim() == 1:
            G_vals = G_vals.unsqueeze(0)

        # 4) Risk-neutral parameter nets
        mu = self.K(z)              # (B,d)
        sigmas, rhos = self.H(z)    # sigmas: (B,d), rhos: (B,d(d-1)/2)
        r_tilde = self.R(z)         # (B,1) or (B,)

        sigma = L_from_sigmas_rhos(sigmas, rhos)  # (B,d,d) Cholesky

        # 5) Derivatives needed for alpha/beta/gamma
        def G_single(z_single: torch.Tensor) -> torch.Tensor:
            # returns (N,)
            return self.G(z_single.unsqueeze(0), tau).squeeze(0)


        dG_dtau = d_tau_autograd_nodewise(self.G, z, tau)  # (B,N)
        grad_z_G, trace_cov_hess = grad_and_trace_cov_hess_G(G_single, z, sigma)  # (B,N,d), (B,N)

        alpha, beta, gamma = paper_alpha_beta_gamma_trace(
            G=G_vals,
            dG_dtau=dG_dtau,
            grad_z_G=grad_z_G,
            trace_cov_hess=trace_cov_hess,
            mu=mu,
            sigma=sigma,
            r_tilde=r_tilde,
        )  # all (B,N)


        # 6) Solve ODE for (A,B)
        A_vals, B_vals = solve_AB(tau, alpha, beta, gamma)  # both (B,N)

        # SHARPE RATIO
        # r shape -> (B,1) for broadcast
        r = r_tilde if r_tilde.ndim == 2 else r_tilde.unsqueeze(1)  # (B,1)
        r = r.expand(-1, G_vals.shape[1])  # (B,N)

        # gTmu = (∇G)^T μ  (B,N)
        gTmu = (grad_z_G * mu.unsqueeze(1)).sum(dim=2)

        # bracket = ∂τG - (∇G)^T μ - 1/2 Tr[σ^T H(G) σ]
        bracket = dG_dtau - gTmu - 0.5 * trace_cov_hess  # (B,N)

        # Use ODE RHS to get A' and B' on-grid (no finite diff needed)
        dB_dtau = alpha * B_vals + beta            # (B,N)
        dA_dtau = gamma * (B_vals ** 2)            # (B,N)

        # Residual R = LN/P  (should be ~0)
        R_tau = (
            -r
            - dA_dtau
            + G_vals * dB_dtau
            + B_vals * bracket
            + (B_vals ** 2) * gamma
        )  # (B,N)

        # Approx Sharpe ratio like Andreasen (optional)
        sigma_bar = 0.006
        tau_safe = torch.clamp(tau.unsqueeze(0), min=1e-8)  # (1,N)
        SR_tau = R_tau / (tau_safe * sigma_bar)

        arb = {
            "R_tau": R_tau,      # (B,N) main no-arb residual (LN/P)
            "SR_tau": SR_tau,    # (B,N) approximate SR
            "max_abs_R": R_tau.abs().max(dim=1).values,      # (B,)
            "max_abs_SR_1to30": SR_tau[:, 1:].abs().max(dim=1).values,  # (B,) ignore tau=0
        }


        # Hard asserts (instead of silent expand)
        assert A_vals.shape == G_vals.shape, f"A_vals {A_vals.shape} != G_vals {G_vals.shape}"
        assert B_vals.shape == G_vals.shape, f"B_vals {B_vals.shape} != G_vals {G_vals.shape}"

        # 7) Discount factors
        Xexp = A_vals - B_vals * G_vals

        P = torch.exp(Xexp)  # (B,N)

        if tau[0].item() == 0:
            P = P[:, 1:]

        # 8) Swap rates at observed tenors (drop tau=0)
        S_hat = par_swap_from_discount(P, self.tenors)  # (B,8)

        if squeeze_back:
            S_hat = S_hat.squeeze(0)

        if torch.is_grad_enabled() and (not S_hat.requires_grad):
            raise RuntimeError("S_hat is detached inside FullModel.forward()")

        return S_hat, z, P, A_vals, B_vals, G_vals, mu, sigma, r_tilde, arb