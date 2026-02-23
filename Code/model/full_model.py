# full_model.py

# %pip install torch
# dbutils.library.restartPython()

# DO NOT RUN THIS FILE

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

from Code.utils.helpers import check_monotonicity, instantaneous_forward, finite_minmax

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
            g_bias=True,
            hr_bias=False,
            ab_solver = "chen"
    ):

        super().__init__()

        # Store dimensions
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.ab_solver = ab_solver

        self.tau_max = tau_max  # build discount curve 1..tau_max

        # observed tenors
        self.tenors = tenors if tenors is not None else [1, 2, 3, 5, 10, 15, 20, 30]

        # Networks
        self.encoder = Encoder(input_dim, latent_dim)
        self.G = DecoderG(latent_dim, g_hidden, g_bias)
        self.K = KMu(latent_dim, bias=True)
        self.H = HSigma(latent_dim, h_hidden, hr_bias)
        self.R = RShort(latent_dim, r_hidden, hr_bias)

    def forward(self, S_in, do_arb_checks: bool = False):
        # ensure batch
        squeeze_back = False

        if S_in.dim() == 1:
            S_in = S_in.unsqueeze(0)
            squeeze_back = True

        device = S_in.device
        dtype = S_in.dtype

        # 1) Encode: (B,8) -> (B, latent_dim)
        z = self.encoder(S_in)

        # 2) maturity grid
        tau = torch.linspace(0.0, float(self.tau_max), self.tau_max + 1, device=device, dtype=dtype)

        # 3) Evaluate G(z,tau) in the grid -> (B,T)
        G_vals = self.G(z, tau)

        mu = self.K(z)  # (batch,d)
        sigmas, rhos = self.H(z)  # sigmas: (B,d), rhos: (B,d(d-1)/2)

        r_tilde = self.R(z)

        sigma = L_from_sigmas_rhos(sigmas, rhos) # (B,d,d) = Cholesky L

        def G_single(z_single):
            return self.G(z_single.unsqueeze(0), tau).squeeze(0)  # (N,)

        dG_dtau = d_tau_autograd_nodewise(self.G, z, tau)  # exact ∂G/∂τ (years)

        grad_z_G, trace_cov_hess = grad_and_trace_cov_hess_G(G_single, z, sigma)

        alpha, beta, gamma = paper_alpha_beta_gamma_trace(
            G=G_vals,
            dG_dtau=dG_dtau,
            grad_z_G=grad_z_G,
            trace_cov_hess=trace_cov_hess,
            mu=mu,
            sigma=sigma,
            r_tilde=r_tilde,
        )

        A_vals, B_vals = solve_AB(tau, alpha, beta, gamma, solver=self.ab_solver)

        if A_vals.dim() == 1:
            A_vals = A_vals.unsqueeze(0).expand(G_vals.shape[0], -1)
        if B_vals.dim() == 1:
            B_vals = B_vals.unsqueeze(0).expand(G_vals.shape[0], -1)

        Xexp = A_vals - B_vals * G_vals

        if self.training:
            Xexp = Xexp.clamp(-80.0, 80.0)

        if do_arb_checks:
            print("finite Xexp:", torch.isfinite(Xexp).all().item())
            xmin, xmax = finite_minmax(Xexp)
            print("Xexp finite min/max:", xmin, xmax)

        P = torch.exp(Xexp)

        if do_arb_checks:
            with torch.no_grad():
                if not torch.allclose(P[:, 0], torch.ones_like(P[:, 0]), atol=1e-3, rtol=1e-3):
                    print("Warning: P(tau=0) not ~1. min/max:", float(P[:, 0].min()), float(P[:, 0].max()))

                if torch.any(P <= 0):
                    print("Negative discount factor detected!")

                viol = check_monotonicity(P)
                neg_forwards = (instantaneous_forward(P, tau) < -1e-8).sum().item()

                print("Monotonicity violations:", int(viol))
                print("Negative forward rates:", int(neg_forwards))

        # 6) Swap rates at observed tenors
        P_1T = P[:, 1:]  # drop tau=0, keep 1..tau_max
        S_hat = par_swap_from_discount(P_1T, self.tenors)  # (B,8)

        if squeeze_back:
            S_hat = S_hat.squeeze(0)  # (8,)

        if torch.is_grad_enabled() and (not S_hat.requires_grad):
            raise RuntimeError("S_hat is detached inside FullModel.forward()")

        return S_hat, z, P, A_vals, B_vals, G_vals, mu, sigma, r_tilde

# ------------------------------------------------
# Notes:
# ------------------------------------------------
#
# - __init__():
#   Builds all model components (encoder, G, K, H, R).
#   The output_head is a temporary placeholder that maps
#   latent factors z directly back to 8 swap rates.
#   It is NOT part of the paper and will be removed later.
#
# - forward():
#   Defines how data flows through the model:
#   1) Ensures input has batch shape (8,) -> (1,8)
#   2) Encodes input to latent factors z
#   3) Skips ODE + pricing (for now) and uses output_head
#   4) Removes batch dimension if needed
#
# - Current version is a scaffold for debugging.
#   The real decoder (ODE + bond pricing + swap formula)
#   will later replace output_head.
#
# - Remember that this script is only the forwardpassing of the model. The solving of the model, where we will be using the backpropagation etc. is in the output files, e.g. debug_single_curve.py.

