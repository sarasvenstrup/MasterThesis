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
    solve_AB_rk38,
)

class FullModel(nn.Module):

    def __init__(
            self,
            input_dim=8,
            latent_dim=2,
            tau_max=30,
            B_scale=10.0,
            tenors=None,
            g_hidden=10,
            h_hidden=4,
            r_hidden=4,
            g_bias=True,
            hr_bias=False
    ):

        super().__init__()

        # Store dimensions
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.tau_max = tau_max  # build discount curve 1..tau_max
        self.B_scale = B_scale  # c in B(tau)=tau/c

        # observed tenors
        self.tenors = tenors if tenors is not None else [1, 2, 3, 5, 10, 15, 20, 30]

        # Networks
        self.encoder = Encoder(input_dim, latent_dim)
        self.G = DecoderG(latent_dim, g_hidden, g_bias)
        self.K = KMu(latent_dim, bias=True)
        self.H = HSigma(latent_dim, h_hidden, hr_bias)
        self.R = RShort(latent_dim, r_hidden, hr_bias)

    def params_from_z(self, z: torch.Tensor):
        """
        Returns (mu, L, r) needed by Sharpe ratio code.
        mu: (B,d)
        L:  (B,d,d)   (Cholesky diffusion)
        r:  (B,)      short rate mapping
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)

        mu = self.K(z)  # (B,d)
        sigmas, rhos = self.H(z)  # (B,d), (B, d(d-1)/2)
        L = L_from_sigmas_rhos(sigmas, rhos)  # (B,d,d)
        r = self.R(z)  # (B,1) or (B,)
        if r.ndim == 2 and r.shape[1] == 1:
            r = r.squeeze(1)
        return mu, L, r

    def bond_price_from_z_grid(self, z: torch.Tensor, tau_grid: torch.Tensor) -> torch.Tensor:
        """
        No-interp ZCB curve: returns P(z, tau_grid) for a shared tau_grid (N,).
        tau_grid in YEARS, must be 1D increasing, within [0, tau_max].

        Returns: (B, N)
        Autograd flows through z and tau_grid.
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)
        if tau_grid.ndim == 0:
            tau_grid = tau_grid.view(1)
        if tau_grid.ndim != 1:
            raise ValueError("tau_grid must be 1D (N,)")

        B = z.shape[0]
        device, dtype = z.device, z.dtype

        tau = tau_grid.to(device=device, dtype=dtype)

        if torch.any(tau < 0) or torch.any(tau > float(self.tau_max)):
            raise ValueError(f"tau_grid must be within [0, {self.tau_max}]")
        if torch.any(tau[1:] <= tau[:-1]):
            raise ValueError("tau_grid must be strictly increasing")

        # normalized u in [0,1]
        u = tau / float(self.tau_max)  # (N,)

        # --- same pipeline as forward(), but on this u-grid ---
        G_vals = self.G(z, u)  # (B,N)

        mu = self.K(z)  # (B,d)
        sigmas, rhos = self.H(z)  # (B,d), (B, d(d-1)/2)
        r_tilde = self.R(z)  # (B,1) or (B,)
        sigma = L_from_sigmas_rhos(sigmas, rhos)  # (B,d,d)

        def G_single(z_single):
            return self.G(z_single.unsqueeze(0), u).squeeze(0)  # (N,)

        dG_du = d_tau_autograd_nodewise(self.G, z, u)  # exact ∂G/∂u, (B,N)
        dG_dtau = dG_du / float(self.tau_max)  # chain rule

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

        # IMPORTANT: solve on the same tau grid (works with your new RK38)
        A_vals, B_vals = solve_AB_rk38(tau, alpha, beta, gamma)  # (B,N), (B,N)

        P = torch.exp(A_vals - B_vals * G_vals)  # (B,N)
        return P

    def bond_price_from_z(self, z: torch.Tensor, tau_query: torch.Tensor) -> torch.Tensor:
        # convenience wrapper
        if not torch.is_tensor(tau_query):
            tau_query = torch.tensor(tau_query, device=z.device, dtype=z.dtype)
        if tau_query.ndim == 0:
            tau_query = tau_query.view(1)
        P = self.bond_price_from_z_grid(z, tau_query)  # (B,N)
        return P.squeeze(1) if P.shape[1] == 1 else P

    def forward(self, S_in):
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
        tau = torch.arange(0, self.tau_max + 1, device=device, dtype=dtype)
        tau_in = tau / float(self.tau_max)  # [0,1]

        # 3) Evaluate G(z,tau) in the grid -> (B,T)
        G_vals = self.G(z, tau_in)

        mu = self.K(z)  # (batch,d)
        sigmas, rhos = self.H(z)  # sigmas: (B,d), rhos: (B,d(d-1)/2)

        r_tilde = self.R(z)

        sigma = L_from_sigmas_rhos(sigmas, rhos) # (B,d,d) = Cholesky L

        def G_single(z_single):
            return self.G(z_single.unsqueeze(0), tau_in).squeeze(0)  # (N,)

        dG_du = d_tau_autograd_nodewise(self.G, z, tau_in)  # exact ∂G/∂u
        dG_dtau = dG_du / float(self.tau_max)

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

        A_vals, B_vals = solve_AB_rk38(tau, alpha, beta, gamma)

        # 5) Discount Factors
        P = torch.exp(A_vals - B_vals * G_vals)  # (B,T)

        # should be ~1 at tau=0
        if torch.is_grad_enabled():
            if not torch.allclose(P[:, 0], torch.ones_like(P[:, 0]), atol=1e-3, rtol=1e-3):
                print("Warning: P(tau=0) not ~1. min/max:", float(P[:, 0].min()), float(P[:, 0].max()))

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

