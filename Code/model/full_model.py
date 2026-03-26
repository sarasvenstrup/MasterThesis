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
    solve_AB,
)


class FullModel(nn.Module):
    """
    Paper-faithful 2-factor version.

    Returns:
        - S_hat only, by default
        - (S_hat, aux) if return_aux=True
    """

    def __init__(
        self,
        input_dim: int = 8,
        latent_dim: int = 2,
        tau_max: int = 30,
        tenors: list[int] | None = None,
        g_hidden: int = 10,
        h_hidden: int = 4,
        r_hidden: int = 4,
        g_bias: bool = True,
        hr_bias: bool = False,
    ):
        super().__init__()

        # First reproduce the paper in 2D.
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.tau_max = tau_max
        self.tenors = tenors if tenors is not None else [1, 2, 3, 5, 10, 15, 20, 30]
        assert max(self.tenors) <= self.tau_max, "All tenors must be <= tau_max."

        # Fixed annual maturity grid: 0,1,...,tau_max
        self.register_buffer(
            "_tau_grid",
            torch.arange(0, tau_max + 1, dtype=torch.float32),
            persistent=False,
        )

        # Networks
        self.encoder = Encoder(input_dim, latent_dim)          # (8 -> 2)
        self.G = DecoderG(latent_dim, g_hidden, g_bias)        # (z, tau) -> scalar
        self.K = KMu(latent_dim=latent_dim, bias=True)         # mu(z) = Mz + N
        self.H = HSigma(latent_dim, h_hidden, hr_bias)         # -> log sigmas, atanh rho
        self.R = RShort(latent_dim, r_hidden, hr_bias)         # -> r_tilde(z)

    def _tau(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return self._tau_grid.to(device=device, dtype=dtype)

    def _compute_arb_diagnostics(
        self,
        tau: torch.Tensor,              # (N,)
        G_vals: torch.Tensor,           # (B,N)
        grad_z_G: torch.Tensor,         # (B,N,d)
        trace_cov_hess: torch.Tensor,   # (B,N)
        mu: torch.Tensor,               # (B,d)
        r_tilde: torch.Tensor,          # (B,)
        alpha: torch.Tensor,            # (B,N)
        beta: torch.Tensor,             # (B,N)
        gamma: torch.Tensor,            # (B,N)
        A_vals: torch.Tensor,           # (B,N)
        B_vals: torch.Tensor,           # (B,N)
        dG_dtau: torch.Tensor,          # (B,N)
    ) -> dict:
        # r shape -> (B,N)
        r = r_tilde.unsqueeze(1).expand(-1, G_vals.shape[1])

        # (∇G)^T μ
        gTmu = (grad_z_G * mu.unsqueeze(1)).sum(dim=2)  # (B,N)

        # bracket = ∂τG - (∇G)^T μ - 1/2 Tr[σ^T H(G) σ]
        bracket = dG_dtau - gTmu - 0.5 * trace_cov_hess  # (B,N)

        # ODE RHS on-grid
        dB_dtau = alpha * B_vals + beta
        dA_dtau = gamma * (B_vals ** 2)

        # Residual from PDE rewrite; should be close to zero
        R_tau = (
            -r
            - dA_dtau
            + G_vals * dB_dtau
            + B_vals * bracket
            + (B_vals ** 2) * gamma
        )  # (B,N)

        sigma_bar = 0.006
        tau_safe = torch.clamp(tau.unsqueeze(0), min=1e-8)  # (1,N)
        SR_tau = R_tau / (tau_safe * sigma_bar)

        return {
            "R_tau": R_tau[:, 1:],                    # skip tau=0
            "SR_tau": SR_tau[:, 1:],                  # skip tau=0
            "tau_grid": tau[1:],                      # 1..tau_max
            "max_abs_R": R_tau[:, 1:].abs().max(dim=1).values,
            "max_abs_SR_1to30": SR_tau[:, 1:].abs().max(dim=1).values,
        }

    def forward(
        self,
        S_in: torch.Tensor,
        do_arb_checks: bool = False,
        return_aux: bool = False,
    ):
        """
        S_in:
            (B,8) or (8,)

        Returns:
            - If return_aux=False (default): S_hat only
            - If return_aux=True: (S_hat, aux_dict)
              where aux_dict contains all computed quantities:
                'z': latent factors (B,d)
                'P_mkt': market discount factors (B, tau_max)
                'P_full': full discount factors (B, tau_max+1)
                'A_vals': ODE solution A (B, tau_max+1)
                'B_vals': ODE solution B (B, tau_max+1)
                'G_vals': G function values (B, tau_max+1)
                'mu': drift parameters (B,d)
                'sigma': volatility matrix (B,d,d)
                'r_tilde': short rate (B,)
                'alpha': ODE coefficient (B, tau_max+1)
                'beta': ODE coefficient (B, tau_max+1)
                'gamma': ODE coefficient (B, tau_max+1)
                'arb': arbitrage diagnostics dict (if do_arb_checks=True)
                'tau_grid': maturity grid (tau_max+1,)
        """
        squeeze_back = False
        if S_in.dim() == 1:
            S_in = S_in.unsqueeze(0)
            squeeze_back = True

        device = S_in.device
        dtype = S_in.dtype
        tau = self._tau(device=device, dtype=dtype)   # (N,) with N=tau_max+1

        # 1) Encode observed swap curve -> latent factors
        z = self.encoder(S_in)                        # (B,d)

        # 2) Evaluate G(z, tau) on the whole annual grid
        G_vals = self.G(z, tau)                       # (B,N)
        if G_vals.dim() == 1:
            G_vals = G_vals.unsqueeze(0)

        # 3) Risk-neutral parameter networks
        mu = self.K(z)                                # (B,d)
        sigmas, rhos = self.H(z)                      # (B,d), (B,1)
        r_tilde = self.R(z)                           # (B,1) or (B,)
        if r_tilde.ndim == 2 and r_tilde.shape[-1] == 1:
            r_tilde = r_tilde.squeeze(-1)             # (B,)

        sigma = L_from_sigmas_rhos(sigmas, rhos)      # (B,d,d)

        # 4) Derivatives of G wrt tau and z
        def G_single(z_single: torch.Tensor) -> torch.Tensor:
            # returns (N,)
            return self.G(z_single.unsqueeze(0), tau).squeeze(0)

        dG_dtau = d_tau_autograd_nodewise(self.G, z, tau)              # (B,N)
        grad_z_G, trace_cov_hess = grad_and_trace_cov_hess_G(
            G_single, z, sigma
        )                                                              # (B,N,2), (B,N)

        # 5) ODE coefficients alpha, beta, gamma
        alpha, beta, gamma = paper_alpha_beta_gamma_trace(
            G=G_vals,
            dG_dtau=dG_dtau,
            grad_z_G=grad_z_G,
            trace_cov_hess=trace_cov_hess,
            mu=mu,
            sigma=sigma,
            r_tilde=r_tilde,
        )  # all (B,N)

        # 6) Solve coupled ODE for A and B
        A_vals, B_vals = solve_AB(tau, alpha, beta, gamma)             # (B,N), (B,N)

        # 7) Hard sanity checks at tau=0
        assert A_vals.shape == G_vals.shape, f"A_vals {A_vals.shape} != G_vals {G_vals.shape}"
        assert B_vals.shape == G_vals.shape, f"B_vals {B_vals.shape} != G_vals {G_vals.shape}"

        assert torch.allclose(A_vals[:, 0], torch.zeros_like(A_vals[:, 0]), atol=1e-6)
        assert torch.allclose(B_vals[:, 0], torch.zeros_like(B_vals[:, 0]), atol=1e-6)

        # 8) Bond prices on full grid 0..tau_max
        P_full = torch.exp(A_vals - B_vals * G_vals)                   # (B,N)
        assert torch.allclose(P_full[:, 0], torch.ones_like(P_full[:, 0]), atol=1e-6)
        assert torch.isfinite(P_full).all()

        # Market grid starts at 1Y, not 0Y
        P_mkt = P_full[:, 1:]                                          # (B,tau_max)

        # 9) Convert annual discount factors to par swap rates at chosen tenors
        S_hat = par_swap_from_discount(P_mkt, self.tenors)             # (B,len(tenors))

        arb = None
        if do_arb_checks:
            arb = self._compute_arb_diagnostics(
                tau=tau,
                G_vals=G_vals,
                grad_z_G=grad_z_G,
                trace_cov_hess=trace_cov_hess,
                mu=mu,
                r_tilde=r_tilde,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                A_vals=A_vals,
                B_vals=B_vals,
                dG_dtau=dG_dtau,
            )

        aux = {
            "z": z,
            "P_mkt": P_mkt,
            "P_full": P_full,
            "A_vals": A_vals,
            "B_vals": B_vals,
            "G_vals": G_vals,
            "mu": mu,
            "sigma": sigma,
            "r_tilde": r_tilde,
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "arb": arb,
            "tau_grid": tau,
        }

        if squeeze_back:
            S_hat = S_hat.squeeze(0)

        if torch.is_grad_enabled() and (not S_hat.requires_grad):
            raise RuntimeError("S_hat is detached inside FullModel.forward().")

        if return_aux:
            return S_hat, aux
        return S_hat
