import torch
import torch.nn as nn

from .Encoder import Encoder
from .DecoderG import DecoderG

# Baseline components only — no stable imports, no config dependency
from .K_mu import KMu as KMuBaseline
from .R_short import RShort
from .H_sigma import HSigma as HSigmaBaseline

from Code.utils.rates import par_swap_from_discount
from .sigma_matrix import L_from_sigmas_rhos
from Code.utils.ode import (
    d_tau_autograd_nodewise,
    grad_and_trace_cov_hess_G,
    paper_alpha_beta_gamma_trace,
    solve_AB,
)

VARIANT = "baseline"  # frozen — never changes


class FullModel(nn.Module):
    """
    Baseline-only FullModel.  Stable variant imports and config checks have
    been removed so that changes to the stable pipeline can never affect
    baseline results or initialization.

    Returns:
        - P_mkt only, by default
        - (P_mkt, aux) if return_aux=True
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

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.tau_max = tau_max
        self.tenors = tenors if tenors is not None else [1, 2, 3, 5, 10, 15, 20, 30]

        assert max(self.tenors) <= self.tau_max, "All tenors must be <= tau_max."

        self.register_buffer(
            "_tau_grid",
            torch.arange(0, tau_max + 1, dtype=torch.float32),
            persistent=False,
        )

        self.encoder = Encoder(input_dim, latent_dim)

        self.G = DecoderG(latent_dim, g_hidden, g_bias)

        self.K = KMuBaseline(
            latent_dim=latent_dim,
            bias=True,
        )
        self.H = HSigmaBaseline(
            latent_dim=latent_dim,
            hidden_dim=h_hidden,
            bias=hr_bias,
        )

        self.R = RShort(latent_dim, r_hidden, bias=hr_bias)

    def _tau(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return self._tau_grid.to(device=device, dtype=dtype)

    def _compute_arb_diagnostics(
        self,
        tau: torch.Tensor,
        G_vals: torch.Tensor,
        grad_z_G: torch.Tensor,
        trace_cov_hess: torch.Tensor,
        mu: torch.Tensor,
        r_tilde: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        gamma: torch.Tensor,
        A_vals: torch.Tensor,
        B_vals: torch.Tensor,
        dG_dtau: torch.Tensor,
    ) -> dict:
        r = r_tilde.unsqueeze(1).expand(-1, G_vals.shape[1])

        gTmu = (grad_z_G * mu.unsqueeze(1)).sum(dim=2)
        bracket = dG_dtau - gTmu - 0.5 * trace_cov_hess

        dB_dtau = alpha * B_vals + beta
        dA_dtau = gamma * (B_vals ** 2)

        R_tau = (
            -r
            - dA_dtau
            + G_vals * dB_dtau
            + B_vals * bracket
            + (B_vals ** 2) * gamma
        )

        SR_tau = R_tau

        return {
            "R_tau": R_tau[:, 1:],
            "SR_tau": SR_tau[:, 1:],
            "tau_grid": tau[1:],
            "max_abs_R": R_tau[:, 1:].abs().max(dim=1).values,
            "max_abs_SR_1to30": SR_tau[:, 1:].abs().max(dim=1).values,
        }

    def decode_from_z(
            self,
            z: torch.Tensor,
            tau: torch.Tensor | None = None,
            do_arb_checks: bool = False,
            return_aux: bool = False,
    ):
        """
        Decode latent states directly to discount factors and swap rates.

        Args:
            z:   shape (B, latent_dim) or (latent_dim,)
            tau: optional custom tau grid. If None, uses the model's default grid 0,1,...,tau_max

        Returns:
            - P_mkt by default
            - (P_mkt, aux) if return_aux=True
        """
        squeeze_back = False
        if z.dim() == 1:
            z = z.unsqueeze(0)
            squeeze_back = True

        device = z.device
        dtype = z.dtype

        if tau is None:
            tau = self._tau(device=device, dtype=dtype)
        else:
            tau = tau.to(device=device, dtype=dtype)

        # 1) Evaluate G(z, tau)
        G_vals = self.G(z, tau)
        if G_vals.dim() == 1:
            G_vals = G_vals.unsqueeze(0)

        # 2) Risk-neutral parameter networks
        mu = self.K(z)

        sigmas, rhos = self.H(z)
        sigma = L_from_sigmas_rhos(sigmas, rhos, validate=False)

        r_tilde = self.R(z)
        if r_tilde.ndim == 2 and r_tilde.shape[-1] == 1:
            r_tilde = r_tilde.squeeze(-1)

        # 3) Derivatives
        def G_single(z_single: torch.Tensor) -> torch.Tensor:
            return self.G(z_single.unsqueeze(0), tau).squeeze(0)

        dG_dtau = d_tau_autograd_nodewise(self.G, z, tau)
        grad_z_G, trace_cov_hess = grad_and_trace_cov_hess_G(G_single, z, sigma)

        # 4) ODE coefficients
        alpha, beta, gamma = paper_alpha_beta_gamma_trace(
            G=G_vals,
            dG_dtau=dG_dtau,
            grad_z_G=grad_z_G,
            trace_cov_hess=trace_cov_hess,
            mu=mu,
            sigma=sigma,
            r_tilde=r_tilde,
        )

        # 5) Solve ODEs
        A_vals, B_vals = solve_AB(tau, alpha, beta, gamma)

        # 6) Bond prices
        log_P = A_vals - B_vals * G_vals
        P_full = torch.exp(log_P)
        P_mkt = P_full[:, 1:]

        # 7) Swap rates only if tau matches annual market grid
        S_hat = None
        if tau.numel() == self.tau_max + 1:
            tau_default = self._tau(device=device, dtype=dtype)
            if torch.allclose(tau, tau_default, atol=1e-10, rtol=0.0):
                S_hat = par_swap_from_discount(P_mkt, self.tenors)

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
            "P_full": P_full,
            "P_mkt": P_mkt,
            "S_hat": S_hat,
            "A_vals": A_vals,
            "B_vals": B_vals,
            "G_vals": G_vals,
            "mu": mu,
            "sigmas": sigmas,
            "rhos": rhos,
            "sigma": sigma,
            "r_tilde": r_tilde,
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "arb": arb,
            "tau_grid": tau,
            "z": z,
        }

        if squeeze_back:
            P_mkt = P_mkt.squeeze(0)
            aux = {k: (v.squeeze(0) if torch.is_tensor(v) and v.shape[0] == 1 else v) for k, v in aux.items()}

        if return_aux:
            return P_mkt, aux
        return P_mkt

    def forward(
            self,
            S_in: torch.Tensor,
            do_arb_checks: bool = False,
            return_aux: bool = False,
    ):
        squeeze_back = False
        if S_in.dim() == 1:
            S_in = S_in.unsqueeze(0)
            squeeze_back = True

        z = self.encoder(S_in)
        _, aux = self.decode_from_z(z, tau=None, do_arb_checks=do_arb_checks, return_aux=True)

        S_hat = aux["S_hat"]
        if S_hat is None:
            raise RuntimeError("Default tau grid should produce S_hat, but got None.")

        if squeeze_back:
            S_hat = S_hat.squeeze(0)

        if torch.is_grad_enabled() and (not S_hat.requires_grad):
            raise RuntimeError("S_hat is detached inside FullModel.forward().")

        if return_aux:
            return S_hat, aux
        return S_hat
