"""
FullModelPrice: thin subclass of FullModel that adds k_override and
sigma_scale support to decode_from_z.

full_model_stable.py is not touched at all.  This subclass adds two
capabilities so that the same pricing dynamics can be used consistently in
*both* simulation and the no-arbitrage ODE:

  k_override  — replaces self.K in the ODE (drift correction)
  sigma_scale — scales self.H's sigmas output in the ODE (vol correction)

Background
----------
simulate_to_expiry_differentiable already accepts k_override and sigma_scale
to modify the simulated paths.  Without this class the terminal call

    model.decode_from_z(z_T, return_aux=True)

would still use the original unscaled K and H, creating an inconsistency
between simulation dynamics and decoded bond prices.

sigma_scale implementation note
--------------------------------
L_from_sigmas_rhos builds L such that L[b,i,j] = sigmas[b,i] * f(rhos).
The simulation shock is:

    shock = sigma_scale * (L @ dW)  ≡  diag(sigma_scale) @ L @ dW

diag(sigma_scale) @ L  is identical to  L_from_sigmas_rhos(sigmas * scale, rhos)

so applying sigma_scale in the ODE is equivalent to pre-multiplying the
sigmas vector returned by H(z) by sigma_scale.  A lightweight _ScaledHWrapper
handles this without touching any ODE utilities.

Usage
-----
    from Code.model.full_model_price import FullModelPrice

    model = FullModelPrice(latent_dim=4)
    # strict=False is required: checkpoint has DecoderG weights but not the
    # new G.z_scale buffer introduced by DecoderGStable.
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    # Calibrate the tanh input scale from in-sample latent states.
    model.set_z_scale(z_train)   # z_train: (N, latent_dim)

    # Reconstruction — uses original K and H (unchanged)
    S_hat = model(x)

    # Pricing — drift + vol consistent in both simulation and ODE
    _, aux = model.decode_from_z(
        z_T,
        return_aux=True,
        k_override=wrapper,          # wrapper.forward(z) returns K*(z)
        sigma_scale=wrapper.sigma_vec,  # shape (d,), learnable
    )
"""

import torch
import torch.nn as nn
from .full_model_stable import FullModel
from .DecoderG_stable import DecoderGStable


class _ScaledHWrapper(nn.Module):
    """
    Thin wrapper around an H module that scales each sigma_i by scale[i].

    H(z) returns (sigmas, rhos) where sigmas has shape (B, d).
    This wrapper returns (sigmas * scale.unsqueeze(0), rhos) so that the
    resulting L matrix satisfies

        L_scaled = diag(scale) @ L_original

    which is exactly the scaling applied in simulate_to_expiry_differentiable
    when sigma_scale is a (d,) tensor.
    """

    def __init__(self, h_module: nn.Module, scale: torch.Tensor):
        super().__init__()
        self._h     = h_module
        # Register as buffer so device/dtype moves are handled automatically,
        # but do NOT make it a Parameter (it must not be trained here).
        self.register_buffer("_scale", scale)

    def forward(self, z: torch.Tensor):
        sigmas, rhos = self._h(z)
        return sigmas * self._scale.unsqueeze(0), rhos


class FullModelPrice(FullModel):
    """
    Identical to FullModel except:

    1. self.G is replaced by DecoderGStable, which applies tanh input
       normalisation on z before the MLP.  This prevents OOD latent states
       from extrapolating G wildly and blowing up the ODE coefficients.

    2. decode_from_z accepts two extra keyword arguments:

      k_override   : nn.Module | None
          If None  → uses self.K (original, reconstruction drift).
          If given → temporarily replaces self.K so the ODE sees K*(z).

      sigma_scale  : Tensor (d,) | float | None
          If None  → uses self.H unchanged (original, reconstruction vol).
          If given → wraps self.H with _ScaledHWrapper so that the ODE uses
                     sigma_eff[i] = sigma_scale[i] * sigma_i(z).

    Pass both together when calling decode_from_z at the terminal state
    during pricing to make the ODE fully consistent with the simulation:

        _, aux_T = model.decode_from_z(
            z_T, return_aux=True,
            k_override=wrapper,
            sigma_scale=wrapper.sigma_vec,
        )

    After loading a pre-trained checkpoint, copy MLP weights from the old
    DecoderG and calibrate z_scale::

        model.G.net.load_state_dict(old_model.G.net.state_dict())
        model.set_z_scale(z_train)   # (N, latent_dim) in-sample latent states
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Replace base DecoderG with the stable tanh-normalised version,
        # preserving the same hidden dimension.
        old_G = self.G
        self.G = DecoderGStable(
            latent_dim=old_G.latent_dim,
            hidden_dim=old_G.net[0].out_features,   # hidden_dim from first Linear
            bias=old_G.net[0].bias is not None,
        )

    def set_z_scale(self, z_train: torch.Tensor) -> None:
        """
        Convenience wrapper: set z_scale on the stable decoder from
        in-sample latent states.

        Args:
            z_train: (N, latent_dim) tensor of training latent states.
        """
        self.G.set_z_scale(z_train)

    def decode_from_z(
            self,
            z: torch.Tensor,
            tau=None,
            do_arb_checks: bool = False,
            return_aux: bool = False,
            k_override=None,
            sigma_scale=None,
    ):
        # Fast path: no overrides — identical to FullModel
        if k_override is None and sigma_scale is None:
            return super().decode_from_z(z, tau, do_arb_checks, return_aux)

        original_K = self.K
        original_H = self.H

        try:
            if k_override is not None:
                self.K = k_override

            if sigma_scale is not None:
                if not torch.is_tensor(sigma_scale):
                    sigma_scale = torch.tensor(
                        sigma_scale, device=z.device, dtype=z.dtype
                    )
                else:
                    sigma_scale = sigma_scale.to(device=z.device, dtype=z.dtype)
                self.H = _ScaledHWrapper(original_H, sigma_scale)

            result = super().decode_from_z(z, tau, do_arb_checks, return_aux)

        finally:
            # Always restore even if an exception occurs
            self.K = original_K
            self.H = original_H

        return result
