"""
FullModelPrice: thin subclass of FullModel that adds k_override support.

full_model_stable.py is not touched at all. This subclass adds a single
capability: passing a separate K_Q module into decode_from_z so that the
Q-measure drift is used consistently in both the ODE and the simulation
during pricing, while reconstruction continues to use the original K^P.

Usage
-----
    from Code.model.full_model_price import FullModelPrice

    model = FullModelPrice(latent_dim=4)
    model.load_state_dict(...)

    # Reconstruction — uses K^P (original, unchanged)
    S_hat = model(x)

    # Pricing — uses K_Q in both the ODE and simulation
    _, aux = model.decode_from_z(z_T, return_aux=True, k_override=K_Q)
"""

import torch
from .full_model_stable import FullModel


class FullModelPrice(FullModel):
    """
    Identical to FullModel except decode_from_z accepts an optional k_override.

    k_override : nn.Module | None
        If None  → behaves exactly like FullModel (uses self.K, K^P).
        If given → temporarily replaces self.K with k_override (K^Q) for the
                   duration of the ODE solve, then restores self.K.

    This makes K^Q consistent in both the simulation and the ODE without
    duplicating any code or modifying full_model_stable.py.
    """

    def decode_from_z(
            self,
            z: torch.Tensor,
            tau=None,
            do_arb_checks: bool = False,
            return_aux: bool = False,
            k_override=None,
    ):
        if k_override is None:
            # No override — identical to FullModel
            return super().decode_from_z(z, tau, do_arb_checks, return_aux)

        # Temporarily swap self.K for k_override so the parent's ODE sees K^Q
        original_K = self.K
        self.K = k_override
        try:
            result = super().decode_from_z(z, tau, do_arb_checks, return_aux)
        finally:
            self.K = original_K   # always restore, even if an exception occurs
        return result
