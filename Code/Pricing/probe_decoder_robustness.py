"""
Decoder robustness probe:
For each checkpoint, take z = encoder(curve) then z_perturbed = z + scale * eps,
and measure what fraction of decoded P_full are finite.
This isolates decoder brittleness independent of SDE drift/diffusion.
"""
import torch
import numpy as np
import os, sys

sys.path.insert(0, os.path.abspath('../..'))
os.environ.setdefault("SKIP_VARIANT_CONFIRM", "1")

from Code import config
config.confirm_variant()

from Code.load_swapdata import my_data
from Code.model.full_model import FullModel as FM_base
from Code.model.full_model_stable import FullModel as FM_stable

device = torch.device('cpu')
meta, X_tensor, *_ = my_data(use='bbg')
X_eur = X_tensor[meta['ccy'] == 'EUR'].float()

torch.manual_seed(42)
scales = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
n_samples = 200  # rows of X_eur to use
n_noise   = 50   # noise draws per row

X_sub = X_eur[:n_samples]

configs = [
    ('dim2_baseline', 2, False),
    ('dim3_baseline', 3, False),
    ('dim4_baseline', 4, False),
    ('dim2_stable',   2, True),
    ('dim3_stable',   3, True),
    ('dim4_stable',   4, True),
]

print(f"{'Model':<20} " + "  ".join(f"eps={s:.2f}" for s in scales))
print("-" * (20 + 10 * len(scales)))

for label, dim, is_stable in configs:
    path = f'Figures/TrainingResults/{label}/ep5000/checkpoint_{label.split("_")[0]}_ep5000.pt'
    if not os.path.isfile(path):
        print(f"{label:<20}  MISSING")
        continue

    raw = torch.load(path, map_location=device)
    ModelClass = FM_stable if is_stable else FM_base
    model = ModelClass(latent_dim=dim)
    model.load_state_dict(raw)
    model.eval()

    row = []
    with torch.no_grad():
        z0_full = model.encoder(X_sub.to(device))   # (<=200, dim)
        n_actual = z0_full.shape[0]

        for scale in scales:
            if scale == 0.0:
                z_test = z0_full.unsqueeze(1).expand(-1, n_noise, -1).reshape(-1, dim)
            else:
                eps = torch.randn(n_actual, n_noise, dim)
                z_test = (z0_full.unsqueeze(1) + scale * eps).reshape(-1, dim)

            _, aux = model.decode_from_z(z_test, tau=None, return_aux=True)
            P = aux['P_full']
            finite_frac = float(torch.isfinite(P).all(dim=1).float().mean())
            row.append(finite_frac)

    print(f"{label:<20} " + "  ".join(f"  {v:.3f}" for v in row))


