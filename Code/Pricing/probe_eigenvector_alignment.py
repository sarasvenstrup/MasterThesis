"""Verify the slow-eigenvector alignment claim."""
import torch, numpy as np, os, sys
sys.path.insert(0, os.path.abspath('../..'))
os.environ.setdefault("SKIP_VARIANT_CONFIRM", "1")
from Code import config; config.confirm_variant()
from Code.load_swapdata import my_data
from Code.model.full_model_stable import FullModel as FM_stable

meta, X_tensor, *_ = my_data(use='bbg')
X_eur = X_tensor[meta['ccy'] == 'EUR'].float()
mid_idx = len(X_eur) // 2

path = '../../Figures/TrainingResults/dim4_stable/ep5000/checkpoint_dim4_ep5000.pt'
raw = torch.load(path, map_location='cpu')
m = FM_stable(latent_dim=4); m.load_state_dict(raw); m.eval()

with torch.no_grad():
    z0     = m.encoder(X_eur[mid_idx:mid_idx+1]).float()
    M      = m.K.stable_matrix()
    N      = m.K.N
    z_star = -torch.linalg.solve(M, N)

delta = (z_star - z0.squeeze()).detach()   # z* - z_0  shape (4,)
print(f"z* - z_0 = {delta.numpy()}")
print(f"||z* - z_0|| = {delta.norm():.4f}")

# Eigendecomposition of M (real symmetric — use eigh for stability)
# M = -(V^T V + eps I) so it's symmetric negative-definite
eigvals, eigvecs = torch.linalg.eigh(M)   # eigvals ascending, eigvecs columns
print(f"\nEigenvalues of M: {eigvals.detach().numpy()}")
print(f"Timescales 1/|λ|: {(1/eigvals.abs()).detach().numpy()}")

# Project delta onto eigenvectors
projections = (eigvecs.T @ delta).detach().numpy()   # (4,)
print(f"\nProjections of (z* - z_0) onto eigenvectors: {projections}")
print(f"Squared projections: {projections**2}")
print(f"Total squared norm: {(projections**2).sum():.4f}  (should be {delta.norm()**2:.4f})")

# Fraction of squared norm in slowest mode
slowest_idx = eigvals.abs().argmin().item()
print(f"\nSlowest eigenvalue: λ={eigvals[slowest_idx]:.5f}, timescale={1/abs(float(eigvals[slowest_idx])):.1f}y")
slow_frac = (projections[slowest_idx]**2) / (projections**2).sum()
print(f"Fraction of ||z*-z0||² in slowest mode: {slow_frac:.6f}  ({slow_frac*100:.3f}%)")

# Drift-only prediction: z(T) - z_0 = (I - exp(M*T)) @ (z* - z_0)
import scipy.linalg
T = 5.0
M_np = M.detach().numpy()
exp_MT = scipy.linalg.expm(M_np * T)
drift_displacement_vec = (np.eye(4) - exp_MT) @ delta.numpy()
print(f"\nDrift-only displacement vector at T=5Y: {drift_displacement_vec}")
print(f"||drift-only displacement|| at T=5Y: {np.linalg.norm(drift_displacement_vec):.4f}")

# Per-mode contribution table
print(f"\nPer-mode contributions:")
print(f"{'Mode':>4} {'lambda':>10} {'timescale':>12} {'|a_i|':>8} {'(1-exp(lT))':>13} {'contribution':>13}")
for i in range(4):
    lam = float(eigvals[i])
    ts  = 1/abs(lam)
    ai  = abs(projections[i])
    factor = abs(1 - np.exp(lam * T))
    contrib = ai * factor
    print(f"{i:>4} {lam:>10.4f} {ts:>12.1f} {ai:>8.4f} {factor:>13.6f} {contrib:>13.6f}")

