"""
Check the mean-reversion strength of the trained stable model.
"""
import torch
import numpy as np
import sys
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Code.model.full_model_stable import FullModel

checkpoint_path = r"C:\Users\Bruger\PycharmProjects\MasterThesis\Figures\TrainingResults\dim4_stable\ep5000\checkpoint_dim4_ep5000.pt"

device = torch.device("cpu")
state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)

model = FullModel(latent_dim=4).double()
model.load_state_dict(state_dict, strict=False)
model.eval()

print("="*70)
print("DRIFT STRENGTH ANALYSIS")
print("="*70)

# Get the drift matrix M
M = model.K.stable_matrix().detach().cpu().numpy()

print("\nDrift matrix M = -(V^T·V + ε·I):")
print(M)

# Compute eigenvalues
eigenvalues = np.linalg.eigvals(M)
eigenvalues_sorted = np.sort(eigenvalues)

print("\nEigenvalues of M (sorted):")
for i, eig in enumerate(eigenvalues_sorted):
    print(f"  λ_{i+1} = {eig:.6f}")

print("\nMean-reversion time scales (1/|λ|):")
for i, eig in enumerate(eigenvalues_sorted):
    if eig < 0:
        tau = -1.0 / eig
        print(f"  τ_{i+1} = {tau:.2f} years  (eigenvalue {eig:.6f})")
    else:
        print(f"  λ_{i+1} = {eig:.6f} is NON-NEGATIVE! ❌ Model is UNSTABLE!")

# Check typical volatility
z_test = torch.zeros(1, 4, dtype=torch.float64)
sigmas, rhos = model.H(z_test)
sigmas_np = sigmas.detach().cpu().numpy()[0]

print("\nVolatility at z=0:")
for i, sig in enumerate(sigmas_np):
    print(f"  σ_{i+1} = {sig:.6f}")

print("\nDrift/Diffusion ratio at z=0:")
print("  (Larger ratio = stronger mean-reversion relative to noise)")
for i in range(4):
    ratio = abs(eigenvalues_sorted[i]) / sigmas_np[i]
    print(f"  |λ_{i+1}|/σ_{i+1} = {ratio:.4f}")
    if ratio < 0.1:
        print(f"    ⚠️  Very weak mean-reversion!")
    elif ratio < 0.5:
        print(f"    ⚠️  Weak mean-reversion")
    elif ratio < 2.0:
        print(f"    ✓  Moderate mean-reversion")
    else:
        print(f"    ✓✓ Strong mean-reversion")

# Simulate a single step to see typical displacement
print("\n" + "="*70)
print("SINGLE-STEP DISPLACEMENT TEST")
print("="*70)

z0 = torch.tensor([[0.1, -0.01, 0.03, -0.01]], dtype=torch.float64)
dt = 1/12  # monthly

mu = model.K(z0).detach().cpu().numpy()[0]
L_obj = model.H(z0)

print(f"\nStarting at z = {z0[0].numpy()}")
print(f"Time step dt = {dt:.4f} (monthly)")

drift_displacement = mu * dt
print(f"\nDrift displacement (μ·dt):")
print(f"  {drift_displacement}")
print(f"  Magnitude: {np.linalg.norm(drift_displacement):.6f}")

# Typical diffusion displacement (RMS)
diff_displacement_rms = sigmas_np * np.sqrt(dt)
print(f"\nTypical diffusion displacement (σ·√dt):")
print(f"  {diff_displacement_rms}")
print(f"  Magnitude: {np.linalg.norm(diff_displacement_rms):.6f}")

ratio = np.linalg.norm(drift_displacement) / np.linalg.norm(diff_displacement_rms)
print(f"\n||Drift|| / ||Diffusion|| = {ratio:.4f}")
if ratio < 0.1:
    print("  ❌ PROBLEM: Drift is much weaker than diffusion!")
    print("     → Paths will wander far from equilibrium")
    print("     → This explains why diffusion_scale is needed")
elif ratio < 0.5:
    print("  ⚠️  WARNING: Drift is weaker than diffusion")
    print("     → Paths may stray outside training support")
else:
    print("  ✓  Drift is comparable to diffusion")
    print("     → Mean-reversion should keep paths controlled")

print("\n" + "="*70)
print("RECOMMENDATION")
print("="*70)

if ratio < 0.5:
    recommended_scale = max(0.2, ratio)
    print(f"\nBased on drift/diffusion ratio, recommended diffusion_scale:")
    print(f"  diffusion_scale = {recommended_scale:.2f}")
    print(f"\nThis will balance drift and diffusion to keep paths stable.")
else:
    print("\nYour model has strong mean-reversion.")
    print("diffusion_scale should not be needed in theory.")
    print("If you still see divergence, check:")
    print("  1. Time step dt (should be small, e.g., 1/12)")
    print("  2. Decoder extrapolation (G, R may be unstable outside training)")

