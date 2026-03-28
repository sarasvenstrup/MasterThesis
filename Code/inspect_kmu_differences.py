"""
Script to inspect and compare differences between K_mu and K_mu_stable.

This script analyzes:
1. Architecture differences
2. Parameter initialization strategies
3. Forward pass computation
4. Numerical stability properties
5. Eigenvalue behavior
"""

import sys
import os

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from Code.model.K_mu import KMu
from Code.model.K_mu_stable import KMuStable

print("=" * 80)
print("COMPARISON: K_mu vs K_mu_stable")
print("=" * 80)

# ============================================================================
# 1. ARCHITECTURAL DIFFERENCES
# ============================================================================
print("\n" + "=" * 80)
print("1. ARCHITECTURAL DIFFERENCES")
print("=" * 80)

print("\n[K_mu]")
print("  - Simple linear transformation: mu(z) = M*z + N")
print("  - M: direct (latent_dim x latent_dim) matrix learned directly")
print("  - N: learned bias vector")
print("  - No stability guarantees on M eigenvalues")

print("\n[K_mu_stable]")
print("  - Parametrized linear transformation: mu(z) = M*z + N")
print("  - M = -(V^T*V + eps*I), where V is learned")
print("  - Guarantees: M has strictly NEGATIVE eigenvalues (mean-reversion)")
print("  - V: orthogonal initialization")
print("  - N: learned bias vector")
print("  - epsilon: regularization parameter (default 1e-3)")

# ============================================================================
# 2. PARAMETER INSPECTION
# ============================================================================
print("\n" + "=" * 80)
print("2. PARAMETER INSPECTION")
print("=" * 80)

latent_dim = 2

kmu = KMu(latent_dim=latent_dim, bias=True)
kmu_stable = KMuStable(latent_dim=latent_dim, bias=True, epsilon=1e-3)

print(f"\nFor latent_dim={latent_dim}:")

print("\n[K_mu parameters]:")
for name, param in kmu.named_parameters():
    print(f"  {name}: shape={param.shape}, requires_grad={param.requires_grad}")

print("\n[K_mu_stable parameters]:")
for name, param in kmu_stable.named_parameters():
    print(f"  {name}: shape={param.shape}, requires_grad={param.requires_grad}")

# ============================================================================
# 3. FORWARD PASS COMPUTATION
# ============================================================================
print("\n" + "=" * 80)
print("3. FORWARD PASS COMPUTATION")
print("=" * 80)

print("\n[K_mu forward pass]")
print("  z (input, shape (B, d))")
print("  → lin(z) = z @ weight.T + bias")
print("  → output (B, d)")

print("\n[K_mu_stable forward pass]")
print("  z (input, shape (B, d))")
print("  → compute M = -(V^T @ V + eps*I)")
print("  → mu = z @ M.T + N")
print("  → output (B, d)")

# ============================================================================
# 4. NUMERICAL TEST: Forward pass equivalence
# ============================================================================
print("\n" + "=" * 80)
print("4. NUMERICAL TEST: Forward pass")
print("=" * 80)

z_test = torch.randn(5, latent_dim)

with torch.no_grad():
    output_kmu = kmu(z_test)
    output_stable = kmu_stable(z_test)

print(f"\nTest input shape: {z_test.shape}")
print(f"K_mu output shape: {output_kmu.shape}")
print(f"K_mu_stable output shape: {output_stable.shape}")
print(f"\nK_mu output:\n{output_kmu}")
print(f"\nK_mu_stable output:\n{output_stable}")

# ============================================================================
# 5. EIGENVALUE ANALYSIS (Key difference!)
# ============================================================================
print("\n" + "=" * 80)
print("5. EIGENVALUE ANALYSIS - KEY DIFFERENCE")
print("=" * 80)

print("\n[K_mu eigenvalues]:")
with torch.no_grad():
    M_kmu = kmu.lin.weight  # Direct weight matrix
    eigs_kmu = torch.linalg.eigvals(M_kmu).numpy()
    print(f"  M matrix:\n{M_kmu.numpy()}")
    print(f"  Eigenvalues: {eigs_kmu}")
    print(f"  All negative? {np.all(eigs_kmu.real < 0)}")
    print(f"  Max eigenvalue (for stability): {np.max(eigs_kmu.real):.6f}")

print("\n[K_mu_stable eigenvalues]:")
with torch.no_grad():
    M_stable = kmu_stable.stable_matrix()
    eigs_stable = torch.linalg.eigvals(M_stable).numpy()
    print(f"  M matrix:\n{M_stable.numpy()}")
    print(f"  Eigenvalues: {eigs_stable}")
    print(f"  All negative? {np.all(eigs_stable.real < 0)}")
    print(f"  Max eigenvalue (for stability): {np.max(eigs_stable.real):.6f}")

# ============================================================================
# 6. STABILITY GUARANTEE EXPLANATION
# ============================================================================
print("\n" + "=" * 80)
print("6. STABILITY GUARANTEE EXPLANATION")
print("=" * 80)

print("""
Mean-reversion property:
  The Langevin equation: dz_t = mu(z_t) dt + sigma(z_t) dW_t
  
  If mu(z) = M*z + N, mean-reversion to z* is guaranteed when:
    - All eigenvalues of M have NEGATIVE real parts
    - This ensures z_t → z* as t → ∞
  
[K_mu]
  - M is unconstrained: eigenvalues can be positive, negative, or complex
  - NO guarantee of mean-reversion
  - Model can exhibit explosive behavior
  - Depends entirely on learning dynamics
  
[K_mu_stable]
  - M = -(V^T*V + eps*I) by construction
  - V^T*V is positive semi-definite (PSD)
  - -(V^T*V + eps*I) is negative definite
  - ALL eigenvalues are strictly negative
  - GUARANTEES mean-reversion by mathematical construction
  - Stable across different initializations and training runs
""")

# ============================================================================
# 7. PARAMETER COUNT COMPARISON
# ============================================================================
print("\n" + "=" * 80)
print("7. PARAMETER COUNT COMPARISON")
print("=" * 80)

def count_params(model):
    return sum(p.numel() for p in model.parameters())

n_kmu = count_params(kmu)
n_stable = count_params(kmu_stable)

print(f"\nK_mu parameters: {n_kmu}")
print(f"  - Weight: {latent_dim * latent_dim}")
print(f"  - Bias: {latent_dim}")
print(f"  - Total: {n_kmu}")

print(f"\nK_mu_stable parameters: {n_stable}")
print(f"  - V: {latent_dim * latent_dim}")
print(f"  - Bias N: {latent_dim}")
print(f"  - Total: {n_stable}")

print(f"\nBoth have same parameter count: {n_kmu == n_stable}")

# ============================================================================
# 8. GRADIENT FLOW TEST
# ============================================================================
print("\n" + "=" * 80)
print("8. GRADIENT FLOW TEST")
print("=" * 80)

z_grad = torch.randn(2, latent_dim, requires_grad=True)

# K_mu
output_kmu_grad = kmu(z_grad)
loss_kmu = output_kmu_grad.sum()
loss_kmu.backward()
grad_kmu = z_grad.grad.clone()
z_grad.grad = None

# K_mu_stable
output_stable_grad = kmu_stable(z_grad)
loss_stable = output_stable_grad.sum()
loss_stable.backward()
grad_stable = z_grad.grad.clone()

print(f"\nK_mu gradient norm: {grad_kmu.norm().item():.6f}")
print(f"K_mu_stable gradient norm: {grad_stable.norm().item():.6f}")

# ============================================================================
# 9. SUMMARY TABLE
# ============================================================================
print("\n" + "=" * 80)
print("9. SUMMARY TABLE")
print("=" * 80)

summary_data = {
    "Feature": [
        "Parameterization",
        "Eigenvalue constraint",
        "Stability guarantee",
        "Parameter count",
        "Bias support",
        "Extra hyperparameters",
        "Computational cost",
        "Mean-reversion",
    ],
    "K_mu": [
        "Direct M",
        "None",
        "No",
        f"{n_kmu}",
        "Yes",
        "None",
        "O(d²)",
        "Not guaranteed",
    ],
    "K_mu_stable": [
        "M = -(V^T*V + eps*I)",
        "Strictly negative",
        "Yes",
        f"{n_stable}",
        "Yes",
        "epsilon (1e-3)",
        "O(d²) [same]",
        "Guaranteed",
    ],
}

for key in summary_data["Feature"]:
    idx = summary_data["Feature"].index(key)
    print(f"{key:30s} | {summary_data['K_mu'][idx]:25s} | {summary_data['K_mu_stable'][idx]:25s}")

# ============================================================================
# 10. WHEN TO USE WHICH
# ============================================================================
print("\n" + "=" * 80)
print("10. RECOMMENDATIONS")
print("=" * 80)

print("""
[Use K_mu]
  - When stability guarantees are NOT critical
  - For rapid experimentation/prototyping
  - When model behavior is well-understood from prior work
  - When computational simplicity is paramount
  
[Use K_mu_stable] ✓ RECOMMENDED FOR THIS PROJECT
  - When mean-reversion is critical (interest rate models!)
  - When you want guaranteed numerical stability
  - For production models with real financial data
  - When reproducibility across runs is important
  - When eigenvalue constraints are a requirement
  
This is a MASTER'S THESIS on interest rate modeling:
  → Use K_mu_stable to ensure realistic financial behavior
  → Interest rates MUST revert to long-term mean
  → Stability guarantees are worth the minimal overhead
""")

print("\n" + "=" * 80)
print("End of comparison")
print("=" * 80)

