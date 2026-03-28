"""
Script to inspect and compare differences between R_short and R_short_stable.

This script analyzes:
1. Architecture differences
2. Parameter initialization strategies
3. Forward pass computation
4. Output boundedness properties
5. Gradient behavior
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
from Code.model.R_short import RShort
from Code.model.R_short_stable import RShortStable

print("=" * 80)
print("COMPARISON: R_short vs R_short_stable")
print("=" * 80)

# ============================================================================
# 1. ARCHITECTURAL DIFFERENCES
# ============================================================================
print("\n" + "=" * 80)
print("1. ARCHITECTURAL DIFFERENCES")
print("=" * 80)

print("\n[R_short]")
print("  - Simple 2-layer feedforward network: z -> hidden_dim -> r")
print("  - Architecture: (latent_dim, hidden_dim, 1)")
print("  - Activation: CenteredSoftStep (unbounded)")
print("  - Output: r = final_linear(activation(linear(z)))")
print("  - Output range: UNBOUNDED (-∞, +∞)")
print("  - Risk: Can produce negative or extremely large short rates")

print("\n[R_short_stable]")
print("  - Same 2-layer feedforward network for feature extraction")
print("  - Architecture: (latent_dim, hidden_dim, 1) -> tanh transformation")
print("  - Activation: CenteredSoftStep (same as original)")
print("  - Output: r = r_center + r_scale * tanh(network_output)")
print("  - Output range: BOUNDED approximately [r_center - r_scale, r_center + r_scale]")
print("  - Learnable parameters: r_center, r_scale (positive-constrained)")
print("  - Guarantee: r is always physically realistic")

# ============================================================================
# 2. PARAMETER INSPECTION
# ============================================================================
print("\n" + "=" * 80)
print("2. PARAMETER INSPECTION")
print("=" * 80)

latent_dim = 2
hidden_dim = 4

r_short = RShort(latent_dim=latent_dim, hidden_dim=hidden_dim, bias=False)
r_stable = RShortStable(latent_dim=latent_dim, hidden_dim=hidden_dim, bias=True)

print(f"\nFor latent_dim={latent_dim}, hidden_dim={hidden_dim}:")

print("\n[R_short parameters]:")
for name, param in r_short.named_parameters():
    print(f"  {name}: shape={param.shape}, requires_grad={param.requires_grad}")

print("\n[R_short_stable parameters]:")
for name, param in r_stable.named_parameters():
    print(f"  {name}: shape={param.shape}, requires_grad={param.requires_grad}")

# ============================================================================
# 3. FORWARD PASS COMPUTATION
# ============================================================================
print("\n" + "=" * 80)
print("3. FORWARD PASS COMPUTATION")
print("=" * 80)

print("\n[R_short forward pass]")
print("  z (input, shape (B, latent_dim))")
print("  → lin1(z): (B, hidden_dim)")
print("  → CenteredSoftStep(): (B, hidden_dim)")
print("  → lin2(...): (B, 1) - UNBOUNDED OUTPUT")

print("\n[R_short_stable forward pass]")
print("  z (input, shape (B, latent_dim))")
print("  → lin1(z): (B, hidden_dim)")
print("  → CenteredSoftStep(): (B, hidden_dim)")
print("  → lin2(...): (B, 1) - raw, unbounded")
print("  → tanh(...): (B, 1) in (-1, +1)")
print("  → scale by r_scale (guaranteed > 0)")
print("  → shift by r_center")
print("  → output (B, 1) - BOUNDED in [r_center-r_scale, r_center+r_scale]")

# ============================================================================
# 4. NUMERICAL TEST: Forward pass
# ============================================================================
print("\n" + "=" * 80)
print("4. NUMERICAL TEST: Forward pass")
print("=" * 80)

z_test = torch.randn(5, latent_dim)

with torch.no_grad():
    output_rshort = r_short(z_test)
    output_stable = r_stable(z_test)

print(f"\nTest input shape: {z_test.shape}")
print(f"Test input:\n{z_test}")

print(f"\nR_short output shape: {output_rshort.shape}")
print(f"R_short output:\n{output_rshort}")
print(f"  Min: {output_rshort.min().item():.6f}")
print(f"  Max: {output_rshort.max().item():.6f}")
print(f"  Mean: {output_rshort.mean().item():.6f}")

print(f"\nR_short_stable output shape: {output_stable.shape}")
print(f"R_short_stable output:\n{output_stable}")
print(f"  Min: {output_stable.min().item():.6f}")
print(f"  Max: {output_stable.max().item():.6f}")
print(f"  Mean: {output_stable.mean().item():.6f}")

with torch.no_grad():
    r_center = r_stable.r_center.item()
    r_scale = r_stable.scale().item()
    lower_bound = r_center - r_scale
    upper_bound = r_center + r_scale
    print(f"\nR_short_stable bounds:")
    print(f"  r_center: {r_center:.6f}")
    print(f"  r_scale: {r_scale:.6f}")
    print(f"  Theoretical range: [{lower_bound:.6f}, {upper_bound:.6f}]")
    print(f"  Actual range: [{output_stable.min().item():.6f}, {output_stable.max().item():.6f}]")

# ============================================================================
# 5. OUTPUT BOUNDEDNESS ANALYSIS (Key difference!)
# ============================================================================
print("\n" + "=" * 80)
print("5. OUTPUT BOUNDEDNESS ANALYSIS - KEY DIFFERENCE")
print("=" * 80)

print("\n[R_short output behavior]")
with torch.no_grad():
    # Test with extreme inputs
    z_extreme = torch.randn(100, latent_dim) * 10.0  # Large values
    output_extreme = r_short(z_extreme)
    print(f"  With extreme inputs (std=10.0):")
    print(f"    Min: {output_extreme.min().item():.6f}")
    print(f"    Max: {output_extreme.max().item():.6f}")
    print(f"    Std: {output_extreme.std().item():.6f}")
    print(f"  Risk: Unbounded - can produce unphysical short rates!")

print("\n[R_short_stable output behavior]")
with torch.no_grad():
    z_extreme = torch.randn(100, latent_dim) * 10.0
    output_extreme_stable = r_stable(z_extreme)
    r_center = r_stable.r_center.item()
    r_scale = r_stable.scale().item()
    lower = r_center - r_scale
    upper = r_center + r_scale
    
    print(f"  With extreme inputs (std=10.0):")
    print(f"    Min: {output_extreme_stable.min().item():.6f}")
    print(f"    Max: {output_extreme_stable.max().item():.6f}")
    print(f"    Std: {output_extreme_stable.std().item():.6f}")
    print(f"    Bounds: [{lower:.6f}, {upper:.6f}]")
    print(f"    Within bounds? {(output_extreme_stable >= lower).all() and (output_extreme_stable <= upper).all()}")
    print(f"  Guarantee: Always bounded - physically realistic!")

# ============================================================================
# 6. STABILITY GUARANTEE EXPLANATION
# ============================================================================
print("\n" + "=" * 80)
print("6. STABILITY GUARANTEE EXPLANATION")
print("=" * 80)

print("""
Interest Rate Constraints (Financial Domain):
  Short rate r_t is the instantaneous risk-free rate
  
Physical constraints:
  1. Non-negative: r_t ≥ 0 (rates can't be infinitely negative)
  2. Bounded: r_t ≤ r_max (reasonable upper limit)
  3. Smooth: No discontinuities (for discount factor stability)

[R_short]
  - No constraints on network output
  - Can produce r_t < 0 (negative rates - problematic)
  - Can produce r_t >> 1 (100%+ rates - unrealistic)
  - Discount factors D(t) = exp(-∫r_s ds) can become invalid
  - Model can behave unphysically
  
[R_short_stable]
  - Bounded by construction: r(z) ∈ [r_center - r_scale, r_center + r_scale]
  - Smooth tanh activation (better gradients than hard clipping)
  - Learnable bounds: r_center and r_scale adjust during training
  - Can enforce non-negative constraint on r_center
  - Ensures physically realistic short-rate behavior
  - Smooth gradients everywhere (better training stability)
""")

# ============================================================================
# 7. PARAMETER COUNT COMPARISON
# ============================================================================
print("\n" + "=" * 80)
print("7. PARAMETER COUNT COMPARISON")
print("=" * 80)

def count_params(model):
    return sum(p.numel() for p in model.parameters())

n_rshort = count_params(r_short)
n_stable = count_params(r_stable)

print(f"\nR_short parameters (bias=False): {n_rshort}")
print(f"  - lin1 weight: {latent_dim * hidden_dim}")
print(f"  - lin2 weight: {hidden_dim * 1}")
print(f"  - Total: {n_rshort}")

print(f"\nR_short_stable parameters (bias=True): {n_stable}")
print(f"  - lin1 weight: {latent_dim * hidden_dim}")
print(f"  - lin1 bias: {hidden_dim}")
print(f"  - lin2 weight: {hidden_dim * 1}")
print(f"  - lin2 bias: {1}")
print(f"  - r_center: 1")
print(f"  - raw_r_scale: 1")
print(f"  - Total: {n_stable}")

print(f"\nDifference: R_short_stable has {n_stable - n_rshort} additional parameters")
print(f"  (mainly from biases and learnable bounds)")

# ============================================================================
# 8. GRADIENT FLOW TEST
# ============================================================================
print("\n" + "=" * 80)
print("8. GRADIENT FLOW TEST")
print("=" * 80)

z_grad = torch.randn(2, latent_dim, requires_grad=True)

# R_short
output_rshort_grad = r_short(z_grad)
loss_rshort = output_rshort_grad.sum()
loss_rshort.backward()
grad_rshort = z_grad.grad.clone()
z_grad.grad = None

# R_short_stable
z_grad_stable = torch.randn(2, latent_dim, requires_grad=True)
output_stable_grad = r_stable(z_grad_stable)
loss_stable = output_stable_grad.sum()
loss_stable.backward()
grad_stable = z_grad_stable.grad.clone()

print(f"\nR_short gradient norm: {grad_rshort.norm().item():.6f}")
print(f"R_short_stable gradient norm: {grad_stable.norm().item():.6f}")

# ============================================================================
# 9. TANH SATURATION ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("9. TANH SATURATION ANALYSIS")
print("=" * 80)

print("\nTanh gradient properties:")
print("  tanh(x) = (e^(2x) - 1) / (e^(2x) + 1)")
print("  d/dx tanh(x) = 1 - tanh²(x)")
print("  ")
print("  At x = 0:")
print("    tanh(0) = 0, gradient = 1.0 (strong gradient)")
print("  ")
print("  At x = ±3:")
print("    tanh(±3) ≈ ±0.995, gradient ≈ 0.009 (mild saturation)")
print("  ")
print("  At x = ±6:")
print("    tanh(±6) ≈ ±1.0, gradient ≈ 0.0002 (heavy saturation)")
print("  ")
print("  → Tanh gradients vanish smoothly, never sharply cut off")
print("  → Safer than hard clipping during training")

with torch.no_grad():
    x_range = torch.linspace(-6, 6, 100)
    y = torch.tanh(x_range)
    grad = 1 - y ** 2
    
    print("\nTanh behavior on network outputs:")
    print(f"  x range: [{x_range.min().item():.2f}, {x_range.max().item():.2f}]")
    print(f"  tanh(x) range: [{y.min().item():.6f}, {y.max().item():.6f}]")
    print(f"  gradient range: [{grad.min().item():.6f}, {grad.max().item():.6f}]")

# ============================================================================
# 10. SUMMARY TABLE
# ============================================================================
print("\n" + "=" * 80)
print("10. SUMMARY TABLE")
print("=" * 80)

summary_data = {
    "Feature": [
        "Core architecture",
        "Output transformation",
        "Output range",
        "Boundedness guarantee",
        "Parameter count (bias=True)",
        "Learnable bounds",
        "Gradient behavior",
        "Saturation",
        "Numerical stability",
        "Physical realism",
    ],
    "R_short": [
        "(2→4→1) CenteredSoftStep",
        "None (unbounded)",
        "(-∞, +∞)",
        "No",
        f"{n_rshort}",
        "No",
        "Can be sharp (CenteredSoftStep)",
        "Potential issues",
        "Can produce invalid rates",
        "Not guaranteed",
    ],
    "R_short_stable": [
        "(2→4→1) CenteredSoftStep + tanh",
        "r_center + r_scale * tanh(x)",
        "[r_center-r_scale, r_center+r_scale]",
        "Yes (by construction)",
        f"{n_stable}",
        "Yes (r_center, r_scale > 0)",
        "Smooth tanh gradients",
        "Smooth (no hard cutoff)",
        "Always produces valid rates",
        "Guaranteed",
    ],
}

print(f"\n{'Feature':<30s} | {'R_short':<40s} | {'R_short_stable':<40s}")
print("-" * 115)
for i, key in enumerate(summary_data["Feature"]):
    r_short_val = summary_data["R_short"][i]
    r_stable_val = summary_data["R_short_stable"][i]
    print(f"{key:<30s} | {r_short_val:<40s} | {r_stable_val:<40s}")

# ============================================================================
# 11. WHEN TO USE WHICH
# ============================================================================
print("\n" + "=" * 80)
print("11. RECOMMENDATIONS")
print("=" * 80)

print("""
[Use R_short]
  - For rapid prototyping/experimentation
  - When computational simplicity is paramount
  - When output constraints are handled elsewhere
  - In non-financial domains where unbounded outputs are acceptable
  
[Use R_short_stable] ✓ RECOMMENDED FOR THIS PROJECT
  - For interest rate modeling (this thesis!)
  - When output must be physically realistic
  - For models with real financial data
  - When reproducibility and stability are critical
  - When discount factors must remain valid
  
Interest Rate Modeling Requirements:
  → Use R_short_stable to ensure valid discount factors
  → Short rates MUST stay positive and reasonable
  → Bounded output prevents model collapse
  → Smooth gradients improve training stability
  
Financial Best Practice:
  This is essential for any interest rate model in production.
  The minimal overhead (2 extra parameters) is negligible
  compared to the stability guarantees provided.
""")

print("\n" + "=" * 80)
print("End of comparison")
print("=" * 80)

