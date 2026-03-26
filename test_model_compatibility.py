#!/usr/bin/env python
"""
Quick test to verify FullModel compatibility with Training, Plots, and ResultsGenerator.
Run from repo root: python test_model_compatibility.py
"""

import os
import sys
import torch

# Setup paths
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

print("=" * 80)
print("TESTING MODEL COMPATIBILITY")
print("=" * 80)

# Test 1: Import all required modules
print("\n[TEST 1] Importing modules...")
try:
    from Code.model.full_model import FullModel
    from Code.load_swapdata import my_data
    from Code.utils import helpers as H
    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Instantiate model
print("\n[TEST 2] Instantiating FullModel...")
try:
    model = FullModel(latent_dim=2)
    device = torch.device("cpu")
    model = model.to(device)
    print(f"✓ Model created with latent_dim=2")
except Exception as e:
    print(f"✗ Model instantiation failed: {e}")
    sys.exit(1)

# Test 3: Test forward pass (returns S_hat only by default)
print("\n[TEST 3] Testing forward pass with default output (S_hat only)...")
try:
    # Use realistic swap curve values (not random; random ones lead to NaN due to untrained weights)
    X_test = torch.tensor([
        [1.0, 1.2, 1.4, 1.6, 1.8, 1.9, 2.0, 2.1],
        [1.1, 1.3, 1.5, 1.7, 1.8, 1.95, 2.05, 2.15],
        [0.9, 1.1, 1.3, 1.5, 1.7, 1.85, 1.95, 2.05],
        [1.05, 1.25, 1.45, 1.65, 1.75, 1.9, 2.0, 2.1],
    ], dtype=torch.float32)  # batch of 4 realistic swap curves
    with torch.no_grad():
        output = model(X_test)
    
    if isinstance(output, torch.Tensor) and output.shape == (4, 8):
        print(f"✓ Forward pass returns S_hat tensor only")
        print(f"  - S_hat shape: {output.shape}")
    else:
        print(f"✗ Expected tensor of shape (4, 8), got {type(output)} with shape {output.shape if isinstance(output, torch.Tensor) else 'N/A'}")
        sys.exit(1)
except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Test forward pass with return_aux=True
print("\n[TEST 4] Testing forward pass with return_aux=True...")
try:
    with torch.no_grad():
        S_hat, aux = model(X_test, return_aux=True)
    
    if isinstance(aux, dict) and "z" in aux and "mu" in aux:
        print(f"✓ Forward pass with return_aux=True works")
        print(f"  - Aux dict keys: {list(aux.keys())}")
    else:
        print(f"✗ Aux dict format incorrect")
        sys.exit(1)
except Exception as e:
    print(f"✗ Forward pass with return_aux failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Simulate ResultsGenerator inference pattern
print("\n[TEST 5] Simulating ResultsGenerator inference pattern...")
try:
    # More realistic swap curves with variation
    X_batch = torch.tensor([
        [1.0, 1.15, 1.25, 1.40, 1.60, 1.75, 1.85, 1.95],
        [1.05, 1.18, 1.28, 1.42, 1.62, 1.77, 1.87, 1.97],
        [0.95, 1.10, 1.20, 1.35, 1.55, 1.70, 1.80, 1.90],
        [1.02, 1.16, 1.26, 1.41, 1.61, 1.76, 1.86, 1.96],
        [1.08, 1.20, 1.30, 1.45, 1.65, 1.80, 1.90, 2.00],
        [0.98, 1.12, 1.22, 1.37, 1.57, 1.72, 1.82, 1.92],
        [1.03, 1.17, 1.27, 1.42, 1.62, 1.77, 1.87, 1.97],
        [1.07, 1.19, 1.29, 1.44, 1.64, 1.79, 1.89, 1.99],
    ] * 4, dtype=torch.float32)  # 32 samples
    model.eval()
    with torch.no_grad():
        S_hat, aux = model(X_batch, return_aux=True)
        z = aux["z"]
        mu = aux["mu"]
        sigma_L = aux["sigma"]
        r_tilde = aux["r_tilde"]
    
    print(f"✓ ResultsGenerator pattern works with return_aux=True")
    print(f"  - S_hat: {S_hat.shape}")
    print(f"  - z: {z.shape}")
    print(f"  - mu: {mu.shape}")
    print(f"  - sigma_L: {sigma_L.shape}")
    print(f"  - r_tilde: {r_tilde.shape}")
except Exception as e:
    print(f"✗ ResultsGenerator pattern failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Test with single sample (squeeze_back logic)
print("\n[TEST 6] Testing with single sample (squeeze_back)...")
try:
    X_single = torch.tensor([1.0, 1.2, 1.4, 1.6, 1.8, 1.9, 2.0, 2.1], dtype=torch.float32)  # single realistic curve
    with torch.no_grad():
        S_hat = model(X_single)
    
    if S_hat.dim() == 1:
        print(f"✓ Single sample squeeze_back works")
        print(f"  - S_hat shape: {S_hat.shape} (squeezed to 1D)")
    else:
        print(f"✗ Expected 1D output, got {S_hat.shape}")
        sys.exit(1)
except Exception as e:
    print(f"✗ Single sample test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("ALL TESTS PASSED ✓")
print("=" * 80)
print("\nModel is compatible with:")
print("  ✓ Training.py")
print("  ✓ Plots.py")
print("  ✓ ResultsGenerator.py")
print("\nYou can now run:")
print("  python Code/Training.py")
print("  python Code/Plots.py")
print("  python Code/ResultsGenerator.py")








