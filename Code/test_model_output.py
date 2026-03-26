"""
Test script to verify FullModel output includes all necessary variables.
Run from repo root: python Code/test_model_output.py
"""
import torch
import sys
import os

try:
    REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
    REPO_ROOT = os.path.dirname(REPO_ROOT)
except NameError:
    REPO_ROOT = os.getcwd()

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Code.model.full_model import FullModel

# Create dummy data
B = 2  # batch size
model = FullModel(latent_dim=3)
model.eval()

# Create random input swap curve
S_in = torch.randn(B, 8) * 0.01 + 0.02  # (B, 8) - random swap rates around 2%

print("=" * 70)
print("Testing FullModel Output Variables")
print("=" * 70)

# Test 1: Default output (S_hat only)
print("\n[1] Default output (return_aux=False):")
with torch.no_grad():
    S_hat = model(S_in)
print(f"  ✓ S_hat shape: {S_hat.shape}")
print(f"  ✓ S_hat dtype: {S_hat.dtype}")

# Test 2: Full output with aux_dict
print("\n[2] Full output (return_aux=True):")
with torch.no_grad():
    S_hat, aux = model(S_in, return_aux=True)

print(f"  ✓ S_hat shape: {S_hat.shape}")
print(f"\n  Available variables in aux_dict:")
for key in sorted(aux.keys()):
    val = aux[key]
    if isinstance(val, torch.Tensor):
        print(f"    • {key:15s} : {str(val.shape):20s} dtype={val.dtype}")
    elif isinstance(val, dict):
        print(f"    • {key:15s} : dict with {len(val)} keys")
    else:
        print(f"    • {key:15s} : {type(val).__name__}")

# Test 3: With arbitrage checks
print("\n[3] With arbitrage diagnostics (do_arb_checks=True):")
with torch.no_grad():
    S_hat, aux = model(S_in, return_aux=True, do_arb_checks=True)

if aux['arb'] is not None:
    print(f"  ✓ Arbitrage diagnostics computed:")
    for key in sorted(aux['arb'].keys()):
        val = aux['arb'][key]
        if isinstance(val, torch.Tensor):
            print(f"    • {key:20s} : {str(val.shape):20s}")
        else:
            print(f"    • {key:20s} : {type(val).__name__}")
else:
    print(f"  ! No arbitrage diagnostics available")

# Test 4: Verify all expected keys are present
print("\n[4] Verification of expected keys:")
expected_keys = [
    'z', 'P_mkt', 'P_full', 'A_vals', 'B_vals', 'G_vals',
    'mu', 'sigma', 'r_tilde', 'alpha', 'beta', 'gamma',
    'tau_grid', 'arb'
]
for key in expected_keys:
    if key in aux:
        print(f"  ✓ {key}")
    else:
        print(f"  ✗ {key} MISSING!")

# Test 5: Verify shapes are consistent
print("\n[5] Shape consistency checks:")
tau_max = model.tau_max
print(f"  Model tau_max: {tau_max}")
print(f"  Latent dim: {model.latent_dim}")

checks = [
    ('z', (B, model.latent_dim)),
    ('mu', (B, model.latent_dim)),
    ('sigma', (B, model.latent_dim, model.latent_dim)),
    ('r_tilde', (B,)),
    ('P_full', (B, tau_max + 1)),
    ('P_mkt', (B, tau_max)),
    ('A_vals', (B, tau_max + 1)),
    ('B_vals', (B, tau_max + 1)),
    ('G_vals', (B, tau_max + 1)),
    ('alpha', (B, tau_max + 1)),
    ('beta', (B, tau_max + 1)),
    ('gamma', (B, tau_max + 1)),
    ('tau_grid', (tau_max + 1,)),
]

all_correct = True
for key, expected_shape in checks:
    actual_shape = aux[key].shape
    if actual_shape == expected_shape:
        print(f"  ✓ {key:15s} : {str(actual_shape)}")
    else:
        print(f"  ✗ {key:15s} : expected {expected_shape}, got {actual_shape}")
        all_correct = False

print("\n" + "=" * 70)
if all_correct:
    print("✓ All tests passed!")
else:
    print("✗ Some shape checks failed")
print("=" * 70)

