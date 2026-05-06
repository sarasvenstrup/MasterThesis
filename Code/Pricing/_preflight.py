"""Pre-flight check before running Training_joint.py."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

print("--- imports ---")
from Code import config
config.confirm_variant()
from Code.utils import helpers as H
from Code.load_swapdata import my_data
from Code.model.full_model_stable import FullModel
from Code.Pricing.load_swapvol_ois import load_swaption_vol_data
from Code.Pricing.pricing import bachelier_price_torch, swap_rate_torch
from Code.Simulation.simulate_model import simulate_to_expiry_differentiable
import inspect, torch

sig = inspect.signature(simulate_to_expiry_differentiable)
params = list(sig.parameters.keys())
print(f"simulate_to_expiry_differentiable params: {params}")

print("\n--- load swap data ---")
meta, X, *_ = my_data(use="bbg")
print(f"  X shape : {X.shape}")
print(f"  CCYs    : {sorted(meta['ccy'].unique().tolist())}")

print("\n--- load swaption vol data ---")
df_vol = load_swaption_vol_data(currency="EUR")
print(f"  rows    : {len(df_vol)}")
print(f"  expiries: {sorted(df_vol['option_maturity'].unique().tolist())}")
print(f"  tenors  : {sorted(df_vol['swap_tenor'].unique().tolist())}")

print("\n--- date overlap ---")
import pandas as pd
dates_swap = set(pd.to_datetime(meta[meta["ccy"]=="EUR"]["as_of_date"]).dt.normalize())
df_vol["as_of_date"] = pd.to_datetime(df_vol["as_of_date"]).dt.normalize()
overlap = df_vol[df_vol["as_of_date"].isin(dates_swap)]
print(f"  overlapping vol rows: {len(overlap)} across {overlap['as_of_date'].nunique()} dates")
if len(overlap) == 0:
    print("  WARNING: no date overlap - pricing loss will be DISABLED")

print("\n--- model init ---")
m = FullModel(latent_dim=4)
n_params = sum(p.numel() for p in m.parameters())
print(f"  total params: {n_params}")

print("\n--- per-group param counts ---")
groups = [("H", m.H), ("K", m.K), ("G", m.G), ("encoder", m.encoder), ("R", m.R)]
for name, module in groups:
    n = sum(p.numel() for p in module.parameters())
    print(f"  {name:10s}: {n} params")

print("\n=== ALL CHECKS PASSED — ready to run Training_joint.py ===")

