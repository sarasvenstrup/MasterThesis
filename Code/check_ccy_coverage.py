import pandas as pd
import numpy as np
import os

base = r'C:\Users\Bruger\PycharmProjects\MasterThesis\Figures\OOSResults\Roll'
path = os.path.join(base, 'OOS_roll_dim4_stable', 'train5Y_test6M_step6M', 'ep2500',
                    'oos_rolling_bbg_dim4_train5Y_test6M_step6M.csv')

df = pd.read_csv(path)
ccy_cols = [c for c in df.columns if c.startswith('rmse_bps_')]
ccys = [c.replace('rmse_bps_', '') for c in ccy_cols]

print("Per-currency NaN count across 18 windows:")
for col, ccy in zip(ccy_cols, ccys):
    n_nan = df[col].isna().sum()
    n_total = len(df)
    print(f"  {ccy}: {n_nan}/{n_total} windows missing")

# Which currencies are present in ALL windows?
always_present = [ccy for col, ccy in zip(ccy_cols, ccys) if df[col].isna().sum() == 0]
print(f"\nCurrencies present in ALL windows: {always_present}")

# Recompute avg over fixed set
fixed_cols = [f'rmse_bps_{c}' for c in always_present]
df['avg_rmse_bps_fixed'] = df[fixed_cols].mean(axis=1)

print("\nOriginal avg vs Fixed-set avg (only currencies in all windows):")
print(df[['test_start', 'avg_rmse_bps', 'avg_rmse_bps_fixed']].to_string(index=False))

print(f"\nOverall mean  - original: {df['avg_rmse_bps'].mean():.2f} bps")
print(f"Overall mean  - fixed:    {df['avg_rmse_bps_fixed'].mean():.2f} bps")
print(f"Overall median - original: {df['avg_rmse_bps'].median():.2f} bps")
print(f"Overall median - fixed:    {df['avg_rmse_bps_fixed'].median():.2f} bps")

