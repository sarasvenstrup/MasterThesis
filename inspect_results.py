import pandas as pd
import numpy as np

swap = pd.read_csv('Figures/simulations/simulated_swap_curves_bbg_dim2_ep100_paths200_steps120.csv')
latent = pd.read_csv('Figures/simulations/simulated_latent_bbg_dim2_ep100_paths200_steps120.csv')

rate_cols = [c for c in swap.columns if c.startswith('swap_')]

print('=== MEAN SWAP CURVES AT KEY HORIZONS ===')
for t_val in [0.0, 1.0, 2.0, 5.0, 10.0]:
    rows = swap[(swap.time - t_val).abs() < 0.05]
    means = rows[rate_cols].mean()
    parts = ['  '.join(f'{c.replace("swap_","")}={v*100:.2f}%' for c, v in means.items())]
    print(f't={t_val:.0f}Y: {parts[0]}')

print()
print('=== CROSS-PATH STD AT KEY HORIZONS ===')
for t_val in [0.0, 1.0, 5.0, 10.0]:
    rows = swap[(swap.time - t_val).abs() < 0.05]
    stds = rows[rate_cols].std()
    parts = ['  '.join(f'{c.replace("swap_","")}={v*10000:.0f}bp' for c, v in stds.items())]
    print(f't={t_val:.0f}Y: {parts[0]}')

print()
print('=== NEGATIVE RATE FRACTION OVER TIME ===')
for t_val in [1.0, 2.0, 5.0, 10.0]:
    rows = swap[(swap.time - t_val).abs() < 0.05]
    neg = (rows['swap_10Y'] < 0).mean()
    print(f't={t_val:.0f}Y: {neg:.1%} of paths have negative 10Y rate')

print()
print('=== z MEAN REVERSION CHECK ===')
latent_mean = latent.groupby('time')[['z0', 'z1']].mean()
latent_std  = latent.groupby('time')[['z0', 'z1']].std()
for t_val in [0.0, 1.0, 2.0, 5.0, 10.0]:
    idx = np.argmin(np.abs(latent_mean.index.values - t_val))
    print(f't={t_val:.0f}Y: z0={latent_mean.z0.iloc[idx]:+.4f}+/-{latent_std.z0.iloc[idx]:.4f}  '
          f'z1={latent_mean.z1.iloc[idx]:+.4f}+/-{latent_std.z1.iloc[idx]:.4f}')

print()
print('=== SHORT RATE OVER TIME ===')
r_mean = latent.groupby('time')['r'].mean()
r_std  = latent.groupby('time')['r'].std()
for t_val in [0.0, 1.0, 2.0, 5.0, 10.0]:
    idx = np.argmin(np.abs(r_mean.index.values - t_val))
    print(f't={t_val:.0f}Y: r={r_mean.iloc[idx]*100:.3f}% +/- {r_std.iloc[idx]*100:.3f}%')

print()
print('=== SANITY CHECKS ===')
print(f'NaN in swap: {swap[rate_cols].isna().sum().sum()}')
print(f'Inf in swap: {np.isinf(swap[rate_cols].values).sum()}')
print(f'Negative swap rates: {(swap[rate_cols] < 0).sum().sum()} / {swap[rate_cols].size}')
print(f'Swap rates > 20%: {(swap[rate_cols] > 0.20).sum().sum()}')


