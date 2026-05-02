import pandas as pd
import numpy as np
import os

base = r'C:\Users\Bruger\PycharmProjects\MasterThesis\Figures\OOSResults\Roll'

for dim in [3, 4]:
    for variant in ['stable', 'baseline']:
        folder = os.path.join(base, f'OOS_roll_dim{dim}_{variant}', 'train5Y_test6M_step6M')
        if not os.path.isdir(folder):
            continue
        for ep in os.listdir(folder):
            ep_path = os.path.join(folder, ep)
            csvs = [f for f in os.listdir(ep_path) if f.endswith('.csv')]
            for csv in csvs:
                path = os.path.join(ep_path, csv)
                df = pd.read_csv(path)
                print(f'=== dim{dim} {variant} {ep} ===')
                print(f'  Mean OOS:   {df["avg_rmse_bps"].mean():.2f} bps')
                print(f'  Median OOS: {df["avg_rmse_bps"].median():.2f} bps')
                worst_idx = df["avg_rmse_bps"].idxmax()
                print(f'  Max OOS:    {df["avg_rmse_bps"].max():.2f} bps  ({df.loc[worst_idx,"test_start"]})')
                print(df[['test_start','avg_rmse_bps']].to_string(index=False))
                print()

