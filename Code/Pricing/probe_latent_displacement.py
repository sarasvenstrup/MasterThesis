"""
Latent displacement diagnostic:
Simulate to T=5Y from dim4_baseline and dim4_stable and compare:
  - distribution of z_T vs training z cloud
  - ||z_T - z_0|| statistics
  - fraction of paths that produce finite decoded curves

This is the single figure that cleanly separates:
  "SDE is broken" (baseline) from "decoder is brittle" (stable)
"""
import os, sys, math
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

sys.path.insert(0, os.path.abspath('../..'))
os.environ.setdefault("SKIP_VARIANT_CONFIRM", "1")

from Code import config
config.confirm_variant()

from Code.load_swapdata import my_data
from Code.model.full_model import FullModel as FM_base
from Code.model.full_model_stable import FullModel as FM_stable
from Code.model.sigma_matrix import L_from_sigmas_rhos

# ─── config ───────────────────────────────────────────────────────────────
EXPIRY  = 5          # years
DT      = 1/12       # monthly steps
N_PATHS = 1000
N_STEPS = int(round(EXPIRY / DT))
DIM     = 4
SEED    = 42
OUT_DIR = "../../Figures/Pricing"

CHECKPOINTS = {
    "dim4_baseline": ("Figures/TrainingResults/dim4_baseline/ep5000/checkpoint_dim4_ep5000.pt", False),
    "dim4_stable":   ("Figures/TrainingResults/dim4_stable/ep5000/checkpoint_dim4_ep5000.pt",   True),
}

# ─── helpers ──────────────────────────────────────────────────────────────

def load_model(path, is_stable, dim=4):
    raw = torch.load(path, map_location='cpu')
    cls = FM_stable if is_stable else FM_base
    m = cls(latent_dim=dim)
    m.load_state_dict(raw)
    m.eval()
    return m

@torch.no_grad()
def simulate(model, z0, n_paths, n_steps, dt, seed=42):
    """Simple Euler-Maruyama, returns z_T (n_paths, d)."""
    torch.manual_seed(seed)
    d = z0.shape[1]
    sqrt_dt = math.sqrt(dt)
    z = z0.expand(n_paths, -1).clone().float()
    for _ in range(n_steps):
        sigmas, rhos = model.H(z)
        L = L_from_sigmas_rhos(sigmas, rhos, validate=False)
        dW = torch.randn(n_paths, d) * sqrt_dt
        shock = torch.bmm(L, dW.unsqueeze(-1)).squeeze(-1)
        drift = model.K(z) * dt
        z = z + drift + shock
    return z   # (n_paths, d)

@torch.no_grad()
def finite_frac(model, z):
    _, aux = model.decode_from_z(z, tau=None, return_aux=True)
    P = aux['P_full']
    return float(torch.isfinite(P).all(dim=1).float().mean())

def confidence_ellipse(x, y, ax, n_std=2.0, **kwargs):
    """Draw covariance ellipse for 2-D data."""
    if len(x) < 3:
        return
    cov = np.cov(x, y)
    pearson = cov[0, 1] / (np.sqrt(cov[0, 0]) * np.sqrt(cov[1, 1]) + 1e-12)
    rx, ry = np.sqrt(1 + pearson), np.sqrt(1 - pearson)
    ell = Ellipse((0, 0), width=rx*2, height=ry*2, **kwargs)
    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_x, mean_y = np.mean(x), np.mean(y)
    t = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)
    ell.set_transform(t + ax.transData)
    return ax.add_patch(ell)

# ─── main ─────────────────────────────────────────────────────────────────

device = torch.device('cpu')
meta, X_tensor, *_ = my_data(use='bbg')
X_eur = X_tensor[meta['ccy'] == 'EUR'].float()

# Training z cloud (dim4_stable — use stable model for training cloud ref)
model_stab = load_model(CHECKPOINTS["dim4_stable"][0], True)
with torch.no_grad():
    z_train = model_stab.encoder(X_eur)
z_train_np = z_train.numpy()

print(f"Training cloud: mean={z_train_np.mean(0)}, std={z_train_np.std(0)}")

# Pick a representative starting point (median-ish EUR date)
mid_idx = len(X_eur) // 2
x0 = X_eur[mid_idx:mid_idx+1]

results = {}
for label, (path, is_stable) in CHECKPOINTS.items():
    print(f"\n--- {label} ---")
    model = load_model(path, is_stable)
    with torch.no_grad():
        z0 = model.encoder(x0)
    print(f"  z0 = {z0.numpy().flatten()}")

    z_T = simulate(model, z0, N_PATHS, N_STEPS, DT, seed=SEED)
    disp = (z_T - z0).norm(dim=1).numpy()
    ff   = finite_frac(model, z_T)

    print(f"  ||z_T - z0||  mean={disp.mean():.3f}  median={np.median(disp):.3f}  p95={np.percentile(disp,95):.3f}  max={disp.max():.3f}")
    print(f"  Finite decoded curves at T=5Y: {ff:.1%}")

    # Also get training z for this specific model (baseline uses its own encoder)
    with torch.no_grad():
        z_train_this = model.encoder(X_eur).numpy()

    results[label] = {
        "z0": z0.numpy().flatten(),
        "z_T": z_T.numpy(),
        "z_train": z_train_this,
        "disp": disp,
        "finite_frac": ff,
        "is_stable": is_stable,
    }

# ─── Displacement statistics table ────────────────────────────────────────
print("\n" + "="*70)
print("DISPLACEMENT SUMMARY  (dim4, T=5Y, 1000 paths, dt=1/12)")
print("="*70)
print(f"{'Model':<20} {'Mean':>8} {'Median':>8} {'p95':>8} {'Max':>8} {'Finite%':>9}")
print("-"*70)
for label, r in results.items():
    d = r['disp']
    print(f"{label:<20} {d.mean():>8.3f} {np.median(d):>8.3f} {np.percentile(d,95):>8.3f} {d.max():>8.3f} {r['finite_frac']:>8.1%}")

# ─── Figure: z_T cloud vs training cloud (first 2 dims) ──────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=150)

titles = {"dim4_baseline": "dim4 Baseline  (SDE broken, decoder permissive)",
          "dim4_stable":   "dim4 Stable    (SDE healthy, decoder brittle)"}
colors  = {"train": "#2196F3", "sim": "#F44336"}

for ax, (label, r) in zip(axes, results.items()):
    zt   = r['z_train']
    z_T  = r['z_T']
    z0   = r['z0']

    # Training cloud
    ax.scatter(zt[:, 0], zt[:, 1], s=6, alpha=0.25, color=colors["train"],
               label="Training z cloud", zorder=2)
    confidence_ellipse(zt[:, 0], zt[:, 1], ax, n_std=2,
                       facecolor='none', edgecolor=colors["train"], lw=1.5, linestyle='--',
                       label="Train 2σ ellipse")

    # Simulated z_T
    ax.scatter(z_T[:, 0], z_T[:, 1], s=6, alpha=0.20, color=colors["sim"],
               label=f"z_T at T=5Y (N={N_PATHS})", zorder=3)
    confidence_ellipse(z_T[:, 0], z_T[:, 1], ax, n_std=2,
                       facecolor='none', edgecolor=colors["sim"], lw=1.5,
                       label="z_T 2σ ellipse")

    # Starting point
    ax.scatter([z0[0]], [z0[1]], s=120, marker='*', color='black', zorder=5, label="z_0")

    d = r['disp']
    ff = r['finite_frac']
    ax.set_title(f"{titles[label]}\n"
                 f"||z_T−z_0||: mean={d.mean():.2f}, p95={np.percentile(d,95):.2f}  |  "
                 f"Finite decoded: {ff:.0%}",
                 fontsize=9)
    ax.set_xlabel("z[0]")
    ax.set_ylabel("z[1]")
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.25)

fig.suptitle(f"Latent Displacement at T={EXPIRY}Y: dim4 Baseline vs Stable\n"
             f"(First 2 latent dims shown, dt=1/12, {N_PATHS} paths)", fontsize=11)
fig.tight_layout()
out_path = os.path.join(OUT_DIR, "fig_latent_displacement_baseline_vs_stable.png")
fig.savefig(out_path, dpi=200, bbox_inches='tight')
print(f"\nFigure saved: {out_path}")
plt.close(fig)

# ─── Figure 2: Displacement CDFs side by side ─────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
for label, r in results.items():
    d = r['disp']
    xs = np.sort(d)
    ys = np.arange(1, len(xs)+1) / len(xs)
    ls = '-' if r['is_stable'] else '--'
    ax.plot(xs, ys, lw=2, linestyle=ls, label=f"{label}  (finite={r['finite_frac']:.0%})")

ax.axvline(2.0, color='grey', lw=1, ls=':', label='ε=2.0 (probe robustness limit)')
ax.set_xlabel("||z_T − z_0||  (L2 displacement at T=5Y)")
ax.set_ylabel("CDF")
ax.set_title("Latent Displacement CDF: Baseline vs Stable (dim4, T=5Y)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
fig.tight_layout()
out_path2 = os.path.join(OUT_DIR, "fig_latent_displacement_cdf.png")
fig.savefig(out_path2, dpi=200, bbox_inches='tight')
print(f"Figure saved: {out_path2}")
plt.close(fig)

