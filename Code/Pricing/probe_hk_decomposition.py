"""
H-vs-K Decomposition Probe + z* equilibrium check.

Decomposes the 5Y displacement into:
  (a) DIFFUSION-only: one-step ||z - z_0|| with K=0 (isolates H magnitude)
  (b) DRIFT-only: 5Y simulation with H=0 (isolates equilibrium pull from z*)

Also prints ||z_0 - z*|| averaged over EUR curves for both models.

z* = -M^{-1} N  for stable  (closed form from KMuStable)
z* = -W^{-1} b  for baseline (from KMu.lin, but only meaningful if W is invertible)
"""
import os, sys, math
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath('../..'))
os.environ.setdefault("SKIP_VARIANT_CONFIRM", "1")
from Code import config
config.confirm_variant()

from Code.load_swapdata import my_data
from Code.model.full_model import FullModel as FM_base
from Code.model.full_model_stable import FullModel as FM_stable
from Code.model.sigma_matrix import L_from_sigmas_rhos

DIM     = 4
EXPIRY  = 5
DT      = 1/12
N_STEPS = int(round(EXPIRY / DT))
N_PATHS = 500
SEED    = 42
OUT_DIR = "../../Figures/Pricing"

CHECKPOINTS = {
    "dim4_baseline": ("Figures/TrainingResults/dim4_baseline/ep5000/checkpoint_dim4_ep5000.pt", False),
    "dim4_stable":   ("Figures/TrainingResults/dim4_stable/ep5000/checkpoint_dim4_ep5000.pt",   True),
}


def load_model(path, is_stable, dim=4):
    raw = torch.load(path, map_location='cpu')
    cls = FM_stable if is_stable else FM_base
    m = cls(latent_dim=dim)
    m.load_state_dict(raw)
    m.eval()
    return m


def get_zstar(model, is_stable):
    """Compute equilibrium z* = -M^{-1} N."""
    with torch.no_grad():
        if is_stable:
            M = model.K.stable_matrix()           # (d,d), neg-def
            N = model.K.N                          # (d,)
            z_star = -torch.linalg.solve(M, N)    # (d,)
        else:
            W = model.K.lin.weight                 # (d,d)
            b = model.K.lin.bias                   # (d,)
            try:
                z_star = -torch.linalg.solve(W, b)
            except Exception:
                z_star = torch.full((DIM,), float('nan'))
    return z_star


@torch.no_grad()
def one_step_diffusion_only(model, z0, n_paths, dt, seed):
    """Single EM step with drift zeroed out. Returns ||z1 - z0|| per path."""
    torch.manual_seed(seed)
    z = z0.expand(n_paths, -1).clone().float()
    sqrt_dt = math.sqrt(dt)
    sigmas, rhos = model.H(z)
    L = L_from_sigmas_rhos(sigmas, rhos, validate=False)
    dW = torch.randn(n_paths, DIM) * sqrt_dt
    shock = torch.bmm(L, dW.unsqueeze(-1)).squeeze(-1)
    z1 = z + shock   # NO drift
    return (z1 - z).norm(dim=1).numpy()


@torch.no_grad()
def simulate_drift_only(model, z0, n_steps, dt):
    """Simulate with H=0 (no diffusion). Returns z_T and path of ||z_t - z0||."""
    z = z0.float().clone()
    disps = []
    for _ in range(n_steps):
        drift = model.K(z) * dt
        z = z + drift
        disps.append((z - z0).norm(dim=1).item())
    return z, np.array(disps)


@torch.no_grad()
def simulate_full(model, z0, n_paths, n_steps, dt, seed):
    """Full EM simulation. Returns ||z_T - z_0|| per path."""
    torch.manual_seed(seed)
    z = z0.expand(n_paths, -1).clone().float()
    sqrt_dt = math.sqrt(dt)
    for _ in range(n_steps):
        sigmas, rhos = model.H(z)
        L = L_from_sigmas_rhos(sigmas, rhos, validate=False)
        dW = torch.randn(n_paths, DIM) * sqrt_dt
        shock = torch.bmm(L, dW.unsqueeze(-1)).squeeze(-1)
        drift = model.K(z) * dt
        z = z + drift + shock
    return (z - z0).norm(dim=1).numpy()


# ─── load data ────────────────────────────────────────────────────────────
meta, X_tensor, *_ = my_data(use='bbg')
X_eur = X_tensor[meta['ccy'] == 'EUR'].float()
mid_idx = len(X_eur) // 2
x0 = X_eur[mid_idx:mid_idx+1]

# ─── run probes ───────────────────────────────────────────────────────────
rows = []
drift_curves = {}

print(f"\n{'='*72}")
print("H-vs-K DECOMPOSITION  (dim4, dt=1/12)")
print(f"{'='*72}\n")

for label, (path, is_stable) in CHECKPOINTS.items():
    model = load_model(path, is_stable)

    with torch.no_grad():
        z0     = model.encoder(x0)                  # (1,4)
        z_star = get_zstar(model, is_stable)

    # ��─ z* diagnostics ────────────────────────────────────────────────────
    # over all EUR curves
    z_all = model.encoder(X_eur)
    dist_to_zstar = (z_all - z_star.unsqueeze(0)).norm(dim=1).detach().numpy()

    print(f"--- {label} ---")
    print(f"  z*        = {z_star.numpy()}")
    print(f"  z_0       = {z0.numpy().flatten()}")
    print(f"  z_0 - z*  = {(z0 - z_star).numpy().flatten()}")
    print(f"  ||z_0 - z*||  = {float((z0 - z_star).norm()):.4f}")
    print(f"  ||z - z*|| over EUR train: mean={dist_to_zstar.mean():.4f}  "
          f"std={dist_to_zstar.std():.4f}  max={dist_to_zstar.max():.4f}")

    # H magnitude: σ eigenvalues
    with torch.no_grad():
        sigmas, rhos = model.H(z0)
    print(f"  H sigmas at z_0 = {sigmas.numpy().flatten()}")

    # M eigenvalues for stable
    if is_stable:
        M = model.K.stable_matrix()
        eigvals = torch.linalg.eigvals(M).real.detach().numpy()
        print(f"  K eigenvalues (real) = {np.sort(eigvals)}")

    # (a) Diffusion-only: one-step ||z1 - z0||
    diff_disp = one_step_diffusion_only(model, z0, N_PATHS, DT, SEED)
    # annualise: one step is DT years, 5Y has N_STEPS steps, rough scale = sqrt(N_STEPS)
    print(f"\n  (a) DIFFUSION ONLY (1 step, dt=1/12):")
    print(f"      ||z1-z0||  mean={diff_disp.mean():.4f}  "
          f"p95={np.percentile(diff_disp,95):.4f}  "
          f"max={diff_disp.max():.4f}")
    print(f"      Annualised (×√{N_STEPS}): ~{diff_disp.mean()*math.sqrt(N_STEPS):.3f}")

    # (b) Drift-only: 5Y ||z_T - z0||
    z_T_drift, drift_disp_curve = simulate_drift_only(model, z0, N_STEPS, DT)
    print(f"\n  (b) DRIFT ONLY (T=5Y, H=0):")
    print(f"      ||z_T-z0|| = {drift_disp_curve[-1]:.4f}")
    print(f"      z_T        = {z_T_drift.numpy().flatten()}")

    # (c) Full simulation for reference
    full_disp = simulate_full(model, z0, N_PATHS, N_STEPS, DT, SEED)
    print(f"\n  (c) FULL SIMULATION (T=5Y):")
    print(f"      ||z_T-z0||  mean={full_disp.mean():.4f}  "
          f"p95={np.percentile(full_disp,95):.4f}")

    rows.append({
        "label":         label,
        "z_star":        z_star.numpy(),
        "dist_z0_zstar": float((z0 - z_star).norm()),
        "dist_EUR_mean": dist_to_zstar.mean(),
        "h_sigma_mean":  float(sigmas.mean()),
        "diff_1step_mean": diff_disp.mean(),
        "diff_ann":      diff_disp.mean() * math.sqrt(N_STEPS),
        "drift_5y":      drift_disp_curve[-1],
        "full_5y_mean":  full_disp.mean(),
    })
    drift_curves[label] = drift_disp_curve
    print()

# ─── summary table ────────────────────────────────────────────────────────
print(f"\n{'='*72}")
print("DECOMPOSITION SUMMARY")
print(f"{'='*72}")
print(f"{'Model':<20} {'||z0-z*||':>10} {'H_sigma':>9} {'Diff_ann':>10} "
      f"{'Drift_5Y':>10} {'Full_5Y':>10}")
print("-"*72)
for r in rows:
    print(f"{r['label']:<20} {r['dist_z0_zstar']:>10.4f} {r['h_sigma_mean']:>9.4f} "
          f"{r['diff_ann']:>10.4f} {r['drift_5y']:>10.4f} {r['full_5y_mean']:>10.4f}")

print("""
Columns:
  ||z0-z*||   : distance from starting z_0 to equilibrium (mechanism 2)
  H_sigma     : mean diagonal sigma of H at z_0 (proxy for diffusion scale)
  Diff_ann    : one-step diffusion × sqrt(N_steps), approx. 5Y diffusive displacement
  Drift_5Y    : ||z_T - z_0|| with H=0 over 5Y (pure drift pull, mechanism 2)
  Full_5Y     : mean ||z_T - z_0|| with full EM simulation
""")

# ─── figure: drift-only displacement over time ────────────────────────────
times = np.arange(1, N_STEPS+1) * DT
fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
for label, curve in drift_curves.items():
    ls = '-' if 'stable' in label else '--'
    ax.plot(times, curve, lw=2, linestyle=ls, label=label)
ax.set_xlabel("Time (years)")
ax.set_ylabel("||z_t − z_0||  (drift only, H=0)")
ax.set_title("Drift-only displacement over time: Baseline vs Stable (dim4)\n"
             "Measures equilibrium pull (mechanism 2)")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
out = os.path.join(OUT_DIR, "fig_drift_only_displacement.png")
fig.savefig(out, dpi=200, bbox_inches='tight')
print(f"Figure saved: {out}")
plt.close(fig)

