import os
import sys
import numpy as np
import matplotlib.pyplot as plt

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
try:
    CODE_ROOT = os.path.dirname(os.path.abspath(__file__))
except NameError:
    CODE_ROOT = os.getcwd()

PROJECT_ROOT = os.path.abspath(os.path.join(CODE_ROOT, ".."))
THESIS_ROOT = os.path.abspath(os.path.join(CODE_ROOT, "..", ".."))

if CODE_ROOT not in sys.path:
    sys.path.insert(0, CODE_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if THESIS_ROOT not in sys.path:
    sys.path.insert(0, THESIS_ROOT)

# ---------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------
import pricing


def to_numpy(x):
    if x is None:
        return None
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def get_from_ctx(ctx, candidates, name):
    if isinstance(ctx, dict):
        for k in candidates:
            if k in ctx:
                return ctx[k]

    for k in candidates:
        if hasattr(ctx, k):
            return getattr(ctx, k)

    available = list(ctx.keys()) if isinstance(ctx, dict) else dir(ctx)
    raise KeyError(
        f"Could not find {name}. Tried {candidates}.\n"
        f"Available keys/attrs start with:\n{available[:50]}"
    )


def get_ctx_or_attr(ctx, candidates):
    if isinstance(ctx, dict):
        for k in candidates:
            if k in ctx:
                return ctx[k]

    for k in candidates:
        if hasattr(ctx, k):
            return getattr(ctx, k)

    return None


def normalize_D(D_raw, n_paths=None, n_steps=None):
    D = to_numpy(D_raw)
    D = np.squeeze(D)

    if D.ndim != 2:
        raise ValueError(f"D must be 2D after squeeze, got shape {D.shape}")

    n_times_target = None if n_steps is None else (n_steps + 1)

    if n_paths is not None and n_times_target is not None:
        if D.shape == (n_paths, n_times_target):
            return D
        if D.shape == (n_times_target, n_paths):
            return D.T

    if n_times_target is not None:
        if D.shape[0] == n_times_target:
            return D.T
        if D.shape[1] == n_times_target:
            return D

    return D


def normalize_P(P_raw, n_paths=None, n_steps=None):
    P = to_numpy(P_raw)
    P = np.squeeze(P)

    if P.ndim != 3:
        raise ValueError(f"P must be 3D after squeeze, got shape {P.shape}")

    n_times_target = None if n_steps is None else (n_steps + 1)

    if n_paths is not None and n_times_target is not None:
        if P.shape[0] == n_paths and P.shape[1] == n_times_target:
            return P
        if P.shape[0] == n_times_target and P.shape[1] == n_paths:
            return np.transpose(P, (1, 0, 2))

    if n_times_target is not None:
        if P.shape[0] == n_times_target:
            return np.transpose(P, (1, 0, 2))
        if P.shape[1] == n_times_target:
            return P

    return P


def nearest_index(grid, target):
    grid = np.asarray(grid, dtype=float)
    return int(np.argmin(np.abs(grid - target)))


def summarize_bad_curves(P):
    P = np.asarray(P)

    dP_dtau = np.diff(P, axis=2)
    mono_viol = dP_dtau > 1e-10
    bad_P = (
        np.any(mono_viol, axis=2)
        | np.any(P > 1.0, axis=2)
        | np.any(P <= 0.0, axis=2)
    )

    return {
        "P_min": float(np.nanmin(P)),
        "P_max": float(np.nanmax(P)),
        "share_P_gt_1": float((P > 1.0).mean()),
        "share_P_le_0": float((P <= 0.0).mean()),
        "share_mono_viol": float(mono_viol.mean()),
        "share_bad_path_time": float(bad_P.mean()),
        "bad_mask": bad_P,
    }


def try_decode_clipped_paths(ctx, Z_clip):
    if torch is None:
        raise RuntimeError("Torch is required for clipped-latent decoding.")

    for key in ["decode_latent_paths", "decode_z_paths", "decode_paths", "decoder_fn"]:
        fn = get_ctx_or_attr(ctx, [key])
        if callable(fn):
            out = fn(Z_clip)
            return to_numpy(out)

    model = get_ctx_or_attr(ctx, ["model", "net", "full_model"])
    if model is None:
        raise RuntimeError("Could not find model in ctx.")

    model.eval()

    try:
        param0 = next(model.parameters())
        model_dtype = param0.dtype
        model_device = param0.device
    except StopIteration:
        model_dtype = torch.float64
        model_device = torch.device("cpu")

    z_t = torch.as_tensor(Z_clip, dtype=model_dtype, device=model_device)
    flat = z_t.reshape(-1, z_t.shape[-1])

    with torch.no_grad():
        if hasattr(model, "decode_from_z"):
            P_mkt, aux = model.decode_from_z(flat, tau=None, do_arb_checks=False, return_aux=True)

            if isinstance(aux, dict) and "P_full" in aux:
                out = aux["P_full"]
            else:
                out = P_mkt

            out = to_numpy(out)
            return out.reshape(z_t.shape[0], z_t.shape[1], -1)

    raise RuntimeError("Found model but no decoder entry point.")


# =====================================================================
# NEW: dynamics diagnostics
# =====================================================================
def print_dynamics_diagnostics(ctx, z_train_std=None):
    """
    Print key quantities that determine whether mean-reversion can
    keep simulated paths inside the training distribution:

        kappa       — global drift scale (larger = stronger reversion)
        M eigenvalues — must all be negative for stability
        sigma at z0 — diffusion size (should be << kappa for stability)
        implied std  — sigma * sqrt(T) over 10 years (should be < training std)
        theta        — long-run mean the drift pulls toward
        reversion ratio — kappa / sigma: rule of thumb > 5 is healthy
    """
    if torch is None:
        print("torch not available — skipping dynamics diagnostics")
        return

    model = get_ctx_or_attr(ctx, ["model", "net", "full_model"])
    if model is None:
        print("No model found in ctx — skipping dynamics diagnostics")
        return

    z0_tensor = get_ctx_or_attr(ctx, ["z0"])
    if z0_tensor is None:
        print("No z0 found in ctx — skipping dynamics diagnostics")
        return

    model.eval()

    print("\n" + "=" * 60)
    print("DYNAMICS DIAGNOSTICS")
    print("=" * 60)

    with torch.no_grad():
        # --- K: drift ---
        if hasattr(model.K, "raw_kappa"):
            kappa = F.softplus(model.K.raw_kappa).item()
            print(f"  kappa (drift scale)  : {kappa:.6f}")
        else:
            print("  kappa                : (not available — not KMuStable)")

        if hasattr(model.K, "drift_matrix"):
            M = model.K.drift_matrix()
            eigs = torch.linalg.eigvals(M).real.detach().cpu().numpy()
            print(f"  M eigenvalues (real) : {eigs}")
            print(f"  All negative?        : {bool((eigs < 0).all())}")
        else:
            print("  M eigenvalues        : (not available)")

        if hasattr(model.K, "theta"):
            theta = model.K.theta.detach().cpu().numpy()
            print(f"  theta (long-run mean): {theta}")

        # --- H: diffusion ---
        z0 = z0_tensor
        if z0.dim() == 1:
            z0 = z0.unsqueeze(0)

        try:
            sigmas, rhos = model.H(z0)
            sigma_vals = sigmas.detach().cpu().numpy().flatten()
            print(f"  sigma at z0          : {sigma_vals}")

            # Implied std of z over T=10 years under pure diffusion (no drift)
            T = 10.0
            implied_std = sigma_vals * np.sqrt(T)
            print(f"  implied std (T=10y)  : {implied_std}  "
                  f"[sigma * sqrt(10)]")

            if z_train_std is not None:
                print(f"  training z std       : {z_train_std}")
                ratio = implied_std / (np.asarray(z_train_std) + 1e-12)
                print(f"  implied/training std : {ratio}  "
                      f"[> 1 means diffusion dominates over 10y]")

            if hasattr(model.K, "raw_kappa"):
                rev_ratio = kappa / (sigma_vals + 1e-12)
                print(f"  kappa / sigma        : {rev_ratio}  "
                      f"[rule of thumb: > 5 is healthy]")

                # Half-life: time for mean reversion to halve displacement
                # For OU: half_life = log(2) / |lambda_min|
                if hasattr(model.K, "drift_matrix"):
                    lambda_min = float(np.abs(eigs).min())
                    if lambda_min > 0:
                        half_life = np.log(2) / lambda_min
                        print(f"  mean-reversion half-life: {half_life:.2f}y  "
                              f"[should be << 10y]")

        except Exception as e:
            print(f"  [WARN] Could not compute sigma diagnostics: {e}")

        # --- displacement of z0 from theta ---
        if hasattr(model.K, "theta"):
            z0_np = z0.detach().cpu().numpy().flatten()
            theta_np = model.K.theta.detach().cpu().numpy()
            displacement = z0_np - theta_np
            print(f"  z0 - theta           : {displacement}  "
                  f"[initial displacement from long-run mean]")

    print("=" * 60)


def run_P_D_diagnostics(
    checkpoint_path,
    as_of_date,
    ccy="EUR",
    n_paths=2000,
    n_steps=120,
    dt=1 / 12,
    n_plot_paths=20,
):
    print(f"Running diagnostics for {as_of_date}")

    ctx = pricing.run_simulation(
        checkpoint_path=checkpoint_path,
        ccy_filter=ccy,
        as_of_date=str(as_of_date),
        n_paths=n_paths,
        n_steps=n_steps,
        dt=dt,
        show_plot=False,
    )

    # -----------------------------------------------------------------
    # ── NEW: print dynamics diagnostics immediately after simulation ──
    # -----------------------------------------------------------------
    z_train_std_raw = get_ctx_or_attr(ctx, ["z_train_std"])
    z_train_std_np  = to_numpy(z_train_std_raw) if z_train_std_raw is not None else None
    print_dynamics_diagnostics(ctx, z_train_std=z_train_std_np)

    # -----------------------------------------------------------------
    # Extract latent paths
    # -----------------------------------------------------------------
    Z_raw = get_from_ctx(
        ctx,
        ["z_paths", "latent_paths", "z_full_paths", "z"],
        "latent paths z",
    )

    Z = to_numpy(Z_raw)
    Z = np.squeeze(Z)

    if Z.ndim != 3:
        raise ValueError(f"Z must be 3D after squeeze, got shape {Z.shape}")

    if Z.shape[0] == n_paths and Z.shape[1] == n_steps + 1:
        pass
    elif Z.shape[0] == n_steps + 1 and Z.shape[1] == n_paths:
        Z = np.transpose(Z, (1, 0, 2))
    else:
        print("Unexpected Z shape:", Z.shape)

    print("\nZ shape:", Z.shape)
    print("Z overall min:", Z.min(axis=(0, 1)))
    print("Z overall max:", Z.max(axis=(0, 1)))
    print("Z mean:", Z.mean(axis=(0, 1)))
    print("Z std :", Z.std(axis=(0, 1)))

    # Empirical training box from your earlier diagnostics
    train_min = np.array([-0.071420, -0.064361], dtype=float)
    train_max = np.array([0.005746,  0.012604],  dtype=float)

    below   = (Z < train_min).any(axis=2)
    above   = (Z > train_max).any(axis=2)
    outside = below | above

    print("\nLatent OOD diagnostics")
    print(f"Share of path-times outside training box: {outside.mean():.4%}")
    print(f"Share with z[0] below min: {(Z[:, :, 0] < train_min[0]).mean():.4%}")
    print(f"Share with z[0] above max: {(Z[:, :, 0] > train_max[0]).mean():.4%}")
    print(f"Share with z[1] below min: {(Z[:, :, 1] < train_min[1]).mean():.4%}")
    print(f"Share with z[1] above max: {(Z[:, :, 1] > train_max[1]).mean():.4%}")

    # -----------------------------------------------------------------
    # Extract discount factors and decoded curves
    # -----------------------------------------------------------------
    D_raw = get_from_ctx(
        ctx,
        ["discount_paths", "D_t", "D_paths", "discount_factors", "D"],
        "discount factor paths D",
    )

    P_raw = get_from_ctx(
        ctx,
        ["P_full_paths", "P_full", "P_paths", "discount_curves", "P"],
        "decoded discount curves P",
    )

    time_grid_raw = None
    tau_grid_raw  = None

    for candidates in [["t_grid", "time_grid", "times", "t"]]:
        try:
            time_grid_raw = get_from_ctx(ctx, candidates, "time grid")
            break
        except Exception:
            pass

    for candidates in [["tau_grid", "taus", "tau", "maturity_grid"]]:
        try:
            tau_grid_raw = get_from_ctx(ctx, candidates, "tau grid")
            break
        except Exception:
            pass

    D = normalize_D(D_raw, n_paths=n_paths, n_steps=n_steps)
    P = normalize_P(P_raw, n_paths=n_paths, n_steps=n_steps)

    n_paths_eff, n_times, n_tau = P.shape

    if time_grid_raw is None:
        time_grid = np.arange(n_times) * dt
    else:
        time_grid = np.asarray(to_numpy(time_grid_raw)).reshape(-1)

    if tau_grid_raw is None:
        tau_grid = np.arange(n_tau)
    else:
        tau_grid = np.asarray(to_numpy(tau_grid_raw)).reshape(-1)

    # -----------------------------------------------------------------
    # Link bad curves to latent extrapolation
    # -----------------------------------------------------------------
    bad_P = np.any(np.diff(P, axis=2) > 1e-10, axis=2) | np.any(P > 1.0, axis=2)

    print("\nLink between bad curves and latent extrapolation")
    print("Share bad_P overall:", bad_P.mean())
    print("Share outside overall:", outside.mean())
    print("Share bad_P among outside:", bad_P[outside].mean() if outside.any() else 0.0)
    print("Share bad_P among inside :", bad_P[~outside].mean() if (~outside).any() else 0.0)

    print("\nShapes")
    print("D shape:", D.shape)
    print("P shape:", P.shape)
    print("time_grid shape:", time_grid.shape)
    print("tau_grid shape :", tau_grid.shape)

    # -----------------------------------------------------------------
    # Diagnostics for D
    # -----------------------------------------------------------------
    ED   = D.mean(axis=0)
    D_q01 = np.quantile(D, 0.01, axis=0)
    D_q50 = np.quantile(D, 0.50, axis=0)
    D_q99 = np.quantile(D, 0.99, axis=0)

    print("\nD diagnostics")
    print(f"E[D_t] min/max       : [{ED.min():.6f}, {ED.max():.6f}]")
    print(f"D overall min/max    : [{D.min():.6f}, {D.max():.6f}]")
    print(f"Share D_t > 1        : {(D > 1.0).mean():.4%}")
    print(f"Share D_t <= 0       : {(D <= 0.0).mean():.4%}")

    # -----------------------------------------------------------------
    # Diagnostics for P
    # -----------------------------------------------------------------
    print("\nP diagnostics")
    print(f"P overall min/max    : [{np.nanmin(P):.6f}, {np.nanmax(P):.6f}]")
    print(f"Share P > 1          : {(P > 1.0).mean():.4%}")
    print(f"Share P <= 0         : {(P <= 0.0).mean():.4%}")

    dP_dtau  = np.diff(P, axis=2)
    mono_viol = dP_dtau > 1e-10
    viol_any  = np.any(mono_viol, axis=2)

    print(f"Share mono violations: {mono_viol.mean():.4%}")
    print(f"Share path-time with any maturity violation: {viol_any.mean():.4%}")

    # -----------------------------------------------------------------
    # Selected maturities
    # -----------------------------------------------------------------
    tau_targets = [1, 5, 10]
    tau_idxs = []
    for target in tau_targets:
        if len(tau_grid) > 0:
            tau_idxs.append(nearest_index(tau_grid, target))
    tau_idxs = sorted(set(tau_idxs))

    print("\nSelected maturities")
    for idx in tau_idxs:
        print(f"tau index {idx}, tau ~ {tau_grid[idx]}")

    # -----------------------------------------------------------------
    # Clipped-latent sanity check
    # -----------------------------------------------------------------
    P_clip = None
    Z_clip = np.clip(Z, train_min, train_max)

    print("\nRunning clipped-latent sanity check...")
    try:
        P_clip_raw = try_decode_clipped_paths(ctx, Z_clip)
        P_clip = normalize_P(P_clip_raw, n_paths=n_paths, n_steps=n_steps)

        stats_orig = summarize_bad_curves(P)
        stats_clip = summarize_bad_curves(P_clip)

        print("\nOriginal decoded P diagnostics")
        print(f"P range                 : [{stats_orig['P_min']:.6f}, {stats_orig['P_max']:.6f}]")
        print(f"Share P > 1             : {stats_orig['share_P_gt_1']:.4%}")
        print(f"Share P <= 0            : {stats_orig['share_P_le_0']:.4%}")
        print(f"Share mono violations   : {stats_orig['share_mono_viol']:.4%}")
        print(f"Share bad path-times    : {stats_orig['share_bad_path_time']:.4%}")

        print("\nClipped decoded P diagnostics")
        print(f"P_clip range            : [{stats_clip['P_min']:.6f}, {stats_clip['P_max']:.6f}]")
        print(f"Share P_clip > 1        : {stats_clip['share_P_gt_1']:.4%}")
        print(f"Share P_clip <= 0       : {stats_clip['share_P_le_0']:.4%}")
        print(f"Share mono viol (clip)  : {stats_clip['share_mono_viol']:.4%}")
        print(f"Share bad path-times    : {stats_clip['share_bad_path_time']:.4%}")

        time_targets_clip = [0.0, 1.0, 5.0, 10.0]
        time_idxs_clip = sorted(set(nearest_index(time_grid, t) for t in time_targets_clip))

        plt.figure(figsize=(8, 5))
        for j in time_idxs_clip:
            plt.plot(tau_grid, P[:, j, :].mean(axis=0),
                     linestyle="--", label=f"orig t≈{time_grid[j]:.2f}")
            plt.plot(tau_grid, P_clip[:, j, :].mean(axis=0),
                     label=f"clip t≈{time_grid[j]:.2f}")
        plt.axhline(1.0, linestyle=":", linewidth=1)
        plt.xlabel("tau")
        plt.ylabel("Mean discount curve")
        plt.title(f"Original vs clipped decoded curves: {as_of_date}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print("\nClipped-latent sanity check could not be completed.")
        print("Reason:", repr(e))

    # -----------------------------------------------------------------
    # Plot 1: E[D_t] with bands
    # -----------------------------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(time_grid, ED, label="E[D_t]")
    plt.plot(time_grid, D_q50, linestyle="--", label="Median D_t")
    plt.fill_between(time_grid, D_q01, D_q99, alpha=0.2, label="1%-99% band")
    plt.axhline(1.0, linestyle=":", linewidth=1)
    plt.xlabel("t")
    plt.ylabel("D_t")
    plt.title(f"Discount-factor diagnostics: {as_of_date}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # -----------------------------------------------------------------
    # Plot 2: sample D paths
    # -----------------------------------------------------------------
    plt.figure(figsize=(8, 5))
    n_show = min(n_plot_paths, D.shape[0])
    for i in range(n_show):
        plt.plot(time_grid, D[i], alpha=0.5)
    plt.axhline(1.0, linestyle=":", linewidth=1)
    plt.xlabel("t")
    plt.ylabel("D_t")
    plt.title(f"Sample D paths: {as_of_date}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # -----------------------------------------------------------------
    # Plot 3: mean P(t, tau) for selected maturities
    # -----------------------------------------------------------------
    plt.figure(figsize=(8, 5))
    for idx in tau_idxs:
        mean_P_tau = P[:, :, idx].mean(axis=0)
        plt.plot(time_grid, mean_P_tau, label=f"tau≈{tau_grid[idx]:.2f}")
    plt.axhline(1.0, linestyle=":", linewidth=1)
    plt.xlabel("t")
    plt.ylabel("E[P_t(tau)]")
    plt.title(f"Mean decoded bond prices at selected maturities: {as_of_date}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # -----------------------------------------------------------------
    # Plot 4: mean curve snapshots at selected times
    # -----------------------------------------------------------------
    time_targets = [0.0, 1.0, 5.0, 10.0]
    time_idxs = sorted(set(nearest_index(time_grid, t) for t in time_targets))

    plt.figure(figsize=(8, 5))
    for idx in time_idxs:
        mean_curve = P[:, idx, :].mean(axis=0)
        plt.plot(tau_grid, mean_curve, label=f"t≈{time_grid[idx]:.2f}")
    plt.axhline(1.0, linestyle=":", linewidth=1)
    plt.xlabel("tau")
    plt.ylabel("E[P_t(tau)]")
    plt.title(f"Mean decoded discount curves: {as_of_date}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # -----------------------------------------------------------------
    # Plot 5: share of P>1 over time for selected tau
    # -----------------------------------------------------------------
    plt.figure(figsize=(8, 5))
    for idx in tau_idxs:
        share_gt1 = (P[:, :, idx] > 1.0).mean(axis=0)
        plt.plot(time_grid, share_gt1, label=f"tau≈{tau_grid[idx]:.2f}")
    plt.xlabel("t")
    plt.ylabel("Share[P_t(tau) > 1]")
    plt.title(f"Frequency of P>1 by maturity: {as_of_date}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # -----------------------------------------------------------------
    # Plot 6: latent path fans for each dimension
    # -----------------------------------------------------------------
    n_show = min(n_plot_paths, Z.shape[0])
    n_dims = Z.shape[2]

    fig, axes = plt.subplots(1, n_dims, figsize=(6 * n_dims, 5))
    if n_dims == 1:
        axes = [axes]

    for d, ax in enumerate(axes):
        for i in range(n_show):
            ax.plot(time_grid, Z[i, :, d], alpha=0.4, linewidth=0.8)
        ax.axhline(train_min[d], color="red",   linestyle="--", linewidth=1.5,
                   label=f"train min ({train_min[d]:.4f})")
        ax.axhline(train_max[d], color="red",   linestyle="--", linewidth=1.5,
                   label=f"train max ({train_max[d]:.4f})")
        ax.set_xlabel("t (years)")
        ax.set_ylabel(f"z[{d}]")
        ax.set_title(f"Latent paths z[{d}]")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Latent path fans: {as_of_date}", fontsize=12)
    fig.tight_layout()
    plt.show()

    return {
        "ctx": ctx,
        "D": D,
        "P": P,
        "P_clip": P_clip,
        "Z": Z,
        "time_grid": time_grid,
        "tau_grid": tau_grid,
        "ED": ED,
    }


def main():
    checkpoint_path = (
        r"C:\Users\Bruger\PycharmProjects\MasterThesis\Figures\TrainingResults"
        r"\dim2_stable\pricing_continuation\final_checkpoint.pt"
    )

    _out = run_P_D_diagnostics(
        checkpoint_path=checkpoint_path,
        as_of_date="2010-10-29",
        ccy="EUR",
        n_paths=2000,
        n_steps=120,
        dt=1 / 12,
        n_plot_paths=20,
    )


if __name__ == "__main__":
    main()