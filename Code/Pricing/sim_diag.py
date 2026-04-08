# =============================================================================
# pricing_monitor.py
# =============================================================================
# Snapshot diagnostics for documenting model quality before, during, and after
# pricing continuation training. Run this script at any checkpoint to get a
# clean, printable summary of the key quantities.
#
# Captures:
#   1. Drift parameters  — kappa, eigenvalues, half-lives, theta
#   2. Diffusion         — sigma mean/std across EUR latent cloud
#   3. Short rate        — r0 and distribution across EUR cloud
#   4. Martingale check  — E[F_T] vs K for all 9 (expiry, tenor) pairs
#                          (the key diagnostic: should be ~0 after training)
#   5. Implied vol table — MC-implied normal vol vs market for a single date
#   6. Curve RMSE        — encoder-decoder reconstruction quality (bp)
#
# Usage:
#   python pricing_monitor.py                          # uses defaults below
#   python pricing_monitor.py --checkpoint path.pt --date 2015-06-30 --tag "ep50"
#   python pricing_monitor.py --no-mc                  # skip MC (fast mode)
#
# Output: prints a clean table and appends a row to pricing_monitor_log.csv
# =============================================================================

import argparse
import math
import os
import sys
import csv
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()

CODE_ROOT   = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
THESIS_ROOT = os.path.abspath(os.path.join(CODE_ROOT,  ".."))
for p in [THESIS_ROOT, CODE_ROOT, SCRIPT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_CHECKPOINT = (
    r"C:\Users\Bruger\PycharmProjects\MasterThesis\Figures\TrainingResults\dim2_stable\pricing_simple_v3\final_checkpoint.pt"
)
DEFAULT_CCY      = "EUR"
DEFAULT_DATE     = "2010-10-29"
DEFAULT_TAG      = "ep200_pricing"
MC_N_PATHS       = 2000
MC_N_STEPS       = 120       # 10 years at monthly dt
MC_DT            = 1 / 12
EXPIRIES         = [1, 5, 10]
TENORS           = [1, 5, 10]

LOG_FILE = os.path.join(SCRIPT_DIR, "pricing_monitor_log.csv")

# ---------------------------------------------------------------------------
# Market vols for 2010-10-29 (hardcoded reference for the baseline date)
# Add more dates here if needed
# ---------------------------------------------------------------------------
MARKET_VOLS_BP = {
    "2010-10-29": {
        (1, 1): 44.0, (1, 5): 32.7, (1, 10): 28.4,
        (5, 1): 26.9, (5, 5): 22.7, (5, 10): 21.6,
        (10, 1): 18.6, (10, 5): 18.3, (10, 10): 19.6,
    }
}

from Code import config
from Code.model.full_model import FullModel
from Code.model.sigma_matrix import L_from_sigmas_rhos
from Code.load_swapdata import my_data
from Code.Pricing.simulate_model import (
    simulate_latent_paths, compute_discount_paths, resolve_curve_index
)
from Code.Pricing.pricing import (
    time0_forward_swap_and_annuity,
    swap_from_discount_curve_at_expiry,
    implied_bachelier_vol,
)


# =============================================================================
# Helpers
# =============================================================================

def load_model(checkpoint_path, device):
    raw   = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = raw["model_state_dict"] if (isinstance(raw, dict) and "model_state_dict" in raw) else raw
    model = FullModel(latent_dim=2).to(device).double()
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [WARN] missing keys: {missing}")
    model.eval()
    return model


def encode_all(model, X_tensor, device):
    dtype = next(model.parameters()).dtype
    zs = []
    with torch.no_grad():
        for i in range(0, X_tensor.shape[0], 256):
            zs.append(model.encoder(X_tensor[i:i+256].to(device=device, dtype=dtype)).detach())
    return torch.cat(zs, dim=0)


def sep(char="-", width=70):
    print(char * width)


# =============================================================================
# 1. Drift parameters
# =============================================================================

def check_drift(model):
    K = model.K
    kappa  = F.softplus(K.raw_kappa).item()
    M      = K.drift_matrix().detach().cpu()
    eigs   = torch.linalg.eigvals(M).real.numpy()
    eigs_s = np.sort(eigs)          # most negative last
    theta  = K.theta.detach().cpu().numpy()
    hls    = [math.log(2) / abs(e) if abs(e) > 0 else float("inf") for e in eigs_s]

    sep()
    print("1. DRIFT  (K network — KMuStable)")
    sep()
    print(f"  kappa           = {kappa:.6f}")
    print(f"  eigenvalues(M)  = {np.round(eigs_s, 6)}")
    print(f"  half-lives      = {[f'{h:.2f}y' for h in hls]}")
    print(f"  theta           = {np.round(theta, 6)}")

    verdict = []
    if min(hls) < 1.5:
        verdict.append(f"WARN: fast mode hl={min(hls):.2f}y — paths decay before 5Y expiry")
    if max(hls) > 50:
        verdict.append(f"INFO: slow mode hl={max(hls):.2f}y — good for long-expiry vol")
    for v in verdict:
        print(f"  >> {v}")

    return {
        "kappa": kappa,
        "eig_fast": float(eigs_s[0]),
        "eig_slow": float(eigs_s[-1]),
        "hl_fast": float(min(hls)),
        "hl_slow": float(max(hls)),
        "theta_0": float(theta[0]),
        "theta_1": float(theta[1]),
    }


# =============================================================================
# 2. Diffusion parameters
# =============================================================================

def check_diffusion(model, X_ccy, device):
    dtype = next(model.parameters()).dtype
    with torch.no_grad():
        z = model.encoder(X_ccy[:64].to(device=device, dtype=dtype))
        sigmas, rhos = model.H(z)
        s = sigmas.detach().cpu().numpy()
        r = rhos.detach().cpu().numpy()

    sigma_mean = float(s.mean())
    sigma_std  = float(s.std())

    sep()
    print("2. DIFFUSION  (H network — sigma)")
    sep()
    print(f"  sigma mean (both dims) = {sigma_mean:.6f}")
    print(f"  sigma std              = {sigma_std:.6f}")
    print(f"  sigma per dim mean     = {s.mean(axis=0).round(6)}")
    print(f"  rho mean               = {r.mean(axis=0).round(6)}")

    if sigma_mean < 0.003:
        print(f"  >> WARN: sigma too small — SDE barely diffuses")
    elif sigma_mean > 0.05:
        print(f"  >> WARN: sigma large — may over-produce vol")
    else:
        print(f"  >> OK: sigma in plausible range [0.003, 0.05]")

    return {"sigma_mean": sigma_mean, "sigma_std": sigma_std}


# =============================================================================
# 3. Short rate
# =============================================================================

def check_short_rate(model, X_ccy, device, z0):
    dtype = next(model.parameters()).dtype
    with torch.no_grad():
        z_all = model.encoder(X_ccy.to(device=device, dtype=dtype))
        r_all = model.R(z_all).detach().cpu().numpy().flatten()
        r0    = float(model.R(z0).detach().cpu().numpy().flatten()[0])

    sep()
    print("3. SHORT RATE  (R network)")
    sep()
    print(f"  r0 at chosen date      = {r0*100:.4f}%  ({r0*10000:.1f} bp)")
    print(f"  r across EUR cloud:  mean={r_all.mean()*100:.3f}%  "
          f"std={r_all.std()*100:.3f}%  "
          f"min={r_all.min()*100:.3f}%  max={r_all.max()*100:.3f}%")

    if r_all.min() < -0.10:
        print(f"  >> WARN: short rate goes very negative ({r_all.min()*100:.2f}%)")

    return {"r0_pct": r0 * 100, "r_mean_pct": float(r_all.mean() * 100)}


# =============================================================================
# 4. Martingale check  E[F_T] vs K
# =============================================================================

def check_martingale(model, z0, device, n_paths=MC_N_PATHS, n_steps=MC_N_STEPS, dt=MC_DT):
    torch.manual_seed(42)

    sep()
    print("4. MARTINGALE CHECK  E[F_T] vs K  (should be ~0 after training)")
    sep()
    print(f"  Paths={n_paths}, steps={n_steps}, dt={dt:.4f}  ({n_steps*dt:.1f}y horizon)")

    z_paths, r_paths, _, _ = simulate_latent_paths(
        model=model, z0=z0, n_paths=n_paths, n_steps=n_steps, dt=dt, device=device
    )

    with torch.no_grad():
        _, aux0 = model.decode_from_z(z0, tau=None, do_arb_checks=False, return_aux=True)
    P_full_0  = aux0["P_full"].detach().cpu().numpy()
    tau_grid0 = aux0["tau_grid"].detach().cpu().numpy()

    annual_steps = {1: int(round(1/dt)), 5: int(round(5/dt)), 10: int(round(10/dt))}

    print(f"\n  {'Pair':>8}  {'K (bp)':>8}  {'E[F_T] (bp)':>13}  "
          f"{'drift (bp)':>12}  {'std (bp)':>10}  {'|drift|/std':>12}")
    print(f"  {'-'*8}  {'-'*8}  {'-'*13}  {'-'*12}  {'-'*10}  {'-'*12}")

    results = {}
    for expiry in EXPIRIES:
        exp_step = annual_steps[expiry]
        if exp_step >= z_paths.shape[1]:
            continue

        z_at_exp = z_paths[:, exp_step, :]

        # Decode in chunks
        P_full_list, tau_exp = [], None
        with torch.no_grad():
            for i in range(0, z_at_exp.shape[0], 256):
                _, aux = model.decode_from_z(
                    z_at_exp[i:i+256], tau=None, do_arb_checks=False, return_aux=True
                )
                P_full_list.append(aux["P_full"].detach().cpu().numpy())
                if tau_exp is None:
                    tau_exp = aux["tau_grid"].detach().cpu().numpy()
        P_full_exp = np.concatenate(P_full_list, axis=0)

        for tenor in TENORS:
            q0 = time0_forward_swap_and_annuity(P_full_0, tau_grid0, expiry, tenor)
            K  = q0["forward_swap"]

            rates = []
            for pi in range(n_paths):
                try:
                    res = swap_from_discount_curve_at_expiry(P_full_exp[pi], tau_exp, tenor)
                    r = res["swap_rate"]
                    if np.isfinite(r):
                        rates.append(r)
                except Exception:
                    pass

            if not rates:
                print(f"  {expiry}Y×{tenor}Y  {'n/a':>8}  {'n/a':>13}  {'n/a':>12}  {'n/a':>10}  {'n/a':>12}  (no valid paths)")
                continue

            F_T   = np.array(rates)
            drift = (F_T.mean() - K) * 10000
            std   = F_T.std() * 10000
            ratio = abs(drift) / std if std > 0 else float("inf")

            results[(expiry, tenor)] = {
                "K_bp": K * 10000, "drift_bp": drift, "std_bp": std, "ratio": ratio,
                "n_valid": len(rates)
            }
            flag = " <<" if abs(drift) > 50 else ""
            print(f"  {expiry}Y×{tenor}Y  {K*10000:>8.1f}  {F_T.mean()*10000:>13.1f}  "
                  f"{drift:>+12.1f}  {std:>10.1f}  {ratio:>12.2f}  ({len(rates)}/{n_paths}){flag}")

    mean_abs_drift = float(np.nanmean([abs(v["drift_bp"]) for v in results.values()])) if results else float("nan")
    print(f"\n  Mean |drift| across all pairs: {mean_abs_drift:.1f} bp")
    if mean_abs_drift > 50:
        print(f"  >> FAIL: large drift — theta points to wrong rate level")
        print(f"           Pricing training (Fix A) must correct this.")
    elif mean_abs_drift > 10:
        print(f"  >> CAUTION: moderate drift — monitor during training")
    else:
        print(f"  >> PASS: martingale condition approximately satisfied")

    return results, mean_abs_drift, z_paths, P_full_0, tau_grid0


# =============================================================================
# 5. Implied vol table (MC)
# =============================================================================

def check_implied_vols(model, z0, z_paths, P_full_0, tau_grid0, date_str, device,
                       n_paths=MC_N_PATHS, n_steps=MC_N_STEPS, dt=MC_DT):
    sep()
    print("5. IMPLIED VOL TABLE  (MC normal vol vs market)")
    sep()

    market = MARKET_VOLS_BP.get(date_str, {})
    if not market:
        print(f"  No market vols loaded for {date_str} — skipping vol comparison")

    r_paths = None  # reuse z_paths from martingale check, recompute discount
    with torch.no_grad():
        r_list = []
        for i in range(0, z_paths.shape[0] * z_paths.shape[1], 256):
            path_i = i // z_paths.shape[1]
            step_i = i % z_paths.shape[1]
            if path_i >= z_paths.shape[0]:
                break
        # compute r_paths efficiently
        n_p, n_t, d = z_paths.shape
        z_flat = z_paths.reshape(-1, d)
        r_flat = []
        for i in range(0, z_flat.shape[0], 512):
            r_chunk = model.R(z_flat[i:i+512]).detach()
            if r_chunk.ndim == 2 and r_chunk.shape[-1] == 1:
                r_chunk = r_chunk.squeeze(-1)
            r_flat.append(r_chunk)
        r_paths = torch.cat(r_flat).reshape(n_p, n_t)

    discount_paths = compute_discount_paths(r_paths, dt=dt)

    # Decode P_full paths (annual steps only to save time)
    annual_steps = {1: int(round(1/dt)), 5: int(round(5/dt)), 10: int(round(10/dt))}
    P_exp_cache = {}
    for expiry, exp_step in annual_steps.items():
        z_at_exp = z_paths[:, exp_step, :]
        P_full_list, tau_exp = [], None
        with torch.no_grad():
            for i in range(0, z_at_exp.shape[0], 256):
                _, aux = model.decode_from_z(
                    z_at_exp[i:i+256], tau=None, do_arb_checks=False, return_aux=True
                )
                P_full_list.append(aux["P_full"].detach().cpu().numpy())
                if tau_exp is None:
                    tau_exp = aux["tau_grid"].detach().cpu().numpy()
        P_exp_cache[expiry] = (np.concatenate(P_full_list, axis=0), tau_exp)

    print(f"\n  {'Pair':>8}  {'Mkt (bp)':>9}  {'Model (bp)':>11}  "
          f"{'Error (bp)':>11}  {'|Error|/Mkt':>13}")
    print(f"  {'-'*8}  {'-'*9}  {'-'*11}  {'-'*11}  {'-'*13}")

    vol_results = {}
    for expiry in EXPIRIES:
        exp_step = annual_steps[expiry]
        P_full_exp, tau_exp = P_exp_cache[expiry]
        disc_exp = discount_paths[:, exp_step].detach().cpu().numpy()

        for tenor in TENORS:
            q0  = time0_forward_swap_and_annuity(P_full_0, tau_grid0, expiry, tenor)
            K   = q0["forward_swap"]
            A0  = q0["annuity"]

            # Pathwise payoffs — filter invalid paths and discount factors
            pvs = []
            for pi in range(n_paths):
                try:
                    res = swap_from_discount_curve_at_expiry(P_full_exp[pi], tau_exp, tenor)
                    if not np.isfinite(res["swap_rate"]) or not np.isfinite(res["annuity"]):
                        continue
                    payoff = max(res["swap_rate"] - K, 0.0) * res["annuity"]
                    pv = disc_exp[pi] * payoff
                    if np.isfinite(pv):
                        pvs.append(pv)
                except Exception:
                    pass

            if not pvs:
                iv_bp = float("nan")
                mc_price = float("nan")
            else:
                mc_price = float(np.mean(pvs))
                if not np.isfinite(mc_price) or mc_price < 0:
                    iv_bp = float("nan")
                else:
                    iv = implied_bachelier_vol(
                        market_price=mc_price, forward=K, strike=K,
                        expiry=expiry, annuity=A0, payer=True
                    )
                    iv_bp = iv * 10000 if np.isfinite(iv) else float("nan")

            mkt_bp  = market.get((expiry, tenor), float("nan"))
            err_bp  = iv_bp - mkt_bp if np.isfinite(iv_bp) and np.isfinite(mkt_bp) else float("nan")
            rel_err = abs(err_bp) / mkt_bp if np.isfinite(err_bp) and mkt_bp > 0 else float("nan")

            vol_results[(expiry, tenor)] = {
                "model_bp": iv_bp, "market_bp": mkt_bp, "error_bp": err_bp
            }

            mkt_str = f"{mkt_bp:>9.1f}" if np.isfinite(mkt_bp) else f"{'n/a':>9}"
            err_str = f"{err_bp:>+11.1f}" if np.isfinite(err_bp) else f"{'n/a':>11}"
            rel_str = f"{rel_err:>12.1%}" if np.isfinite(rel_err) else f"{'n/a':>12}"
            print(f"  {expiry}Y×{tenor}Y  {mkt_str}  {iv_bp:>11.2f}  {err_str}  {rel_str}")

    valid_errors = [abs(v["error_bp"]) for v in vol_results.values()
                    if np.isfinite(v.get("error_bp", float("nan")))]
    mae = float(np.mean(valid_errors)) if valid_errors else float("nan")
    print(f"\n  MAE across all pairs: {mae:.1f} bp")

    return vol_results, mae


# =============================================================================
# 6. Curve RMSE
# =============================================================================

def check_curve_rmse(model, X_ccy, device):
    dtype = next(model.parameters()).dtype
    outs  = []
    with torch.no_grad():
        for i in range(0, X_ccy.shape[0], 256):
            outs.append(model(X_ccy[i:i+256].to(device=device, dtype=dtype)).detach().cpu())
    S_hat = torch.cat(outs, dim=0)
    X_cpu = X_ccy.cpu()
    mask  = torch.isfinite(X_cpu).all(dim=1) & torch.isfinite(S_hat).all(dim=1)
    rmse  = float(((X_cpu[mask] - S_hat[mask]) ** 2).mean().sqrt().item() * 10000)

    sep()
    print("6. CURVE RMSE  (encoder-decoder reconstruction)")
    sep()
    print(f"  EUR curve RMSE = {rmse:.2f} bp")
    if rmse > 5.0:
        print(f"  >> WARN: RMSE > 5 bp — G has drifted from autoencoder quality")
    else:
        print(f"  >> OK")

    return {"curve_rmse_bp": rmse}


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Pricing monitor — pre/during/post training")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--ccy",        default=DEFAULT_CCY)
    parser.add_argument("--date",       default=DEFAULT_DATE)
    parser.add_argument("--tag",        default=DEFAULT_TAG,
                        help="Label for this snapshot (e.g. 'ep200_baseline', 'ep50', 'final')")
    parser.add_argument("--no-mc",      action="store_true",
                        help="Skip MC checks (sections 4 and 5) — fast mode for quick drift/sigma checks")
    parser.add_argument("--paths",      type=int, default=MC_N_PATHS)
    args = parser.parse_args()

    device = torch.device("cpu")
    print("\n" + "=" * 70)
    print(f"PRICING MONITOR  —  {args.tag}")
    print(f"  checkpoint : {os.path.basename(args.checkpoint)}")
    print(f"  date       : {args.date}")
    print(f"  run time   : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    model    = load_model(args.checkpoint, device)
    meta, X_tensor, *_ = my_data(ccy_filter=args.ccy)
    dtype    = next(model.parameters()).dtype
    X_tensor = X_tensor.to(dtype=dtype)

    mask_ccy = meta["ccy"].astype(str).str.upper() == args.ccy.upper()
    X_ccy    = X_tensor[mask_ccy.to_numpy()]

    idx = resolve_curve_index(meta, as_of_date=args.date)
    S0  = X_tensor[idx:idx+1].to(device=device, dtype=dtype)
    with torch.no_grad():
        z0 = model.encoder(S0)
    print(f"\n  z0 = {z0.detach().cpu().numpy().flatten()}")

    # Run all checks
    drift_stats  = check_drift(model)
    diff_stats   = check_diffusion(model, X_ccy, device)
    rate_stats   = check_short_rate(model, X_ccy, device, z0)
    rmse_stats   = check_curve_rmse(model, X_ccy, device)

    mart_results = vol_results = None
    mean_drift   = mae = float("nan")

    if not args.no_mc:
        mart_results, mean_drift, z_paths, P_full_0, tau_grid0 = check_martingale(
            model, z0, device, n_paths=args.paths
        )
        vol_results, mae = check_implied_vols(
            model, z0, z_paths, P_full_0, tau_grid0,
            date_str=args.date, device=device, n_paths=args.paths
        )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    sep("=")
    print("SUMMARY")
    sep("=")
    print(f"  Tag              : {args.tag}")
    print(f"  kappa            : {drift_stats['kappa']:.4f}")
    print(f"  half-life (fast) : {drift_stats['hl_fast']:.2f} y")
    print(f"  half-life (slow) : {drift_stats['hl_slow']:.2f} y")
    print(f"  theta            : [{drift_stats['theta_0']:.4f}, {drift_stats['theta_1']:.4f}]")
    print(f"  sigma_mean       : {diff_stats['sigma_mean']:.6f}")
    print(f"  r0               : {rate_stats['r0_pct']:.3f}%")
    print(f"  curve RMSE       : {rmse_stats['curve_rmse_bp']:.2f} bp")
    print(f"  mean |drift|     : {mean_drift:.1f} bp  (martingale, target < 10)")
    print(f"  vol MAE          : {mae:.1f} bp  (vs market)")

    # -------------------------------------------------------------------------
    # Append to CSV log
    # -------------------------------------------------------------------------
    log_row = {
        "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M"),
        "tag":          args.tag,
        "checkpoint":   os.path.basename(args.checkpoint),
        "date":         args.date,
        "kappa":        drift_stats["kappa"],
        "hl_fast_y":    drift_stats["hl_fast"],
        "hl_slow_y":    drift_stats["hl_slow"],
        "theta_0":      drift_stats["theta_0"],
        "theta_1":      drift_stats["theta_1"],
        "sigma_mean":   diff_stats["sigma_mean"],
        "r0_pct":       rate_stats["r0_pct"],
        "curve_rmse_bp": rmse_stats["curve_rmse_bp"],
        "mean_drift_bp": mean_drift,
        "vol_mae_bp":   mae,
    }

    write_header = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(log_row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(log_row)
    print(f"\n  Log appended to: {LOG_FILE}")


if __name__ == "__main__":
    main()