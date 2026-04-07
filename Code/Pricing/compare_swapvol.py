import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

# ---------------------------------------------------------------------
# User setup
# ---------------------------------------------------------------------
DEFAULT_PRICING_RUN_NAME = "pricing_dyn_ep200"
USE_PRICING_CHECKPOINT = True
EXPLICIT_CHECKPOINT_PATH = None
PRICING_RUN_NAME = DEFAULT_PRICING_RUN_NAME

USE = "bbg"
LATENT_DIM = 2
EPOCHS = 200                  # used only when loading a non-pricing checkpoint path
CCY = "EUR"
IDX_CHOICE_FALLBACK = 0       # kept for reference; not used when matching by date
SEED = 1234

# Simulation settings
N_PATHS = 1000
N_STEPS = 120                 # 10 years at monthly steps
DT = 1 / 12
DISCRETIZATION = "euler"
SIM_MODE = "full"
DIFFUSION_SCALE = 1.0
G0_FLOOR = 1e-5
ACCRUAL = 1.0
PAYER = True
NOTIONAL = 1.0

# What to run
RUN_SINGLE_EXAMPLE = True
RUN_PANEL_COMPARISON = False

# Single example settings
EXAMPLE_DATE = "2012-05-31"   # set to None to auto-pick the first overlapping market date
EXAMPLE_STRUCTURES = [(1, 5), (1, 10), (5, 5), (5, 10)]

# Panel settings
MAX_DATES = 10
NEUTRAL_DATES = [
    "2012-05-31", "2014-07-31", "2013-04-30", "2012-08-31",
    "2014-06-30", "2012-07-31", "2012-12-31", "2014-05-30",
    "2014-08-29", "2013-03-29",
]

PLOT_DPI = 200
SHOW_PLOTS = False

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
try:
    CODE_ROOT = os.path.dirname(os.path.abspath(__file__))
except NameError:
    CODE_ROOT = os.getcwd()

PROJECT_ROOT = os.path.abspath(os.path.join(CODE_ROOT, ".."))
THESIS_ROOT = os.path.abspath(os.path.join(CODE_ROOT, "..", ".."))

for _p in [CODE_ROOT, PROJECT_ROOT, THESIS_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

OUT_DIR = os.path.join(THESIS_ROOT, "Figures", "Pricing", "vol_comparison")
EXCEL_PATH = os.path.join(THESIS_ROOT, "SwapData", "SwapVol.xlsx")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------
# Imports from current pricing setup
# ---------------------------------------------------------------------
from Code import config
from Code.load_swapdata import my_data

from Code.Pricing.pricing import (
    set_seed,
    resolve_checkpoint_path_current,
    load_model,
    build_decoder_tau_grid,
    simulate_latent_paths,
    compute_discount_paths,
    time0_forward_swap_and_annuity_from_z,
    price_swaption_mc,
    implied_bachelier_vol,
)

print(f"Repo root: {THESIS_ROOT}")
print(f"Code root: {CODE_ROOT}")
print(f"Active model variant from config.py: {config.VARIANT}")


def normalize_compare_checkpoint_settings():
    explicit = EXPLICIT_CHECKPOINT_PATH
    if explicit is not None:
        explicit = str(explicit).strip()
        if explicit == "":
            explicit = None

    use_pricing = bool(USE_PRICING_CHECKPOINT)
    run_name = str(PRICING_RUN_NAME).strip() if PRICING_RUN_NAME is not None else DEFAULT_PRICING_RUN_NAME
    if run_name == "":
        run_name = DEFAULT_PRICING_RUN_NAME

    return {
        "use_pricing_checkpoint": use_pricing,
        "explicit_checkpoint_path": explicit,
        "pricing_run_name": run_name,
    }


# ---------------------------------------------------------------------
# Bloomberg code parser
# ---------------------------------------------------------------------
def _parse_bbg_code(cs: str):
    if not cs.isdigit():
        return None, None
    n = len(cs)
    if n == 2:
        return int(cs[0]), int(cs[1])
    if n == 3:
        if cs[:2] == "10":
            return 10, int(cs[2])
        return int(cs[0]), int(cs[1:])
    if n == 4:
        return int(cs[:2]), int(cs[2:])
    return None, None


# ---------------------------------------------------------------------
# Market vol loader
# ---------------------------------------------------------------------
def load_market_vols(excel_path: str) -> pd.DataFrame:
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Market vol file not found: {excel_path}")

    raw = pd.read_excel(excel_path, sheet_name=0, header=None)
    code_row = raw.iloc[0]
    data = pd.read_excel(excel_path, sheet_name=0, header=4)

    date_col = data.columns[1]

    col_map = {}
    for idx, col in enumerate(data.columns):
        raw_code = code_row.iloc[idx] if idx < len(code_row) else None
        cs = str(raw_code).strip() if pd.notna(raw_code) else ""
        col_map[col] = _parse_bbg_code(cs)

    keep = [c for c in data.columns if not data[c].isnull().all()]
    data = data[keep].copy()

    melted = data.melt(id_vars=[date_col], var_name="swap_col", value_name="vol")
    melted["option_maturity"] = melted["swap_col"].map(lambda c: col_map.get(c, (None, None))[0])
    melted["swap_tenor"] = melted["swap_col"].map(lambda c: col_map.get(c, (None, None))[1])

    melted = melted.dropna(subset=["option_maturity", "swap_tenor"])
    melted = melted.rename(columns={date_col: "as_of_date"})
    melted["as_of_date"] = pd.to_datetime(melted["as_of_date"], errors="coerce")
    melted = melted.dropna(subset=["as_of_date", "vol"])

    melted["option_maturity"] = melted["option_maturity"].astype(int)
    melted["swap_tenor"] = melted["swap_tenor"].astype(int)
    melted["vol"] = pd.to_numeric(melted["vol"], errors="coerce")
    melted = melted.dropna(subset=["vol"])

    med = float(melted["vol"].median())
    if med > 1.0:
        print(f"  [units] median={med:.2f} -> basis-points -> dividing by 10,000")
        melted["vol"] /= 10_000.0
    elif med > 0.01:
        print(f"  [units] median={med:.4f} -> percent -> dividing by 100")
        melted["vol"] /= 100.0
    else:
        print(f"  [units] median={med:.6f} -> already decimal")

    return melted[["as_of_date", "option_maturity", "swap_tenor", "vol"]].reset_index(drop=True)


# ---------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------
def nearest_idx(dates: pd.Series, target: pd.Timestamp) -> int:
    return int((dates - target).abs().argmin())


def nearest_date(dates, target: pd.Timestamp) -> pd.Timestamp:
    dates = pd.to_datetime(pd.Series(dates)).sort_values().reset_index(drop=True)
    idx = nearest_idx(dates, target)
    return pd.Timestamp(dates.iloc[idx])


def structure_label(expiry: int, tenor: int) -> str:
    return f"{int(expiry)}Yx{int(tenor)}Y"


def save_table(df: pd.DataFrame, path_csv: str, path_xlsx: str = None):
    df.to_csv(path_csv, index=False)
    print(f"Saved CSV -> {path_csv}")

    if path_xlsx is None:
        return

    try:
        with pd.ExcelWriter(path_xlsx, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Results")
            ws = writer.sheets["Results"]
            ws.freeze_panes = "A2"
            for column in ws.columns:
                max_length = 0
                col_letter = column[0].column_letter
                for cell in column:
                    try:
                        max_length = max(max_length, len(str(cell.value)))
                    except Exception:
                        pass
                ws.column_dimensions[col_letter].width = min(max_length + 2, 40)
        print(f"Saved XLSX -> {path_xlsx}")
    except Exception as exc:
        print(f"[WARNING] Could not save XLSX: {exc}")


# ---------------------------------------------------------------------
# Data/model context
# ---------------------------------------------------------------------
def build_context():
    os.makedirs(OUT_DIR, exist_ok=True)
    set_seed(SEED)

    ckpt_cfg = normalize_compare_checkpoint_settings()

    checkpoint_path = resolve_checkpoint_path_current(
        thesis_root=THESIS_ROOT,
        use=USE,
        latent_dim=LATENT_DIM,
        epochs=EPOCHS,
        use_pricing_checkpoint=ckpt_cfg["use_pricing_checkpoint"],
        pricing_run_name=ckpt_cfg["pricing_run_name"],
        explicit_checkpoint_path=ckpt_cfg["explicit_checkpoint_path"],
    )

    model = load_model(
        checkpoint_path=checkpoint_path,
        device=DEVICE,
        latent_dim=LATENT_DIM,
        use_pricing_checkpoint=ckpt_cfg["use_pricing_checkpoint"],
        pricing_run_name=ckpt_cfg["pricing_run_name"],
    )

    _, _, meta_full, X_tensor_full, _, _, _, _ = my_data(use=USE)
    meta_full = meta_full.reset_index(drop=True).copy()
    dates_full = pd.to_datetime(meta_full["as_of_date"])

    ccy_mask = meta_full["ccy"].astype(str).str.upper() == CCY.upper()
    meta_eur = meta_full.loc[ccy_mask].reset_index(drop=True)
    dates_eur = pd.to_datetime(meta_eur["as_of_date"]).reset_index(drop=True)

    X_tensor_full = X_tensor_full.double()
    X_eur = X_tensor_full[ccy_mask.to_numpy()]

    decoder_tau_grid_base = build_decoder_tau_grid(
        model=model,
        device=DEVICE,
        dtype=torch.float64,
        fine_step=1 / 52,
        fine_horizon=1.0,
    )

    market_vols = load_market_vols(EXCEL_PATH)

    print(f"Using pricing checkpoint: {ckpt_cfg['use_pricing_checkpoint']}")
    if ckpt_cfg["explicit_checkpoint_path"] is not None:
        print(f"Explicit checkpoint path: {ckpt_cfg['explicit_checkpoint_path']}")
    else:
        print(f"Pricing run name       : {ckpt_cfg['pricing_run_name']}")
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"{CCY} curves: {len(meta_eur)} rows | {dates_eur.min().date()} -> {dates_eur.max().date()}")
    print(
        f"Market vol quotes: {len(market_vols)} rows | "
        f"{market_vols['as_of_date'].min().date()} -> {market_vols['as_of_date'].max().date()}"
    )

    return {
        "model": model,
        "checkpoint_path": checkpoint_path,
        "meta_eur": meta_eur,
        "X_eur": X_eur,
        "dates_eur": dates_eur,
        "decoder_tau_grid_base": decoder_tau_grid_base,
        "market_vols": market_vols,
    }


# ---------------------------------------------------------------------
# Pricing helpers
# ---------------------------------------------------------------------
def simulate_once_for_date(model, z0):
    with torch.no_grad():
        z_paths, r_paths, _, _ = simulate_latent_paths(
            model=model,
            z0=z0,
            n_paths=N_PATHS,
            n_steps=N_STEPS,
            dt=DT,
            device=DEVICE,
            discretization=DISCRETIZATION,
            sim_mode=SIM_MODE,
            diffusion_scale=DIFFUSION_SCALE,
        )
        discount_paths = compute_discount_paths(r_paths, dt=DT, method="trapezoid")

    return {
        "z_paths": z_paths,
        "r_paths": r_paths,
        "discount_paths": discount_paths,
        "dt": DT,
    }


def price_structure_for_date(model, decoder_tau_grid_base, z0, sim, expiry: int, tenor: int):
    t0_quote = time0_forward_swap_and_annuity_from_z(
        model=model,
        z0=z0,
        decoder_tau_grid_base=decoder_tau_grid_base,
        expiry=float(expiry),
        tenor=int(tenor),
        g0_floor=G0_FLOOR,
        accrual=ACCRUAL,
    )

    strike = t0_quote["forward_swap"]

    mc = price_swaption_mc(
        model=model,
        z_paths=sim["z_paths"],
        r_paths=sim["r_paths"],
        decoder_tau_grid_base=decoder_tau_grid_base,
        dt=sim["dt"],
        strike=strike,
        expiry=float(expiry),
        tenor=int(tenor),
        notional=NOTIONAL,
        payer=PAYER,
        g0_floor=G0_FLOOR,
        accrual=ACCRUAL,
        discount_paths=sim["discount_paths"],
    )

    model_vol = implied_bachelier_vol(
        market_price=mc["price"],
        forward=t0_quote["forward_swap"],
        strike=strike,
        expiry=float(expiry),
        annuity=t0_quote["annuity"],
        notional=NOTIONAL,
        payer=PAYER,
    )

    return {
        "strike": strike,
        "forward_swap_t0": t0_quote["forward_swap"],
        "annuity_t0": t0_quote["annuity"],

        "mc_price": mc["price"],
        "mc_stderr": mc["stderr"],
        "valid_decode_frac": mc["frac_valid_paths"],
        "actual_expiry": mc["actual_expiry"],
        "model_vol": model_vol,

        "mean_swap_rate_at_expiry": mc["mc_mean_swap_rate_at_expiry"],
        "mean_annuity_at_expiry": mc["mc_mean_annuity_at_expiry"],

        # pass through full pathwise expiry quantities
        "swap_rate": mc["swap_rate"],
        "annuity": mc["annuity"],
        "discount_to_expiry": mc["discount_to_expiry"],
    }


# ---------------------------------------------------------------------
# Example run
# ---------------------------------------------------------------------
def run_single_example(context):
    print("\n" + "=" * 70)
    print("RUNNING SINGLE MARKET-VOL COMPARISON EXAMPLE")
    print("=" * 70)

    market_vols = context["market_vols"]
    dates_eur = context["dates_eur"]

    market_dates = pd.to_datetime(sorted(market_vols["as_of_date"].unique()))
    overlap_dates = market_dates[(market_dates >= dates_eur.min()) & (market_dates <= dates_eur.max())]
    if len(overlap_dates) == 0:
        raise RuntimeError("No overlapping dates between market vol data and EUR curve data.")

    if EXAMPLE_DATE is None:
        market_date = pd.Timestamp(overlap_dates[0])
        requested_date = market_date
    else:
        requested_date = pd.Timestamp(EXAMPLE_DATE)
        market_date = nearest_date(overlap_dates, requested_date)

    curve_idx = nearest_idx(dates_eur, market_date)
    curve_date = pd.Timestamp(dates_eur.iloc[curve_idx])

    market_gap_days = abs((market_date - requested_date).days)
    curve_gap_days = abs((curve_date - market_date).days)

    if market_gap_days > 31:
        warnings.warn(
            f"Nearest market date is {market_gap_days} days away from requested example date {requested_date.date()}.",
            RuntimeWarning,
        )
    if curve_gap_days > 31:
        warnings.warn(
            f"Nearest EUR curve is {curve_gap_days} days away from market date {market_date.date()}.",
            RuntimeWarning,
        )

    print(f"Requested example date: {requested_date.date()}")
    print(f"Matched market date   : {market_date.date()} (gap={market_gap_days} days)")
    print(f"Matched EUR curve date: {curve_date.date()} (gap={curve_gap_days} days, idx={curve_idx})")

    S0 = context["X_eur"][curve_idx: curve_idx + 1].to(DEVICE).double()
    with torch.no_grad():
        z0 = context["model"].encoder(S0)

    eps = 1e-3

    def forward_swap_from_z(z, expiry, tenor):
        out = time0_forward_swap_and_annuity_from_z(
            model=context["model"],
            z0=z,
            decoder_tau_grid_base=context["decoder_tau_grid_base"],
            expiry=float(expiry),
            tenor=int(tenor),
            g0_floor=G0_FLOOR,
            accrual=ACCRUAL,
        )
        return float(out["forward_swap"])

    z_base = z0.detach().clone()

    for expiry in [1, 5]:
        print(f"\nLATENT SENSITIVITIES AT EXPIRY {expiry}Y")
        for tenor in [5, 10]:
            s0 = forward_swap_from_z(z_base, expiry, tenor)

            z_up_1 = z_base.clone()
            z_dn_1 = z_base.clone()
            z_up_1[:, 0] += eps
            z_dn_1[:, 0] -= eps

            z_up_2 = z_base.clone()
            z_dn_2 = z_base.clone()
            z_up_2[:, 1] += eps
            z_dn_2[:, 1] -= eps

            ds_dz1 = (forward_swap_from_z(z_up_1, expiry, tenor) - forward_swap_from_z(z_dn_1, expiry, tenor)) / (
                        2 * eps)
            ds_dz2 = (forward_swap_from_z(z_up_2, expiry, tenor) - forward_swap_from_z(z_dn_2, expiry, tenor)) / (
                        2 * eps)

            print(f"{expiry}Yx{tenor}Y: swap={s0:.6f}, dS/dz1={ds_dz1:.6f}, dS/dz2={ds_dz2:.6f}")

    sim = simulate_once_for_date(context["model"], z0)

    print("\n" + "-" * 70)
    print("LATENT PATH DIAGNOSTICS")
    print("-" * 70)

    for expiry in [1, 5]:
        expiry_idx = int(round(expiry / DT))
        z_exp = sim["z_paths"][:, expiry_idx, :].detach().cpu().numpy()

        z1 = z_exp[:, 0]
        z2 = z_exp[:, 1]

        print(f"\nExpiry {expiry}Y")
        print(f"  z1 mean/std : {np.mean(z1):.6f} / {np.std(z1, ddof=0):.6f}")
        print(f"  z2 mean/std : {np.mean(z2):.6f} / {np.std(z2, ddof=0):.6f}")
        print(f"  corr(z1,z2) : {np.corrcoef(z1, z2)[0, 1]:.6f}")
        print(
            f"  z1 q05/q50/q95 : "
            f"{np.quantile(z1, 0.05):.6f} / {np.quantile(z1, 0.50):.6f} / {np.quantile(z1, 0.95):.6f}"
        )
        print(
            f"  z2 q05/q50/q95 : "
            f"{np.quantile(z2, 0.05):.6f} / {np.quantile(z2, 0.50):.6f} / {np.quantile(z2, 0.95):.6f}"
        )

    print("\n" + "-" * 70)
    print("APPROX VARIANCE DECOMPOSITION")
    print("-" * 70)

    sens = {
        (1, 5): (-0.267063, -0.173559),
        (1, 10): (-0.301869, -0.049661),
        (5, 5): (-0.330469, 0.050264),
        (5, 10): (-0.336045, 0.073840),
    }

    latent_stats = {
        1: {"std1": 0.013009, "std2": 0.009918, "corr": 0.108524},
        5: {"std1": 0.028170, "std2": 0.014244, "corr": 0.503064},
    }

    for (expiry, tenor), (g1, g2) in sens.items():
        s = latent_stats[expiry]
        var1 = s["std1"] ** 2
        var2 = s["std2"] ** 2
        cov12 = s["corr"] * s["std1"] * s["std2"]

        c1 = (g1 ** 2) * var1
        c2 = (g2 ** 2) * var2
        c12 = 2.0 * g1 * g2 * cov12
        total = c1 + c2 + c12

        print(f"\n{expiry}Yx{tenor}Y")
        print(f"  z1 contribution   : {c1 / total:.3f}")
        print(f"  z2 contribution   : {c2 / total:.3f}")
        print(f"  covariance term   : {c12 / total:.3f}")
        print(f"  approx std (bp)   : {np.sqrt(total) * 10000:.2f}")

    rows = []
    day_market = market_vols[market_vols["as_of_date"] == market_date].copy()

    pathwise_by_structure = {}

    for expiry, tenor in EXAMPLE_STRUCTURES:
        label = structure_label(expiry, tenor)
        print(f"\nPricing {label} ...")

        try:
            result = price_structure_for_date(
                model=context["model"],
                decoder_tau_grid_base=context["decoder_tau_grid_base"],
                z0=z0,
                sim=sim,
                expiry=expiry,
                tenor=tenor,
            )
        except Exception as exc:
            warnings.warn(f"Skipping {label}: {exc}", RuntimeWarning)
            continue

        swap_rate_paths = result["swap_rate"].detach().cpu()
        annuity_paths = result["annuity"].detach().cpu()

        pathwise_by_structure[(int(expiry), int(tenor))] = {
            "label": label,
            "swap_rate": swap_rate_paths.numpy().reshape(-1),
            "annuity": annuity_paths.numpy().reshape(-1),
        }

        swap_mean = float(swap_rate_paths.mean().item())
        swap_std = float(swap_rate_paths.std(unbiased=False).item())
        swap_q05 = float(torch.quantile(swap_rate_paths, 0.05).item())
        swap_q50 = float(torch.quantile(swap_rate_paths, 0.50).item())
        swap_q95 = float(torch.quantile(swap_rate_paths, 0.95).item())

        ann_mean = float(annuity_paths.mean().item())
        ann_std = float(annuity_paths.std(unbiased=False).item())

        market_row = day_market[
            (day_market["option_maturity"] == int(expiry))
            & (day_market["swap_tenor"] == int(tenor))
        ]
        market_vol = float(market_row["vol"].iloc[0]) if not market_row.empty else np.nan
        error_bp = (
            (result["model_vol"] - market_vol) * 10000
            if np.isfinite(market_vol) and np.isfinite(result["model_vol"])
            else np.nan
        )

        print(f"  Forward swap t0 : {result['forward_swap_t0']:.6f}")
        print(f"  ATM strike      : {result['strike']:.6f}")
        print(f"  MC price        : {result['mc_price']:.8f}")
        print(f"  MC stderr       : {result['mc_stderr']:.8f}")
        print(f"  Model vol (bp)  : {result['model_vol'] * 10000:.2f}" if np.isfinite(result["model_vol"]) else "  Model vol (bp)  : nan")
        print(f"  Market vol (bp) : {market_vol * 10000:.2f}" if np.isfinite(market_vol) else "  Market vol (bp) : missing")
        print(f"  Error (bp)      : {error_bp:.2f}" if np.isfinite(error_bp) else "  Error (bp)      : nan")

        print(f"  Swap@expiry mean : {swap_mean:.6f}")
        print(f"  Swap@expiry std  : {swap_std * 10000:.2f} bp")
        print(f"  Swap@expiry q05  : {swap_q05:.6f}")
        print(f"  Swap@expiry q50  : {swap_q50:.6f}")
        print(f"  Swap@expiry q95  : {swap_q95:.6f}")
        print(f"  Annuity@expiry mean : {ann_mean:.6f}")
        print(f"  Annuity@expiry std  : {ann_std:.6f}")

        rows.append({
            "requested_date": requested_date,
            "market_date": market_date,
            "curve_date": curve_date,
            "label": label,
            "option_maturity": int(expiry),
            "swap_tenor": int(tenor),
            "forward_swap_t0_pct": result["forward_swap_t0"] * 100,
            "annuity_t0": result["annuity_t0"],
            "strike_pct": result["strike"] * 100,
            "mc_price": result["mc_price"],
            "mc_stderr": result["mc_stderr"],
            "valid_decode_frac": result["valid_decode_frac"],
            "mean_swap_rate_at_expiry_pct": swap_mean * 100,
            "std_swap_rate_at_expiry_bp": swap_std * 10000,
            "q05_swap_rate_at_expiry_pct": swap_q05 * 100,
            "q50_swap_rate_at_expiry_pct": swap_q50 * 100,
            "q95_swap_rate_at_expiry_pct": swap_q95 * 100,
            "mean_annuity_at_expiry": ann_mean,
            "std_annuity_at_expiry": ann_std,
            "actual_expiry": result["actual_expiry"],
            "model_vol_bp": result["model_vol"] * 10000 if np.isfinite(result["model_vol"]) else np.nan,
            "market_vol_bp": market_vol * 10000 if np.isfinite(market_vol) else np.nan,
            "vol_error_bp": error_bp,
        })

    print("\n" + "-" * 70)
    print("TENOR DIFFERENTIATION DIAGNOSTICS")
    print("-" * 70)

    compare_pairs = [
        ((1, 5), (1, 10)),
        ((5, 5), (5, 10)),
    ]

    for left_key, right_key in compare_pairs:
        if left_key not in pathwise_by_structure or right_key not in pathwise_by_structure:
            continue

        left = pathwise_by_structure[left_key]
        right = pathwise_by_structure[right_key]

        x = left["swap_rate"]
        y = right["swap_rate"]
        spread = y - x  # 10Y minus 5Y

        corr = float(np.corrcoef(x, y)[0, 1])
        mean_spread_bp = float(np.mean(spread) * 10000.0)
        std_spread_bp = float(np.std(spread, ddof=0) * 10000.0)
        q05_spread_bp = float(np.quantile(spread, 0.05) * 10000.0)
        q50_spread_bp = float(np.quantile(spread, 0.50) * 10000.0)
        q95_spread_bp = float(np.quantile(spread, 0.95) * 10000.0)

        print(f"\nExpiry {left_key[0]}Y: {left['label']} vs {right['label']}")
        print(f"  Corr(swap_5Y, swap_10Y)      : {corr:.6f}")
        print(f"  Mean spread (10Y-5Y)         : {mean_spread_bp:.2f} bp")
        print(f"  Std spread  (10Y-5Y)         : {std_spread_bp:.2f} bp")
        print(f"  Spread q05 / q50 / q95       : "
              f"{q05_spread_bp:.2f} / {q50_spread_bp:.2f} / {q95_spread_bp:.2f} bp")

    if not rows:
        raise RuntimeError("Single example produced no successful rows.")

    df = pd.DataFrame(rows).sort_values(["option_maturity", "swap_tenor"]).reset_index(drop=True)
    save_table(
        df,
        os.path.join(OUT_DIR, "example_comparison.csv"),
        os.path.join(OUT_DIR, "example_comparison.xlsx"),
    )

    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = np.arange(len(df))
    width = 0.36
    ax.bar(x - width / 2, df["market_vol_bp"], width=width, label="Market")
    ax.bar(x + width / 2, df["model_vol_bp"], width=width, label="Model")
    ax.set_xticks(x)
    ax.set_xticklabels(df["label"], rotation=45)
    ax.set_ylabel("Normal vol (bp)")
    ax.set_title(f"{CCY} ATM swaption vols on {market_date.date()}")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, "example_comparison.png")
    plt.savefig(plot_path, dpi=PLOT_DPI, bbox_inches="tight")
    print(f"Saved plot -> {plot_path}")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)

    print("\nExample summary:")
    print(df[["label", "model_vol_bp", "market_vol_bp", "vol_error_bp"]].to_string(index=False))
    return df


# ---------------------------------------------------------------------
# Panel comparison
# ---------------------------------------------------------------------
def run_panel_comparison(context):
    print("\n" + "=" * 70)
    print("RUNNING PANEL COMPARISON")
    print("=" * 70)

    market_vols = context["market_vols"]
    dates_eur = context["dates_eur"]
    model = context["model"]

    structures_all = sorted({
        (int(e), int(t))
        for e, t in zip(market_vols["option_maturity"], market_vols["swap_tenor"])
    })

    max_horizon = int(round(N_STEPS * DT))
    structures = [
        (e, t)
        for e, t in structures_all
        if e <= max_horizon and e + t <= int(model.tau_max)
    ]
    if not structures:
        raise RuntimeError("No market structures fit within the simulation/model horizon.")

    dates = pd.to_datetime(sorted(market_vols["as_of_date"].unique()))
    dates = dates[(dates >= dates_eur.min()) & (dates <= dates_eur.max())]
    if len(dates) == 0:
        raise RuntimeError("No overlapping dates between market vol data and EUR curve data.")

    if NEUTRAL_DATES is not None:
        neutral = pd.to_datetime(NEUTRAL_DATES)
        dates = dates[dates.isin(neutral)]

    if len(dates) > MAX_DATES:
        dates = dates[:MAX_DATES]

    print(f"Using {len(dates)} dates and {len(structures)} structures.")

    rows = []
    for i, market_date in enumerate(dates, start=1):
        curve_idx = nearest_idx(dates_eur, market_date)
        curve_date = pd.Timestamp(dates_eur.iloc[curve_idx])
        gap_days = abs((curve_date - market_date).days)

        print(f"\n[{i}/{len(dates)}] market={market_date.date()} | curve={curve_date.date()} | gap={gap_days}d")
        if gap_days > 31:
            print("  skipped because nearest curve is too far away")
            continue

        S0 = context["X_eur"][curve_idx: curve_idx + 1].to(DEVICE).double()
        with torch.no_grad():
            z0 = model.encoder(S0)

        sim = simulate_once_for_date(model, z0)
        day_market = market_vols[market_vols["as_of_date"] == market_date].copy()

        for expiry, tenor in structures:
            label = structure_label(expiry, tenor)
            try:
                result = price_structure_for_date(
                    model=model,
                    decoder_tau_grid_base=context["decoder_tau_grid_base"],
                    z0=z0,
                    sim=sim,
                    expiry=expiry,
                    tenor=tenor,
                )
            except Exception as exc:
                warnings.warn(f"Skipping {market_date.date()} {label}: {exc}", RuntimeWarning)
                continue

            market_row = day_market[
                (day_market["option_maturity"] == int(expiry))
                & (day_market["swap_tenor"] == int(tenor))
            ]
            market_vol = float(market_row["vol"].iloc[0]) if not market_row.empty else np.nan
            error_bp = (
                (result["model_vol"] - market_vol) * 10000
                if np.isfinite(market_vol) and np.isfinite(result["model_vol"])
                else np.nan
            )

            rows.append({
                "as_of_date": market_date,
                "curve_date": curve_date,
                "label": label,
                "option_maturity": int(expiry),
                "swap_tenor": int(tenor),
                "model_vol_bp": result["model_vol"] * 10000 if np.isfinite(result["model_vol"]) else np.nan,
                "market_vol_bp": market_vol * 10000 if np.isfinite(market_vol) else np.nan,
                "vol_error_bp": error_bp,
                "mc_price": result["mc_price"],
                "mc_stderr": result["mc_stderr"],
                "valid_decode_frac": result["valid_decode_frac"],
                "forward_swap_t0_pct": result["forward_swap_t0"] * 100,
                "strike_pct": result["strike"] * 100,
            })

    if not rows:
        raise RuntimeError("Panel comparison produced no rows.")

    df = pd.DataFrame(rows).sort_values(["as_of_date", "option_maturity", "swap_tenor"]).reset_index(drop=True)
    save_table(
        df,
        os.path.join(OUT_DIR, "vol_comparison.csv"),
        os.path.join(OUT_DIR, "vol_comparison.xlsx"),
    )

    df_v = df.dropna(subset=["model_vol_bp", "market_vol_bp"]).copy()
    if df_v.empty:
        print("[WARNING] No rows have both model and market vols. Skipping plots.")
        return df

    summary = (
        df_v.groupby("label")
        .agg(
            mean_err_bp=("vol_error_bp", "mean"),
            mae_bp=("vol_error_bp", lambda x: np.abs(x).mean()),
            rmse_bp=("vol_error_bp", lambda x: np.sqrt((x ** 2).mean())),
            n=("vol_error_bp", "count"),
        )
        .reset_index()
        .sort_values("mean_err_bp")
    )
    save_table(summary, os.path.join(OUT_DIR, "vol_comparison_summary.csv"))

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(df_v["market_vol_bp"], df_v["model_vol_bp"], alpha=0.65, s=30)
    lo = min(df_v["market_vol_bp"].min(), df_v["model_vol_bp"].min()) - 2
    hi = max(df_v["market_vol_bp"].max(), df_v["model_vol_bp"].max()) + 2
    ax.plot([lo, hi], [lo, hi], "r--", lw=1)
    ax.set_xlabel("Market normal vol (bp)")
    ax.set_ylabel("Model implied normal vol (bp)")
    ax.set_title(f"{CCY} ATM swaption vols: model vs market")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    scatter_path = os.path.join(OUT_DIR, "vol_scatter.png")
    plt.savefig(scatter_path, dpi=PLOT_DPI, bbox_inches="tight")
    print(f"Saved plot -> {scatter_path}")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(summary)), 4.5))
    ax.bar(summary["label"], summary["mean_err_bp"])
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_ylabel("Mean vol error (bp)")
    ax.set_title(f"{CCY} ATM swaption vol error by structure")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    bars_path = os.path.join(OUT_DIR, "vol_error_by_type.png")
    plt.savefig(bars_path, dpi=PLOT_DPI, bbox_inches="tight")
    print(f"Saved plot -> {bars_path}")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)

    print("\nPanel summary:")
    print(summary.to_string(index=False))
    return df


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------
def main():
    context = build_context()

    example_df = None
    panel_df = None

    if RUN_SINGLE_EXAMPLE:
        example_df = run_single_example(context)

    if RUN_PANEL_COMPARISON:
        panel_df = run_panel_comparison(context)

    return {
        "example_df": example_df,
        "panel_df": panel_df,
    }


if __name__ == "__main__":
    main()