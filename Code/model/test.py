import torch
import numpy as np

# adjust these imports if your module path differs
from Code.model.full_model import FullModel
from Code.load_swapdata import my_data
from Code.Pricing.simulate_model import (
    simulate_latent_paths,
    compute_latent_statistics,
    resolve_curve_index,
)

# =========================
# SETTINGS
# =========================
MODE = "both"   # "real", "sim", or "both"
CHECKPOINT_PATH = r"C:\Users\Bruger\PycharmProjects\MasterThesis\Figures\TrainingResults\dim2_baseline\ep3500\checkpoint_dim2_ep3500.pt"

USE = "bbg"
CCY_FILTER = "EUR"
AS_OF_DATE = None          # e.g. "2016-08-30"
N_PATHS = 500
N_STEPS = 24
DT = 1 / 365
DIFFUSION_SCALE = 0.1
DEVICE = "cpu"
DTYPE = torch.float64
SEED = 1234


def load_model(checkpoint_path, device="cpu", dtype=torch.float64):
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = FullModel().to(device)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device=device, dtype=dtype)
    model.eval()
    return model


@torch.no_grad()
def stability_report(model, z, tau=None):
    _, aux = model.decode_from_z(
        z, tau=tau, do_arb_checks=True, return_aux=True
    )

    expo = aux["A_vals"] - aux["B_vals"] * aux["G_vals"]
    Pmkt = aux["P_mkt"]

    print("max |A|:", aux["A_vals"].abs().max().item())
    print("max |B|:", aux["B_vals"].abs().max().item())

    expo = aux["A_vals"] - aux["B_vals"] * aux["G_vals"]
    idx = torch.nonzero(torch.abs(expo) == torch.abs(expo).max(), as_tuple=False)[0]
    print("worst index:", idx)

    report = {
        "n_points": int(z.shape[0]),
        "finite_G": torch.isfinite(aux["G_vals"]).all().item(),
        "finite_sigmas": torch.isfinite(aux["sigmas"]).all().item(),
        "finite_rhos": torch.isfinite(aux["rhos"]).all().item(),
        "finite_alpha": torch.isfinite(aux["alpha"]).all().item(),
        "finite_beta": torch.isfinite(aux["beta"]).all().item(),
        "finite_gamma": torch.isfinite(aux["gamma"]).all().item(),
        "finite_A": torch.isfinite(aux["A_vals"]).all().item(),
        "finite_B": torch.isfinite(aux["B_vals"]).all().item(),
        "finite_P": torch.isfinite(aux["P_full"]).all().item(),

        "min_abs_G": aux["G_vals"].abs().min().item(),
        "max_abs_alpha": aux["alpha"].abs().max().item(),
        "max_abs_beta": aux["beta"].abs().max().item(),
        "max_abs_gamma": aux["gamma"].abs().max().item(),
        "max_abs_exponent": expo.abs().max().item(),

        "min_sigma": aux["sigmas"].min().item(),
        "max_sigma": aux["sigmas"].max().item(),
        "max_abs_rho": aux["rhos"].abs().max().item() if aux["rhos"].numel() else 0.0,

        "P_min": aux["P_full"].min().item(),
        "P_max": aux["P_full"].max().item(),

        "num_P_le_0": (Pmkt <= 0).sum().item(),
        "num_P_gt_1": (Pmkt > 1.0 + 1e-8).sum().item(),
        "num_monotonicity_violations": (Pmkt[:, 1:] > Pmkt[:, :-1] + 1e-8).sum().item(),

        "max_abs_R": (
            aux["arb"]["max_abs_R"].max().item()
            if aux["arb"] is not None else float("nan")
        ),
    }
    return report, aux


def print_report(name, report):
    print(f"\n=== {name} ===")
    for k, v in report.items():
        print(f"{k:28s}: {v}")


def latent_summary(name, z):
    print(f"\n--- {name} latent summary ---")
    print("shape:", tuple(z.shape))
    print("z min :", z.min(dim=0).values)
    print("z max :", z.max(dim=0).values)
    print("z mean:", z.mean(dim=0))
    print("z std :", z.std(dim=0))


def fraction_outside_3std(z_ref, z_test):
    mu = z_ref.mean(dim=0)
    sd = z_ref.std(dim=0)
    lo = mu - 3.0 * sd
    hi = mu + 3.0 * sd

    outside_any = ((z_test < lo) | (z_test > hi)).any(dim=1)
    return outside_any.float().mean().item(), lo, hi


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model = load_model(CHECKPOINT_PATH, device=DEVICE, dtype=DTYPE)

    meta, X_tensor, meta_full, X_tensor_full, tenors, df_wide, df_wide_all, SCALE_IS_PERCENT = my_data(
        use=USE,
        ccy_filter=CCY_FILTER,
    )
    X_tensor = X_tensor.to(device=DEVICE, dtype=DTYPE)

    # -------------------------
    # REAL DATA TEST
    # -------------------------
    z_real = None
    if MODE in {"real", "both", "sim"}:
        with torch.no_grad():
            z_real = model.encoder(X_tensor)

    if MODE in {"real", "both"}:
        latent_summary("real", z_real)
        report_real, _ = stability_report(model, z_real)
        print_report("real_encoded_data", report_real)

    # -------------------------
    # SIMULATION TEST
    # -------------------------
    if MODE in {"sim", "both"}:
        start_idx = resolve_curve_index(meta, as_of_date=AS_OF_DATE)
        S0 = X_tensor[start_idx:start_idx + 1].to(device=DEVICE, dtype=DTYPE)

        with torch.no_grad():
            z0 = model.encoder(S0)

        z_train_mean, z_train_cov, z_train_std = compute_latent_statistics(
            model, X_tensor, DEVICE, model.latent_dim
        )

        z_paths, r_paths, mu_paths, L_paths = simulate_latent_paths(
            model=model,
            z0=z0,
            n_paths=N_PATHS,
            n_steps=N_STEPS,
            dt=DT,
            device=DEVICE,
            diffusion_scale=DIFFUSION_SCALE,
        )

        z_sim = z_paths.reshape(-1, z_paths.shape[-1])

        latent_summary("simulated", z_sim)
        report_sim, _ = stability_report(model, z_sim)
        print_report("simulated_latent_points", report_sim)

        frac_out, lo, hi = fraction_outside_3std(z_real, z_sim)
        print("\n--- sim vs real latent cloud ---")
        print("z0:", z0.squeeze(0))
        print("real 3std low :", lo)
        print("real 3std high:", hi)
        print("fraction of simulated points outside real ±3 std:", frac_out)


if __name__ == "__main__":
    main()