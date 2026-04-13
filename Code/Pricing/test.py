import torch

# adjust this import to your project structure
from Code.model.full_model import FullModel


@torch.no_grad()
def stability_report(model, z, tau=None):
    P, aux = model.decode_from_z(
        z, tau=tau, do_arb_checks=True, return_aux=True
    )

    expo = aux["A_vals"] - aux["B_vals"] * aux["G_vals"]
    Pmkt = aux["P_mkt"]

    report = {
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


def basic_pass_fail(report, g_tol=1e-6, exp_tol=80.0, rho_tol=0.9999):
    bad = []

    for key in [
        "finite_G", "finite_sigmas", "finite_rhos", "finite_alpha",
        "finite_beta", "finite_gamma", "finite_A", "finite_B", "finite_P"
    ]:
        if not report[key]:
            bad.append(f"{key}=False")

    if report["min_abs_G"] < g_tol:
        bad.append(f"min_abs_G<{g_tol}")

    if report["max_abs_exponent"] > exp_tol:
        bad.append(f"max_abs_exponent>{exp_tol}")

    if report["max_abs_rho"] > rho_tol:
        bad.append(f"max_abs_rho>{rho_tol}")

    if report["num_P_le_0"] > 0:
        bad.append("P<=0 found")

    if report["num_P_gt_1"] > 0:
        bad.append("P>1 found")

    if report["num_monotonicity_violations"] > 0:
        bad.append("discount curve not decreasing")

    return bad


def make_test_batches(latent_dim, device, dtype):
    return {
        "zeros": torch.zeros(8, latent_dim, device=device, dtype=dtype),
        "small_normal": 0.1 * torch.randn(8, latent_dim, device=device, dtype=dtype),
        "unit_normal": 1.0 * torch.randn(8, latent_dim, device=device, dtype=dtype),
        "wide_normal": 3.0 * torch.randn(8, latent_dim, device=device, dtype=dtype),
        "extreme_manual": torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [-1.0, -1.0],
                [3.0, 0.0],
                [0.0, 3.0],
                [-3.0, 0.0],
                [0.0, -3.0],
                [5.0, -5.0],
            ],
            device=device,
            dtype=dtype,
        ),
    }


def load_model(checkpoint_path, device="cpu", dtype=torch.float64):
    model = FullModel()
    ckpt = torch.load(checkpoint_path, map_location=device)

    # common checkpoint layouts
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)

    model = model.to(device=device, dtype=dtype)
    model.eval()
    return model


def main():
    torch.manual_seed(0)

    checkpoint_path = "your_checkpoint.pt"
    device = "cpu"
    dtype = torch.float64   # use float64 first for stability diagnostics

    model = load_model(checkpoint_path, device=device, dtype=dtype)

    test_batches = make_test_batches(
        latent_dim=model.latent_dim,
        device=device,
        dtype=dtype,
    )

    for name, z in test_batches.items():
        report, aux = stability_report(model, z)
        print_report(name, report)

        bad = basic_pass_fail(report)
        if bad:
            print("FAIL:", ", ".join(bad))
        else:
            print("PASS")


if __name__ == "__main__":
    main()