import torch


def _default_eigen_tol(dtype: torch.dtype) -> float:
    return max(100.0 * torch.finfo(dtype).eps, 1e-12)


def _num_rhos_for_dim(d: int) -> int:
    if d < 1:
        raise ValueError(f"Dimension must be >= 1, got {d}.")
    return d * (d - 1) // 2


def _check_input_shapes(sigmas: torch.Tensor, rhos: torch.Tensor | None = None) -> int:
    if sigmas.ndim != 2:
        raise ValueError(f"sigmas must have shape (B,d), got {tuple(sigmas.shape)}.")

    d = sigmas.shape[1]
    expected_rhos = _num_rhos_for_dim(d)

    if d == 1:
        if rhos is not None and rhos.numel() > 0:
            if rhos.ndim != 2 or rhos.shape[1] != 0:
                raise ValueError(f"rhos must be None or empty for d=1, got {tuple(rhos.shape)}.")
        return d

    if rhos is None:
        raise ValueError(f"rhos is required for d={d} (shape (B,{expected_rhos})).")

    if rhos.ndim != 2:
        raise ValueError(f"rhos must have shape (B,{expected_rhos}), got {tuple(rhos.shape)}.")

    if rhos.shape[0] != sigmas.shape[0]:
        raise ValueError(
            f"sigmas and rhos must have the same batch size, got {sigmas.shape[0]} and {rhos.shape[0]}."
        )

    if rhos.shape[1] != expected_rhos:
        raise ValueError(f"rhos is required for d={d} (shape (B,{expected_rhos})).")

    return d


def corr_from_rhos(sigmas: torch.Tensor, rhos: torch.Tensor | None = None) -> torch.Tensor:
    """
    Build the batch of correlation matrices implied by rhos.

    Args:
      sigmas: (B,d)
      rhos:   (B,d*(d-1)//2) or None for d=1

    Returns:
      R: (B,d,d)
    """
    d = _check_input_shapes(sigmas, rhos)
    B = sigmas.shape[0]
    device, dtype = sigmas.device, sigmas.dtype

    R = torch.eye(d, device=device, dtype=dtype).unsqueeze(0).repeat(B, 1, 1)

    if d == 1:
        return R
    if d == 2:
        rho12 = rhos[:, 0]
        R[:, 0, 1] = rho12
        R[:, 1, 0] = rho12
        return R
    if d == 3:
        rho12, rho13, rho23 = rhos.unbind(dim=1)
        R[:, 0, 1] = R[:, 1, 0] = rho12
        R[:, 0, 2] = R[:, 2, 0] = rho13
        R[:, 1, 2] = R[:, 2, 1] = rho23
        return R
    if d == 4:
        rho12, rho13, rho14, rho23, rho24, rho34 = rhos.unbind(dim=1)
        R[:, 0, 1] = R[:, 1, 0] = rho12
        R[:, 0, 2] = R[:, 2, 0] = rho13
        R[:, 0, 3] = R[:, 3, 0] = rho14
        R[:, 1, 2] = R[:, 2, 1] = rho23
        R[:, 1, 3] = R[:, 3, 1] = rho24
        R[:, 2, 3] = R[:, 3, 2] = rho34
        return R

    raise NotImplementedError("Only d=1,2,3,4 implemented for (sigmas, rhos) parameterization.")


def sigma_matrix_from_sigmas_rhos(sigmas: torch.Tensor, rhos: torch.Tensor | None = None) -> torch.Tensor:
    """
    Build the covariance matrix Sigma = D R D from sigmas and rhos.
    """
    R = corr_from_rhos(sigmas, rhos)
    D = torch.diag_embed(sigmas)
    return D @ R @ D


def check_sigmas_rhos(
    sigmas: torch.Tensor,
    rhos: torch.Tensor | None = None,
    *,
    eigen_tol: float | None = None,
    require_pd: bool = True,
    sigma_tol: float = 0.0,
) -> dict[str, torch.Tensor | torch.Tensor | None]:
    """
    Validate the batch of (sigmas, rhos) inputs.

    Args:
      sigmas:     (B,d)
      rhos:       (B,d*(d-1)//2) or None for d=1
      eigen_tol:  tolerance for eigenvalue tests; defaults by dtype
      require_pd: if True, require min eigenvalue > eigen_tol (strictly PD)
                  if False, require min eigenvalue >= -eigen_tol (PSD)
      sigma_tol:  sigmas must be > sigma_tol when require_pd=True,
                  otherwise sigmas must be >= -sigma_tol

    Returns:
      dict with boolean masks and diagnostics.
    """
    d = _check_input_shapes(sigmas, rhos)

    if eigen_tol is None:
        eigen_tol = _default_eigen_tol(sigmas.dtype)

    if require_pd:
        bad_sigma = (sigmas <= sigma_tol).any(dim=1)
    else:
        bad_sigma = (sigmas < -sigma_tol).any(dim=1)

    if d == 1:
        min_eig = torch.ones(sigmas.shape[0], device=sigmas.device, dtype=sigmas.dtype)
        bad_rho = torch.zeros_like(bad_sigma)
        if require_pd:
            bad_eig = torch.zeros_like(bad_sigma)
        else:
            bad_eig = torch.zeros_like(bad_sigma)
    else:
        bad_rho = (rhos.abs() > 1.0 + eigen_tol).any(dim=1)
        R = corr_from_rhos(sigmas, rhos)
        R = 0.5 * (R + R.transpose(-1, -2))
        eigvals = torch.linalg.eigvalsh(R)
        min_eig = eigvals[:, 0]
        if require_pd:
            bad_eig = min_eig <= eigen_tol
        else:
            bad_eig = min_eig < -eigen_tol

    is_valid = ~(bad_sigma | bad_rho | bad_eig)

    return {
        "is_valid": is_valid,
        "bad_sigma": bad_sigma,
        "bad_rho": bad_rho,
        "bad_eig": bad_eig,
        "min_eig": min_eig,
    }


def validate_sigmas_rhos(
    sigmas: torch.Tensor,
    rhos: torch.Tensor | None = None,
    *,
    eigen_tol: float | None = None,
    require_pd: bool = True,
    sigma_tol: float = 0.0,
) -> None:
    """
    Raise ValueError when the batch of (sigmas, rhos) is not valid.
    """
    checks = check_sigmas_rhos(
        sigmas,
        rhos,
        eigen_tol=eigen_tol,
        require_pd=require_pd,
        sigma_tol=sigma_tol,
    )

    is_valid = checks["is_valid"]
    if bool(is_valid.all()):
        return

    bad_idx = (~is_valid).nonzero(as_tuple=False).squeeze(-1)
    first_bad = int(bad_idx[0].item())

    reasons = []
    if bool(checks["bad_sigma"][first_bad]):
        if require_pd:
            reasons.append(f"sigmas must be strictly positive (> {sigma_tol})")
        else:
            reasons.append(f"sigmas must be nonnegative (>= {-sigma_tol})")
    if bool(checks["bad_rho"][first_bad]):
        reasons.append(f"some correlations exceed 1 in magnitude (tol={eigen_tol})")
    if bool(checks["bad_eig"][first_bad]):
        if require_pd:
            reasons.append(
                f"correlation matrix is not positive definite: min_eig={checks['min_eig'][first_bad].item():.3e}"
            )
        else:
            reasons.append(
                f"correlation matrix is not positive semidefinite: min_eig={checks['min_eig'][first_bad].item():.3e}"
            )

    mode = "PD" if require_pd else "PSD"
    raise ValueError(
        f"Invalid (sigmas, rhos) for {mode} Cholesky/covariance construction at batch index {first_bad}: "
        + "; ".join(reasons)
    )


def L_from_sigmas_rhos_1d(sigmas: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    sigmas: (B,1) positive [sigma1]
    returns:
      L: (B,1,1) with Sigma = L L^T = sigma^2
    """
    device, dtype = sigmas.device, sigmas.dtype
    B = sigmas.shape[0]
    s1 = sigmas[:, 0]

    L = torch.zeros(B, 1, 1, device=device, dtype=dtype)
    L[:, 0, 0] = s1
    return L


def L_from_sigmas_rhos_2d(sigmas: torch.Tensor, rhos: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    sigmas: (B,2) positive [sigma1, sigma2]
    rhos:   (B,1) in (-1,1) [rho12]
    returns:
      L: (B,2,2) lower-triangular with Sigma = L L^T
    """
    device, dtype = sigmas.device, sigmas.dtype
    B = sigmas.shape[0]

    s1 = sigmas[:, 0]
    s2 = sigmas[:, 1]
    rho = rhos[:, 0]

    # After validation, the clamp is only a small numerical safeguard.
    sqrt1mr2 = torch.sqrt(torch.clamp(1.0 - rho**2, min=eps))

    L = torch.zeros(B, 2, 2, device=device, dtype=dtype)
    L[:, 0, 0] = s1
    L[:, 1, 0] = rho * s2
    L[:, 1, 1] = sqrt1mr2 * s2

    return L


def L_from_sigmas_rhos_3d(sigmas: torch.Tensor, rhos: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    sigmas: (B,3) positive [sigma1, sigma2, sigma3]
    rhos:   (B,3) in (-1,1) ordered as [rho12, rho13, rho23]
    returns:
      L: (B,3,3) lower-triangular Cholesky factor such that Sigma = L L^T
    """
    device, dtype = sigmas.device, sigmas.dtype
    B = sigmas.shape[0]

    s1 = sigmas[:, 0]
    s2 = sigmas[:, 1]
    s3 = sigmas[:, 2]

    rho12 = rhos[:, 0]
    rho13 = rhos[:, 1]
    rho23 = rhos[:, 2]

    # After validation, the clamp is only a small numerical safeguard.
    one_minus_r12_sq = torch.clamp(1.0 - rho12**2, min=eps)
    sqrt_one_minus_r12_sq = torch.sqrt(one_minus_r12_sq)

    L00 = s1
    L10 = rho12 * s2
    L11 = s2 * sqrt_one_minus_r12_sq

    L20 = rho13 * s3
    L21 = s3 * (rho23 - rho12 * rho13) / sqrt_one_minus_r12_sq

    inside = 1.0 - rho13**2 - ((rho23 - rho12 * rho13) ** 2) / one_minus_r12_sq
    inside = torch.clamp(inside, min=eps)
    L22 = s3 * torch.sqrt(inside)

    L = torch.zeros(B, 3, 3, device=device, dtype=dtype)
    L[:, 0, 0] = L00
    L[:, 1, 0] = L10
    L[:, 1, 1] = L11
    L[:, 2, 0] = L20
    L[:, 2, 1] = L21
    L[:, 2, 2] = L22

    return L


def L_from_sigmas_rhos_4d(sigmas: torch.Tensor, rhos: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    sigmas: (B,4) positive [sigma1, sigma2, sigma3, sigma4]
    rhos:   (B,6) in (-1,1) ordered as
            [rho12, rho13, rho14, rho23, rho24, rho34]

    returns:
      L: (B,4,4) lower-triangular Cholesky factor such that Sigma = L L^T
    """
    device, dtype = sigmas.device, sigmas.dtype
    B = sigmas.shape[0]

    s1 = sigmas[:, 0]
    s2 = sigmas[:, 1]
    s3 = sigmas[:, 2]
    s4 = sigmas[:, 3]

    rho12 = rhos[:, 0]
    rho13 = rhos[:, 1]
    rho14 = rhos[:, 2]
    rho23 = rhos[:, 3]
    rho24 = rhos[:, 4]
    rho34 = rhos[:, 5]

    # After validation, the clamp is only a small numerical safeguard.
    one_minus_r12_sq = torch.clamp(1.0 - rho12**2, min=eps)
    sqrt_one_minus_r12_sq = torch.sqrt(one_minus_r12_sq)

    L00 = s1

    L10 = rho12 * s2
    L11 = s2 * sqrt_one_minus_r12_sq

    L20 = rho13 * s3
    L21 = s3 * (rho23 - rho12 * rho13) / sqrt_one_minus_r12_sq

    inside3 = 1.0 - rho13**2 - ((rho23 - rho12 * rho13) ** 2) / one_minus_r12_sq
    inside3 = torch.clamp(inside3, min=eps)
    L22 = s3 * torch.sqrt(inside3)

    L30 = rho14 * s4
    L31 = s4 * (rho24 - rho12 * rho14) / sqrt_one_minus_r12_sq

    numer32 = (
        rho34
        - rho13 * rho14
        - ((rho23 - rho12 * rho13) * (rho24 - rho12 * rho14)) / one_minus_r12_sq
    )
    L32 = s4 * numer32 / torch.sqrt(inside3)

    inside4 = (
        1.0
        - rho14**2
        - ((rho24 - rho12 * rho14) ** 2) / one_minus_r12_sq
        - (numer32 ** 2) / inside3
    )
    inside4 = torch.clamp(inside4, min=eps)
    L33 = s4 * torch.sqrt(inside4)

    L = torch.zeros(B, 4, 4, device=device, dtype=dtype)
    L[:, 0, 0] = L00

    L[:, 1, 0] = L10
    L[:, 1, 1] = L11

    L[:, 2, 0] = L20
    L[:, 2, 1] = L21
    L[:, 2, 2] = L22

    L[:, 3, 0] = L30
    L[:, 3, 1] = L31
    L[:, 3, 2] = L32
    L[:, 3, 3] = L33

    return L


def L_from_sigmas_rhos(
    sigmas: torch.Tensor,
    rhos: torch.Tensor | None = None,
    eps: float = 1e-12,
    *,
    validate: bool = True,
    require_pd: bool = True,
    eigen_tol: float | None = None,
    sigma_tol: float = 0.0,
) -> torch.Tensor:
    """
    Construct the lower-triangular factor L implied by (sigmas, rhos).

    Args:
      validate:   if True, check the implied correlation matrix before building L
      require_pd: if True, require the implied correlation matrix to be positive definite
                  and sigmas to be strictly positive. If False, allow PSD / singular inputs.
      eigen_tol:  tolerance for PD/PSD validation; defaults by dtype
      sigma_tol:  tolerance for sigma positivity / nonnegativity
    """
    d = _check_input_shapes(sigmas, rhos)

    if validate:
        validate_sigmas_rhos(
            sigmas,
            rhos,
            eigen_tol=eigen_tol,
            require_pd=require_pd,
            sigma_tol=sigma_tol,
        )

    if d == 1:
        return L_from_sigmas_rhos_1d(sigmas, eps=eps)
    if d == 2:
        return L_from_sigmas_rhos_2d(sigmas, rhos, eps=eps)
    if d == 3:
        return L_from_sigmas_rhos_3d(sigmas, rhos, eps=eps)
    if d == 4:
        return L_from_sigmas_rhos_4d(sigmas, rhos, eps=eps)
    raise NotImplementedError("Only d=1,2,3,4 implemented for (sigmas, rhos) parameterization.")
