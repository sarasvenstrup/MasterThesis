import torch

def L_from_sigmas_rhos_1d(sigmas: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    sigmas: (B,1) positive [sigma1]
    returns:
      L: (B,1,1) with Sigma = L L^T = sigma^2
    """
    device, dtype = sigmas.device, sigmas.dtype
    B = sigmas.shape[0]
    s1 = torch.clamp(sigmas[:, 0], min=eps)  # just to be safe

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

    one_minus_r12_sq = torch.clamp(1.0 - rho12**2, min=eps)
    sqrt_one_minus_r12_sq = torch.sqrt(one_minus_r12_sq)

    # L entries from the derived formulas
    L00 = s1
    L10 = rho12 * s2
    L11 = s2 * sqrt_one_minus_r12_sq

    L20 = rho13 * s3
    L21 = s3 * (rho23 - rho12 * rho13) / sqrt_one_minus_r12_sq

    # last diagonal must be real
    inside = 1.0 - rho13**2 - ((rho23 - rho12 * rho13)**2) / one_minus_r12_sq
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

    one_minus_r12_sq = torch.clamp(1.0 - rho12**2, min=eps)
    sqrt_one_minus_r12_sq = torch.sqrt(one_minus_r12_sq)

    # --- row 1 ---
    L00 = s1

    # --- row 2 ---
    L10 = rho12 * s2
    L11 = s2 * sqrt_one_minus_r12_sq

    # --- row 3 ---
    L20 = rho13 * s3
    L21 = s3 * (rho23 - rho12 * rho13) / sqrt_one_minus_r12_sq

    inside3 = 1.0 - rho13**2 - ((rho23 - rho12 * rho13)**2) / one_minus_r12_sq
    inside3 = torch.clamp(inside3, min=eps)
    L22 = s3 * torch.sqrt(inside3)

    # --- row 4 ---
    L30 = rho14 * s4
    L31 = s4 * (rho24 - rho12 * rho14) / sqrt_one_minus_r12_sq

    L32 = s4 * (
        rho34
        - rho13 * rho14
        - ((rho23 - rho12 * rho13) * (rho24 - rho12 * rho14)) / one_minus_r12_sq
    ) / torch.sqrt(inside3)

    inside4 = 1.0 - rho14**2 - ((rho24 - rho12 * rho14)**2) / one_minus_r12_sq \
              - (
                  (
                      rho34
                      - rho13 * rho14
                      - ((rho23 - rho12 * rho13) * (rho24 - rho12 * rho14)) / one_minus_r12_sq
                  )**2
                ) / inside3

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

def L_from_sigmas_rhos(sigmas: torch.Tensor, rhos: torch.Tensor | None = None, eps: float = 1e-12) -> torch.Tensor:
    d = sigmas.shape[1]
    if d == 1:
        return L_from_sigmas_rhos_1d(sigmas, eps=eps)
    elif d == 2:
        # rhos should be (B,1)
        if rhos is None:
            raise ValueError("rhos is required for d=2 (shape (B,1)).")
        return L_from_sigmas_rhos_2d(sigmas, rhos, eps=eps)
    elif d == 3:
        # rhos should be (B,3)
        if rhos is None:
            raise ValueError("rhos is required for d=3 (shape (B,3)).")
        return L_from_sigmas_rhos_3d(sigmas, rhos, eps=eps)
    elif d == 4:
        # rhos should be (B,3)
        if rhos is None:
            raise ValueError("rhos is required for d=4 (shape (B,4)).")
        return L_from_sigmas_rhos_4d(sigmas, rhos, eps=eps)

    else:
        raise NotImplementedError("Only d=1,2,3 implemented for (sigmas, rhos) parameterization.")
