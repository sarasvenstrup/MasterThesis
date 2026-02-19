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

    # PSD check via det(R) >= 0 for 3x3 correlation matrix
    detR = 1.0 - rho12**2 - rho13**2 - rho23**2 + 2.0 * rho12 * rho13 * rho23
    if torch.any(detR < 0):
        # Keep this lightweight; printing every batch can slow training a lot.
        print("Warning: correlation matrix not PSD. min detR:", float(detR.min()))

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
    else:
        raise NotImplementedError("Only d=1,2,3 implemented for (sigmas, rhos) parameterization.")
