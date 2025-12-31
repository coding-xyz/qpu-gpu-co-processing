import math
import itertools
from dataclasses import dataclass
from typing import Dict, Tuple, List

import torch

from utils import *

# ----------------------------
# Build POVM elements for settings
# ----------------------------
def build_povm_for_settings(settings: List[str], n: int, device, dtype=torch.complex64):
    """
    For each setting (like 'XZ'), build POVM elements M_{setting, outcome}
    where outcome is bitstring length n.
    Returns:
      M: (K, d, d) complex tensor
      setting_outcome_index: list of (setting, outcome_str)
    """
    d = 2**n
    Ms = []
    idx = []

    for setting in settings:
        Us = [U_for_basis(b, device=device, dtype=dtype) for b in setting]
        U = kron_n(Us)  # apply before Z measurement
        Udag = U.conj().T

        for bits in itertools.product([0, 1], repeat=n):
            Pz = kron_n([projector(bi, device=device, dtype=dtype) for bi in bits])  # Z-basis projector
            M = Udag @ Pz @ U  # POVM element in original frame
            Ms.append(M)
            outcome = "".join(str(bi) for bi in bits)
            idx.append((setting, outcome))

    M = torch.stack(Ms, dim=0)  # (K, d, d)
    return M, idx


# ----------------------------
# MLE tomography with Cholesky parameterization
# ----------------------------
@dataclass
class TomographyResult:
    rho: torch.Tensor
    loss_history: List[float]


def cholesky_param_to_rho(params: torch.Tensor, n: int) -> torch.Tensor:
    """
    params: real tensor that encodes complex lower-triangular T (d x d)
    We pack real and imag parts.
    """
    d = 2**n
    # params shape: (d, d, 2) real, but only lower triangle used
    re = params[..., 0]
    im = params[..., 1]
    T = torch.tril(re + 1j * im)
    A = T @ T.conj().T
    rho = A / torch.trace(A)
    return rho


def mle_tomography(
    counts: Dict[str, Dict[str, int]],
    n: int,
    iters: int = 2000,
    lr: float = 0.05,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    dtype=torch.complex64,
    eps: float = 1e-12,
    verbose_every: int = 200,
) -> TomographyResult:
    """
    counts[setting][outcome_bitstring] = count
    setting is string length n in {'X','Y','Z'}^n
    """
    device = torch.device(device)
    d = 2**n

    settings = sorted(counts.keys())
    M, idx = build_povm_for_settings(settings, n=n, device=device, dtype=dtype)  # (K,d,d)

    # Build observed frequencies vector y and total shots per (setting,outcome)
    y = torch.zeros(len(idx), device=device, dtype=torch.float32)
    Nset = {s: sum(counts[s].values()) for s in settings}

    for k, (s, outcome) in enumerate(idx):
        y[k] = float(counts[s].get(outcome, 0))

    # Initialize parameters for T (lower triangle), small random
    params = torch.zeros((d, d, 2), device=device, dtype=torch.float32, requires_grad=True)
    with torch.no_grad():
        params.uniform_(-0.01, 0.01)
        # add diagonal bias so it's not near-zero
        for i in range(d):
            params[i, i, 0] += 1.0

    opt = torch.optim.Adam([params], lr=lr)
    loss_hist = []

    # Precompute for likelihood: group indices per setting to normalize to probs
    # But our M already defines a valid POVM for each setting; sum over outcomes == I.
    # The predicted probabilities for each (setting,outcome) are p_k = Tr(M_k rho).
    for t in range(1, iters + 1):
        opt.zero_grad(set_to_none=True)

        rho = cholesky_param_to_rho(params, n=n)  # (d,d) complex
        # p = real(Tr(M rho)) for all k in batch:
        # Using einsum: Tr(M_k rho) = sum_{ij} M_k[i,j] * rho[j,i]
        # Equivalent: (M * rho^T).sum over i,j
        p = torch.einsum("kij,ji->k", M, rho).real  # (K,)
        p = torch.clamp(p, min=eps, max=1.0)

        # Negative log-likelihood for multinomial counts:
        # L = - sum_k y_k log p_k  (constants omitted)
        loss = -(y * torch.log(p)).sum()

        loss.backward()
        opt.step()

        lv = float(loss.detach().cpu().item())
        loss_hist.append(lv)

        if verbose_every and (t % verbose_every == 0 or t == 1 or t == iters):
            # sanity: trace and min eigenvalue (CPU eig for small d only)
            tr = torch.trace(rho).real.detach().cpu().item()
            print(f"[{t:5d}/{iters}] loss={lv:.3f} Tr(rho)={tr:.6f}")

    with torch.no_grad():
        rho = cholesky_param_to_rho(params, n=n).detach()

    return TomographyResult(rho=rho, loss_history=loss_hist)


# ----------------------------
# Demo: simulate 1-qubit |+> state, do tomography from synthetic counts
# ----------------------------
def simulate_counts_from_rho(
    rho: torch.Tensor,
    n: int,
    shots_per_setting: int = 2000,
    settings: List[str] = None,
    seed: int = 0,
) -> Dict[str, Dict[str, int]]:
    device = rho.device
    dtype = rho.dtype
    if settings is None:
        settings = ["".join(s) for s in itertools.product(["X", "Y", "Z"], repeat=n)]

    M, idx = build_povm_for_settings(settings, n=n, device=device, dtype=dtype)
    p = torch.einsum("kij,ji->k", M, rho).real
    p = torch.clamp(p, min=1e-12)
    # Group by setting
    counts = {s: {} for s in settings}
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    # For each setting, sample outcomes
    # idx is ordered by settings then outcomes
    offset = 0
    for s in settings:
        probs = p[offset: offset + 2**n].detach().cpu()
        probs = probs / probs.sum()
        samples = torch.multinomial(probs, num_samples=shots_per_setting, replacement=True, generator=g)
        # accumulate
        for k in samples.tolist():
            outcome = format(k, f"0{n}b")
            counts[s][outcome] = counts[s].get(outcome, 0) + 1
        offset += 2**n
    return counts
