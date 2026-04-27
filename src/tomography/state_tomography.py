import time
import torch
import numpy as np
import itertools
from typing import Dict, Optional, Tuple

def _projectors_1q(device, cdtype=torch.complex64):
    I = torch.eye(2, device=device, dtype=cdtype)
    X = torch.tensor([[0,1],[1,0]], device=device, dtype=cdtype)
    Y = torch.tensor([[0,-1j],[1j,0]], device=device, dtype=cdtype)
    Z = torch.tensor([[1,0],[0,-1]], device=device, dtype=cdtype)
    # (+,-) projectors
    Px = torch.stack([0.5*(I+X), 0.5*(I-X)], dim=0)
    Py = torch.stack([0.5*(I+Y), 0.5*(I-Y)], dim=0)
    Pz = torch.stack([0.5*(I+Z), 0.5*(I-Z)], dim=0)
    return torch.stack([Px, Py, Pz], dim=0)  # [3,2,2,2], order X,Y,Z

def _rho_from_T(T: torch.Tensor) -> torch.Tensor:
    X = T @ T.conj().transpose(-1, -2)
    return X / torch.trace(X)

def fit_1q_mle_spam(
    counts_3x2: np.ndarray,
    A_meas_2x2: np.ndarray,
    steps: int = 2000,
    lr: float = 0.05,
    seed: int = 0,
    device: Optional[str] = None,
    dtype: str = "complex64",
) -> Dict[str, object]:
    """
    SPAM-aware MLE state tomography for 1 qubit.

    Input:
      counts_3x2: shape (3,2), settings [X,Y,Z], outcomes [+, -] -> [0,1]
      A_meas_2x2: shape (2,2), q_obs = A_meas @ p_true
                 columns=true outcomes, rows=observed outcomes
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    cdtype = torch.complex64 if dtype == "complex64" else torch.complex128
    rdtype = torch.float32 if cdtype == torch.complex64 else torch.float64

    M = _projectors_1q(device=device, cdtype=cdtype)              # [3,2,2,2]
    C = torch.tensor(counts_3x2, device=device, dtype=rdtype)     # [3,2]
    A = torch.tensor(A_meas_2x2, device=device, dtype=rdtype)     # [2,2]

    d = 2
    T_re = torch.randn(d, d, device=device, dtype=rdtype, requires_grad=True)
    T_im = torch.randn(d, d, device=device, dtype=rdtype, requires_grad=True)
    opt = torch.optim.Adam([T_re, T_im], lr=lr)

    eps = 1e-12
    t0 = time.perf_counter()
    nll_val = None

    for _ in range(steps):
        opt.zero_grad()
        T = torch.complex(T_re, T_im)
        rho = _rho_from_T(T)  # [2,2]

        p_true = torch.einsum("moij,ji->mo", M, rho).real          # [3,2]
        p_true = torch.clamp(p_true, min=eps, max=1-eps)

        q_obs = (A @ p_true.transpose(0,1)).transpose(0,1)         # [3,2]
        q_obs = torch.clamp(q_obs, min=eps, max=1-eps)

        nll = -(C * torch.log(q_obs)).sum()
        nll.backward()
        opt.step()
        nll_val = float(nll.detach().cpu())

    runtime = time.perf_counter() - t0
    with torch.no_grad():
        T = torch.complex(T_re, T_im)
        rho_np = _rho_from_T(T).detach().cpu().numpy().astype(np.complex64)

    return {
        "rho": rho_np,
        "nll_final": nll_val,
        "runtime_sec": float(runtime),
        "device_used": device,
        "steps": int(steps),
        "lr": float(lr),
        "seed": int(seed),
    }

def _projectors_2q(device, cdtype=torch.complex64) -> Tuple[torch.Tensor, list]:
    sq = _projectors_1q(device, cdtype=cdtype)
    settings = []
    labels = []
    # fixed order: XX,XY,XZ,YX,YY,YZ,ZX,ZY,ZZ
    for a,b in itertools.product(["X","Y","Z"], repeat=2):
        labels.append(a+b)
        P = torch.stack([
            torch.kron(sq[a][0], sq[b][0]),
            torch.kron(sq[a][0], sq[b][1]),
            torch.kron(sq[a][1], sq[b][0]),
            torch.kron(sq[a][1], sq[b][1]),
        ], dim=0)  # [4,4,4]
        settings.append(P)
    return torch.stack(settings, dim=0), labels  # [9,4,4,4]

def fit_2q_mle_spam(
    counts_9x4: np.ndarray,
    A_meas_4x4: np.ndarray,
    steps: int = 3000,
    lr: float = 0.03,
    seed: int = 0,
    device: Optional[str] = None,
    dtype: str = "complex64",
) -> Dict[str, object]:
    """
    SPAM-aware MLE state tomography for 2 qubits.

    Input:
      counts_9x4: shape (9,4), settings [XX..ZZ],
        outcomes [++, +-, -+, --] -> [00,01,10,11]
      A_meas_4x4: shape (4,4), q_obs = A_meas @ p_true
        columns=true outcomes, rows=observed outcomes
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    cdtype = torch.complex64 if dtype == "complex64" else torch.complex128
    rdtype = torch.float32 if cdtype == torch.complex64 else torch.float64

    M, labels = _projectors_2q(device=device, cdtype=cdtype)          # [9,4,4,4]
    C = torch.tensor(counts_9x4, device=device, dtype=rdtype)         # [9,4]
    A = torch.tensor(A_meas_4x4, device=device, dtype=rdtype)         # [4,4]

    d = 4
    T_re = torch.randn(d, d, device=device, dtype=rdtype, requires_grad=True)
    T_im = torch.randn(d, d, device=device, dtype=rdtype, requires_grad=True)
    opt = torch.optim.Adam([T_re, T_im], lr=lr)

    eps = 1e-12
    t0 = time.perf_counter()
    nll_val = None

    for _ in range(steps):
        opt.zero_grad()
        T = torch.complex(T_re, T_im)
        rho = _rho_from_T(T)  # [4,4]

        p_true = torch.einsum("moij,ji->mo", M, rho).real            # [9,4]
        p_true = torch.clamp(p_true, min=eps, max=1-eps)

        q_obs = (A @ p_true.transpose(0,1)).transpose(0,1)           # [9,4]
        q_obs = torch.clamp(q_obs, min=eps, max=1-eps)

        nll = -(C * torch.log(q_obs)).sum()
        nll.backward()
        opt.step()
        nll_val = float(nll.detach().cpu())

    runtime = time.perf_counter() - t0
    with torch.no_grad():
        T = torch.complex(T_re, T_im)
        rho_np = _rho_from_T(T).detach().cpu().numpy().astype(np.complex64)

    return {
        "rho": rho_np,
        "nll_final": nll_val,
        "runtime_sec": float(runtime),
        "device_used": device,
        "steps": int(steps),
        "lr": float(lr),
        "seed": int(seed),
        "settings_order": labels,
    }
