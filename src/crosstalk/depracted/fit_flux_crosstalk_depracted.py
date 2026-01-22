# src/flux/fit.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import numpy as np
import torch


@dataclass
class FluxFitConfig:
    epochs: int = 2000
    lr: float = 2e-2
    fit_quadratic: bool = False
    seed: int = 0
    dtype: torch.dtype = torch.float32


@dataclass
class FluxFitResult:
    C_hat: np.ndarray            # (n,n) float64
    D_hat: Optional[np.ndarray]  # (n,n) float64 or None
    rmse: float
    rel_C_error_if_true_available: Optional[float]
    fit_time_sec: float
    loss_last: float
    benchmark_numpy_lstsq_time_sec: Optional[float]
    summary: Dict[str, Any]


@torch.no_grad()
def numpy_baseline(Phi: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """
    Least squares solve for C in omega ≈ Phi @ C^T
    Returns C (n,n) such that Phi @ C^T ≈ omega.
    """
    C_T, *_ = np.linalg.lstsq(Phi, omega, rcond=None)
    return C_T.T


def fit_flux_crosstalk_torch(
    Phi: np.ndarray,
    omega: np.ndarray,
    device: torch.device,
    cfg: FluxFitConfig,
) -> Tuple[np.ndarray, Optional[np.ndarray], float, float, float]:
    """
    Fit C (and optional D) by minimizing MSE:
      pred = Phi @ C^T + (Phi^2) @ D^T
    Returns (C_hat, D_hat, rmse, fit_time_sec, loss_last)
    """
    # Validate
    if Phi.ndim != 2 or omega.ndim != 2:
        raise ValueError("Phi and omega must be 2D arrays (k,n).")
    if Phi.shape != omega.shape:
        raise ValueError(f"Shape mismatch: Phi {Phi.shape} vs omega {omega.shape}")

    torch.manual_seed(cfg.seed)

    Phi_t = torch.as_tensor(Phi, dtype=cfg.dtype, device=device)
    omega_t = torch.as_tensor(omega, dtype=cfg.dtype, device=device)

    k, n = Phi.shape

    C = torch.nn.Parameter(0.01 * torch.randn(n, n, device=device, dtype=cfg.dtype))
    D = torch.nn.Parameter(torch.zeros(n, n, n, device=device, dtype=cfg.dtype)) if cfg.fit_quadratic else None

    opt = torch.optim.Adam([C] + ([D] if D is not None else []), lr=cfg.lr)

    import time
    t0 = time.perf_counter()

    loss_last = 0.0
    for _ in range(cfg.epochs):
        opt.zero_grad(set_to_none=True)
        pred = Phi_t @ C.T
        if D is not None:
            pred = pred + (Phi_t ** 2) @ D.T
        loss = torch.mean((pred - omega_t) ** 2)
        loss.backward()
        opt.step()
        loss_last = float(loss.detach().item())

    fit_time = time.perf_counter() - t0

    C_hat = C.detach().cpu().numpy().astype(np.float64)
    D_hat = D.detach().cpu().numpy().astype(np.float64) if D is not None else None

    # RMSE on CPU
    pred_cpu = Phi @ C_hat.T
    if D_hat is not None:
        Phi_tensor = Phi[:, :, None] * Phi[:, None, :]  # Shape (k, n, n)
        pred_cpu += np.tensordot(Phi_tensor, D_hat, axes=([1, 2], [1, 2]))
    resid = (pred_cpu - omega).astype(np.float64)
    rmse = float(np.sqrt(np.mean(resid ** 2)))

    return C_hat, D_hat, rmse, float(fit_time), float(loss_last)


def evaluate_rel_error(C_hat: np.ndarray, C_true: Optional[np.ndarray]) -> Optional[float]:
    if C_true is None:
        return None
    C_true = C_true.astype(np.float64, copy=False)
    return float(np.linalg.norm(C_hat - C_true) / (np.linalg.norm(C_true) + 1e-12))


def run_flux_crosstalk_fit(
    Phi: np.ndarray,
    omega: np.ndarray,
    device: torch.device,
    cfg: FluxFitConfig,
    C_true: Optional[np.ndarray] = None,
    benchmark_numpy: bool = False,
) -> FluxFitResult:
    """
    High-level runner: fits, computes metrics, optionally benchmarks numpy lstsq.
    """
    C_hat, D_hat, rmse, fit_time, loss_last = fit_flux_crosstalk_torch(Phi, omega, device, cfg)

    rel_err = evaluate_rel_error(C_hat, C_true)

    np_time = None
    if benchmark_numpy:
        import time
        t1 = time.perf_counter()
        _ = numpy_baseline(Phi.astype(np.float64), omega.astype(np.float64))
        np_time = time.perf_counter() - t1

    summary = {
        "abs_max": float(np.max(np.abs(C_hat))),
        "abs_mean": float(np.mean(np.abs(C_hat))),
    }

    return FluxFitResult(
        C_hat=C_hat,
        D_hat=D_hat,
        rmse=rmse,
        rel_C_error_if_true_available=rel_err,
        fit_time_sec=fit_time,
        loss_last=loss_last,
        benchmark_numpy_lstsq_time_sec=float(np_time) if np_time is not None else None,
        summary=summary,
    )
