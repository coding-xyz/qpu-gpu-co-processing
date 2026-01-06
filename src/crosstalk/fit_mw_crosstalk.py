# src/crosstalk/fit_mw_crosstalk.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import numpy as np
import torch

@dataclass
class MWFitConfig:
    eps: float = 1e-3          # Tikhonov reg
    f0_mode: str = "third"     # plotting bin selection: "third" or "mid"
    dtype: torch.dtype = torch.complex64

@dataclass
class MWFitResult:
    Hf: np.ndarray                  # (F,n,n) complex64/128 on CPU
    Gf: np.ndarray                  # (F,n,n) complex64/128 on CPU
    x_pre: np.ndarray               # (n,T) float32/64 on CPU (time domain)
    x_reconstruction_rmse: float
    fit_time_sec: float
    invert_time_sec: float
    f0_bin: int
    summary: Dict[str, Any]


def rfft_stack(x: np.ndarray) -> np.ndarray:
    """
    x: (n,T) real -> X_f: (n,F) complex
    """
    if x.ndim != 2:
        raise ValueError("x must be 2D (n,T)")
    return np.fft.rfft(x, axis=1)


def irfft_stack(X_f: np.ndarray, T: int) -> np.ndarray:
    """
    X_f: (n,F) complex -> x: (n,T) real
    """
    return np.fft.irfft(X_f, n=T, axis=1)


def estimate_Hf_rank1(
    X_f: torch.Tensor,  # (n,F) complex
    Y_f: torch.Tensor,  # (n,F) complex
    eps: float,
) -> torch.Tensor:
    """
    Rank-1 outer-product regularized estimate per frequency:
      H(f) = Y X^H (X X^H + eps I)^{-1}
    Returns Hf: (F,n,n)
    """
    if X_f.ndim != 2 or Y_f.ndim != 2:
        raise ValueError("X_f, Y_f must be (n,F)")
    n, F = X_f.shape
    if Y_f.shape != (n, F):
        raise ValueError("X_f and Y_f shape mismatch")

    # reshape to (F,n,1)
    Xv = X_f.transpose(0, 1).unsqueeze(-1)  # (F,n,1)
    Yv = Y_f.transpose(0, 1).unsqueeze(-1)  # (F,n,1)
    Xh = torch.conj(Xv).transpose(1, 2)     # (F,1,n)

    XXh = Xv @ Xh                           # (F,n,n) rank-1
    YXh = Yv @ Xh                           # (F,n,n)

    I = torch.eye(n, device=X_f.device, dtype=X_f.dtype).unsqueeze(0).expand(F, n, n)
    Hf = YXh @ torch.linalg.inv(XXh + eps * I)
    return Hf


def build_predistorter(
    Hf: torch.Tensor,  # (F,n,n)
    eps: float,
) -> torch.Tensor:
    """
    G(f) = inv(H(f) + eps I)
    """
    F, n, _ = Hf.shape
    I = torch.eye(n, device=Hf.device, dtype=Hf.dtype).unsqueeze(0).expand(F, n, n)
    return torch.linalg.inv(Hf + eps * I)


def apply_predistorter_to_Y(
    Gf: torch.Tensor,  # (F,n,n)
    Y_des_f: torch.Tensor,  # (n,F)
) -> torch.Tensor:
    """
    x_pre_f(:,f) = G(f) @ y_des_f(:,f)
    Returns x_pre_f: (n,F)
    """
    # (F,n,n) @ (F,n,1) -> (F,n,1)
    Yv = Y_des_f.transpose(0, 1).unsqueeze(-1)      # (F,n,1)
    Xv = (Gf @ Yv).squeeze(-1)                      # (F,n)
    return Xv.transpose(0, 1).contiguous()          # (n,F)


def pick_f0(F: int, mode: str) -> int:
    if F <= 1:
        return 0
    if mode == "mid":
        return F // 2
    # default "third"
    return max(0, min(F - 1, F // 3))


def run_mw_crosstalk_fit(
    x: np.ndarray,   # (n,T) real
    y: np.ndarray,   # (n,T) real
    device: torch.device,
    cfg: MWFitConfig,
) -> MWFitResult:
    """
    Full pipeline:
      - rfft: X_f, Y_f
      - estimate Hf
      - invert -> Gf
      - reconstruct x_pre by x_pre_f = Gf @ Y_f
      - irfft -> x_pre
      - metrics
    """
    if x.shape != y.shape:
        raise ValueError(f"x shape {x.shape} != y shape {y.shape}")
    n, T = x.shape

    X_f_np = rfft_stack(x)
    Y_f_np = rfft_stack(y)
    F = X_f_np.shape[1]

    X_f = torch.as_tensor(X_f_np, device=device, dtype=cfg.dtype)
    Y_f = torch.as_tensor(Y_f_np, device=device, dtype=cfg.dtype)

    import time
    t0 = time.perf_counter()
    Hf = estimate_Hf_rank1(X_f, Y_f, eps=float(cfg.eps))
    fit_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    Gf = build_predistorter(Hf, eps=float(cfg.eps))
    inv_time = time.perf_counter() - t1

    # Use measured Y as desired output proxy (demo)
    x_pre_f = apply_predistorter_to_Y(Gf, Y_des_f=Y_f)  # (n,F)
    x_pre = torch.fft.irfft(x_pre_f, n=T, dim=1).detach().cpu().numpy()

    err = (x_pre - x).astype(np.float64)
    rmse = float(np.sqrt(np.mean(err**2)))

    f0 = pick_f0(F, cfg.f0_mode)

    Hf_cpu = Hf.detach().cpu().numpy()
    Gf_cpu = Gf.detach().cpu().numpy()

    summary: Dict[str, Any] = {
        "Hf_abs_max_at_f0": float(np.max(np.abs(Hf_cpu[f0]))),
        "Hf_abs_mean_at_f0": float(np.mean(np.abs(Hf_cpu[f0]))),
        "Gf_abs_max_at_f0": float(np.max(np.abs(Gf_cpu[f0]))),
        "Gf_abs_mean_at_f0": float(np.mean(np.abs(Gf_cpu[f0]))),
    }

    return MWFitResult(
        Hf=Hf_cpu,
        Gf=Gf_cpu,
        x_pre=x_pre,
        x_reconstruction_rmse=rmse,
        fit_time_sec=float(fit_time),
        invert_time_sec=float(inv_time),
        f0_bin=int(f0),
        summary=summary,
    )
