# src/crosstalk/fit_flux_crosstalk.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, Literal

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from .models import TrainConfig, build_model, ModelName

@dataclass
class InvertConfig:
    steps: int = 800
    lr: float = 5e-2
    l2: float = 1e-3
    clip: Optional[float] = None
    seed: int = 0
    dtype: torch.dtype = torch.float32


@dataclass
class FluxFitResult:
    model: ModelName
    params: Dict[str, np.ndarray]          # exported params (C/D/NN weights summary)
    rmse_test: float
    fit_time_sec: float
    loss_last: float
    summary: Dict[str, Any]


@torch.no_grad()
def rmse(pred: torch.Tensor, y: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((pred - y) ** 2)).item())


def train(
    Z: np.ndarray,
    domega: np.ndarray,
    device: torch.device,
    cfg:TrainConfig,
) -> Tuple[torch.nn.Module, float, float, float]:
    """
    ML-style training with DataLoader + train/test split.
    """
    if Z.ndim != 2 or domega.ndim != 2:
        raise ValueError("Z and domega must be 2D arrays (k,n).")
    if Z.shape != domega.shape:
        raise ValueError(f"Shape mismatch: Z {Z.shape} vs domega {domega.shape}")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    Z_t = torch.as_tensor(Z, dtype=cfg.dtype)
    Y_t = torch.as_tensor(domega, dtype=cfg.dtype)
    ds = TensorDataset(Z_t, Y_t)

    n_total = len(ds)
    n_test = max(1, int(cfg.test_frac * n_total))
    n_train = n_total - n_test

    ds_train, ds_test = random_split(
        ds, [n_train, n_test],
        generator=torch.Generator().manual_seed(cfg.seed)
    )
    train_loader = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(ds_test, batch_size=max(1024, cfg.batch_size), shuffle=False)

    n = Z.shape[1]
    model = build_model(cfg, n=n, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    import time
    t0 = time.perf_counter()

    loss_last = 0.0
    for _ in range(cfg.epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = torch.mean((pred - yb) ** 2)
            loss.backward()
            opt.step()
            loss_last = float(loss.detach().item())

    fit_time = time.perf_counter() - t0

    # eval
    model.eval()
    rmses = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            pred = model(xb)
            rmses.append(rmse(pred, yb))
    rmse_test = float(np.mean(rmses))

    return model, rmse_test, float(fit_time), float(loss_last)


@torch.no_grad()
def predict(model: torch.nn.Module, Z: np.ndarray, device: torch.device, dtype: torch.dtype) -> np.ndarray:
    model.eval()
    Z_t = torch.as_tensor(Z, dtype=dtype, device=device)
    pred = model(Z_t).detach().cpu().numpy().astype(np.float64)
    return pred


def invert_Z_for_target(
    model: torch.nn.Module,
    domega_target: np.ndarray,   # (n,) or (B,n)
    device: torch.device,
    inv: InvertConfig,
    Z_init: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Solve Z by differentiable optimization:
      min_Z || model(Z) - domega_target ||^2 + l2 * ||Z||^2
    """
    torch.manual_seed(inv.seed)

    tgt = np.asarray(domega_target, dtype=np.float64)
    if tgt.ndim == 1:
        tgt = tgt[None, :]
        squeeze = True
    elif tgt.ndim == 2:
        squeeze = False
    else:
        raise ValueError("domega_target must be shape (n,) or (B,n).")

    B, n = tgt.shape
    y_t = torch.as_tensor(tgt, dtype=inv.dtype, device=device)

    if Z_init is None:
        z0 = torch.zeros((B, n), dtype=inv.dtype, device=device)
    else:
        z0 = torch.as_tensor(np.asarray(Z_init, dtype=np.float64), dtype=inv.dtype, device=device)
        if z0.ndim == 1:
            z0 = z0[None, :]
        if z0.shape != (B, n):
            raise ValueError(f"Z_init shape {z0.shape} does not match target {(B,n)}")

    Z_var = torch.nn.Parameter(z0.clone())
    opt = torch.optim.Adam([Z_var], lr=inv.lr)

    model.eval()
    for _ in range(inv.steps):
        opt.zero_grad(set_to_none=True)
        pred = model(Z_var)
        loss_data = torch.mean((pred - y_t) ** 2)
        loss_reg = inv.l2 * torch.mean(Z_var ** 2)
        loss = loss_data + loss_reg
        loss.backward()
        opt.step()
        if inv.clip is not None:
            with torch.no_grad():
                Z_var.clamp_(-float(inv.clip), float(inv.clip))

    Z_sol = Z_var.detach().cpu().numpy().astype(np.float64)
    if squeeze:
        Z_sol = Z_sol[0]
    return Z_sol


# # @torch.no_grad()
# def eval_response(model: torch.nn.Module, z0: torch.Tensor) -> Dict[str, np.ndarray]:
#     """
#     """
#     out: Dict[str, np.ndarray] = {}

#     C = torch.autograd.functional.jacobian(model, z0, create_graph=False)
#     out["C"] = C.detach().cpu().numpy().astype(np.float64)

#     model.eval()
#     y0 = model(z0)
#     n_out = y0.numel()
#     n_in = z0.numel()
#     D = torch.zeros((n_in, n_in, n_out), device=z0.device, dtype=z0.dtype)
#     # For each output component y_j, compute Hessian wrt z
#     for j in range(n_out):
#         def fj(z):
#             return model(z)[j]  # scalar
#         H = torch.autograd.functional.hessian(fj, z0)  # (n_in, n_in)
#         # H = 0.5 * (H + H.T)
#         D[:, :, j] = H
#     out["D"] = D.detach().cpu().numpy().astype(np.float64)

#     return out

def run_flux_fit(
    Z: np.ndarray,
    domega: np.ndarray,
    device: torch.device,
    cfg: TrainConfig,
) -> FluxFitResult:
    
    model, rmse_test, fit_time, loss_last = train(Z, domega, device, cfg)
    Z_t = torch.as_tensor(Z, dtype=cfg.dtype)
    params = eval_response(model, torch.zeros_like(Z_t))

    summary: Dict[str, Any] = {
        "model": cfg.model,
        "rmse_test": rmse_test,
    }

    # add a bit more for linear-ish models
    if "C" in params:
        C = params["C"]
        summary["abs_max_C"] = float(np.max(np.abs(C)))
        summary["abs_mean_C"] = float(np.mean(np.abs(C)))
    if "D" in params:
        D = params["D"]
        summary["abs_max_D"] = float(np.max(np.abs(D)))
        summary["abs_mean_D"] = float(np.mean(np.abs(D)))

    return FluxFitResult(
        model=cfg.model,
        params=params,
        rmse_test=float(rmse_test),
        fit_time_sec=float(fit_time),
        loss_last=float(loss_last),
        summary=summary,
    )
