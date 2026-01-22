from __future__ import annotations

import numpy as np
import torch
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, Optional, Union, Tuple, List

from .calibration_models import *  # CalibrationModelBase, LinearModel, MLPModel, ResidualMLPModel, ModelName


# ============================================================
# Configs
# ============================================================

@dataclass
class TrainConfig:
    model: ModelName
    epochs: int = 1000
    lr: float = 2e-3
    batch_size: int = 256
    seed: int = 0
    weight_decay: float = 0.0
    grad_clip: Optional[float] = None
    dtype: torch.dtype = torch.float32
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    init_ridge: float = 1e-8          # linear init regularization
    do_init_from_data: bool = True    # whether to do init_from_data on all data


@dataclass
class TrainResult:
    cfg: Dict[str, Any]
    loss: List[float]                 # training MSE history on full data
    rmse: float                       # RMSE on full data (proxy)
    extra: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# utils
# ============================================================

def _set_seed(seed: int, device: torch.device):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

def _rmse(pred: torch.Tensor, tgt: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((pred - tgt) ** 2)).detach().cpu().item())

def build_model(cfg: TrainConfig, n: int, device: torch.device) -> CalibrationModelBase:
    kw = cfg.model_kwargs or {}
    if cfg.model == "linear":
        m = LinearModel(n, **kw)
    elif cfg.model == "mlp":
        m = MLPModel(n, **kw)
    elif cfg.model == "residual_mlp":
        m = ResidualMLPModel(n, **kw)
    else:
        raise ValueError(f"Unknown model {cfg.model}")
    return m.to(device=device, dtype=cfg.dtype)

def _linearization_artifacts_from_local(
    model: CalibrationModelBase,
    target0: np.ndarray,
    *,
    pinv_rcond: float = 1e-10,
) -> Dict[str, Any]:
    """
    Uses calibration_models.local_linear_calibrator() as the single source of truth.

    Given:
        control ≈ control0 + (target - target0) @ G^T
    Derive:
        control ≈ target @ A_inv^T + d_inv
        target  ≈ control @ C_fwd^T + e_fwd   (derived, local)
    """
    G, control0 = local_linear_calibrator(model, target0)  # G:(n,n), control0:(n,)

    target0 = np.asarray(target0, dtype=np.float64).reshape(model.n,)
    control0 = np.asarray(control0, dtype=np.float64).reshape(model.n,)

    # inverse affine: control ≈ target @ G^T + d_inv
    d_inv = control0 - (target0 @ G.T)

    # derived forward affine (local): target ≈ control @ C^T + e
    C_fwd = np.linalg.pinv(G, rcond=float(pinv_rcond))
    e_fwd = (-d_inv[None, :] @ np.linalg.pinv(G.T, rcond=float(pinv_rcond))).squeeze(0)

    return {
        "target_ref": target0,
        "control_ref": control0,
        "G_dcontrol_dtarget": G,   # same as A_inv
        "d_inv": d_inv,
        "C_fwd_dtarget_dcontrol": C_fwd,
        "e_fwd": e_fwd,
    }

# ============================================================
# Training: target -> control
#   Forward physics: target = f(control)
#   Inverse calibrator: control = g(target)  <-- train this
# ============================================================

def train_calibrator(
    cfg: TrainConfig,
    target_data: np.ndarray,   # (N, n)
    control_data: np.ndarray,  # (N, n)
    device: Union[str, torch.device] = "cuda",
) -> Tuple[CalibrationModelBase, TrainResult]:
    dev = torch.device(device)
    _set_seed(cfg.seed, dev)

    target = np.asarray(target_data, dtype=np.float64)
    control = np.asarray(control_data, dtype=np.float64)
    if target.ndim != 2 or control.ndim != 2 or target.shape != control.shape:
        raise ValueError(
            f"Expect target,control both (N,n) and same shape, got "
            f"target{target.shape}, control{control.shape}"
        )

    N, n = target.shape
    model = build_model(cfg, n=n, device=dev)

    # init_from_data expects (control_data, target_data) and fits:
    #   control ≈ target @ A^T + d
    init_info: Dict[str, Any] = {}
    if cfg.do_init_from_data:
        init_info = model.init_from_data(control, target, ridge=float(cfg.init_ridge))

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.lr),
        weight_decay=float(cfg.weight_decay),
    )

    T = torch.as_tensor(target, device=dev, dtype=cfg.dtype)
    C = torch.as_tensor(control, device=dev, dtype=cfg.dtype)

    def iter_batches(Tb: torch.Tensor, Cb: torch.Tensor, bs: int):
        Nn = Tb.shape[0]
        perm = torch.randperm(Nn, device=Tb.device)
        for i in range(0, Nn, bs):
            idx = perm[i:i + bs]
            yield Tb[idx], Cb[idx]

    loss_hist: List[float] = []

    for _ in range(int(cfg.epochs)):
        model.train()
        loss_sum = 0.0
        nb = 0
        for tb, cb in iter_batches(T, C, int(cfg.batch_size)):
            pred_control = model(tb)                       # control = g(target)
            loss = torch.mean((pred_control - cb) ** 2)    # MSE in control space (proxy)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip))
            opt.step()
            loss_sum += float(loss.detach().item())
            nb += 1
        loss_hist.append(loss_sum / max(1, nb))

    model.eval()
    with torch.no_grad():
        rmse_all = _rmse(model(T), C)

    # ---- diagnostics: linearization (inverse + derived forward) ----
    target_ref_zero = np.zeros(n, dtype=np.float64)
    target_ref_mean = target.mean(axis=0).astype(np.float64)

    lin_zero = _linearization_artifacts_from_local(model, target_ref_zero)
    lin_mean = _linearization_artifacts_from_local(model, target_ref_mean)

    extra = {
        "init_from_data": init_info,
        "rmse_proxy_note": "RMSE is computed on full dataset in control space; for calibration quality use end-to-end metrics.",
        "linearization_at_zero": lin_zero,   # contains A_inv,d_inv and derived C_fwd,e_fwd
        "linearization_at_mean": lin_mean,
    }

    res = TrainResult(
        cfg=asdict(cfg),
        loss=loss_hist,
        rmse=rmse_all,
        extra=extra,
    )
    return model, res
