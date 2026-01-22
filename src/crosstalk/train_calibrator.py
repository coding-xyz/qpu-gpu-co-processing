from __future__ import annotations

import numpy as np
import torch
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, Literal, Optional, Union, Tuple, List

# NOTE:
# calibration_models.py is assumed to define:
#   - ModelName = Literal["linear", "mlp", "residual_mlp"]
#   - CalibrationModelBase
#   - LinearModel / MLPModel / ResidualMLPModel
from .calibration_models import *  # keep if you prefer; explicit imports are cleaner

# ============================================================
# Configs
# ============================================================

@dataclass
class TrainConfig:
    model: ModelName
    epochs: int = 1000
    lr: float = 2e-3
    test_frac: float = 0.1
    batch_size: int = 256
    seed: int = 0
    split_seed: int = 0
    weight_decay: float = 0.0
    grad_clip: Optional[float] = None
    dtype: torch.dtype = torch.float32
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    init_ridge: float = 1e-8          # linear init regularization
    do_init_from_data: bool = True    # whether to do init_from_data on train split


@dataclass
class TrainResult:
    cfg: Dict[str, Any]
    train_loss: List[float]
    test_loss: List[float]
    rmse_train: float
    rmse_test: float
    extra: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# utils
# ============================================================

def _set_seed(seed: int, device: torch.device):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _train_test_split(N: int, test_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(N)
    rng.shuffle(idx)
    n_test = int(round(N * float(test_frac)))
    return idx[n_test:], idx[:n_test]


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


# ============================================================
# Training: target -> control
#   Crosstalk (forward physics): target = f(control)
#   Calibration (inverse model): control = g(target)
# Here we train g: (target_data -> control_data)
# ============================================================

def train_calibrator(
    cfg: TrainConfig,
    target_data: np.ndarray,   # (N, n) desired/target system response
    control_data: np.ndarray,  # (N, n) applied control vector
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
    tr_idx, te_idx = _train_test_split(N, cfg.test_frac, cfg.split_seed)

    model = build_model(cfg, n=n, device=dev)

    # NOTE: init_from_data in CalibrationModelBase is defined as:
    #   init_from_data(control_data, target_data, ...)
    # i.e. it fits: control ≈ target @ A^T + d
    init_info: Dict[str, Any] = {}
    if cfg.do_init_from_data:
        init_info = model.init_from_data(control[tr_idx], target[tr_idx], ridge=float(cfg.init_ridge))

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.lr),
        weight_decay=float(cfg.weight_decay),
    )

    Ttr = torch.as_tensor(target[tr_idx], device=dev, dtype=cfg.dtype)
    Ctr = torch.as_tensor(control[tr_idx], device=dev, dtype=cfg.dtype)
    Tte = torch.as_tensor(target[te_idx], device=dev, dtype=cfg.dtype)
    Cte = torch.as_tensor(control[te_idx], device=dev, dtype=cfg.dtype)

    def iter_batches(Xb: torch.Tensor, Tb: torch.Tensor, bs: int):
        Nn = Xb.shape[0]
        perm = torch.randperm(Nn, device=Xb.device)
        for i in range(0, Nn, bs):
            idx = perm[i:i + bs]
            yield Xb[idx], Tb[idx]

    train_loss_hist: List[float] = []
    test_loss_hist: List[float] = []

    for _ in range(int(cfg.epochs)):
        model.train()
        loss_sum = 0.0
        nb = 0
        for tb, cb in iter_batches(Ttr, Ctr, int(cfg.batch_size)):
            pred_control = model(tb)                       # control = g(target)
            loss = torch.mean((pred_control - cb) ** 2)    # MSE in control space
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip))
            opt.step()
            loss_sum += float(loss.detach().item())
            nb += 1
        train_loss_hist.append(loss_sum / max(1, nb))

        model.eval()
        with torch.no_grad():
            te = torch.mean((model(Tte) - Cte) ** 2).item()
        test_loss_hist.append(float(te))

    model.eval()
    with torch.no_grad():
        rmse_tr = _rmse(model(Ttr), Ctr)
        rmse_te = _rmse(model(Tte), Cte)

    # ---- diagnostics (no Hessian) ----
    target0 = np.zeros(n, dtype=np.float64)
    J0 = model.jacobian_at(target0).cpu().numpy()  # d control / d target at 0
    diag = np.abs(np.diag(J0))
    off = np.abs(J0 - np.diag(np.diag(J0)))

    extra = {
        "init_from_data": init_info,
        "J_dcontrol_dtarget_at0": J0,
        "J_diag_mean_abs": float(diag.mean()),
        "J_offdiag_mean_abs": float(off.mean()),
        "J_offdiag_over_diag": float(off.mean() / (diag.mean() + 1e-12)),
    }

    res = TrainResult(
        cfg=asdict(cfg),
        train_loss=train_loss_hist,
        test_loss=test_loss_hist,
        rmse_train=rmse_tr,
        rmse_test=rmse_te,
        extra=extra,
    )
    return model, res
