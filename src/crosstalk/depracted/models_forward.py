from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, Literal, Optional, Union, Tuple, overload

ModelName = Literal["linear", "quadratic", "mlp", "residual_mlp"]
@dataclass
class TrainConfig:
    model: ModelName
    epochs: int = 1000
    lr: float = 2e-3
    test_frac: float = 0.1
    batch_size: int = 256
    seed: int = 0
    split_seed: int = 0
    dtype: torch.dtype = torch.float32
    model_kwargs: Dict[str, Any] = field(default_factory=dict)  # model-specific

@dataclass
class InvertConfig:
    steps: int = 800
    lr: float = 5e-2
    l2: float = 1e-3
    clip: Optional[float] = None
    seed: int = 0
    dtype: torch.dtype = torch.float32

# ----------------------------
# Models: Z -> domega
# ----------------------------

@dataclass(frozen=True)
class ModelInfo:
    name: str
    n_in: int
    n_out: int

class FluxCrosstalkModelBase(nn.Module, ABC):
    def __init__(self, n: int):
        super().__init__()
        self.n = int(n)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def extra_repr(self) -> str:
        """print models with n and parameters"""
        n_param = sum(p.numel() for p in self.parameters())
        return f"n={self.n}, params={n_param/1e3:.1f}k, device={self.device.type}, dtype={self.dtype}"

    @abstractmethod
    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """z: (B,n) -> y: (B,n)"""

    def export_params(self) -> Dict[str, Any]:
        """export important parameters (default empty, can be overwritten)"""
        return {}
    
    @overload
    def predict(self, Z: np.ndarray) -> np.ndarray: ...
    @overload
    def predict(self, Z: torch.Tensor) -> torch.Tensor: ...

    @torch.no_grad()
    def predict(self, Z: Union[np.ndarray, torch.Tensor]):
        self.eval()
        if isinstance(Z, np.ndarray):
            Z_t = torch.as_tensor(Z, device=self.device, dtype=self.dtype)
            pred = self(Z_t)
            return pred.detach().cpu().numpy()
        elif isinstance(Z, torch.Tensor):
            Z_t = Z.to(device=self.device, dtype=self.dtype)
            return self(Z_t)
        else:
            raise TypeError(f"predict expects np.ndarray or torch.Tensor, got {type(Z)}")

    JacobianReturn = Union[torch.Tensor, Tuple[torch.Tensor, ...]]
    def response_at(self, z0: torch.Tensor, need_hessian: bool = False) -> JacobianReturn:
        """Return linear response with the Jacobian matrix"""
        self.eval()
        z0 = z0.detach().to(device=self.device, dtype=self.dtype).requires_grad_(True)
        J = torch.autograd.functional.jacobian(lambda z: self(z), z0, create_graph=False)
        return J # Note: the shape depends on batch，keep as torch.Tensor of shape (B,n,B,n)

    @overload
    def invert_Z_for_target(self, domega: np.ndarray, inv: InvertConfig, ...) -> np.ndarray: ...
    @overload
    def invert_Z_for_target(self, domega: torch.Tensor, inv: InvertConfig, ...) -> torch.Tensor: ...

    def invert_Z_for_target(
        self,
        domega_target: Union[np.ndarray, torch.Tensor],
        cfg: InvertConfig,
        Z_init: Optional[Union[np.ndarray, torch.Tensor]] = None,
        return_loss: bool = False,
    ):
        """
        Solve Z by differentiable optimization:
            min_Z  mean(|| self(Z) - domega_target ||^2) + l2 * mean(||Z||^2)

        Inputs:
          domega_target: (n,) or (B,n), numpy or torch
          Z_init:        (n,) or (B,n), numpy or torch, optional

        Returns:
          Z_sol with same type as domega_target, shape (n,) or (B,n).
          If return_loss=True, returns (Z_sol, loss_last_float).
        """
        # ---- seeds (best-effort determinism) ----
        torch.manual_seed(cfg.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(cfg.seed)

        # ---- normalize target to torch tensor on model device ----
        input_is_numpy = isinstance(domega_target, np.ndarray)

        if input_is_numpy:
            tgt_np = np.asarray(domega_target, dtype=np.float64)
            if tgt_np.ndim == 1:
                tgt_np = tgt_np[None, :]
                squeeze = True
            elif tgt_np.ndim == 2:
                squeeze = False
            else:
                raise ValueError("domega_target must be shape (n,) or (B,n).")

            B, n = tgt_np.shape
            y_t = torch.as_tensor(tgt_np, device=self.device, dtype=cfg.dtype)
        else:
            if not isinstance(domega_target, torch.Tensor):
                raise TypeError(f"domega_target must be np.ndarray or torch.Tensor, got {type(domega_target)}")

            tgt_t = domega_target.detach()
            if tgt_t.ndim == 1:
                tgt_t = tgt_t[None, :]
                squeeze = True
            elif tgt_t.ndim == 2:
                squeeze = False
            else:
                raise ValueError("domega_target must be shape (n,) or (B,n).")

            B, n = tgt_t.shape
            y_t = tgt_t.to(device=self.device, dtype=cfg.dtype)

        # optional: check model I/O dim
        if hasattr(self, "n") and int(getattr(self, "n")) != int(n):
            raise ValueError(f"Target dim n={n} does not match model.n={getattr(self, 'n')}")

        # ---- init Z ----
        if Z_init is None:
            z0 = torch.zeros((B, n), device=self.device, dtype=cfg.dtype)
        else:
            if input_is_numpy:
                z0_np = np.asarray(Z_init, dtype=np.float64)
                z0 = torch.as_tensor(z0_np, device=self.device, dtype=cfg.dtype)
            else:
                if not isinstance(Z_init, torch.Tensor):
                    raise TypeError(f"Z_init must be same type as domega_target (torch.Tensor), got {type(Z_init)}")
                z0 = Z_init.detach().to(device=self.device, dtype=cfg.dtype)

            if z0.ndim == 1:
                z0 = z0[None, :]
            if tuple(z0.shape) != (B, n):
                raise ValueError(f"Z_init shape {tuple(z0.shape)} does not match target {(B, n)}")

        # ---- optimize Z ----
        Z_var = torch.nn.Parameter(z0.clone())
        opt = torch.optim.Adam([Z_var], lr=float(cfg.lr))

        self.eval()
        loss_last = 0.0
        for _ in range(int(cfg.steps)):
            opt.zero_grad(set_to_none=True)
            pred = self(Z_var)
            loss_data = torch.mean((pred - y_t) ** 2)
            loss_reg = float(cfg.l2) * torch.mean(Z_var ** 2)
            loss = loss_data + loss_reg
            loss.backward()
            opt.step()

            loss_last = float(loss.detach().item())

            if cfg.clip is not None:
                with torch.no_grad():
                    Z_var.clamp_(-float(cfg.clip), float(cfg.clip))

        # ---- return with same type as input ----
        Z_sol_t = Z_var.detach()
        if squeeze:
            Z_sol_t = Z_sol_t[0]

        if input_is_numpy:
            Z_sol = Z_sol_t.cpu().numpy().astype(np.float64)
        else:
            # return torch tensor on model device with inv.dtype
            Z_sol = Z_sol_t

        if return_loss:
            return Z_sol, loss_last
        return Z_sol


class LinearModel(FluxCrosstalkModelBase):
    # domega = Z @ C^T
    def __init__(self, n: int):
        super().__init__(n)
        self.C = torch.nn.Parameter(0.01 * torch.randn(n, n))

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        return Z @ self.C.T

    def export_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return {
            "C": self.C.detach().cpu().numpy(),
            "n_params": int(n_params),
        }

class QuadraticModel(FluxCrosstalkModelBase):
    # domega = Z @ C^T + sum_{p,q} Z_p Z_q D[p,q,j]
    def __init__(self, n: int):
        super().__init__(n)
        self.C = torch.nn.Parameter(0.01 * torch.randn(n, n))
        self.D = torch.nn.Parameter(torch.zeros(n, n, n))

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        pred = Z @ self.C.T
        pred = pred + torch.einsum("bp,bq,pqj->bj", Z, Z, self.D)
        return pred
   
    def export_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return {
            "C": self.C.detach().cpu().numpy(), 
            "D": self.D.detach().cpu().numpy(),
            "n_params": int(n_params),
        }
    
class MLPModel(FluxCrosstalkModelBase):
    def __init__(
        self,
        n: int,
        hidden: int = 128,
        depth: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__(n)
        self.hidden = hidden
        self.depth = depth
        self.dropout = dropout

        layers = []
        in_dim = n
        for _ in range(max(1, depth)):
            layers.append(torch.nn.Linear(in_dim, hidden))
            layers.append(torch.nn.GELU())
            if dropout > 0:
                layers.append(torch.nn.Dropout(p=float(dropout)))
            in_dim = hidden
        layers.append(torch.nn.Linear(in_dim, n))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        return self.net(Z)
    
    def export_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return {
            "hidden": self.hidden,
            "depth": self.depth,
            "dropout":self.dropout,
            "n_params": int(n_params),
        }


class ResidualMLPModel(torch.nn.Module):
    """
    domega = linear(Z) + mlp_residual(Z)
    This is usually stable/strong in calibration contexts.
    """
    def __init__(
        self,
        n: int,
        hidden: int = 128,
        depth: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.lin = LinearModel(n)
        self.mlp = MLPModel(n, hidden=hidden, depth=depth, dropout=dropout)

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        return self.lin(Z) + self.mlp(Z)

    def export_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return {
            "components": {
                "linear": self.lin.export_params(),
                "mlp": self.mlp.export_params(),
            },
            "n_params": int(n_params),
        }
    
def build_model(cfg: TrainConfig, n: int, device: torch.device) -> torch.nn.Module:
    kw = cfg.model_kwargs or {}
    if   cfg.model == "linear":         model = LinearModel(n, **kw)
    elif cfg.model == "quadratic":      model = QuadraticModel(n, **kw)
    elif cfg.model == "mlp":            model = MLPModel(n, **kw)
    elif cfg.model == "residual_mlp":   model = ResidualMLPModel(n, **kw)
    else: raise ValueError(f"Unknown model {cfg.model}")
    return model.to(device=device, dtype=cfg.dtype)

