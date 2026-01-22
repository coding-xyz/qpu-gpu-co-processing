from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Literal, Optional, Tuple, List
from ..common import ArrayLike

ModelName = Literal["linear", "mlp", "residual_mlp"]

# ============================================================
# Base: inverse calibration model target -> control
# ============================================================

class CalibrationModelBase(nn.Module, ABC):
    """
    Crosstalk model (forward physics):
        target = f(control)

    Calibration model (inverse):
        control = g(target)

    This base class describes the inverse map g().
    """

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
        n_param = sum(p.numel() for p in self.parameters())
        return f"n={self.n}, params={n_param/1e3:.1f}k, device={self.device.type}, dtype={self.dtype}"

    # --------------------------------------------------------
    # Core inverse map
    # --------------------------------------------------------

    @abstractmethod
    def forward(self, target: torch.Tensor) -> torch.Tensor:
        """
        (B, n) target  -> (B, n) control
        """

    # --------------------------------------------------------
    # Tensor normalization
    # --------------------------------------------------------

    def _to_model_tensor(self, x: ArrayLike, *, detach: bool = True) -> torch.Tensor:
        """
        Convert input to a tensor on (self.device, self.dtype).
        """
        if isinstance(x, np.ndarray):
            return torch.as_tensor(x, device=self.device, dtype=self.dtype)

        if isinstance(x, torch.Tensor):
            t = x.detach() if detach else x
            if t.device == self.device and t.dtype == self.dtype:
                return t
            return t.to(device=self.device, dtype=self.dtype)

        raise TypeError(type(x))

    # --------------------------------------------------------
    # Public inference API
    # --------------------------------------------------------

    @torch.no_grad()
    def predict(self, target: ArrayLike) -> ArrayLike:
        """
        Predict control from target.
        """
        self.eval()
        tgt = self._to_model_tensor(target, detach=True)

        out = self(tgt)
        if isinstance(target, np.ndarray):
            return out.detach().cpu().numpy()
        else:
            return out

    # --------------------------------------------------------
    # Linear inverse baseline
    # --------------------------------------------------------

    @torch.no_grad()
    def fit_linear_affine(
        self,
        control: ArrayLike,  # (N, n)
        target: ArrayLike,   # (N, n)
        ridge: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Solve inverse affine model:
            control ≈ target @ A^T + d

        Returns:
            A: (n, n)
            d: (n,)
        """
        X = self._to_model_tensor(control, detach=True)
        Y = self._to_model_tensor(target, detach=True)

        if X.ndim != 2 or Y.ndim != 2 or X.shape != Y.shape:
            raise ValueError(f"Expect control,target both (N,n), got {X.shape}, {Y.shape}")

        N, n = X.shape
        if n != self.n:
            raise ValueError(f"dim mismatch: data n={n}, model.n={self.n}")

        Phi = torch.cat([Y, Y.new_ones((N, 1))], dim=1)  # (N, n+1)

        if ridge > 0:
            PtP = Phi.T @ Phi
            I = torch.eye(n + 1, device=Phi.device, dtype=Phi.dtype)
            M_t = torch.linalg.solve(PtP + ridge * I, Phi.T @ X)
        else:
            M_t = torch.linalg.pinv(Phi) @ X

        M = M_t.T
        A = M[:, :n].contiguous()
        d = M[:, n].contiguous()
        return A, d

    # --------------------------------------------------------
    # Initialization hook
    # --------------------------------------------------------

    @torch.no_grad()
    def init_from_data(
        self,
        control_data: ArrayLike,
        target_data: ArrayLike,
        ridge: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Default: compute linear baseline only for logging.
        """
        A, d = self.fit_linear_affine(control_data, target_data, ridge=ridge)
        return {
            "linear_A": A.detach().cpu().numpy(),
            "linear_d": d.detach().cpu().numpy(),
            "applied": False,
        }

    # --------------------------------------------------------
    # Jacobian and linearization
    # --------------------------------------------------------

    def jacobian_at(self, target: ArrayLike) -> torch.Tensor:
        """
        J = ∂control / ∂target at point `target`.

        target: (n,)  -> returns (n, n)
        """
        self.eval()
        tgt = self._to_model_tensor(target, detach=True)

        if tgt.ndim != 1 or tgt.shape[0] != self.n:
            raise ValueError(f"target must be (n,), got {tgt.shape}")

        tgt = tgt.requires_grad_(True)

        def g(inp: torch.Tensor) -> torch.Tensor:
            return self(inp[None, :])[0]

        J = torch.autograd.functional.jacobian(g, tgt, create_graph=False)
        return J.detach()

    @torch.no_grad()
    def linearize_at(
        self,
        target: ArrayLike,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Local linearization around target:

            control ≈ a + target @ J^T
        """
        tgt = self._to_model_tensor(target, detach=True)

        if tgt.ndim != 1 or tgt.shape[0] != self.n:
            raise ValueError(f"target must be (n,), got {tgt.shape}")

        control0 = self(tgt[None, :])[0].detach()
        J = self.jacobian_at(tgt)
        a = control0 - tgt @ J.T
        return a, J, tgt.detach(), control0


# ============================================================
# Models
# ============================================================

class LinearModel(CalibrationModelBase):
    """
    Inverse linear calibration:
        control = (target - b) @ A^T
    """
    def __init__(self, n: int, bias: bool = True):
        super().__init__(n)
        self.A = nn.Parameter(0.01 * torch.randn(n, n))
        self.b = nn.Parameter(torch.zeros(n)) if bias else None

    def forward(self, target: torch.Tensor) -> torch.Tensor:
        X = target - self.b if self.b is not None else target
        return X @ self.A.T

    @torch.no_grad()
    def init_from_data(self, control_data, target_data, ridge: float = 0.0) -> Dict[str, Any]:
        A, d = self.fit_linear_affine(control_data, target_data, ridge=ridge)
        self.A.copy_(A)

        if self.b is not None:
            b = - (d[None, :] @ torch.linalg.pinv(self.A.T)).squeeze(0)
            self.b.copy_(b)

        return {
            "linear_A": A.detach().cpu().numpy(),
            "linear_d": d.detach().cpu().numpy(),
            "applied": True,
        }


class MLPModel(CalibrationModelBase):
    def __init__(self, n: int, hidden: int = 128, depth: int = 2, dropout: float = 0.0):
        super().__init__(n)
        self.hidden = int(hidden)
        self.depth = int(depth)
        self.dropout = float(dropout)

        layers: List[nn.Module] = []
        in_dim = n
        for _ in range(max(1, self.depth)):
            layers += [nn.Linear(in_dim, self.hidden), nn.GELU()]
            if self.dropout > 0:
                layers += [nn.Dropout(self.dropout)]
            in_dim = self.hidden
        layers += [nn.Linear(in_dim, n)]
        self.net = nn.Sequential(*layers)

    def forward(self, target: torch.Tensor) -> torch.Tensor:
        return self.net(target)


class ResidualMLPModel(CalibrationModelBase):
    """
    control = linear(target) + mlp(target)
    """
    def __init__(self, n: int, hidden: int = 128, depth: int = 2, dropout: float = 0.0, bias: bool = True):
        super().__init__(n)
        self.lin = LinearModel(n, bias=bias)
        self.mlp = MLPModel(n, hidden=hidden, depth=depth, dropout=dropout)

    def forward(self, target: torch.Tensor) -> torch.Tensor:
        return self.lin(target) + self.mlp(target)

    @torch.no_grad()
    def init_from_data(self, control_data, target_data, ridge: float = 0.0) -> Dict[str, Any]:
        info = self.lin.init_from_data(control_data, target_data, ridge=ridge)
        info["applied"] = True
        info["target"] = "linear_branch"
        return info


# ============================================================
# Jacobian variation metric
# ============================================================

@torch.no_grad()
def jacobian_variation_metric(
    model: CalibrationModelBase,
    target_center: Optional[np.ndarray] = None,
    target_scale: float = 1.0,
    n_probe: int = 64,
    seed: int = 0,
) -> Dict[str, float]:
    """
    Measure nonlinearity via Jacobian variation.
    """
    n = model.n
    rng = np.random.default_rng(seed)

    if target_center is None:
        target_center = np.zeros(n, dtype=np.float64)
    target_center = np.asarray(target_center, dtype=np.float64)

    J0 = model.jacobian_at(target_center)
    J0n = float(torch.linalg.norm(J0, ord="fro").cpu()) + 1e-12

    rel = []
    for _ in range(n_probe):
        tk = target_center + rng.normal(size=n) * target_scale
        Jk = model.jacobian_at(tk)
        rel.append(float(torch.linalg.norm(Jk - J0, ord="fro").cpu()) / J0n)

    rel = np.asarray(rel)
    return {
        "jacobian_var_mean_fro_rel": float(rel.mean()),
        "jacobian_var_max_fro_rel": float(rel.max()),
        "jacobian_var_p95_fro_rel": float(np.quantile(rel, 0.95)),
    }


# ============================================================
# Local linear calibrator (engineering interface)
# ============================================================

@torch.no_grad()
def local_linear_calibrator(
    model: CalibrationModelBase,
    target0: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Engineering interface:

        control ≈ control0 + (target - target0) @ G^T
    """
    target0 = np.asarray(target0, dtype=np.float64).reshape(model.n,)

    control0 = model.predict(target0[None, :])
    if isinstance(control0, torch.Tensor):
        control0 = control0.detach().cpu().numpy()
    control0 = control0[0]

    G = model.jacobian_at(target0).cpu().numpy()
    return G, control0
