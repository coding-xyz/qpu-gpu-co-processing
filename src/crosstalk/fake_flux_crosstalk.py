# src/flux/pseudo_experiment.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

import numpy as np


def make_sparse_matrix(
    n: int,
    seed: int = 0,
    diag_loc: float = 1.0,
    diag_scale: float = 0.2,
    off_scale: float = 0.08,
    m_min: int = 2,
    m_max: int = 6,
) -> np.ndarray:
    """Sparse-ish (n,n) matrix: diagonal + a few off-diagonals per row."""
    rng = np.random.default_rng(seed)
    M = np.zeros((n, n), dtype=np.float64)

    diag = rng.normal(loc=diag_loc, scale=diag_scale, size=n)
    M[np.arange(n), np.arange(n)] = diag

    for i in range(n):
        m = rng.integers(m_min, m_max)  # 2~5
        js = rng.choice(n, size=m, replace=False)
        for j in js:
            if j == i:
                continue
            M[i, j] += rng.normal(loc=0.0, scale=off_scale)

    return M


def make_sparse_tensor_D(
    n: int,
    seed: int = 0,
    base_scale: float = 0.06,
    symmetrize_pq: bool = True,
) -> np.ndarray:
    """
    D[p,q,j] tensor for quadratic term contributing to output channel j.
    """
    D = np.zeros((n, n, n), dtype=np.float64)
    for j in range(n):
        Dj = make_sparse_matrix(
            n,
            seed=int(seed + 101 * (j + 1)),
            diag_loc=0.0,
            diag_scale=base_scale,
            off_scale=base_scale,
            m_min=2,
            m_max=6,
        )
        if symmetrize_pq:
            Dj = 0.5 * (Dj + Dj.T)
        D[:, :, j] = Dj
    return D


@dataclass
class CrosstalkConfig:
    n: int = 24
    seed: int = 0

    # Nonlinearity strengths
    quad_strength: float = 0.02       # quadratic tensor term
    sat_strength: float = 0.05        # tanh saturation term
    sat_scale: float = 0.8            # tanh scale
    per_strength: float = 0.02        # sinusoid term (proxy periodicity)
    per_period: float = 1.0

    # Noise
    noise_white: float = 0.01         # additive Gaussian noise
    noise_pattern: float = 0.005      # scales with |Z|
    drift_strength: float = 0.0       # optional slow drift per batch (set >0 to enable)


class ZToZCrosstalkPlant:
    """
    A reusable pseudo-lab mapping: Z (k,n) -> domega_meas (k,n)
    Includes:
      - linear sparse mixing (C_true)
      - quadratic term with D_true
      - saturating nonlinearity tanh(Z/sat_scale)
      - periodic term sin(2π Z / per_period)
      - measurement noise (white + |Z|-dependent)
      - optional slow drift (per call)
    """

    def __init__(self, cfg: CrosstalkConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

        n = cfg.n
        # "truth" parameters
        self.C_true = make_sparse_matrix(n, seed=cfg.seed, diag_loc=1.0, diag_scale=0.2, off_scale=0.08)

        self.D_true = cfg.quad_strength * make_sparse_tensor_D(
            n, seed=cfg.seed + 1, base_scale=0.06, symmetrize_pq=True
        )

        # drift state
        self._drift = np.zeros((n,), dtype=np.float64)

    def forward(self, Z: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """
        Deterministic (if add_noise=False) forward mapping.
        Z: (k,n) float64
        returns: (k,n) float64
        """
        Z = np.asarray(Z, dtype=np.float64)
        if Z.ndim != 2 or Z.shape[1] != self.cfg.n:
            raise ValueError(f"Z must be (k,{self.cfg.n}) but got {Z.shape}")

        # --- linear ---
        domega = Z @ self.C_true.T  # (k,n)

        # --- quadratic ---
        if self.cfg.quad_strength != 0.0:
            Z_tensor = Z[:, :, None] * Z[:, None, :]  # (k,n,n)
            domega += np.tensordot(Z_tensor, self.D_true, axes=([1, 2], [0, 1]))  # (k,n)

        # --- saturating ---
        if self.cfg.sat_strength != 0.0:
            domega += self.cfg.sat_strength * (np.tanh(Z / self.cfg.sat_scale) @ self.C_true.T)

        # --- periodic ---
        if self.cfg.per_strength != 0.0:
            domega += self.cfg.per_strength * (np.sin(2 * np.pi * Z / self.cfg.per_period) @ self.C_true.T)

        # --- slow drift (per call) ---
        if self.cfg.drift_strength != 0.0:
            step = self.rng.normal(0.0, self.cfg.drift_strength, size=(self.cfg.n,))
            self._drift = 0.99 * self._drift + step
            domega = domega + self._drift[None, :]

        if not add_noise:
            return domega

        # --- noise ---
        noise = self.rng.normal(0.0, self.cfg.noise_white, size=domega.shape)
        noise += self.rng.normal(0.0, self.cfg.noise_pattern, size=domega.shape) * (
            np.abs(Z) / (np.std(Z) + 1e-12)
        )
        return domega + noise

    def get_truth(self) -> Dict[str, Any]:
        return {
            "C_true": self.C_true.copy(),
            "D_true": self.D_true.copy(),
            "cfg": self.cfg,
        }


def sample_random_Z(k: int, n: int, z_scale: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, z_scale, size=(k, n)).astype(np.float64)
