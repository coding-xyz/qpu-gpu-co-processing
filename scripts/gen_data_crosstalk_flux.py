# flux/gen_flux_npz.py
from __future__ import annotations
import argparse
import numpy as np


def make_sparse_matrix(n: int, seed: int = 0,
                       diag_loc: float = 1.0, diag_scale: float = 0.2,
                       off_scale: float = 0.08,
                       m_min: int = 2, m_max: int = 6) -> np.ndarray:
    """Sparse-ish (n,n) matrix with diagonal + a few off-diagonals per row."""
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


def make_sparse_tensor_D(n: int, seed: int = 0,
                         base_scale: float = 0.05,
                         symmetrize_pq: bool = True) -> np.ndarray:
    """
    D[p,q,j] tensor for quadratic term contributing to output j.
    Each slice D[:,:,j] is sparse-ish.
    """
    D = np.zeros((n, n, n), dtype=np.float64)
    for j in range(n):
        Dj = make_sparse_matrix(
            n,
            seed=int(seed + 101 * (j + 1)),
            diag_loc=0.0,
            diag_scale=base_scale,
            off_scale=base_scale,
            m_min=2, m_max=6
        )
        if symmetrize_pq:
            Dj = 0.5 * (Dj + Dj.T)
        D[:, :, j] = Dj
    return D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_npz", required=True)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--k", type=int, default=4000)

    # Z patterns
    ap.add_argument("--z_scale", type=float, default=0.30, help="typical Z-line amplitude (voltage/code)")
    ap.add_argument("--seed", type=int, default=0)

    # nonlinearity knobs
    ap.add_argument("--quad_strength", type=float, default=0.02, help="strength of quadratic term")
    ap.add_argument("--sat_strength", type=float, default=0.05, help="strength of saturating nonlinearity")
    ap.add_argument("--sat_scale", type=float, default=0.8, help="scale for tanh saturation (bigger->more linear)")
    ap.add_argument("--per_strength", type=float, default=0.02, help="strength of periodic nonlinearity (sin)")
    ap.add_argument("--per_period", type=float, default=1.0, help="period scale for sin term")

    # noise
    ap.add_argument("--noise_white", type=float, default=0.01, help="white measurement noise std (freq units)")
    ap.add_argument("--noise_pattern", type=float, default=0.005, help="pattern-dependent noise (multiplies |Z|)")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    n, k = args.n, args.k

    # ---- truth linear coupling ----
    C_true = make_sparse_matrix(n, seed=args.seed, diag_loc=1.0, diag_scale=0.2, off_scale=0.08)

    # ---- truth quadratic coupling ----
    D_true = args.quad_strength * make_sparse_tensor_D(n, seed=args.seed + 1, base_scale=0.06, symmetrize_pq=True)

    # ---- random Z patterns (what you dial) ----
    Z = rng.normal(0.0, args.z_scale, size=(k, n)).astype(np.float64)

    # ---- linear response ----
    domega = Z @ C_true.T  # (k,n)

    # ---- quadratic nonlinearity ----
    Z_tensor = Z[:, :, None] * Z[:, None, :]                      # (k,n,n)
    domega += np.tensordot(Z_tensor, D_true, axes=([1, 2], [0, 1]))  # (k,n)

    # ---- saturating nonlinearity (elementwise) ----
    # emulate flux-frequency curve bending / actuator saturation
    if args.sat_strength != 0.0:
        domega += args.sat_strength * np.tanh(Z / args.sat_scale) @ C_true.T

    # ---- periodic nonlinearity (elementwise) ----
    # emulate periodicity in flux (coarse proxy)
    if args.per_strength != 0.0:
        domega += args.per_strength * np.sin(2 * np.pi * Z / args.per_period) @ C_true.T

    # ---- noise ----
    noise = rng.normal(0.0, args.noise_white, size=domega.shape).astype(np.float64)
    # pattern-dependent noise: bigger |Z| -> more noise
    noise += rng.normal(0.0, args.noise_pattern, size=domega.shape).astype(np.float64) * (np.abs(Z) / (args.z_scale + 1e-12))

    domega_meas = domega + noise

    np.savez_compressed(
        args.out_npz,
        Z=Z,
        domega_meas=domega_meas,
        C_true=C_true,
        D_true=D_true,
        meta=np.array(
            [n, k, args.z_scale, args.quad_strength, args.sat_strength, args.per_strength,
             args.noise_white, args.noise_pattern],
            dtype=np.float64
        )
    )

    print(f"Wrote {args.out_npz}")
    print(f"Shapes: Z {Z.shape}, domega_meas {domega_meas.shape}, C_true {C_true.shape}, D_true {D_true.shape}")


if __name__ == "__main__":
    main()
