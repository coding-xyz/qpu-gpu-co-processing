# flux/gen_flux_npz.py
from __future__ import annotations
import argparse
import numpy as np

def make_sparse_crosstalk(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    C = np.zeros((n, n), dtype=np.float64)

    # Diagonal self-coupling
    diag = rng.normal(loc=1.0, scale=0.2, size=n)
    C[np.arange(n), np.arange(n)] = diag

    # Local-ish sparse leakage: a few off-diagonals per row
    for i in range(n):
        m = rng.integers(2, 6)  # 2~5 leak terms
        js = rng.choice(n, size=m, replace=False)
        for j in js:
            if j == i:
                continue
            C[i, j] += rng.normal(loc=0.0, scale=0.08)

    return C

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_npz", required=True)
    ap.add_argument("--n", type=int, default=24, help="number of flux lines / qubits")
    ap.add_argument("--k", type=int, default=4000, help="number of random bias patterns")
    ap.add_argument("--phi_scale", type=float, default=0.35, help="typical flux amplitude")
    ap.add_argument("--noise", type=float, default=0.01, help="measurement noise (freq units)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nonlinear", action="store_true", help="add weak quadratic term")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    n, k = args.n, args.k

    C_true = make_sparse_crosstalk(n, seed=args.seed)

    # Random bias patterns Phi: shape (k, n)
    Phi = rng.normal(0.0, args.phi_scale, size=(k, n)).astype(np.float64)

    # Linear response: dω = Phi @ C^T  (so output shape (k,n))
    domega = Phi @ C_true.T

    # Optional weak quadratic effects: sum_j D_ij * Phi_j^2
    if args.nonlinear:
        D_true = 0.02 * make_sparse_crosstalk(n, seed=args.seed + 1)
        domega = domega + (Phi**2) @ D_true.T
    else:
        D_true = np.zeros_like(C_true)

    # Add measurement noise
    omega_meas = domega + rng.normal(0.0, args.noise, size=domega.shape)

    np.savez_compressed(
        args.out_npz,
        Phi=Phi,
        omega_meas=omega_meas,
        C_true=C_true,
        D_true=D_true,
        meta=np.array([args.n, args.k, args.phi_scale, args.noise, int(args.nonlinear)], dtype=np.float64),
    )
    print(f"Wrote {args.out_npz}")
    print(f"Shapes: Phi {Phi.shape}, omega_meas {omega_meas.shape}, C_true {C_true.shape}")

if __name__ == "__main__":
    main()
