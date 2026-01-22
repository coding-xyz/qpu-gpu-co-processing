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

def make_sparse_crosstalk_3d(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    D = np.zeros((n, n, n), dtype=np.float64)  # Now D is a 3D tensor
    
    # Diagonal self-coupling for each slice
    diag = rng.normal(loc=1.0, scale=0.2, size=n)
    D[np.arange(n), np.arange(n), np.arange(n)] = diag

    # Local-ish sparse leakage: a few off-diagonals per row for each slice
    for i in range(n):
        js = rng.choice(n, size=rng.integers(2, 6), replace=False)  # 2~5 leak terms
        for j in js:
            ks = rng.choice(n, size=rng.integers(2, 6), replace=False)  # 2~5 leak terms
            for k in ks:
                if j == i and k == i:
                    continue
                D[i, j, k] += rng.normal(loc=0.0, scale=0.2)

    return D

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_npz", required=True)
    ap.add_argument("--n", type=int, default=24, help="number of flux lines / qubits")
    ap.add_argument("--k", type=int, default=4000, help="number of random bias patterns")
    ap.add_argument("--phi_scale", type=float, default=0.35, help="typical flux amplitude")
    ap.add_argument("--noise", type=float, default=0.01, help="measurement noise (freq units)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nonlinear", type=float, default=0, help="add weak quadratic term")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    n, k = args.n, args.k

    C_true = make_sparse_crosstalk(n, seed=args.seed)

    # Random bias patterns Phi: shape (k, n)
    Phi = rng.normal(0.0, args.phi_scale, size=(k, n)).astype(np.float64)

    # Linear response: dω = Phi @ C^T  (so output shape (k,n))
    domega = Phi @ C_true.T

    # Optional weak quadratic effects: sum_p,q D_i,p,q * Phi_k,p * Phi_k,q
    if args.nonlinear != 0:
        Phi_tensor = Phi[:,:,None] * Phi[:,None,:]
        D_true = args.nonlinear * make_sparse_crosstalk_3d(n, seed=args.seed + 1)
        domega += np.tensordot(Phi_tensor, D_true, axes=([1,2],[1,2]))
    else:
        D_true = np.zeros((n,n,n))

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
    print(f"Shapes: Phi {Phi.shape}, omega_meas {omega_meas.shape}, C_true {C_true.shape}, D_true {D_true.shape}")

if __name__ == "__main__":
    main()
