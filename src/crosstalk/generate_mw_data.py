# mw/gen_mw_npz.py
from __future__ import annotations
import argparse
import numpy as np

def random_stable_fir(n: int, L: int, seed: int = 0, leak: float = 0.15) -> np.ndarray:
    """
    H: shape (n_out, n_in, L)
    Mostly diagonal + small cross terms, with decaying taps.
    """
    rng = np.random.default_rng(seed)
    H = np.zeros((n, n, L), dtype=np.float64)

    # diagonal: main path
    for i in range(n):
        taps = rng.normal(0, 1, size=L)
        decay = np.exp(-np.linspace(0, 3.0, L))
        taps = 0.7 * decay * taps
        taps[0] += 1.0  # direct
        H[i, i, :] = taps

    # cross terms
    for i in range(n):
        m = rng.integers(2, min(6, n))  # number of cross couplings
        js = rng.choice(n, size=m, replace=False)
        for j in js:
            if j == i: 
                continue
            taps = rng.normal(0, 1, size=L)
            decay = np.exp(-np.linspace(0, 3.5, L))
            H[i, j, :] += leak * decay * taps
    return H

def apply_mimo_fir(H: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    y_i[t] = sum_j sum_k H[i,j,k] x_j[t-k]
    H: (n,n,L), x: (n,T) -> y: (n,T)
    """
    n, _, L = H.shape
    T = x.shape[1]
    y = np.zeros((n, T), dtype=np.float64)
    for k in range(L):
        y[:, k:] += H[:, :, k] @ x[:, :T-k]
    return y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_npz", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--t", type=int, default=16384, help="time samples")
    ap.add_argument("--L", type=int, default=33, help="FIR length")
    ap.add_argument("--noise", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    n, T, L = args.n, args.t, args.L

    H_true = random_stable_fir(n, L, seed=args.seed, leak=0.12)

    # Excitation x: random multi-tone-like waveform (baseband I/Q packed as real here)
    x = rng.normal(0, 1, size=(n, T))
    # Smooth a bit to mimic AWG-ish bandwidth (cheap lowpass)
    for i in range(n):
        x[i] = np.convolve(x[i], np.ones(9)/9.0, mode="same")

    y = apply_mimo_fir(H_true, x)
    y += rng.normal(0, args.noise, size=y.shape)

    np.savez_compressed(
        args.out_npz,
        x=x.astype(np.float32),
        y=y.astype(np.float32),
        H_true=H_true.astype(np.float32),
        meta=np.array([n, T, L, args.noise], dtype=np.float64),
    )
    print(f"Wrote {args.out_npz}")
    print(f"Shapes: x {x.shape}, y {y.shape}, H_true {H_true.shape}")

if __name__ == "__main__":
    main()
