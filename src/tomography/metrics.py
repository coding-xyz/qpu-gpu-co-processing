import numpy as np
from typing import Dict

def purity(rho: np.ndarray) -> float:
    return float(np.real(np.trace(rho @ rho)))

def bloch_vector_1q(rho: np.ndarray) -> Dict[str, float]:
    X = np.array([[0,1],[1,0]], dtype=np.complex64)
    Y = np.array([[0,-1j],[1j,0]], dtype=np.complex64)
    Z = np.array([[1,0],[0,-1]], dtype=np.complex64)
    return {
        "x": float(np.real(np.trace(rho @ X))),
        "y": float(np.real(np.trace(rho @ Y))),
        "z": float(np.real(np.trace(rho @ Z))),
    }

def pauli_expectations_2q(rho: np.ndarray) -> Dict[str, float]:
    I = np.eye(2, dtype=np.complex64)
    X = np.array([[0,1],[1,0]], dtype=np.complex64)
    Y = np.array([[0,-1j],[1j,0]], dtype=np.complex64)
    Z = np.array([[1,0],[0,-1]], dtype=np.complex64)
    paulis = {"I": I, "X": X, "Y": Y, "Z": Z}
    out: Dict[str, float] = {}
    for a in "IXYZ":
        for b in "IXYZ":
            P = np.kron(paulis[a], paulis[b])
            out[a+b] = float(np.real(np.trace(rho @ P)))
    return out
