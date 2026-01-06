import argparse
import json
import math
import numpy as np

from ..src.common import save_npz

def _ket_state(name: str, n_qubits: int) -> np.ndarray:
    if n_qubits == 1:
        if name == "0": return np.array([1,0], dtype=np.complex64)
        if name == "1": return np.array([0,1], dtype=np.complex64)
        if name == "+": return np.array([1,1], dtype=np.complex64)/math.sqrt(2)
        if name == "-": return np.array([1,-1], dtype=np.complex64)/math.sqrt(2)
        if name.lower() == "i": return np.array([1,1j], dtype=np.complex64)/math.sqrt(2)
        if name.lower() == "-i": return np.array([1,-1j], dtype=np.complex64)/math.sqrt(2)
        raise ValueError(f"Unknown 1q state: {name}")
    if n_qubits == 2:
        n = name.lower()
        if n in ["phi+", "bell_phi+"]:
            return np.array([1,0,0,1], dtype=np.complex64)/math.sqrt(2)
        if n in ["phi-", "bell_phi-"]:
            return np.array([1,0,0,-1], dtype=np.complex64)/math.sqrt(2)
        if n in ["psi+", "bell_psi+"]:
            return np.array([0,1,1,0], dtype=np.complex64)/math.sqrt(2)
        if n in ["psi-", "bell_psi-"]:
            return np.array([0,1,-1,0], dtype=np.complex64)/math.sqrt(2)
        if name == "00": return np.array([1,0,0,0], dtype=np.complex64)
        if name == "11": return np.array([0,0,0,1], dtype=np.complex64)
        raise ValueError(f"Unknown 2q state: {name}")
    raise ValueError("n_qubits must be 1 or 2")

def _proj_1q():
    I = np.eye(2, dtype=np.complex64)
    X = np.array([[0,1],[1,0]], dtype=np.complex64)
    Y = np.array([[0,-1j],[1j,0]], dtype=np.complex64)
    Z = np.array([[1,0],[0,-1]], dtype=np.complex64)
    Px = np.stack([0.5*(I+X), 0.5*(I-X)], axis=0)
    Py = np.stack([0.5*(I+Y), 0.5*(I-Y)], axis=0)
    Pz = np.stack([0.5*(I+Z), 0.5*(I-Z)], axis=0)
    return np.stack([Px,Py,Pz], axis=0)  # [3,2,2,2] order X,Y,Z

def _proj_2q():
    sq = _proj_1q()
    labels = ["X","Y","Z"]
    settings = []
    setting_labels = []
    for ia,a in enumerate(labels):
        for ib,b in enumerate(labels):
            setting_labels.append(a+b)
            Pa = sq[ia]
            Pb = sq[ib]
            P = np.stack([
                np.kron(Pa[0], Pb[0]),
                np.kron(Pa[0], Pb[1]),
                np.kron(Pa[1], Pb[0]),
                np.kron(Pa[1], Pb[1]),
            ], axis=0)  # [4,4,4]
            settings.append(P)
    return np.stack(settings, axis=0), setting_labels  # [9,4,4,4]

def _sample_multinomial(probs: np.ndarray, shots: int, rng: np.random.Generator) -> np.ndarray:
    draws = rng.choice(len(probs), size=shots, p=probs)
    return np.array([(draws==k).sum() for k in range(len(probs))], dtype=np.int64)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_qubits", type=int, choices=[1,2], required=True)
    ap.add_argument("--state", type=str, default="+",
                    help="1q: 0,1,+,-,i,-i ; 2q: 00,11,phi+,phi-,psi+,psi-")
    ap.add_argument("--shots", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--p01", type=float, default=0.04, help="single-qubit P(obs=1|true=0)")
    ap.add_argument("--p10", type=float, default=0.05, help="single-qubit P(obs=0|true=1)")
    ap.add_argument("--out", type=str, required=True, help="output npz path")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    ket = _ket_state(args.state, args.n_qubits)
    rho = np.outer(ket, ket.conj()).astype(np.complex64)
    rho = (rho + rho.conj().T)/2
    rho = rho / np.trace(rho)

    # confusion: q_obs = A @ p_true (cols=true, rows=obs)
    A1 = np.array([[1-args.p01, args.p10],
                   [args.p01, 1-args.p10]], dtype=np.float32)

    if args.n_qubits == 1:
        M = _proj_1q()
        counts = np.zeros((3,2), dtype=np.int64)
        for m in range(3):
            p_true = np.array([np.real(np.trace(rho @ M[m,0])),
                               np.real(np.trace(rho @ M[m,1]))], dtype=np.float64)
            p_true = np.clip(p_true, 1e-12, 1-1e-12)
            p_true = p_true / p_true.sum()
            q_obs = A1 @ p_true
            q_obs = q_obs / q_obs.sum()
            counts[m] = _sample_multinomial(q_obs, args.shots, rng)

        payload = dict(
            n_qubits=np.int64(1),
            counts=counts,
            A_meas=A1,
            settings=np.array(["X","Y","Z"], dtype=object),
            meta_json=json.dumps({
                "state": args.state,
                "shots_per_setting": args.shots,
                "seed": args.seed,
                "confusion_single_qubit": {"p01": args.p01, "p10": args.p10},
                "convention": "q_obs = A_meas @ p_true ; columns=true, rows=observed",
                "settings_order_1q": ["X","Y","Z"],
                "outcome_order_1q": ["+","-"]
            })
        )
        save_npz(args.out, payload)
    
    elif args.n_qubits == 2:
        M2, labels = _proj_2q()
        A2 = np.kron(A1, A1).astype(np.float32)
        counts = np.zeros((9,4), dtype=np.int64)
        for m in range(9):
            p_true = np.array([np.real(np.trace(rho @ M2[m,o])) for o in range(4)], dtype=np.float64)
            p_true = np.clip(p_true, 1e-12, 1-1e-12)
            p_true = p_true / p_true.sum()
            q_obs = A2 @ p_true
            q_obs = q_obs / q_obs.sum()
            counts[m] = _sample_multinomial(q_obs, args.shots, rng)

        payload = dict(
            n_qubits=np.int64(2),
            counts=counts,
            A_meas=A2,
            settings=np.array(labels, dtype=object),
            meta_json=json.dumps({
                "state": args.state,
                "shots_per_setting": args.shots,
                "seed": args.seed,
                "confusion_single_qubit": {"p01": args.p01, "p10": args.p10},
                "A_meas_constructed_as": "kron(A1, A1)",
                "convention": "q_obs = A_meas @ p_true ; columns=true, rows=observed",
                "settings_order_2q": labels,
                "outcome_order_2q": ["++","+-","-+","--"]
            })
        )
        save_npz(args.out, payload)
    else:
        return 1
    
    return 0

if __name__ == "__main__":
    main()
