import argparse
import numpy as np

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json

from src.tomography.io import load_npz, save_json
from src.tomography.tomo_1q import fit_1q_mle_spam
from src.tomography.tomo_2q import fit_2q_mle_spam
from src.tomography.metrics import purity, bloch_vector_1q, pauli_expectations_2q

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", default=None)
    ap.add_argument("--device", default=None, help="cpu or cuda (default auto)")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ds = load_npz(args.data)
    n = int(ds["n_qubits"])
    counts = ds["counts"]
    A = ds["A_meas"]

    if n == 1:
        steps = args.steps if args.steps is not None else 2000
        lr = args.lr if args.lr is not None else 0.05
        out = fit_1q_mle_spam(counts, A, steps=steps, lr=lr, seed=args.seed, device=args.device)
        rho = out["rho"]
        result = {
            "n_qubits": 1,
            "device_used": out["device_used"],
            "optimizer": {"steps": out["steps"], "lr": out["lr"], "seed": out["seed"]},
            "nll_final": out["nll_final"],
            "runtime_sec": out["runtime_sec"],
            "rho_real": np.real(rho).tolist(),
            "rho_imag": np.imag(rho).tolist(),
            "purity": purity(rho),
            "bloch": bloch_vector_1q(rho),
            "meta": ds.get("meta_json", None),
        }
        if args.output is not None:
            save_json(args.output, result)
            print(f"Wrote {args.output}")
        return result


    if n == 2:
        steps = args.steps if args.steps is not None else 3000
        lr = args.lr if args.lr is not None else 0.03
        out = fit_2q_mle_spam(counts, A, steps=steps, lr=lr, seed=args.seed, device=args.device)
        rho = out["rho"]
        result = {
            "n_qubits": 2,
            "device_used": out["device_used"],
            "optimizer": {"steps": out["steps"], "lr": out["lr"], "seed": out["seed"]},
            "nll_final": out["nll_final"],
            "runtime_sec": out["runtime_sec"],
            "rho_real": np.real(rho).tolist(),
            "rho_imag": np.imag(rho).tolist(),
            "purity": purity(rho),
            "pauli_expectations": pauli_expectations_2q(rho),
            "settings_order": out.get("settings_order"),
            "meta": ds.get("meta_json", None),
        }
        if args.output is not None:
            save_json(args.output, result)
            print(f"Wrote {args.output}")
        return result

    raise ValueError("n_qubits must be 1 or 2")

if __name__ == "__main__":
    res = main()    
    print(json.dumps(res, indent=2, ensure_ascii=False))

