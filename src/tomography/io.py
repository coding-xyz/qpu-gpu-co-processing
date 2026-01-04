import json
import numpy as np
from pathlib import Path
from typing import Any, Dict

def save_npz(path: str, payload: Dict[str, Any]) -> None:
    """
    Save a tomography dataset payload to NPZ.

    Required keys in payload:
      - n_qubits: int (1 or 2)
      - counts: np.ndarray
          * 1q: shape (3,2), settings order [X,Y,Z], outcome order [+, -] mapped to [0,1]
          * 2q: shape (9,4), settings order [XX,XY,XZ,YX,YY,YZ,ZX,ZY,ZZ],
                outcome order [++, +-, -+, --] mapped to [00,01,10,11]
      - A_meas: np.ndarray
          * 1q: shape (2,2), 2q: shape (4,4)
          * convention: q_obs = A_meas @ p_true
            columns = true outcomes, rows = observed outcomes

    Optional:
      - settings: np.ndarray[str]
      - meta_json: str (JSON-encoded metadata)
    """
    arr = {}
    for k, v in payload.items():
        if isinstance(v, (dict, list)):
            arr[k] = np.array([json.dumps(v)], dtype=object)
        elif isinstance(v, str):
            arr[k] = np.array([v], dtype=object)
        else:
            arr[k] = v
    np.savez_compressed(path, **arr)

def load_npz(path: str) -> Dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    out: Dict[str, Any] = {}
    for k in data.files:
        v = data[k]

        # unwrap numpy scalars / 0-d arrays
        if isinstance(v, np.ndarray) and v.ndim == 0:
            scalar = v.item()
            if isinstance(scalar, (str, bytes)):
                try:
                    out[k] = json.loads(scalar)
                except Exception:
                    out[k] = scalar
            else:
                out[k] = scalar
            continue

        # unwrap 1-element arrays (often stored for strings/objects)
        if isinstance(v, np.ndarray) and v.shape == (1,):
            scalar = v[0]
            if isinstance(scalar, (str, bytes)):
                try:
                    out[k] = json.loads(scalar)
                except Exception:
                    out[k] = scalar
                continue
            if v.dtype == object:
                try:
                    out[k] = json.loads(str(scalar))
                except Exception:
                    out[k] = scalar
                continue
            out[k] = scalar
            continue

        out[k] = v

    if "n_qubits" in out:
        out["n_qubits"] = int(out["n_qubits"])
    return out

def save_json(path: str, obj: Dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False))

def load_json(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())
