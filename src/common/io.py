from __future__ import annotations
import json
import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional

def ensure_dir(d: str | Path) -> Path:
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    return d

def save_json(path: str | Path, obj: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        print(f"Wrote {path}")

def maybe_write_json(out_json: Optional[str], result: Dict[str, Any]) -> Dict[str, Any]:
    if out_json:
        save_json(out_json, result)
    return result

def load_json(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())

def save_npz(path: str, payload: Dict[str, Any]) -> None:
    arr = {}
    for k, v in payload.items():
        if isinstance(v, (dict, list)):
            arr[k] = np.array([json.dumps(v)], dtype=object)
        elif isinstance(v, str):
            arr[k] = np.array([v], dtype=object)
        else:
            arr[k] = v
    np.savez_compressed(path, **arr)
    print(f"Wrote {path}")

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