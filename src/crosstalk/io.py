# common/io_utils.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, Optional

def save_json(path: str | Path, obj: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def maybe_write_json(out_json: Optional[str], result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Follow your rule:
    - if out_json is provided: write & return result
    - else: just return result
    """
    if out_json:
        save_json(out_json, result)
        print(f"Wrote {out_json}")
    return result
