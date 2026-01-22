from __future__ import annotations
import torch
from torch.utils.data import Dataset, DataLoader
import subprocess
from pathlib import Path


def pick_device(s: str) -> torch.device:
    if s == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if s == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available but --device cuda requested.")
        return torch.device("cuda")
    return torch.device("cpu")

def to_tensor(x, device: torch.device, dtype=None):
    t = torch.as_tensor(x)
    if dtype is not None:
        t = t.to(dtype)
    return t.to(device)

class TraceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

def make_loader(X, y, batch_size=1024, shuffle=True, num_workers=0):
    ds = TraceDataset(X, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

def run_script(
    entry: str,
    args: dict,
    *,
    cwd=None,
    python: str = "python",
    capture: bool = True,
):
    """
    Run a python module (`-m xxx`) or a script path (`xxx.py`) with CLI args.

    Parameters
    ----------
    entry:
        - module name: "scripts.xxx"
        - script path: "scripts/xxx.py"
    args:
        dict of CLI args, e.g.
            {
              "--in_npz": "data/test_flux.npz",
              "--epochs": 1000,
              "--stdout_json": True,
            }
    cwd:
        working directory (recommended: project root)
    python:
        python executable (default: "python")
    capture:
        capture stdout/stderr if True
    """

    entry = str(entry)

    # ----------------------------
    # Decide: module or script
    # ----------------------------
    is_script = entry.endswith(".py") or Path(entry).exists()

    if is_script:
        # script path
        cmd = [python, entry]
    else:
        # module
        cmd = [python, "-m", entry]

    # ----------------------------
    # Append CLI arguments
    # ----------------------------
    for k, v in args.items():
        if v is None:
            continue
        if isinstance(v, bool):
            if v:
                cmd.append(k)
        else:
            cmd.extend([k, str(v)])

    # ----------------------------
    # Run
    # ----------------------------
    print("Running command:")
    print(" ", " ".join(cmd))

    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        text=True,
        check=False,   # notebook-friendly
    )

    return result
