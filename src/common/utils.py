from __future__ import annotations
import torch
from torch.utils.data import Dataset, DataLoader

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
