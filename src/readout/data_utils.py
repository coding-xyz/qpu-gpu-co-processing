import torch
from torch.utils.data import Dataset, DataLoader

class TraceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

def make_loader(X, y, batch_size=1024, shuffle=True, num_workers=0):
    ds = TraceDataset(X, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
