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


#----------------
# Experiment data
#----------------
import os
import numpy as np 

def load_experiment_data(dir):
    f = {k: sorted(os.listdir(os.path.join(dir, k))) for k in ["0","1"]}

    def read_data(p, k):
        r = np.load(os.path.join(dir,k,p), allow_pickle=True)
        return r["data"][1][0][0][0].real

    x0 = np.concatenate([read_data(x, "0") for x in f["0"]], axis=0)
    x1 = np.concatenate([read_data(x, "1") for x in f["1"]], axis=0)

    x = np.concatenate([x0, x1], axis=0)
    y = np.concatenate([np.zeros(x0.shape[0]), np.ones(x1.shape[0])])

    return x, y

def split_train_test(x, y, ratio=0.9, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    rng.shuffle(idx)
    n_train = int(ratio * len(y))
    tr, te = idx[:n_train], idx[n_train:]
    return x[tr], y[tr], x[te], y[te]