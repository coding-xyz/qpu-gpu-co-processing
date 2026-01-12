import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LDA(nn.Module):
    def __init__(self, T):
        super(LDA, self).__init__()
        self.fc = nn.Linear(2,2)

    def forward(self, X):
        I = X[:, :, 0].sum(dim=1)  # (batch_size,)
        Q = X[:, :, 1].sum(dim=1)  # (batch_size,)
        z = torch.stack([I, Q], dim=-1)
        return self.fc(z)

class MatchedFilter(nn.Module):
    def __init__(self, T):
        super(MatchedFilter, self).__init__()
        self.window = nn.Parameter(torch.ones(T))  # (time_steps,)
        self.fc = nn.Linear(2,2)

    def forward(self, X):
        w = self.window.view(1, -1)      # (1,T)
        I = (X[:, :, 0] * w).sum(dim=1)  # (batch_size,)
        Q = (X[:, :, 1] * w).sum(dim=1)  # (batch_size,)
        z = torch.stack([I, Q], dim=-1)
        return self.fc(z)

class TinyCNN(nn.Module):
    def __init__(self, T, hidden=32):
        super().__init__()
        self.conv1 = nn.Conv1d(2, hidden, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=7, padding=3)
        self.conv3 = nn.Conv1d(hidden, hidden, kernel_size=5, padding=2)
        self.fc = nn.Linear(hidden, 2)

    def forward(self, x):
        x = x.transpose(1,2)  # (B,2,T)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.mean(dim=-1)
        return self.fc(x)

class AmortizedBayesNet(nn.Module):
    def __init__(self, T, hidden=64, predict_sigma=True):
        super().__init__()
        self.predict_sigma = predict_sigma
        self.encoder = nn.Sequential(
            nn.Conv1d(2, hidden, 9, padding=4),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, 9, padding=4),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, 7, padding=3),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden, 2)
        if predict_sigma:
            self.sigma_head = nn.Linear(hidden, 1)

    def forward(self, x):
        x = x.transpose(1,2)
        h = self.encoder(x).mean(dim=-1)
        logits = self.head(h)
        if self.predict_sigma:
            sigma = F.softplus(self.sigma_head(h)) + 1e-4
            return logits, sigma.squeeze(-1)
        return logits, None


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        self.pe: torch.Tensor
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TinyTransformer(nn.Module):
    def __init__(self, T, d_model=64, nhead=4, num_layers=2, dim_ff=128, dropout=0.1):
        super().__init__()
        self.inp = nn.Linear(2, d_model)
        self.pos = PositionalEncoding(d_model, max_len=max(512, T+5))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, activation="gelu"
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 2)

    def forward(self, x):
        h = self.inp(x)
        h = self.pos(h)
        h = self.enc(h)
        h = h.mean(dim=1)
        return self.head(h)

