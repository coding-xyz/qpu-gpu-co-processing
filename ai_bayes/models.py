import torch
import torch.nn as nn
import torch.nn.functional as F

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
