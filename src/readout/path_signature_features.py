import torch

def signature2_features(X):
    B,T,C = X.shape
    dX = X[:,1:,:] - X[:,:-1,:]
    dIt = X[:,-1,0] - X[:,0,0]
    dQt = X[:,-1,1] - X[:,0,1]

    I0 = X[:,:-1,0]; Q0 = X[:,:-1,1]
    I1 = X[:,1:,0];  Q1 = X[:,1:,1]
    area = 0.5 * torch.sum(I0*Q1 - Q0*I1, dim=1)

    mean = X.mean(dim=1)
    var = X.var(dim=1, unbiased=False)

    seg_len = torch.sqrt((dX*dX).sum(dim=-1) + 1e-12).sum(dim=1)
    r = torch.sqrt((X*X).sum(dim=-1) + 1e-12)
    rmax = r.max(dim=1).values
    rmin = r.min(dim=1).values

    feats = torch.stack(
        [dIt, dQt, area,
         mean[:,0], mean[:,1], var[:,0], var[:,1],
         seg_len, rmax, rmin],
        dim=-1
    )
    return feats

class SignatureLogReg(torch.nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.lin = torch.nn.Linear(feat_dim, 2)
    def forward(self, feats):
        return self.lin(feats)
