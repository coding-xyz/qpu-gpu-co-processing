import numpy as np
import torch
import torch.nn.functional as F
from .bayes_core import bayes_posterior_from_templates
from .path_signature_features import signature2_features, SignatureLogReg
import time

@torch.no_grad()
def eval_bayes(X, y, mu0, mu1, sigma, device):
    X = torch.as_tensor(X, dtype=torch.float32, device=device)
    y = torch.as_tensor(y, dtype=torch.long, device=device)
    mu0t = torch.as_tensor(mu0, dtype=torch.float32, device=device).unsqueeze(0).repeat(X.shape[0],1,1)
    mu1t = torch.as_tensor(mu1, dtype=torch.float32, device=device).unsqueeze(0).repeat(X.shape[0],1,1)
    post, _ = bayes_posterior_from_templates(X, mu0t, mu1t, torch.tensor(float(sigma), device=device))
    pred = post.argmax(dim=-1)
    acc = (pred==y).float().mean().item()
    nll = (-torch.log(post[torch.arange(y.shape[0]), y]+1e-9)).mean().item()
    return acc, nll

@torch.no_grad()
def eval_nn(model, loader, device):
    model.eval()
    accs=[]; nlls=[]
    for Xb,yb in loader:
        Xb=Xb.to(device); yb=yb.to(device)
        out=model(Xb)
        logits=out[0] if isinstance(out, tuple) else out
        prob=F.softmax(logits, dim=-1)
        accs.append((prob.argmax(dim=-1)==yb).float().mean())
        nlls.append((-torch.log(prob[torch.arange(yb.shape[0]), yb]+1e-9)).mean())
    return float(torch.stack(accs).mean().cpu()), float(torch.stack(nlls).mean().cpu())

@torch.no_grad()
def eval_signature(model, X, y, device, batch=4096):
    X = torch.as_tensor(X, dtype=torch.float32, device=device)
    y = torch.as_tensor(y, dtype=torch.long, device=device)
    n = X.shape[0]
    accs=[]; nlls=[]
    for i in range(0,n,batch):
        xb = X[i:i+batch]
        yb = y[i:i+batch]
        feats = signature2_features(xb)
        logits = model(feats)
        prob = F.softmax(logits, dim=-1)
        accs.append((prob.argmax(dim=-1)==yb).float().mean())
        nlls.append((-torch.log(prob[torch.arange(yb.shape[0]), yb]+1e-9)).mean())
    return float(torch.stack(accs).mean().cpu()), float(torch.stack(nlls).mean().cpu())

def train_simple(model, train_loader, device, epochs=5, lr=2e-3):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    history = {}
    t0 = time.time()
    for _ in range(epochs):
        model.train()
        for X,y in train_loader:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            out = model(X)
            logits = out[0] if isinstance(out, tuple) else out
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    history["training time"] = time.time() - t0
    return model, history