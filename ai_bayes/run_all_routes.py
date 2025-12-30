import argparse, json
import numpy as np
import torch
import torch.nn.functional as F
from bayes_core import bayes_posterior_from_templates
from bayes_em import bayes_em_fit
from models import TinyCNN, AmortizedBayesNet
from data_utils import make_loader

def load_templates_from_meta(meta):
    s0 = np.array(meta["template_s0_re"], np.float32) + 1j*np.array(meta["template_s0_im"], np.float32)
    s1 = np.array(meta["template_s1_re"], np.float32) + 1j*np.array(meta["template_s1_im"], np.float32)
    mu0 = np.stack([s0.real, s0.imag], axis=-1).astype(np.float32)
    mu1 = np.stack([s1.real, s1.imag], axis=-1).astype(np.float32)
    return mu0, mu1

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

def train_simple(model, train_loader, device, epochs=5, lr=2e-3):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.train()
        for X,y in train_loader:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            out = model(X)
            logits = out[0] if isinstance(out, tuple) else out
            loss = torch.nn.functional.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return model

@torch.no_grad()
def eval_nn(model, X, y, device, batch=2048):
    loader = make_loader(X, y, batch_size=batch, shuffle=False)
    model.eval()
    accs=[]; nlls=[]
    for Xb,yb in loader:
        Xb=Xb.to(device); yb=yb.to(device)
        out=model(Xb)
        logits=out[0] if isinstance(out, tuple) else out
        prob=torch.nn.functional.softmax(logits, dim=-1)
        accs.append((prob.argmax(dim=-1)==yb).float().mean())
        nlls.append((-torch.log(prob[torch.arange(yb.shape[0]), yb]+1e-9)).mean())
    return float(torch.stack(accs).mean().cpu()), float(torch.stack(nlls).mean().cpu())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="test_dataset.npz")
    ap.add_argument("--em_iter", type=int, default=25)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=1024)
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=True)
    Xtr, ytr, Xte, yte = d["X_train"], d["y_train"], d["X_test"], d["y_test"]
    meta = json.loads(d["meta"].item())
    mu0_true, mu1_true = load_templates_from_meta(meta)
    sigma_true = float(meta["sigma"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    acc, nll = eval_bayes(Xte, yte, mu0_true, mu1_true, sigma_true, device)
    print(f"[A] Bayes-only: acc={acc:.4f}  NLL={nll:.4f}")

    mu0_em, mu1_em, sigma_em = bayes_em_fit(Xtr, n_iter=args.em_iter, anchor_mu0=None, device=device)
    acc, nll = eval_bayes(Xte, yte, mu0_em.numpy(), mu1_em.numpy(), sigma_em, device)
    print(f"[B] Bayes-EM (unanchored): acc={acc:.4f}  NLL={nll:.4f}  sigma={sigma_em:.3f}")

    mu0_em2, mu1_em2, sigma_em2 = bayes_em_fit(
        Xtr, n_iter=args.em_iter, anchor_mu0=mu0_true, anchor_strength=200.0, device=device
    )
    acc, nll = eval_bayes(Xte, yte, mu0_em2.numpy(), mu1_em2.numpy(), sigma_em2, device)
    print(f"[B'] Bayes-EM (weak anchor): acc={acc:.4f}  NLL={nll:.4f}  sigma={sigma_em2:.3f}")

    train_loader = make_loader(Xtr, ytr, batch_size=args.batch, shuffle=True)
    T = Xtr.shape[1]

    cnn = train_simple(TinyCNN(T), train_loader, device, epochs=args.epochs)
    acc, nll = eval_nn(cnn, Xte, yte, device)
    print(f"[C] CNN: acc={acc:.4f}  NLL={nll:.4f}")

    ab = train_simple(AmortizedBayesNet(T, predict_sigma=True), train_loader, device, epochs=args.epochs)
    acc, nll = eval_nn(ab, Xte, yte, device)
    print(f"[D] Amortized Bayes (NN): acc={acc:.4f}  NLL={nll:.4f}")

if __name__ == "__main__":
    main()
