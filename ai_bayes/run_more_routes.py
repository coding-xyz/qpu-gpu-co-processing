import argparse, json
import numpy as np
import torch
import torch.nn.functional as F
from data_utils import make_loader
from hmm_gaussian import fit_hmm_templates_fixed, hmm_classify
from transformer_model import TinyTransformer
from path_signature_features import signature2_features, SignatureLogReg

def load_templates_from_meta(meta):
    s0 = np.array(meta["template_s0_re"], np.float32) + 1j*np.array(meta["template_s0_im"], np.float32)
    s1 = np.array(meta["template_s1_re"], np.float32) + 1j*np.array(meta["template_s1_im"], np.float32)
    mu0 = np.stack([s0.real, s0.imag], axis=-1).astype(np.float32)
    mu1 = np.stack([s1.real, s1.imag], axis=-1).astype(np.float32)
    return mu0, mu1

def train_simple(model, train_loader, device, epochs=6, lr=2e-3):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.train()
        for X,y in train_loader:
            X=X.to(device, non_blocking=True); y=y.to(device, non_blocking=True)
            logits = model(X)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return model

@torch.no_grad()
def eval_model(model, X, y, device, batch=2048):
    loader = make_loader(X, y, batch_size=batch, shuffle=False)
    accs=[]; nlls=[]
    model.eval()
    for Xb,yb in loader:
        Xb=Xb.to(device); yb=yb.to(device)
        logits = model(Xb)
        prob = F.softmax(logits, dim=-1)
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="test_dataset.npz")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--hmm_iter", type=int, default=25)
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=True)
    Xtr, ytr, Xte, yte = d["X_train"], d["y_train"], d["X_test"], d["y_test"]
    meta = json.loads(d["meta"].item())
    mu0, mu1 = load_templates_from_meta(meta)
    sigma0 = float(meta["sigma"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # [E] HMM
    sigma, p01, p10 = fit_hmm_templates_fixed(Xtr, mu0, mu1, n_iter=args.hmm_iter, init_sigma=sigma0, device=device)
    post = hmm_classify(Xte, mu0, mu1, sigma=sigma, p01=p01, p10=p10, device=device)
    y = torch.as_tensor(yte, dtype=torch.long, device=device)
    pred = post.argmax(dim=-1)
    acc = (pred==y).float().mean().item()
    nll = (-torch.log(post[torch.arange(y.shape[0]), y]+1e-9)).mean().item()
    print(f"[E] HMM: acc={acc:.4f}  NLL={nll:.4f}  sigma={sigma:.3f} p01={p01:.2e} p10={p10:.2e}")

    # [F] Transformer
    train_loader = make_loader(Xtr, ytr, batch_size=args.batch, shuffle=True)
    T = Xtr.shape[1]
    trm = train_simple(TinyTransformer(T), train_loader, device, epochs=args.epochs)
    acc, nll = eval_model(trm, Xte, yte, device)
    print(f"[F] Transformer: acc={acc:.4f}  NLL={nll:.4f}")

    # [G] Signature-like + linear
    Xtr_t = torch.as_tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.as_tensor(ytr, dtype=torch.long, device=device)
    feats = signature2_features(Xtr_t)
    clf = SignatureLogReg(feats.shape[-1]).to(device)
    opt = torch.optim.AdamW(clf.parameters(), lr=3e-3)
    bs = 8192
    for _ in range(args.epochs):
        perm = torch.randperm(feats.shape[0], device=device)
        for i in range(0, feats.shape[0], bs):
            idx = perm[i:i+bs]
            logits = clf(feats[idx])
            loss = F.cross_entropy(logits, ytr_t[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    acc, nll = eval_signature(clf, Xte, yte, device)
    print(f"[G] Signature-like + linear: acc={acc:.4f}  NLL={nll:.4f}")

if __name__ == "__main__":
    main()
