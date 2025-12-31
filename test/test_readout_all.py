import argparse, json
import numpy as np
import torch
import torch.nn.functional as F

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.readout.models import TinyCNN, AmortizedBayesNet, TinyTransformer
from src.readout.bayes_em import bayes_em_fit
from src.readout.hmm_gaussian import fit_hmm_templates_fixed, hmm_classify
from src.readout.path_signature_features import signature2_features, SignatureLogReg

from src.readout.training import load_templates_from_meta, train_simple, eval_bayes, eval_nn, eval_signature
from src.readout.data_utils import make_loader

# Define the flush function to ensure the output is shown immediately
def flush_print(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="test_dataset.npz")
    ap.add_argument("--em_iter", type=int, default=25)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--hmm_iter", type=int, default=25)
    args = ap.parse_args()

    # Load data
    d = np.load(args.data, allow_pickle=True)
    Xtr, ytr, Xte, yte = d["X_train"], d["y_train"], d["X_test"], d["y_test"]
    meta = json.loads(d["meta"].item())
    mu0, mu1 = load_templates_from_meta(meta)
    sigma0 = float(meta["sigma"])

    # Prepare loaders for ML
    train_loader = make_loader(Xtr, ytr, batch_size=args.batch, shuffle=True)
    test_loader = make_loader(Xte, yte, batch_size=2048, shuffle=False)
    T = Xtr.shape[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    flush_print("device:", device)

    # Dictionary to store results
    results = {}

    # [A] Bayes-only
    acc, nll = eval_bayes(Xte, yte, mu0, mu1, sigma0, device)
    results["Bayes-only"] = {"acc": acc, "nll": nll}
    flush_print(f"[A] Bayes-only: acc={acc:.4f}  NLL={nll:.4f}")

    # # [B] Bayes-EM (unanchored)
    # mu0_em, mu1_em, sigma_em = bayes_em_fit(Xtr, n_iter=args.em_iter, anchor_mu0=None, device=device)
    # acc, nll = eval_bayes(Xte, yte, mu0_em.numpy(), mu1_em.numpy(), sigma_em, device)
    # results["Bayes-EM (unanchored)"] = {"acc": acc, "nll": nll, "sigma": sigma_em}
    # flush_print(f"[B] Bayes-EM (unanchored): acc={acc:.4f}  NLL={nll:.4f}  sigma={sigma_em:.3f}")

    # [B] Bayes-EM (weak anchor)
    mu0_em2, mu1_em2, sigma_em2 = bayes_em_fit(
        Xtr, n_iter=args.em_iter, anchor_mu0=mu0, anchor_strength=200.0, device=device
    )
    acc, nll = eval_bayes(Xte, yte, mu0_em2.numpy(), mu1_em2.numpy(), sigma_em2, device)
    results["Bayes-EM (weak anchor)"] = {"acc": acc, "nll": nll, "sigma": sigma_em2}
    flush_print(f"[B'] Bayes-EM (weak anchor): acc={acc:.4f}  NLL={nll:.4f}  sigma={sigma_em2:.3f}")

    # [C] CNN
    cnn = train_simple(TinyCNN(T), train_loader, device, epochs=args.epochs)
    acc, nll = eval_nn(cnn, test_loader, device)
    results["CNN"] = {"acc": acc, "nll": nll}
    flush_print(f"[C] CNN: acc={acc:.4f}  NLL={nll:.4f}")

    # [D] Amortized Bayes + NN
    ab = train_simple(AmortizedBayesNet(T, predict_sigma=True), train_loader, device, epochs=args.epochs)
    acc, nll = eval_nn(ab, test_loader, device)
    results["Amortized Bayes (NN)"] = {"acc": acc, "nll": nll}
    flush_print(f"[D] Amortized Bayes (NN): acc={acc:.4f}  NLL={nll:.4f}")

    # [E] HMM
    sigma, p01, p10 = fit_hmm_templates_fixed(Xtr, mu0, mu1, n_iter=args.hmm_iter, init_sigma=sigma0, device=device)
    post = hmm_classify(Xte, mu0, mu1, sigma=sigma, p01=p01, p10=p10, device=device)
    y = torch.as_tensor(yte, dtype=torch.long, device=device)
    pred = post.argmax(dim=-1)
    acc = (pred==y).float().mean().item()
    nll = (-torch.log(post[torch.arange(y.shape[0]), y]+1e-9)).mean().item()
    results["HMM"] = {"acc": acc, "nll": nll, "sigma": sigma, "p01": p01, "p10": p10}
    flush_print(f"[E] HMM: acc={acc:.4f}  NLL={nll:.4f}  sigma={sigma:.3f} p01={p01:.2e} p10={p10:.2e}")

    # [F] Transformer
    trm = train_simple(TinyTransformer(T), train_loader, device, epochs=args.epochs)
    acc, nll = eval_nn(trm, test_loader, device)
    results["Transformer"] = {"acc": acc, "nll": nll}
    flush_print(f"[F] Transformer: acc={acc:.4f}  NLL={nll:.4f}")

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
    results["Signature-like + linear"] = {"acc": acc, "nll": nll}
    flush_print(f"[G] Signature-like + linear: acc={acc:.4f}  NLL={nll:.4f}")

    # Output all results as a dictionary
    flush_print("\nAll Results:", results)
    return results

if __name__ == "__main__":
    results = main()