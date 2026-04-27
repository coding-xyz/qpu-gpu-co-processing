import argparse, json
import numpy as np
import torch
import torch.nn.functional as F

import sys

from src.readout import LDA, MatchedFilter, TinyCNN, AmortizedBayesNet, TinyTransformer
from src.readout import bayes_em_fit, bayes_init
from src.readout import fit_hmm_templates_fixed, hmm_classify
from src.readout import signature2_features, SignatureLogReg

from src.readout import train_simple, eval_bayes, eval_nn, eval_signature
from src.readout import make_loader

# Define the flush function to ensure the output is shown immediately
def flush_print(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_npz", type=str, default="test_dataset.npz")
    ap.add_argument("--em_iter", type=int, default=25)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--hmm_iter", type=int, default=25)
    args = ap.parse_args()

    # Load data
    d = np.load(args.in_npz, allow_pickle=True)
    Xtr, ytr, Xte, yte = d["X_train"], d["y_train"], d["X_test"], d["y_test"]
    mu0, mu1, sigma0 = bayes_init(Xtr, ytr)

    # Prepare loaders for ML
    train_loader = make_loader(Xtr, ytr, batch_size=args.batch, shuffle=True)
    test_loader = make_loader(Xte, yte, batch_size=2048, shuffle=False)
    T = Xtr.shape[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    flush_print("device:", device)

    # Dictionary to store results
    results = {}

    # [A] Linear Classifier
    lda, his = train_simple(LDA(T), train_loader, device, epochs=args.epochs)
    acc, nll = eval_nn(lda, test_loader, device)
    results["Linear Classifier"] = {"acc": acc, "nll": nll, "time": his["training time"]}
    flush_print(f"[A] Linear Classifier: acc={acc:.4f}  NLL={nll:.4f}, TIME={his['training time']:.4f}")

    # [B] Matched Filter
    mf, his = train_simple(MatchedFilter(T), train_loader, device, epochs=args.epochs)
    acc, nll = eval_nn(mf, test_loader, device)
    results["Matched Filter + LDA"] = {"acc": acc, "nll": nll, "time": his["training time"]}
    flush_print(f"[B] Matched Filter + LDA: acc={acc:.4f}  NLL={nll:.4f}, TIME={his['training time']:.4f}")

    # [C] Bayes-only
    acc, nll = eval_bayes(Xte, yte, mu0, mu1, sigma0, device)
    results["Bayes-only"] = {"acc": acc, "nll": nll}
    flush_print(f"[C] Bayes-only: acc={acc:.4f}  NLL={nll:.4f}")

    # # [D] Bayes-EM (weak anchor)
    # mu0_em2, mu1_em2, sigma_em2 = bayes_em_fit(
    #     Xtr, n_iter=args.em_iter, anchor_mu0=mu0, anchor_strength=20.0, device=device
    # )
    # acc, nll = eval_bayes(Xte, yte, mu0_em2.numpy(), mu1_em2.numpy(), sigma_em2, device)
    # results["Bayes-EM (weak anchor)"] = {"acc": acc, "nll": nll, "sigma": sigma_em2}
    # flush_print(f"[D] Bayes-EM (weak anchor): acc={acc:.4f}  NLL={nll:.4f}  sigma={sigma_em2:.3f}")

    # # [E] CNN
    # cnn, his = train_simple(TinyCNN(T), train_loader, device, epochs=args.epochs)
    # acc, nll = eval_nn(cnn, test_loader, device)
    # results["CNN"] = {"acc": acc, "nll": nll, "time": his["training time"]}
    # flush_print(f"[E] CNN: acc={acc:.4f}  NLL={nll:.4f}, TIME={his['training time']:.4f}")
    
    # # [F] Transformer
    # trm, his = train_simple(TinyTransformer(T), train_loader, device, epochs=args.epochs)
    # acc, nll = eval_nn(trm, test_loader, device)
    # results["Transformer"] = {"acc": acc, "nll": nll, "time": his["training time"]}
    # flush_print(f"[F] Transformer: acc={acc:.4f}  NLL={nll:.4f}, TIME={his['training time']:.4f}")

    # # [G] Amortized Bayes + NN
    # ab, his = train_simple(AmortizedBayesNet(T, predict_sigma=True), train_loader, device, epochs=args.epochs)
    # acc, nll = eval_nn(ab, test_loader, device)
    # results["Amortized Bayes (NN)"] = {"acc": acc, "nll": nll, "time": his["training time"]}
    # flush_print(f"[G] Amortized Bayes (NN): acc={acc:.4f}  NLL={nll:.4f}, TIME={his['training time']:.4f}")

    # # [H] HMM
    # sigma, p01, p10 = fit_hmm_templates_fixed(Xtr, mu0, mu1, n_iter=args.hmm_iter, init_sigma=sigma0, device=device)
    # post = hmm_classify(Xte, mu0, mu1, sigma=sigma, p01=p01, p10=p10, device=device)
    # y = torch.as_tensor(yte, dtype=torch.long, device=device)
    # pred = post.argmax(dim=-1)
    # acc = (pred==y).float().mean().item()
    # nll = (-torch.log(post[torch.arange(y.shape[0]), y]+1e-9)).mean().item()
    # results["HMM"] = {"acc": acc, "nll": nll, "sigma": sigma, "p01": p01, "p10": p10}
    # flush_print(f"[H] HMM: acc={acc:.4f}  NLL={nll:.4f}  sigma={sigma:.3f} p01={p01:.2e} p10={p10:.2e}")

    # # [I] Signature-like + linear
    # Xtr_t = torch.as_tensor(Xtr, dtype=torch.float32, device=device)
    # ytr_t = torch.as_tensor(ytr, dtype=torch.long, device=device)
    # feats = signature2_features(Xtr_t)
    # clf = SignatureLogReg(feats.shape[-1]).to(device)
    # opt = torch.optim.AdamW(clf.parameters(), lr=3e-3)
    # bs = 8192
    # for _ in range(args.epochs):
    #     perm = torch.randperm(feats.shape[0], device=device)
    #     for i in range(0, feats.shape[0], bs):
    #         idx = perm[i:i+bs]
    #         logits = clf(feats[idx])
    #         loss = F.cross_entropy(logits, ytr_t[idx])
    #         opt.zero_grad(set_to_none=True)
    #         loss.backward()
    #         opt.step()
    # acc, nll = eval_signature(clf, Xte, yte, device)
    # results["Signature-like + linear"] = {"acc": acc, "nll": nll}
    # flush_print(f"[I] Signature-like + linear: acc={acc:.4f}  NLL={nll:.4f}")

    # Output all results as a dictionary
    flush_print("\nAll Results:", results)
    return results

if __name__ == "__main__":
    results = main()
