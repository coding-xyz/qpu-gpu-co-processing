# AI + Bayes state readout from I(t), Q(t) (GPU-ready)

This mini-package gives you **seven routes** for superconducting-qubit-style dispersive readout,
starting from time traces `I(t), Q(t)`:

1) **Bayes-only** (calibrated templates, Gaussian noise)
2) **Bayes-EM** (estimate templates/noise from mixed data; optional weak anchor)
3) **NN-only (CNN)** (tiny 1D CNN -> logits)
4) **Bayes + NN (amortized Bayes)** (network predicts posterior fast)
5) **HMM (2-state Gaussian)** (captures possible state flips during readout, e.g. T1)
6) **Transformer** (small Transformer encoder for sequence classification)
7) **Path-signature-like features + linear classifier** (ΔI,ΔQ, area, moments, etc.)

All training/inference uses **PyTorch** and will use **GPU** automatically if available.

## Quick start

```bash
python -m pip install torch numpy
python generate_test_data.py --out test_dataset.npz --n_train 50000 --n_test 10000
python run_all_routes.py --data test_dataset.npz
python run_more_routes.py --data test_dataset.npz
