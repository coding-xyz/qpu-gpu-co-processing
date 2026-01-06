# scripts/mw_fit_fir.py
from __future__ import annotations
import argparse
import numpy as np

from src.common import maybe_write_json
from src.common import pick_device
from src.crosstalk import FluxFitConfig, run_flux_crosstalk_fit
# from src.crosstalk import write_mw_freq_fit_artifacts, write_flux_fit_artifacts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_npz", required=True)
    ap.add_argument("--out_json", default=None)
    ap.add_argument("--out_dir", default="flux_out")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=2e-2)
    ap.add_argument("--fit_quadratic", action="store_true")
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    npz = np.load(args.in_npz, allow_pickle=True)
    Phi = npz["Phi"].astype(np.float32)           # (k,n)
    omega = npz["omega_meas"].astype(np.float32)  # (k,n)
    C_true = npz["C_true"].astype(np.float32) if "C_true" in npz else None

    device = pick_device(args.device)
    cfg = FluxFitConfig(
        epochs=args.epochs,
        lr=args.lr,
        fit_quadratic=bool(args.fit_quadratic),
        seed=args.seed,
    )

    fit_res = run_flux_crosstalk_fit(
        Phi=Phi,
        omega=omega,
        device=device,
        cfg=cfg,
        C_true=C_true,
        benchmark_numpy=bool(args.benchmark),
    )

    # plots = write_flux_fit_artifacts(
    #     out_dir=args.out_dir,
    #     Phi=Phi,
    #     omega=omega,
    #     C_hat=fit_res.C_hat,
    #     C_true=C_true,
    # )

    result = {
        "task": "Flux crosstalk fit",
        "in_npz": str(args.in_npz),
        "device": str(device),
        "k_patterns": int(Phi.shape[0]),
        "n_channels": int(Phi.shape[1]),
        "fit_quadratic": bool(args.fit_quadratic),
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "fit_time_sec": float(fit_res.fit_time_sec),
        "loss_last": float(fit_res.loss_last),
        "rmse": float(fit_res.rmse),
        "rel_C_error_if_true_available": fit_res.rel_C_error_if_true_available,
        "benchmark": {
            "numpy_lstsq_time_sec": fit_res.benchmark_numpy_lstsq_time_sec,
        },
        "C_hat_summary": fit_res.summary,
        # "plots": plots,
    }

    result = maybe_write_json(args.out_json, result)
    if args.out_json is None:
        print(result)
    return result

if __name__ == "__main__":
    main()
