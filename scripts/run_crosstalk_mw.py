# scripts/mw_fit_fir.py
from __future__ import annotations
import argparse
import numpy as np

from src.common import maybe_write_json, load_npz
from src.common import pick_device
from src.crosstalk import MWFitConfig, run_mw_crosstalk_fit
# from src.crosstalk import write_mw_freq_fit_artifacts, write_flux_fit_artifacts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_npz", required=True)
    ap.add_argument("--out_json", default=None)
    ap.add_argument("--out_dir", default="flux_out")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--eps", type=float, default=1e-3, help="Tikhonov reg for inversion")
    ap.add_argument("--f0_mode", default="third", choices=["third", "mid"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    npz = load_npz(args.in_npz)
    x = npz["x"].astype(np.float32)  # (n,T)
    y = npz["y"].astype(np.float32)  # (n,T)

    device = pick_device(args.device)
    cfg = MWFitConfig(eps=float(args.eps), f0_mode=args.f0_mode)

    fit_res = run_mw_crosstalk_fit(x=x, y=y, device=device, cfg=cfg)

    # plots = write_mw_freq_fit_artifacts(
    #     out_dir=args.out_dir,
    #     x=x,
    #     x_pre=fit_res.x_pre,
    #     Hf=fit_res.Hf,
    #     f0_bin=fit_res.f0_bin,
    # )

    result = {
        "task": "mw_crosstalk_fit",
        "in_npz": str(args.in_npz),
        "device": str(device),
        "n_channels": int(x.shape[0]),
        "T": int(x.shape[1]),
        "F_bins_rfft": int(fit_res.Hf.shape[0]),
        "eps": float(args.eps),
        "fit_time_sec": float(fit_res.fit_time_sec),
        "invert_time_sec": float(fit_res.invert_time_sec),
        "x_reconstruction_rmse": float(fit_res.x_reconstruction_rmse),
        "f0_bin": int(fit_res.f0_bin),
        "summary": fit_res.summary,
        # "plots": plots,
        "notes": [
            "This demo uses a single rich excitation record; real systems should stack multiple calibration waveforms and solve LS per frequency.",
            "G(f)=inv(H(f)) gives a predistorter; in hardware you apply x_pre such that y≈H x_pre ≈ y_desired.",
            "Rank-1 estimator is a convenience; for production use, build LS with multiple records to make X full-rank per frequency."
        ],
    }

    result = maybe_write_json(args.out_json, result)
    if args.out_json is None:
        print(result)
    return result


if __name__ == "__main__":
    main()
