# scripts/train_inverse_calibrator_sweep.py
from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any, Dict, List

import numpy as np
import torch

from src.common import pick_device, maybe_write_json, to_json_str
from src.crosstalk.train_calibrator import TrainConfig, train_calibrator


def _to_float(x):
    try:
        return float(x)
    except Exception:
        return x


def _print_report(rows: List[Dict[str, Any]], file=sys.stderr):
    """
    Print a compact comparison table to stderr (human-facing).
    """
    headers = [
        "model",
        "fit_time_sec",
        "rmse_train",
        "rmse_test",
        "train_loss_last",
        "test_loss_last",
        "J_offdiag/diag",
    ]

    def fmt(v):
        if v is None:
            return ""
        if isinstance(v, float):
            return f"{v:.6g}"
        return str(v)

    table = [headers] + [[fmt(r.get(h, "")) for h in headers] for r in rows]
    widths = [max(len(str(table[i][j])) for i in range(len(table))) for j in range(len(headers))]

    def line(sep="-"):
        return "+" + "+".join(sep * (w + 2) for w in widths) + "+"

    def row(vals):
        return "|" + "|".join(f" {vals[j]:<{widths[j]}} " for j in range(len(vals))) + "|"

    print(line("-"), file=file)
    print(row(table[0]), file=file)
    print(line("="), file=file)
    for r in table[1:]:
        print(row(r), file=file)
    print(line("-"), file=file)


def main():
    ap = argparse.ArgumentParser(
        description="Train inverse calibration models (sweep): control = g(target); compare Linear/MLP/Residual-MLP"
    )
    ap.add_argument("--in_npz", required=True, help="npz file containing control and target data")
    ap.add_argument("--control_key", default="Z", help="npz key for control array (default: Z)")
    ap.add_argument("--target_key", default="domega_meas", help="npz key for target array (default: domega_meas)")

    # Training shared
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--grad_clip", type=float, default=None)

    ap.add_argument("--test_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split_seed", type=int, default=0)

    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"])

    ap.add_argument("--no_init_from_data", action="store_true", help="disable linear init_from_data")
    ap.add_argument("--init_ridge", type=float, default=1e-8)

    # MLP/Residual-MLP hyperparams (Linear ignores these)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.0)

    # Which models to run
    ap.add_argument(
        "--models",
        default="linear,mlp,residual_mlp",
        help="comma-separated list from {linear,mlp,residual_mlp} (default: all)",
    )

    # Output behavior
    ap.add_argument("--out_json", default=None, help="write full report json to file")
    ap.add_argument(
        "--stdout_json",
        action="store_true",
        help="print JSON report to stdout (machine-facing). Always prints the table to stderr.",
    )
    ap.add_argument(
        "--no_table",
        action="store_true",
        help="do not print the human-facing table to stderr",
    )

    args = ap.parse_args()

    # -------------------------
    # Load data
    # -------------------------
    npz = np.load(args.in_npz, allow_pickle=True)
    if args.control_key not in npz or args.target_key not in npz:
        raise KeyError(
            f"npz must contain keys '{args.control_key}' and '{args.target_key}', "
            f"found: {list(npz.keys())}"
        )

    control = np.asarray(npz[args.control_key], dtype=np.float64)
    target = np.asarray(npz[args.target_key], dtype=np.float64)

    if control.ndim != 2 or target.ndim != 2 or control.shape != target.shape:
        raise ValueError(f"control and target must both be (N,n) same shape, got {control.shape}, {target.shape}")

    N, n = control.shape

    # -------------------------
    # Device & dtype
    # -------------------------
    device = pick_device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64

    # -------------------------
    # Sweep models
    # -------------------------
    model_list = [m.strip() for m in args.models.split(",") if m.strip()]
    allowed = {"linear", "mlp", "residual_mlp"}
    for m in model_list:
        if m not in allowed:
            raise ValueError(f"Unknown model '{m}'. Allowed: {sorted(allowed)}")

    rows: List[Dict[str, Any]] = []
    per_model: Dict[str, Any] = {}

    for model_name in model_list:
        # model-specific kwargs
        if model_name in ("mlp", "residual_mlp"):
            model_kwargs = dict(hidden=int(args.hidden), depth=int(args.depth), dropout=float(args.dropout))
        else:
            model_kwargs = {}

        cfg = TrainConfig(
            model=model_name,  # type: ignore[arg-type]
            epochs=int(args.epochs),
            lr=float(args.lr),
            test_frac=float(args.test_frac),
            batch_size=int(args.batch_size),
            seed=int(args.seed),
            split_seed=int(args.split_seed),
            weight_decay=float(args.weight_decay),
            grad_clip=args.grad_clip,
            dtype=dtype,
            model_kwargs=model_kwargs,
            do_init_from_data=not bool(args.no_init_from_data),
            init_ridge=float(args.init_ridge),
        )

        t0 = time.time()
        model, res = train_calibrator(
            cfg=cfg,
            # IMPORTANT: if your training.py still uses y_target_data/x_control_data,
            # change these keyword args accordingly.
            target_data=target,
            control_data=control,
            device=device,
        )
        fit_time_sec = time.time() - t0

        J_ratio = res.extra.get("J_offdiag_over_diag", None)
        row = {
            "model": model_name,
            "fit_time_sec": float(fit_time_sec),
            "rmse_train": float(res.rmse_train),
            "rmse_test": float(res.rmse_test),
            "train_loss_last": float(res.train_loss[-1]) if res.train_loss else None,
            "test_loss_last": float(res.test_loss[-1]) if res.test_loss else None,
            "J_offdiag/diag": _to_float(J_ratio) if J_ratio is not None else None,
            "J_diag_mean_abs": _to_float(res.extra.get("J_diag_mean_abs", None)),
            "J_offdiag_mean_abs": _to_float(res.extra.get("J_offdiag_mean_abs", None)),
            "init_from_data": res.extra.get("init_from_data", None),
            "model_kwargs": model_kwargs,
        }
        rows.append(row)

        per_model[model_name] = {
            "row": row,
            "train_result": {
                "cfg": res.cfg,
                "rmse_train": float(res.rmse_train),
                "rmse_test": float(res.rmse_test),
                "train_loss": res.train_loss,
                "test_loss": res.test_loss,
                "extra": res.extra,
            },
        }

    rows_sorted = sorted(rows, key=lambda r: (r["rmse_test"] if r["rmse_test"] is not None else 1e99))
    best = rows_sorted[0]["model"] if rows_sorted else None

    report = {
        "task": "Inverse calibration model sweep (control = g(target))",
        "in_npz": str(args.in_npz),
        "control_key": args.control_key,
        "target_key": args.target_key,
        "device": str(device),
        "dtype": str(dtype),
        "N_samples": int(N),
        "n_channels": int(n),
        "train": {
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "batch_size": int(args.batch_size),
            "test_frac": float(args.test_frac),
            "weight_decay": float(args.weight_decay),
            "grad_clip": _to_float(args.grad_clip) if args.grad_clip is not None else None,
            "seed": int(args.seed),
            "split_seed": int(args.split_seed),
            "do_init_from_data": not bool(args.no_init_from_data),
            "init_ridge": float(args.init_ridge),
        },
        "models_ran": model_list,
        "best_by_rmse_test": best,
        "comparison_rows_sorted_by_rmse_test": rows_sorted,
        "per_model": per_model,
    }

    # Write JSON file if requested
    report = maybe_write_json(args.out_json, report)

    # Human-facing table -> stderr (unless disabled)
    if not args.no_table:
        print("\n=== Inverse calibration sweep summary (sorted by rmse_test) ===", file=sys.stderr)
        _print_report(rows_sorted, file=sys.stderr)

    # Machine-facing JSON -> stdout only when requested
    if args.stdout_json:
        print(to_json_str(report))

    return report


if __name__ == "__main__":
    main()
