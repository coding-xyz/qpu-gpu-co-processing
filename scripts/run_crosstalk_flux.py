# scripts/run_crosstalk_flux.py
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from src.common import pick_device, maybe_write_json
from src.crosstalk.train_calibrator import TrainConfig, train_calibrator


def _default_run_dir(in_npz: str) -> str:
    stem = Path(in_npz).stem
    ts = time.strftime("%Y%m%d_%H%M%S")
    return str(Path("runs") / f"invcal_{stem}_{ts}")


def _save_model_pt(model: torch.nn.Module, path: str, *, meta: Dict[str, Any]):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)


def _print_report(rows: List[Dict[str, Any]], file=sys.stdout):
    """
    Print a compact comparison table to stdout (human-facing).
    (No J_offdiag/diag column.)
    """
    headers = [
        "model",
        "fit_time_sec",
        "rmse_proxy",
        "loss_last",
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
        description="Train calibration models (sweep): control = g(target); compare Linear/MLP/Residual-MLP"
    )
    ap.add_argument("--in_npz", required=True)
    ap.add_argument("--control_key", default="Z")
    ap.add_argument("--target_key", default="domega_meas")

    # Training shared
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--grad_clip", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"])

    ap.add_argument("--no_init_from_data", action="store_true")
    ap.add_argument("--init_ridge", type=float, default=1e-8)

    # MLP/Residual-MLP hyperparams
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.0)

    ap.add_argument("--models", default="linear,mlp,residual_mlp")

    # Output
    ap.add_argument("--out_dir", default=None, help="default: runs/crosstalk_calib_<npz>_<timestamp>/")
    ap.add_argument("--out_json", default=None, help="default: <out_dir>/report.json")
    ap.add_argument("--no_table", action="store_true", help="do not print table to stdout")

    args = ap.parse_args()

    # out_dir / out_json
    out_dir = args.out_dir or _default_run_dir(args.in_npz)
    out_dir = str(Path(out_dir))
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_json = args.out_json or str(Path(out_dir) / "report.json")

    # Load data
    npz = np.load(args.in_npz, allow_pickle=True)
    if args.control_key not in npz or args.target_key not in npz:
        raise KeyError(f"Need keys '{args.control_key}' and '{args.target_key}', found {list(npz.keys())}")

    control = np.asarray(npz[args.control_key], dtype=np.float64)
    target = np.asarray(npz[args.target_key], dtype=np.float64)
    if control.ndim != 2 or target.ndim != 2 or control.shape != target.shape:
        raise ValueError(f"control,target must both be (N,n) same shape, got {control.shape}, {target.shape}")
    N, n = control.shape

    # device & dtype
    device = pick_device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64

    # models
    model_list = [m.strip() for m in args.models.split(",") if m.strip()]
    allowed = {"linear", "mlp", "residual_mlp"}
    for m in model_list:
        if m not in allowed:
            raise ValueError(f"Unknown model '{m}'. Allowed: {sorted(allowed)}")

    rows: List[Dict[str, Any]] = []
    per_model: Dict[str, Any] = {}

    for model_name in model_list:
        if model_name in ("mlp", "residual_mlp"):
            model_kwargs = dict(hidden=int(args.hidden), depth=int(args.depth), dropout=float(args.dropout))
        else:
            model_kwargs = {}

        cfg = TrainConfig(
            model=model_name,  # type: ignore[arg-type]
            epochs=int(args.epochs),
            lr=float(args.lr),
            batch_size=int(args.batch_size),
            seed=int(args.seed),
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
            target_data=target,
            control_data=control,
            device=device,
        )
        fit_time_sec = time.time() - t0

        # save pt
        model_pt = str(Path(out_dir) / f"model_{model_name}.pt")
        meta = {"task": "inverse_calibrator", "model": model_name, "n": int(n), "cfg": asdict(cfg)}
        _save_model_pt(model, model_pt, meta=meta)

        # res expected fields: res.rmse, res.loss, res.extra
        rmse_proxy = float(res.rmse)
        loss_last = float(res.loss[-1]) if res.loss else None

        row = {
            "model": model_name,
            "fit_time_sec": float(fit_time_sec),
            "rmse_proxy": rmse_proxy,
            "loss_last": loss_last,
            "model_pt": model_pt,
            "model_kwargs": model_kwargs,
        }
        rows.append(row)

        per_model[model_name] = {
            "row": row,
            "train_result": {
                "cfg": res.cfg,
                "rmse_proxy": rmse_proxy,
                "loss": res.loss,
                "extra": res.extra,
            },
        }

    rows_sorted = sorted(rows, key=lambda r: r["rmse_proxy"])
    best = rows_sorted[0]["model"] if rows_sorted else None

    report = {
        "task": "Inverse calibration model sweep (control = g(target))",
        "in_npz": str(args.in_npz),
        "control_key": args.control_key,
        "target_key": args.target_key,
        "out_dir": out_dir,
        "out_json": out_json,
        "device": str(device),
        "dtype": str(dtype),
        "N_samples": int(N),
        "n_channels": int(n),
        "train": {
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "batch_size": int(args.batch_size),
            "weight_decay": float(args.weight_decay),
            "grad_clip": float(args.grad_clip) if args.grad_clip is not None else None,
            "seed": int(args.seed),
            "do_init_from_data": not bool(args.no_init_from_data),
            "init_ridge": float(args.init_ridge),
        },
        "models_ran": model_list,
        "best_by_rmse_proxy": best,
        "comparison_rows_sorted_by_rmse_proxy": rows_sorted,
        "per_model": per_model,
    }

    report = maybe_write_json(out_json, report)

    # human-facing table & paths -> stderr
    if not args.no_table:
        print(
            "\n=== Inverse calibration sweep summary (sorted by rmse_proxy) ===",
            file=sys.stderr,
        )
        _print_report(rows_sorted, file=sys.stderr)
        print(f"[artifacts] out_dir  = {out_dir}", file=sys.stderr)
        print(f"[artifacts] report  = {out_json}", file=sys.stderr)

    return report


if __name__ == "__main__":
    main()
