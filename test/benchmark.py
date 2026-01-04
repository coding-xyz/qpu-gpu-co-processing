import argparse
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
import time

from src.tomography.io import load_npz, save_json
from src.tomography.tomo_1q import fit_1q_mle_spam
from src.tomography.tomo_2q import fit_2q_mle_spam

def _run(ds, device, steps=None, lr=None, seed=0):
    n = int(ds["n_qubits"])
    counts = ds["counts"]
    A = ds["A_meas"]
    if n == 1:
        s = steps if steps is not None else 2000
        l = lr if lr is not None else 0.05
        return fit_1q_mle_spam(counts, A, steps=s, lr=l, seed=seed, device=device)
    s = steps if steps is not None else 3000
    l = lr if lr is not None else 0.03
    return fit_2q_mle_spam(counts, A, steps=s, lr=l, seed=seed, device=device)

def median(x):
    xs = sorted(x)
    return xs[len(xs)//2]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_npz", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeat", type=int, default=3)
    args = ap.parse_args()

    ds = load_npz(args.in_npz)

    # warmup CPU
    for _ in range(args.warmup):
        _run(ds, "cpu", steps=args.steps, lr=args.lr, seed=args.seed)

    cpu_times = []
    for _ in range(args.repeat):
        t0 = time.perf_counter()
        _run(ds, "cpu", steps=args.steps, lr=args.lr, seed=args.seed)
        cpu_times.append(time.perf_counter() - t0)
    cpu_med = median(cpu_times)

    gpu_med = None
    try:
        import torch
        if torch.cuda.is_available():
            for _ in range(args.warmup):
                _run(ds, "cuda", steps=args.steps, lr=args.lr, seed=args.seed)
            gpu_times = []
            for _ in range(args.repeat):
                t0 = time.perf_counter()
                _run(ds, "cuda", steps=args.steps, lr=args.lr, seed=args.seed)
                gpu_times.append(time.perf_counter() - t0)
            gpu_med = median(gpu_times)
    except Exception:
        gpu_med = None

    report = {
        "in_npz": args.in_npz,
        "n_qubits": int(ds["n_qubits"]),
        "cpu_time_sec_median": cpu_med,
        "gpu_time_sec_median": gpu_med,
        "speedup_median": (cpu_med / gpu_med) if gpu_med else None,
        "repeat": args.repeat,
        "warmup": args.warmup,
        "steps_override": args.steps,
        "lr_override": args.lr,
    }
    save_json(args.out_json, report)

    print("Benchmark summary")
    print(f"  CPU median: {cpu_med:.4f} s")
    if gpu_med:
        print(f"  GPU median: {gpu_med:.4f} s")
        print(f"  Speedup  : {report['speedup_median']:.2f}x")
    else:
        print("  GPU not available.")
    print(f"Wrote {args.out_json}")

if __name__ == "__main__":
    main()
