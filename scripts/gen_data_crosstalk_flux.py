# scripts/gen_data_crosstalk_flux.py
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from typing import Any, Dict

import numpy as np

from src.crosstalk.fake_flux_crosstalk import CrosstalkConfig, ZToZCrosstalkPlant, sample_random_Z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_npz", required=True)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--k", type=int, default=4000)

    # Z patterns
    ap.add_argument("--z_scale", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=0)

    # nonlinearity knobs
    ap.add_argument("--quad_strength", type=float, default=0.02)
    ap.add_argument("--sat_strength", type=float, default=0.05)
    ap.add_argument("--sat_scale", type=float, default=0.8)
    ap.add_argument("--per_strength", type=float, default=0.02)
    ap.add_argument("--per_period", type=float, default=1.0)

    # noise
    ap.add_argument("--noise_white", type=float, default=0.01)
    ap.add_argument("--noise_pattern", type=float, default=0.005)
    ap.add_argument("--drift_strength", type=float, default=0.0)

    # whether to save a deterministic (noise-free) domega too
    ap.add_argument("--also_save_clean", action="store_true")

    args = ap.parse_args()

    # Plant config
    cfg = CrosstalkConfig(
        n=int(args.n),
        seed=int(args.seed),
        quad_strength=float(args.quad_strength),
        sat_strength=float(args.sat_strength),
        sat_scale=float(args.sat_scale),
        per_strength=float(args.per_strength),
        per_period=float(args.per_period),
        noise_white=float(args.noise_white),
        noise_pattern=float(args.noise_pattern),
        drift_strength=float(args.drift_strength),
    )

    plant = ZToZCrosstalkPlant(cfg)

    # Sample random Z patterns (what you dial)
    Z = sample_random_Z(k=int(args.k), n=int(args.n), z_scale=float(args.z_scale), seed=int(args.seed))
    domega_meas = plant.forward(Z, add_noise=True)

    payload: Dict[str, Any] = {
        "Z": Z.astype(np.float64),
        "domega_meas": domega_meas.astype(np.float64),
        "C_true": plant.C_true.astype(np.float64),
        "D_true": plant.D_true.astype(np.float64),
        "plant_cfg_json": np.array([json.dumps(asdict(cfg))], dtype=object),
        "meta": np.array(
            [args.n, args.k, args.z_scale, args.quad_strength, args.sat_strength, args.per_strength,
             args.noise_white, args.noise_pattern, args.drift_strength],
            dtype=np.float64,
        ),
    }

    if args.also_save_clean:
        payload["domega_clean"] = plant.forward(Z, add_noise=False).astype(np.float64)

    np.savez_compressed(args.out_npz, **payload)

    print(f"Wrote {args.out_npz}")
    print(f"Shapes: Z {Z.shape}, domega_meas {domega_meas.shape}, C_true {plant.C_true.shape}, D_true {plant.D_true.shape}")
    print(f"Plant cfg: {asdict(cfg)}")


if __name__ == "__main__":
    main()
