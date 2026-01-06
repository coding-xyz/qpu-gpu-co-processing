#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np


# -----------------------------
# Noise utilities
# -----------------------------
def colored_noise(beta: float, N: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generate 1/f^beta colored noise using frequency-domain shaping.
    beta=0 -> white, beta=1 -> pink, beta=2 -> brown
    Output is zero-mean, unit-std (approximately).
    """
    # rfftfreq includes 0; avoid division by zero at DC.
    freqs = np.fft.rfftfreq(N, d=1.0)
    scale = np.zeros_like(freqs, dtype=np.float64)
    nonzero = freqs > 0
    scale[nonzero] = 1.0 / (freqs[nonzero] ** (beta / 2.0))

    # Random complex spectrum with gaussian real/imag
    spec = (rng.normal(size=len(freqs)) + 1j * rng.normal(size=len(freqs))) * scale
    x = np.fft.irfft(spec, n=N)

    x = x - np.mean(x)
    std = np.std(x)
    if std < 1e-12:
        return np.zeros(N, dtype=np.float32)
    x = x / std
    return x.astype(np.float32)


def one_pole_lowpass_complex(x: np.ndarray, fs: float, fc: float) -> np.ndarray:
    """
    1st-order lowpass (discrete RC) for complex signal.
    fc: -3 dB cutoff (Hz). If fc<=0, returns copy.
    """
    if fc <= 0:
        return x.astype(np.complex64, copy=True)
    a = np.exp(-2.0 * np.pi * fc / fs)  # smoothing factor
    y = np.empty_like(x, dtype=np.complex64)
    y0 = 0j
    for n in range(x.shape[0]):
        y0 = a * y0 + (1.0 - a) * x[n]
        y[n] = y0
    return y


# -----------------------------
# Signal model
# -----------------------------
def two_step_square_envelope(t: np.ndarray, t1: float, t_total: float, a1: float, a2: float) -> np.ndarray:
    """
    Two contiguous square steps:
      [0, t1): amplitude a1 (higher, shorter)
      [t1, t_total): amplitude a2 (lower, longer)
      else: 0
    """
    env = np.zeros_like(t, dtype=np.float32)
    env[(t >= 0) & (t < t1)] = a1
    env[(t >= t1) & (t < t_total)] = a2
    return env


def generate_one_shot(
    *,
    fs: float,
    f_rf: float,
    T: float,
    t1: float,
    a1: float,
    a2: float,
    # ring-up / bandwidth limit (envelope dynamics)
    fc_lp: float,
    # state-dependent complex gain (amp+phase)
    g0_amp: float,
    g0_phase_deg: float,
    g1_amp: float,
    g1_phase_deg: float,
    # noise knobs (stronger, more realistic)
    snr_db: float,
    phase_slow_std_deg: float,
    phase_fast_std_deg: float,
    phase_fast_beta: float,
    amp_fast_std: float,
    amp_fast_beta: float,
    baseline_std: float,
    baseline_beta: float,
    dc_offset: float,
    # edge shaping / overshoot-ish effect
    env_edge_fc: float,
    # RNG
    rng: np.random.Generator,
    label: int | None = None,
):
    """
    Output:
      t (N,), env (N,), rf (N,), label (int), meta dict
    """
    N = int(np.round(T * fs))
    t = (np.arange(N, dtype=np.float64) / fs)

    # envelope: two-step square
    env_ideal = two_step_square_envelope(t, t1=t1, t_total=T, a1=a1, a2=a2)

    # edge shaping (make square edges less ideal; optional)
    # We do it by lowpassing the envelope as a *real* signal (cheap proxy for bandwidth limits / ringing suppression)
    if env_edge_fc > 0:
        # Reuse one-pole on complex by casting to complex then taking real back
        env_shaped = np.real(one_pole_lowpass_complex(env_ideal.astype(np.complex64), fs=fs, fc=env_edge_fc)).astype(np.float32)
    else:
        env_shaped = env_ideal

    # pick label
    if label is None:
        label = int(rng.integers(0, 2))

    # state-dependent complex gain
    g0 = g0_amp * np.exp(1j * np.deg2rad(g0_phase_deg))
    g1 = g1_amp * np.exp(1j * np.deg2rad(g1_phase_deg))
    g = g1 if label == 1 else g0

    # baseband complex envelope before dynamics
    bb = (env_shaped.astype(np.complex64) * g)

    # model ring-up/down with one-pole lowpass on complex envelope (captures transient)
    bb_dyn = one_pole_lowpass_complex(bb, fs=fs, fc=fc_lp)

    # ----- add phase noise (slow + fast colored) -----
    phi_slow = np.deg2rad(rng.normal(0.0, phase_slow_std_deg))  # constant over shot
    phi_fast = colored_noise(phase_fast_beta, N, rng) * np.deg2rad(phase_fast_std_deg)  # time-varying
    phase_noise = phi_slow + phi_fast.astype(np.float64)

    # ----- add amplitude noise (colored, multiplicative) -----
    amp_fast = colored_noise(amp_fast_beta, N, rng) * amp_fast_std
    amp_noise = (1.0 + amp_fast).astype(np.float64)

    # Upconvert to RF and undersample naturally by using fs
    carrier = np.exp(1j * (2.0 * np.pi * f_rf * t + phase_noise))
    rf_clean = np.real(bb_dyn.astype(np.complex128) * amp_noise * carrier).astype(np.float32)

    # Add DC offset
    rf_clean = rf_clean + np.float32(dc_offset)

    # Baseline wander (brown/pink-ish), like ADC front-end drift
    if baseline_std > 0:
        baseline = colored_noise(baseline_beta, N, rng) * baseline_std
        rf_clean = rf_clean + baseline.astype(np.float32)

    # Add AWGN based on target SNR relative to rf_clean power
    sig_pow = float(np.mean(rf_clean.astype(np.float64) ** 2) + 1e-20)
    snr_lin = 10.0 ** (snr_db / 10.0)
    noise_pow = sig_pow / snr_lin
    noise = rng.normal(0.0, np.sqrt(noise_pow), size=N).astype(np.float32)

    rf = rf_clean + noise

    # Alias frequency info
    k = int(np.round(f_rf / fs))
    f_alias = abs(f_rf - k * fs)

    meta = dict(
        fs=float(fs),
        f_rf=float(f_rf),
        f_alias=float(f_alias),
        T=float(T),
        N=int(N),
        t1=float(t1),
        a1=float(a1),
        a2=float(a2),
        fc_lp=float(fc_lp),
        env_edge_fc=float(env_edge_fc),
        snr_db=float(snr_db),
        phase_slow_std_deg=float(phase_slow_std_deg),
        phase_fast_std_deg=float(phase_fast_std_deg),
        phase_fast_beta=float(phase_fast_beta),
        amp_fast_std=float(amp_fast_std),
        amp_fast_beta=float(amp_fast_beta),
        baseline_std=float(baseline_std),
        baseline_beta=float(baseline_beta),
        dc_offset=float(dc_offset),
        label=int(label),
        g0_amp=float(g0_amp),
        g0_phase_deg=float(g0_phase_deg),
        g1_amp=float(g1_amp),
        g1_phase_deg=float(g1_phase_deg),
    )
    return t, env_ideal, env_shaped, rf, label, meta


def main():
    ap = argparse.ArgumentParser(description="Simulate undersampled RF readout (6GHz on 5GSa/s) with strong realistic noise, save to NPZ.")
    ap.add_argument("--out", type=str, default="undersampled_readout_noisy_2us.npz")
    ap.add_argument("--shots", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1234)

    # core timing / sampling
    ap.add_argument("--fs", type=float, default=5e9, help="ADC sample rate (Hz), e.g. 5e9")
    ap.add_argument("--f_rf", type=float, default=6e9, help="RF carrier (Hz), e.g. 6e9")
    ap.add_argument("--T", type=float, default=2e-6, help="total pulse length (s), e.g. 2e-6")

    # two-step envelope
    ap.add_argument("--t1", type=float, default=0.35e-6, help="duration of step-1 (s)")
    ap.add_argument("--a1", type=float, default=1.0, help="amplitude of step-1 (higher, shorter)")
    ap.add_argument("--a2", type=float, default=0.6, help="amplitude of step-2 (lower, longer)")

    # dynamics / bandwidth limits
    ap.add_argument("--fc_lp", type=float, default=30e6, help="one-pole cutoff for complex envelope dynamics (Hz)")
    ap.add_argument("--env_edge_fc", type=float, default=150e6, help="one-pole cutoff to soften square edges (Hz). 0 disables")

    # state-dependent response (mostly phase difference is typical)
    ap.add_argument("--g0_amp", type=float, default=1.00)
    ap.add_argument("--g0_phase_deg", type=float, default=0.0)
    ap.add_argument("--g1_amp", type=float, default=1.00)
    ap.add_argument("--g1_phase_deg", type=float, default=12.0)

    # noise: make it "noisy" by default
    ap.add_argument("--snr_db", type=float, default=8.0, help="AWGN SNR in dB relative to rf power (lower => noisier)")
    ap.add_argument("--phase_slow_std_deg", type=float, default=3.0, help="shot-constant phase drift std (deg)")
    ap.add_argument("--phase_fast_std_deg", type=float, default=1.2, help="time-varying phase noise std (deg)")
    ap.add_argument("--phase_fast_beta", type=float, default=1.0, help="phase fast colored beta: 0 white, 1 pink, 2 brown")

    ap.add_argument("--amp_fast_std", type=float, default=0.08, help="multiplicative amplitude noise std (e.g. 0.08 = 8%)")
    ap.add_argument("--amp_fast_beta", type=float, default=1.0, help="amplitude noise colored beta")

    ap.add_argument("--baseline_std", type=float, default=0.12, help="baseline wander std (in RF units)")
    ap.add_argument("--baseline_beta", type=float, default=2.0, help="baseline colored beta (2 ~ brown)")
    ap.add_argument("--dc_offset", type=float, default=0.00)

    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    N = int(np.round(args.T * args.fs))
    rf_all = np.zeros((args.shots, N), dtype=np.float32)
    y_all = np.zeros((args.shots,), dtype=np.int64)

    # store envelopes once (they are the same for all shots)
    t = (np.arange(N, dtype=np.float64) / args.fs)
    env_ideal = two_step_square_envelope(t, t1=args.t1, t_total=args.T, a1=args.a1, a2=args.a2)
    if args.env_edge_fc > 0:
        env_shaped = np.real(one_pole_lowpass_complex(env_ideal.astype(np.complex64), fs=args.fs, fc=args.env_edge_fc)).astype(np.float32)
    else:
        env_shaped = env_ideal.copy()

    # generate shots
    for i in range(args.shots):
        # per-shot independent randomness
        label = int(rng.integers(0, 2))
        _, _, _, rf, y, _meta = generate_one_shot(
            fs=args.fs,
            f_rf=args.f_rf,
            T=args.T,
            t1=args.t1,
            a1=args.a1,
            a2=args.a2,
            fc_lp=args.fc_lp,
            env_edge_fc=args.env_edge_fc,
            g0_amp=args.g0_amp,
            g0_phase_deg=args.g0_phase_deg,
            g1_amp=args.g1_amp,
            g1_phase_deg=args.g1_phase_deg,
            snr_db=args.snr_db,
            phase_slow_std_deg=args.phase_slow_std_deg,
            phase_fast_std_deg=args.phase_fast_std_deg,
            phase_fast_beta=args.phase_fast_beta,
            amp_fast_std=args.amp_fast_std,
            amp_fast_beta=args.amp_fast_beta,
            baseline_std=args.baseline_std,
            baseline_beta=args.baseline_beta,
            dc_offset=args.dc_offset,
            rng=rng,
            label=label,
        )
        rf_all[i] = rf
        y_all[i] = y

    k = int(np.round(args.f_rf / args.fs))
    f_alias = abs(args.f_rf - k * args.fs)

    np.savez_compressed(
        args.out,
        rf=rf_all,            # (shots, N) real RF waveform sampled by ADC
        y=y_all,              # (shots,) labels 0/1
        t=t,                  # (N,)
        env_ideal=env_ideal,  # (N,) ideal two-step square
        env_shaped=env_shaped,# (N,) softened edges
        fs=np.float64(args.fs),
        f_rf=np.float64(args.f_rf),
        f_alias=np.float64(f_alias),
        T=np.float64(args.T),
        t1=np.float64(args.t1),
        a1=np.float64(args.a1),
        a2=np.float64(args.a2),
        fc_lp=np.float64(args.fc_lp),
        env_edge_fc=np.float64(args.env_edge_fc),
        snr_db=np.float64(args.snr_db),
        phase_slow_std_deg=np.float64(args.phase_slow_std_deg),
        phase_fast_std_deg=np.float64(args.phase_fast_std_deg),
        phase_fast_beta=np.float64(args.phase_fast_beta),
        amp_fast_std=np.float64(args.amp_fast_std),
        amp_fast_beta=np.float64(args.amp_fast_beta),
        baseline_std=np.float64(args.baseline_std),
        baseline_beta=np.float64(args.baseline_beta),
        dc_offset=np.float64(args.dc_offset),
        g0_amp=np.float64(args.g0_amp),
        g0_phase_deg=np.float64(args.g0_phase_deg),
        g1_amp=np.float64(args.g1_amp),
        g1_phase_deg=np.float64(args.g1_phase_deg),
    )

    binc = np.bincount(y_all, minlength=2)
    print(f"[OK] wrote: {args.out}")
    print(f"     rf shape: {rf_all.shape}, labels: {binc.tolist()}")
    print(f"     undersampling alias frequency: {f_alias/1e9:.3f} GHz")
    print("     tip: lower --snr_db, increase --phase_fast_std_deg/--amp_fast_std/--baseline_std to make it noisier.")


if __name__ == "__main__":
    main()
