#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate ML-ready IQ readout dataset using a two-mode Purcell + readout model.

Outputs:
  X_train: (n_train, n_points, 2) float32   [I, Q]
  y_train: (n_train,) int64                 0=|g>, 1=|e>
  X_test : (n_test,  n_points, 2) float32
  y_test : (n_test,)  int64
  params_json: JSON string with generation parameters
"""

import argparse
import json
import numpy as np

# -------------------------
# pulse
# -------------------------
def tanh_square_pulse(t, T_on, T_off, rise, amp):
    on  = 0.5*(1 + np.tanh((t - T_on)/rise))
    off = 0.5*(1 + np.tanh((T_off - t)/rise))
    return amp * on * off

# -------------------------
# core simulator (single shot)
# -------------------------
def simulate_single_shot(
    rng,
    *,
    n_points,
    T,
    # two-mode params
    delta_a, delta_b, J,
    kappa_a, gamma_a,
    kappa_b, gamma_b,
    chi,
    # drive
    pulse_on, pulse_off, rise,
    drive_amp, drive_phase,
    measure,
    # IQ / measurement chain
    demod_phase,
    omega_if, iq_imbalance,
    meas_tau,
    # noise
    amp_noise_std,
    phase_noise_std,
    slow_detune_std,
    # label
    y,   # 0 or 1
):
    dt = T / n_points
    t = np.arange(n_points) * dt

    # input
    env = tanh_square_pulse(t, pulse_on, pulse_off, rise, drive_amp)
    a_in = env * np.exp(1j * drive_phase)

    # linewidths
    Gamma_a = 0.5 * (kappa_a + gamma_a)
    Gamma_b = 0.5 * (kappa_b + gamma_b)

    # qubit-dependent detuning
    s = +1 if y == 1 else -1
    ddrift = rng.normal(0.0, slow_detune_std)
    da = delta_a + ddrift
    db = delta_b + ddrift + s * chi

    # phase noise (per shot)
    phi = rng.normal(0.0, phase_noise_std)

    # state
    a = 0.0 + 0.0j
    b = 0.0 + 0.0j
    y_lpf = 0.0 + 0.0j

    sig = np.zeros(n_points, dtype=np.complex128)

    # imbalance phasor
    if iq_imbalance != 0:
        ph2 = np.exp(1j * 2.0 * omega_if * t)
    else:
        ph2 = None

    for i in range(n_points):
        adot = -(Gamma_a + 1j*da)*a - 1j*J*b + np.sqrt(kappa_a)*a_in[i]
        bdot = -(Gamma_b + 1j*db)*b - 1j*J*a
        a += dt * adot
        b += dt * bdot

        a_out = a_in[i] - np.sqrt(kappa_a)*a
        b_out = -np.sqrt(kappa_b)*b
        s_out = a_out if measure == "a_out" else b_out

        if meas_tau and meas_tau > 0:
            y_lpf += dt * (-(1.0/meas_tau)*y_lpf + (1.0/meas_tau)*s_out)
            sig[i] = y_lpf
        else:
            sig[i] = s_out

    # IQ imbalance
    if ph2 is not None:
        sig = sig + iq_imbalance * ph2 * np.conj(sig)

    # LO phase noise
    sig *= np.exp(1j * phi)

    # demod rotation
    sig *= np.exp(-1j * demod_phase)

    # additive noise
    if amp_noise_std > 0:
        sig += (
            rng.normal(0, amp_noise_std, n_points)
            + 1j*rng.normal(0, amp_noise_std, n_points)
        )

    # return IQ
    X = np.stack([sig.real, sig.imag], axis=-1).astype(np.float32)
    return X


# -------------------------
# main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_npz", required=True)
    ap.add_argument("--n_train", type=int, default=50000)
    ap.add_argument("--n_test",  type=int, default=10000)
    ap.add_argument("--n_points", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)

    # physics (MHz)
    ap.add_argument("--delta_a_MHz", type=float, default=0.0)
    ap.add_argument("--delta_b_MHz", type=float, default=0.8)
    ap.add_argument("--J_MHz",       type=float, default=25.0)
    ap.add_argument("--kappa_a_MHz", type=float, default=50.0)
    ap.add_argument("--gamma_a_MHz", type=float, default=1.0)
    ap.add_argument("--kappa_b_MHz", type=float, default=6.0)
    ap.add_argument("--gamma_b_MHz", type=float, default=1.0)
    ap.add_argument("--chi_MHz",     type=float, default=1.4)

    # pulse / readout
    ap.add_argument("--T", type=float, default=0.2e-6)
    ap.add_argument("--pulse_on",  type=float, default=0.01e-6)
    ap.add_argument("--pulse_off", type=float, default=0.05e-6)
    ap.add_argument("--rise", type=float, default=4e-9)
    ap.add_argument("--drive_amp", type=float, default=1.0)
    ap.add_argument("--measure", type=str, default="a_out")

    # IQ chain
    ap.add_argument("--demod_phase", type=float, default=0.0)
    ap.add_argument("--beat_freq_MHz", type=float, default=60.0)
    ap.add_argument("--iq_imbalance", type=float, default=0.1)
    ap.add_argument("--meas_tau", type=float, default=50e-9)

    # noise
    ap.add_argument("--amp_noise_std", type=float, default=0.12)
    ap.add_argument("--phase_noise_std", type=float, default=0.015)
    ap.add_argument("--slow_detune_MHz", type=float, default=0.5)

    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    twopi = 2*np.pi
    delta_a = twopi * args.delta_a_MHz * 1e6
    delta_b = twopi * args.delta_b_MHz * 1e6
    J       = twopi * args.J_MHz       * 1e6
    kappa_a = twopi * args.kappa_a_MHz * 1e6
    gamma_a = twopi * args.gamma_a_MHz * 1e6
    kappa_b = twopi * args.kappa_b_MHz * 1e6
    gamma_b = twopi * args.gamma_b_MHz * 1e6
    chi     = twopi * args.chi_MHz     * 1e6
    slow_detune_std = twopi * args.slow_detune_MHz * 1e6
    omega_if = np.pi * args.beat_freq_MHz * 1e6

    def make_split(n):
        X = np.zeros((n, args.n_points, 2), dtype=np.float32)
        y = rng.integers(0, 2, size=n, dtype=np.int64)
        for i in range(n):
            X[i] = simulate_single_shot(
                rng,
                n_points=args.n_points,
                T=args.T,
                delta_a=delta_a,
                delta_b=delta_b,
                J=J,
                kappa_a=kappa_a,
                gamma_a=gamma_a,
                kappa_b=kappa_b,
                gamma_b=gamma_b,
                chi=chi,
                pulse_on=args.pulse_on,
                pulse_off=args.pulse_off,
                rise=args.rise,
                drive_amp=args.drive_amp,
                drive_phase=0.0,
                measure=args.measure,
                demod_phase=args.demod_phase,
                omega_if=omega_if,
                iq_imbalance=args.iq_imbalance,
                meas_tau=args.meas_tau,
                amp_noise_std=args.amp_noise_std,
                phase_noise_std=args.phase_noise_std,
                slow_detune_std=slow_detune_std,
                y=y[i],
            )
        return X, y

    X_train, y_train = make_split(args.n_train)
    X_test,  y_test  = make_split(args.n_test)

    params = vars(args)
    np.savez_compressed(
        args.out_npz,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        meta=json.dumps(params, ensure_ascii=False),
    )

    print(f"Saved {args.out_npz}")
    print("train:", X_train.shape, "test:", X_test.shape)


if __name__ == "__main__":
    main()
