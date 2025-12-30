## generate_test_data.py（生成测试数据）

import argparse, json
import numpy as np

def make_templates(T=200, dt=1.0, amp=1.0, phase_sep=np.deg2rad(35), ringdown=0.06):
    t = np.arange(T) * dt
    env = np.exp(-ringdown * t)
    s0 = amp * env * np.exp(1j * 0.0)
    s1 = amp * env * np.exp(1j * phase_sep)
    return s0, 0.95 * s1

def sample_trace(s, sigma=0.25, iq_imbalance=0.03, dc=(0.02, -0.01), drift_std=0.02):
    T = s.shape[0]
    drift = (np.random.randn() + 1j*np.random.randn()) * drift_std
    z = s + drift
    I = z.real
    Q = z.imag
    I2 = I + iq_imbalance * Q
    Q2 = Q + iq_imbalance * I
    I2 = I2 + dc[0] + sigma*np.random.randn(T)
    Q2 = Q2 + dc[1] + sigma*np.random.randn(T)
    return np.stack([I2, Q2], axis=-1).astype(np.float32)

def generate(n, T, s0, s1, **kwargs):
    X = np.zeros((n, T, 2), np.float32)
    y = np.random.randint(0, 2, size=(n,), dtype=np.int64)
    for i in range(n):
        X[i] = sample_trace(s0 if y[i]==0 else s1, **kwargs)
    return X, y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="test_dataset.npz")
    ap.add_argument("--n_train", type=int, default=50000)
    ap.add_argument("--n_test", type=int, default=10000)
    ap.add_argument("--T", type=int, default=200)
    ap.add_argument("--sigma", type=float, default=0.25)
    ap.add_argument("--phase_deg", type=float, default=35.0)
    ap.add_argument("--iq_imbalance", type=float, default=0.03)
    ap.add_argument("--drift_std", type=float, default=0.02)
    args = ap.parse_args()

    s0, s1 = make_templates(T=args.T, phase_sep=np.deg2rad(args.phase_deg))
    X_train, y_train = generate(args.n_train, args.T, s0, s1,
                                sigma=args.sigma, iq_imbalance=args.iq_imbalance, drift_std=args.drift_std)
    X_test, y_test = generate(args.n_test, args.T, s0, s1,
                              sigma=args.sigma, iq_imbalance=args.iq_imbalance, drift_std=args.drift_std)

    meta = dict(
        T=args.T, sigma=args.sigma, phase_deg=args.phase_deg,
        iq_imbalance=args.iq_imbalance, drift_std=args.drift_std,
        template_s0_re=s0.real.tolist(), template_s0_im=s0.imag.tolist(),
        template_s1_re=s1.real.tolist(), template_s1_im=s1.imag.tolist(),
    )
    np.savez_compressed(args.out, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                        meta=json.dumps(meta))
    print(f"saved: {args.out}  train={X_train.shape} test={X_test.shape}")

if __name__ == "__main__":
    main()
