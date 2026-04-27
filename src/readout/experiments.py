
import numpy as np 
from typing import Tuple, Union, Sequence, Optional

def find_demod_freq(X_data, fs, fmin=1e6, fmax=1.5e9, plot=True):
    """
    X_data: (B, T) real
    fs: sampling rate (Hz)
    fmin,fmax: search band
    return: f_peak (Hz)
    """
    X = np.asarray(X_data, dtype=np.float64)

    x_mean = X.mean(axis=0)
    x_mean = x_mean - x_mean.mean()   # 去DC

    F = np.fft.rfft(x_mean)
    freq = np.fft.rfftfreq(len(x_mean), d=1/fs)
    amp = np.abs(F)

    mask = (freq >= fmin) & (freq <= fmax)
    f_peak = freq[mask][np.argmax(amp[mask])]

    if plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(freq[mask]/1e6, amp[mask])
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("|FFT|")
        plt.title(f"Peak = {f_peak/1e6:.3f} MHz")
        plt.show()

    return f_peak

def demod_iq(x, f_demod, fs, phase=0.0):
    # x: (B,T) real
    t = np.arange(x.shape[1]) / fs
    I = x * np.cos(2*np.pi*f_demod*t + phase)
    Q = x * np.sin(2*np.pi*f_demod*t + phase)
    return np.stack([I, Q], axis=-1)  # (B,T,2)


# -----------------
# Spectral analysis
# -----------------
from scipy.signal import find_peaks, peak_widths
from scipy.ndimage import median_filter
from scipy.stats import gennorm
from scipy.special import gamma

ArrayLike = Union[np.ndarray]

def compute_psd_rfft(
    x: ArrayLike,
    fs: float,
    window: Sequence[int] = (0, -1),
    detrend: str = "per_trace",   # "per_trace" | "global" | "none"
    normalize: bool = False,      # True 时输出更像“功率谱密度”(粗略按采样点数归一)
    dtype=np.float64,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute one-sided PSD via rFFT for 1D or 2D time traces.

    Parameters
    ----------
    x : np.ndarray
        Shape (T,) or (n_shots, T)
    fs : float
        Sampling rate (Hz)
    window : (start, stop)
        Slice on time axis. stop=-1 means to the end.
    detrend : str
        - "per_trace": subtract mean of each trace (recommended)
        - "global": subtract overall mean
        - "none": no mean removal
    normalize : bool
        If True, returns |FFT|^2 / N (rough normalization).
        (If you want true PSD in units per Hz, you’d also divide by fs and handle windowing.)
    dtype : numpy dtype
        Cast input to this dtype before FFT.

    Returns
    -------
    freq : np.ndarray
        Frequency axis, shape (n_freq,)
    psd : np.ndarray
        PSD per trace, shape (n_traces, n_freq)
        (If input was 1D, n_traces=1)
    """
    x = np.asarray(x)

    if x.ndim == 1:
        x2 = x[None, :]  # -> (1, T)
    elif x.ndim == 2:
        x2 = x
    else:
        raise ValueError(f"x must be 1D or 2D, got shape {x.shape}")

    start, stop = int(window[0]), int(window[1])
    if stop == -1:
        stop = x2.shape[1]
    x2 = x2[:, start:stop].astype(dtype, copy=False)

    if x2.shape[1] < 2:
        raise ValueError(f"Window too short: got T={x2.shape[1]} after slicing.")

    # detrend / remove DC
    if detrend == "per_trace":
        x2 = x2 - x2.mean(axis=1, keepdims=True)
    elif detrend == "global":
        x2 = x2 - x2.mean()
    elif detrend == "none":
        pass
    else:
        raise ValueError(f"Unknown detrend={detrend!r}")

    n = x2.shape[1]
    freq = np.fft.rfftfreq(n, d=1.0 / fs)

    F = np.fft.rfft(x2, axis=1)
    psd = (F.real * F.real + F.imag * F.imag)  # = |F|^2, avoids abs() temporary

    if normalize:
        psd = psd / n

    return freq, psd

def peak_snr_and_width_from_psd(f, Pxx, prominence_db=6, noise_med_window_bins=401):
    Pdb = 10*np.log10(Pxx + 1e-30)

    if noise_med_window_bins % 2 == 0:
        noise_med_window_bins += 1
    noise_floor_db = median_filter(Pdb, size=noise_med_window_bins, mode='nearest')

    resid_db = Pdb - noise_floor_db

    peaks, props = find_peaks(resid_db, prominence=prominence_db)
    widths, *_ = peak_widths(resid_db, peaks, rel_height=0.5)

    df = f[1] - f[0]
    fwhm_hz = widths * df

    out = []
    for k, p in enumerate(peaks):
        out.append(dict(
            peak_idx=p,
            f0_hz=float(f[p]),
            snr_db=float(resid_db[p]),
            fwhm_hz=float(fwhm_hz[k]),
            peak_psd_db=float(Pdb[p]),
            noise_floor_db=float(noise_floor_db[p]),
        ))
    return out

def scores_from_psd_using_median_floor(
    p,
    peak_idx: int,
    noise_med_window_bins: int = 401,
    *,
    fast: bool = True,
    eps: float = 1e-30,
):
    """
    Compute single-shot score at peak_idx using median noise floor subtraction.

    score_k = Pdb[k, peak_idx] - noise_floor_db[k, peak_idx]

    Parameters
    ----------
    p : ndarray
        Linear PSD, shape (n_shots, n_freq)
    peak_idx : int
        Index of target frequency bin
    noise_med_window_bins : int
        Median window size (must be odd; will be fixed if even)
    fast : bool
        - False: full median_filter over frequency axis (slower, exact)
        - True : local median only around peak_idx (much faster)
    eps : float
        Small number to avoid log(0)

    Returns
    -------
    score : ndarray
        Shape (n_shots,), residual in dB at peak_idx
    """
    p = np.asarray(p)
    if p.ndim != 2:
        raise ValueError(f"p must be 2D (n_shots, n_freq), got shape {p.shape}")

    W = int(noise_med_window_bins)
    if W % 2 == 0:
        W += 1
    h = W // 2

    # convert to dB once
    Pdb = 10.0 * np.log10(p + eps)  # (n_shots, n_freq)

    if not fast:
        # full median filter along frequency axis
        noise_floor_db = median_filter(
            Pdb, size=(1, W), mode="nearest"
        )
        score = Pdb[:, peak_idx] - noise_floor_db[:, peak_idx]

    else:
        # local median only around peak_idx
        i0 = max(0, peak_idx - h)
        i1 = min(p.shape[1], peak_idx + h + 1)
        noise_floor_at_peak = np.median(Pdb[:, i0:i1], axis=1)
        score = Pdb[:, peak_idx] - noise_floor_at_peak

    return score

def best_threshold_from_scores(scores, y):
    """在1D score上扫描阈值，返回最优 balanced accuracy 的阈值和方向，并给出 F0->0/F1->1。"""
    scores = np.asarray(scores, float)
    y = np.asarray(y).astype(int)
    s0, s1 = scores[y == 0], scores[y == 1]

    cand = np.sort(np.unique(np.r_[s0, s1]))
    thr = np.r_[cand[0]-1e-12, (cand[:-1]+cand[1:])/2, cand[-1]+1e-12]

    best = (-1.0, None, None)
    for t in thr:
        # predict 1 if s>=t
        F0 = (s0 < t).mean()
        F1 = (s1 >= t).mean()
        bacc = 0.5*(F0+F1)
        if bacc > best[0]:
            best = (float(bacc), float(t), "predict_1_if_s_ge_t")

        # predict 1 if s<=t
        F0 = (s0 > t).mean()
        F1 = (s1 <= t).mean()
        bacc = 0.5*(F0+F1)
        if bacc > best[0]:
            best = (float(bacc), float(t), "predict_1_if_s_le_t")

    bacc, t, direction = best
    if direction == "predict_1_if_s_ge_t":
        F0, F1 = float((s0 < t).mean()), float((s1 >= t).mean())
    else:
        F0, F1 = float((s0 > t).mean()), float((s1 <= t).mean())

    return {"F0_to_0": F0, "F1_to_1": F1, "threshold": t, "direction": direction, "balanced_acc": bacc}

def _beta_from_two_widths(w_h1, w_h2, h1, h2):
    """Infer generalized-normal shape beta from two full widths at relative heights h1,h2."""
    if not (0.0 < h1 < 1.0 and 0.0 < h2 < 1.0 and h1 != h2):
        raise ValueError(f"h1/h2 must be in (0,1) and different, got h1={h1}, h2={h2}")
    if w_h1 <= 0 or w_h2 <= 0:
        raise ValueError(f"widths must be > 0, got {w_h1}, {w_h2}")
    num = np.log(np.log(1.0 / h1) / np.log(1.0 / h2))
    den = np.log(float(w_h1) / float(w_h2))
    if np.isclose(den, 0.0):
        return 2.0
    beta = float(num / den)
    return max(beta, 1e-3)

def _alpha_from_width(w_h, h, beta):
    """Convert full width at relative height h to generalized-normal scale alpha."""
    if not (0.0 < h < 1.0):
        raise ValueError(f"h must be in (0,1), got {h}")
    if w_h <= 0 or beta <= 0:
        raise ValueError(f"w_h and beta must be > 0, got {w_h}, {beta}")
    return float(w_h) / (2.0 * (np.log(1.0 / float(h)) ** (1.0 / float(beta))))

def _gennorm_std(alpha, beta):
    return float(alpha * np.sqrt(gamma(3.0 / beta) / gamma(1.0 / beta)))

def metrics_from_peaks_gennorm(
    mu0,
    width0,
    mu1,
    width1,
    *,
    width_level=0.5,
    width0_alt=None,
    width1_alt=None,
    width_alt_level=None,
    beta0=None,
    beta1=None,
    p0=0.5,
    p1=0.5,
    grid_size=4001,
):
    """
    Estimate readout metrics with generalized-normal (non-Gaussian) hypothesis.

    Parameters
    ----------
    mu0, mu1 : float
        Peak center / location for state 0 and state 1.
    width0, width1 : float
        Full width at `width_level` relative height (FWHM when width_level=0.5).
    width_level : float
        Relative height in (0,1) for width0/width1.
    width0_alt, width1_alt : float or None
        Optional second widths measured at `width_alt_level`; used to infer beta.
    width_alt_level : float or None
        Relative height for alternate widths; required when *_alt is provided.
    beta0, beta1 : float or None
        Optional fixed shape parameters. If None, inferred from two-width ratio when
        possible; otherwise defaults to 2.0 (Gaussian).
    p0, p1 : float
        Class priors (weights), will be normalized.
    grid_size : int
        Number of threshold points for numerical optimization.

    Returns
    -------
    dict
        threshold, direction, F0_to_0, F1_to_1, F, e0, e1, and fitted model params.
    """
    mu0, mu1 = float(mu0), float(mu1)
    width0, width1 = float(width0), float(width1)
    p0, p1 = float(p0), float(p1)
    if width0 <= 0 or width1 <= 0:
        raise ValueError(f"width0/width1 must be > 0, got {width0}, {width1}")
    if p0 <= 0 or p1 <= 0:
        raise ValueError(f"p0/p1 must be > 0, got {p0}, {p1}")
    if grid_size < 101:
        raise ValueError(f"grid_size too small: {grid_size}")

    z = p0 + p1
    p0, p1 = p0 / z, p1 / z

    can_infer_beta = (
        width0_alt is not None and
        width1_alt is not None and
        width_alt_level is not None
    )
    if beta0 is None:
        if can_infer_beta:
            beta0 = _beta_from_two_widths(width0, float(width0_alt), width_level, float(width_alt_level))
        else:
            beta0 = 2.0
    if beta1 is None:
        if can_infer_beta:
            beta1 = _beta_from_two_widths(width1, float(width1_alt), width_level, float(width_alt_level))
        else:
            beta1 = 2.0
    beta0, beta1 = float(beta0), float(beta1)
    if beta0 <= 0 or beta1 <= 0:
        raise ValueError(f"beta0/beta1 must be > 0, got {beta0}, {beta1}")

    alpha0 = _alpha_from_width(width0, width_level, beta0)
    alpha1 = _alpha_from_width(width1, width_level, beta1)
    if alpha0 <= 0 or alpha1 <= 0:
        raise ValueError(f"Invalid alpha from widths: alpha0={alpha0}, alpha1={alpha1}")

    d0 = gennorm(beta=beta0, loc=mu0, scale=alpha0)
    d1 = gennorm(beta=beta1, loc=mu1, scale=alpha1)

    lo = min(d0.ppf(1e-6), d1.ppf(1e-6))
    hi = max(d0.ppf(1 - 1e-6), d1.ppf(1 - 1e-6))
    if not np.isfinite(lo) or not np.isfinite(hi) or not (hi > lo):
        s0 = _gennorm_std(alpha0, beta0)
        s1 = _gennorm_std(alpha1, beta1)
        lo = min(mu0 - 8.0 * s0, mu1 - 8.0 * s1)
        hi = max(mu0 + 8.0 * s0, mu1 + 8.0 * s1)

    t_grid = np.linspace(lo, hi, int(grid_size), dtype=float)
    c0 = d0.cdf(t_grid)
    c1 = d1.cdf(t_grid)

    # Direction A: predict 1 if s >= t
    F0_a = c0
    F1_a = 1.0 - c1
    F_a = p0 * F0_a + p1 * F1_a
    ia = int(np.argmax(F_a))

    # Direction B: predict 1 if s <= t
    F0_b = 1.0 - c0
    F1_b = c1
    F_b = p0 * F0_b + p1 * F1_b
    ib = int(np.argmax(F_b))

    if F_a[ia] >= F_b[ib]:
        t = float(t_grid[ia])
        F0 = float(F0_a[ia])
        F1 = float(F1_a[ia])
        F = float(F_a[ia])
        direction = "predict_1_if_s_ge_t"
    else:
        t = float(t_grid[ib])
        F0 = float(F0_b[ib])
        F1 = float(F1_b[ib])
        F = float(F_b[ib])
        direction = "predict_1_if_s_le_t"

    return {
        "threshold": t,
        "direction": direction,
        "F0_to_0": F0,
        "F1_to_1": F1,
        "F": F,
        "e0": float(1.0 - F0),
        "e1": float(1.0 - F1),
        "p0": p0,
        "p1": p1,
        "mu0": mu0,
        "mu1": mu1,
        "alpha0": alpha0,
        "alpha1": alpha1,
        "beta0": beta0,
        "beta1": beta1,
        "sigma0_equiv": _gennorm_std(alpha0, beta0),
        "sigma1_equiv": _gennorm_std(alpha1, beta1),
        "width_level": float(width_level),
    }


def predict_fidelities_from_spectrum(freq, psd, y, peak_idx,
                                     noise_med_window_bins=401, fast=True):
    """
    1) 对每shot算 resid_db@peak_idx 当 score
    2) 阈值分类 → F0->0, F1->1
    """
    scores = scores_from_psd_using_median_floor(
        psd, peak_idx,
        noise_med_window_bins=noise_med_window_bins, fast=fast
    )
    out = best_threshold_from_scores(scores, y)
    out.update({
        "peak_idx": int(peak_idx),
        "peak_f0_hz": float(freq[peak_idx]),
        "scores": scores,
    })
    return out


def train_test_fidelity_from_scores(scores, y, train_ratio=0.8, seed=0, stratified=True):
    """
    Split score/label into train/test, fit threshold on train, and evaluate on test.

    Parameters
    ----------
    scores : ndarray
        1D score per shot.
    y : ndarray
        Labels in {0, 1}.
    train_ratio : float
        Train split ratio in (0, 1).
    seed : int
        RNG seed for reproducible splitting.
    stratified : bool
        Keep class balance in train/test when True.

    Returns
    -------
    out : dict
        train: threshold fit metrics on train
        test : measured metrics on held-out test set
    """
    scores = np.asarray(scores, dtype=float).reshape(-1)
    y = np.asarray(y).astype(int).reshape(-1)

    if scores.shape[0] != y.shape[0]:
        raise ValueError(f"scores/y length mismatch: {scores.shape[0]} vs {y.shape[0]}")
    if not (0.0 < float(train_ratio) < 1.0):
        raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")
    if scores.shape[0] < 4:
        raise ValueError("Need at least 4 samples for train/test split.")

    rng = np.random.default_rng(seed)

    if stratified:
        idx0 = np.where(y == 0)[0]
        idx1 = np.where(y == 1)[0]
        if idx0.size < 2 or idx1.size < 2:
            raise ValueError("Need at least 2 samples for each class when stratified=True.")

        rng.shuffle(idx0)
        rng.shuffle(idx1)
        n0_tr = int(train_ratio * idx0.size)
        n1_tr = int(train_ratio * idx1.size)
        n0_tr = min(max(n0_tr, 1), idx0.size - 1)
        n1_tr = min(max(n1_tr, 1), idx1.size - 1)

        tr_idx = np.r_[idx0[:n0_tr], idx1[:n1_tr]]
        te_idx = np.r_[idx0[n0_tr:], idx1[n1_tr:]]
        rng.shuffle(tr_idx)
        rng.shuffle(te_idx)
    else:
        idx = np.arange(y.size)
        rng.shuffle(idx)
        n_tr = int(train_ratio * idx.size)
        n_tr = min(max(n_tr, 1), idx.size - 1)
        tr_idx, te_idx = idx[:n_tr], idx[n_tr:]

    s_tr, y_tr = scores[tr_idx], y[tr_idx]
    s_te, y_te = scores[te_idx], y[te_idx]

    train_fit = best_threshold_from_scores(s_tr, y_tr)
    t = float(train_fit["threshold"])
    direction = train_fit["direction"]

    s0_te = s_te[y_te == 0]
    s1_te = s_te[y_te == 1]
    if s0_te.size == 0 or s1_te.size == 0:
        raise ValueError("Test split has empty class; use stratified=True or adjust train_ratio.")

    if direction == "predict_1_if_s_ge_t":
        F0 = float((s0_te < t).mean())
        F1 = float((s1_te >= t).mean())
    elif direction == "predict_1_if_s_le_t":
        F0 = float((s0_te > t).mean())
        F1 = float((s1_te <= t).mean())
    else:
        raise ValueError(f"Unknown direction={direction!r}")

    bacc = 0.5 * (F0 + F1)
    acc = float((F0 * s0_te.size + F1 * s1_te.size) / (s0_te.size + s1_te.size))

    return {
        "train": {
            **train_fit,
            "n_train": int(tr_idx.size),
            "n_train_class0": int((y_tr == 0).sum()),
            "n_train_class1": int((y_tr == 1).sum()),
        },
        "test": {
            "F0_to_0": F0,
            "F1_to_1": F1,
            "balanced_acc": float(bacc),
            "acc": acc,
            "n_test": int(te_idx.size),
            "n_test_class0": int(s0_te.size),
            "n_test_class1": int(s1_te.size),
        },
    }
