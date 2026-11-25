"""Signal operations shared by the EEG preprocessing chains.

Robust polynomial detrending (de Cheveigne & Arzounian, 2018), Hamming windowed-sinc FIR filters (one pass, delay
compensated, cutoff at -6 dB) and bad-channel detection on the log standard deviation. The spherical-spline operator and
the ocular ICA of the EEG chains are in preprocessing/cnd_recordings.py.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import scipy.signal as ss


@lru_cache(maxsize=32)
def fir_kernel(cutoff_hz: float, sfreq: float, order: int, highpass: bool) -> np.ndarray:
    """Hamming windowed-sinc FIR of the given order (order + 1 taps), cutoff at -6 dB."""
    if order % 2:
        raise ValueError("FIR order must be even for delay compensation")
    return ss.firwin(order + 1, cutoff_hz, window="hamming", pass_zero=not highpass, fs=sfreq)


def fir_filter(x: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Zero-delay FIR filtering along the last axis with edge-value padding (EEGLAB firfilt)."""
    half = (b.size - 1) // 2
    xp = np.concatenate([np.repeat(x[..., :1], half, axis=-1), x, np.repeat(x[..., -1:], half, axis=-1)], axis=-1)
    return ss.oaconvolve(xp, b[(None,) * (x.ndim - 1)], mode="valid", axes=-1)


def robust_detrend(x: np.ndarray, order: int, threshold_sd: float, n_iter: int = 3) -> np.ndarray:
    """Weighted polynomial detrending with iterative outlier down-weighting (NoiseTools nt_detrend)."""
    n = x.shape[-1]
    t = np.linspace(-1.0, 1.0, n)
    basis = np.polynomial.legendre.legvander(t, order)
    out = np.empty_like(x, dtype=float)
    for c in range(x.shape[0]):
        w = np.ones(n, dtype=bool)
        y = np.asarray(x[c], float)
        for _ in range(n_iter):
            coef, *_ = np.linalg.lstsq(basis[w], y[w], rcond=None)
            r = y - basis @ coef
            sd = r[w].std()
            w_new = np.abs(r) < threshold_sd * sd
            if w_new.sum() < basis.shape[1] * 10 or np.array_equal(w_new, w):
                break
            w = w_new
        out[c] = r
    return out


def bad_channels(broadband: list[np.ndarray], ch_names: list[str], robust_z: float) -> list[str]:
    """Channels whose log SD over the condition deviates by more than ``robust_z`` robust SDs, or are flat."""
    sd = np.concatenate(broadband, axis=1).std(axis=1)
    lsd = np.log(np.maximum(sd, 1e-12))
    med = np.median(lsd)
    mad = 1.4826 * np.median(np.abs(lsd - med))
    bad = (np.abs(lsd - med) / max(mad, 1e-12) > robust_z) | (sd < 1e-3)
    return [c for c, b in zip(ch_names, bad) if b]
