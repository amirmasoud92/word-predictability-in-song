"""Operators of the EEG cleaning shared by the three EEG datasets.

Montage information, the spherical-spline interpolation of bad channels with average reference as one linear operator,
the ocular extended-Infomax ICA (at most 40 components; the most anterior channels serve as ocular proxies; at most three
components with |z| > 3 removed), Hamming-window FIR orders and the band-pass filter on the 64 Hz grid. The per-dataset
chains are in readers/cantisong.py, readers/alice.py and readers/speech_in_noise.py.
"""
from __future__ import annotations

import mne
import numpy as np

from preprocessing import signal_operations as pp


def montage_info(labels: list[str], sfreq: float, montage: str) -> mne.Info:
    info = mne.create_info(labels, sfreq, "eeg")
    info.set_montage(mne.channels.make_standard_montage(montage), match_case=False)
    return info


def spatial_operator(info: mne.Info, bads: list[str]) -> np.ndarray:
    """Interpolation of ``bads`` followed by average reference, as a channels x channels matrix."""
    n = len(info.ch_names)
    raw = mne.io.RawArray(np.eye(n), info.copy(), verbose="error")
    raw.info["bads"] = list(bads)
    if bads:
        raw.interpolate_bads(reset_bads=True, verbose="error")
    op = raw.get_data()
    return op - op.mean(axis=0, keepdims=True)


def ocular_ica(x: np.ndarray, info: mne.Info, rank: int, n_proxies: int, z: float, max_components: int,
               seed: int) -> tuple[np.ndarray, list[int]]:
    """Linear projection removing ocular ICA components; proxies are the most anterior channels."""
    pos = np.array([info["chs"][i]["loc"][:3] for i in range(len(info.ch_names))])
    proxies = [info.ch_names[i] for i in np.argsort(-pos[:, 1])[:n_proxies]]
    raw = mne.io.RawArray(x * 1e-6, info.copy(), verbose="error")
    ica = mne.preprocessing.ICA(n_components=rank, method="infomax", fit_params={"extended": True},
                                random_state=seed, max_iter="auto", verbose="error")
    ica.fit(raw, verbose="error")
    idx, scores = ica.find_bads_eog(raw, ch_name=proxies, threshold=z, measure="zscore", verbose="error")
    score = np.max(np.abs(np.atleast_2d(scores)), axis=0)
    idx = sorted(idx, key=lambda k: -score[k])[:max_components]
    ica.exclude = list(idx)
    n = len(info.ch_names)
    probe = mne.io.RawArray(np.hstack([np.eye(n), np.zeros((n, 1))]) * 1e-6, info.copy(), verbose="error")
    out = ica.apply(probe, verbose="error").get_data()
    return (out[:, :n] - out[:, n:]) / 1e-6, list(idx)


def fir_order(sfreq: float, transition_hz: float) -> int:
    m = int(np.ceil(3.3 / (transition_hz / sfreq)))
    return m + (m % 2)


def band_64(x: np.ndarray, sfreq: float, lo: float, hi: float) -> np.ndarray:
    """1 Hz-transition Hamming band-pass on a 64 Hz grid (anti-alias low-pass before decimation)."""
    step = int(round(sfreq / 64.0))
    y = x
    if step > 1:
        y = pp.fir_filter(y, pp.fir_kernel(25.0, sfreq, fir_order(sfreq, 1.0), False))[:, ::step]
    y = pp.fir_filter(y, pp.fir_kernel(hi, 64.0, fir_order(64.0, 1.0), False))
    return pp.fir_filter(y, pp.fir_kernel(max(lo - 0.5, 0.25), 64.0, fir_order(64.0, 1.0), True))
