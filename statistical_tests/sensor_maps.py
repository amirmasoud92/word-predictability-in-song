"""Sensor maps of the word-level measure and their split-half reliability (descriptive; no test).

A listener's word-level gain is the mean over sensors of the per-sensor gain, r(true) - mean r(permuted), with the
permutations of the Monte Carlo rule. The group map is the mean of the per-sensor gains over listeners. Its reliability is
the median, over 5,000 random splits of the listeners into two halves, of the correlation between the two half-group
maps, stepped up with the Spearman-Brown formula. A map is read as a spatial pattern when this reliability is at least
0.6. For two sessions, the between-session map correlation is the correlation over sensors of the two group maps of the
listeners with both sessions.
"""
from __future__ import annotations

import numpy as np

N_SPLITS, MIN_RELIABILITY = 5000, 0.6
SPLIT_SEED = None                                         # seed of the random splits (an integer; set before use)


def sensor_gain(r_true_ch: np.ndarray, r_perm_ch: np.ndarray, n_used: int | None = None) -> np.ndarray:
    """Per-sensor gain of one listener: true r per sensor minus the mean over the first ``n_used`` permuted versions
    (r_perm_ch: versions x sensors)."""
    r_perm_ch = np.asarray(r_perm_ch)[:n_used] if n_used else np.asarray(r_perm_ch)
    return np.asarray(r_true_ch) - r_perm_ch.mean(0)


def split_half(M: np.ndarray, seed: int = SPLIT_SEED, n_splits: int = N_SPLITS) -> dict:
    """M: listeners x sensors. Median correlation between the mean maps of two random halves of the listeners, and its
    Spearman-Brown value."""
    rng = np.random.default_rng(seed)
    n = M.shape[0]
    r = np.empty(n_splits)
    for k in range(n_splits):
        p = rng.permutation(n)
        r[k] = np.corrcoef(M[p[: n // 2]].mean(0), M[p[n // 2:]].mean(0))[0, 1]
    m = float(np.median(r))
    return {"n": int(n), "split_half_r": m, "spearman_brown": 2 * m / (1 + m)}


def between_session_map_r(session1: np.ndarray, session2: np.ndarray) -> float:
    """Correlation over sensors of the group maps of two sessions (rows: the same listeners in the same order)."""
    return float(np.corrcoef(session1.mean(0), session2.mean(0))[0, 1])


def project_positions(pos: np.ndarray, radius: float = 0.94) -> np.ndarray:
    """Azimuthal-equidistant projection of 3-D sensor positions (x right, y anterior, z up) about the centre of the sphere
    fitted to them, scaled so that the outermost sensor lies at ``radius`` of a unit head circle."""
    p = np.asarray(pos, float)
    A = np.c_[2 * p, np.ones(len(p))]
    c = np.linalg.lstsq(A, (p ** 2).sum(1), rcond=None)[0][:3]
    v = p - c
    th = np.arccos(np.clip(v[:, 2] / np.linalg.norm(v, axis=1), -1, 1))
    ph = np.arctan2(v[:, 1], v[:, 0])
    xy = np.c_[th * np.cos(ph), th * np.sin(ph)]
    return xy / np.linalg.norm(xy, axis=1).max() * radius


def alice_positions(sfp_rows: list[list[str]], labels: list[str], anterior_axis: int, anterior_sign: int) -> np.ndarray:
    """Head-frame positions (m) of the Alice electrodes from the release's sfp file, with the anterior axis and its sign
    taken from the channels that follow the vertical EOG (readers/alice.anterior_axis)."""
    sfp = {r[0]: np.array([float(v) for v in r[1:4]]) for r in sfp_rows}
    lat = 1 - anterior_axis if anterior_axis in (0, 1) else 0
    return np.array([[sfp[c][lat], anterior_sign * sfp[c][anterior_axis], sfp[c][2]] for c in labels]) / 100.0
