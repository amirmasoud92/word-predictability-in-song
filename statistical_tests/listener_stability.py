"""Stability of the word-level gain for the listener (Alice): split-half reliability and listener ICC.

The gain of each held-out segment is r(full) - mean r(permuted) within that segment (word-locked 0-800 ms, mean over
channels), from the same models as the listener-level gain. Split-half reliability: Spearman correlation over listeners
of the mean gain of the odd and of the even segments, stepped up with the Spearman-Brown formula. Listener ICC: the share
of the variance of the single-segment gain (x 1000) due to listeners, from dr ~ 1 + (1|listener) + (1|segment)
(lme4), with a 95 % interval from 1000 parametric bootstrap draws (listener_stability.R).
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from readers import alice as h
from stimulus_features import song_regressors as g

R_SCRIPT = Path(__file__).with_name("listener_stability.R")


def per_segment_delta_r(sub: str, data: dict, stims: dict, seed: int) -> dict:
    """Gain of each held-out segment of one listener, 10 within-segment permutations."""
    segs = [s for s in h.SEGMENTS if s in data and data[s][1].any()]
    if len(segs) < h.MIN_SEGMENTS:
        return {}
    a = int(h.START_S * 64)
    rng = np.random.default_rng([seed, int(sub[1:]), 17])
    blocks, masks = [], []
    for s in segs:
        st = stims[s]
        Y, v = data[s]
        T = min(Y.shape[1], st["acoustic"].shape[1] - a)
        X = {k: g.lagged(st[k][:, a:a + T], T) for k in ("acoustic", "word", "surprisal")}
        m = np.zeros(T, bool)
        for i in st["word_idx"]:
            lo = i - a
            if lo + g.WORD_WIN > 0 and lo < T:
                m[max(lo, 0): min(lo + g.WORD_WIN, T)] = True
        blocks.append({"X": X, "Y": Y[:, :T], "v": v[:T], "st": st, "a": a, "T": T})
        masks.append(m & v[:T])
    _, lam = g.fit_predict(blocks, ["acoustic", "word"], None)
    full = g.fit_predict(blocks, ["acoustic", "word", "surprisal"], lam)[0]
    shuf = []
    for _ in range(g.N_SHUF):
        for bl in blocks:
            st = bl["st"]
            raw = np.zeros_like(st["raw_surprisal"])
            raw[st["word_idx"]] = rng.permutation(st["raw_surprisal"][st["word_idx"]])
            bl["X"]["surprisal_shuf"] = g.lagged(g.bz(raw[None])[:, bl["a"]:bl["a"] + bl["T"]], bl["T"])
        shuf.append(g.fit_predict(blocks, ["acoustic", "word", "surprisal_shuf"], lam)[0])
    out = {"listener": sub}
    for j, s in enumerate(segs):
        one = [blocks[j]]
        rf = g.r_mean(one, [full[j]], [masks[j]])
        rs = np.mean([g.r_mean(one, [sh[j]], [masks[j]]) for sh in shuf])
        out[f"seg{s}"] = rf - rs
    return out


def split_half(wide: pd.DataFrame) -> dict:
    """wide: listeners x segments (columns 'seg<k>' or segment numbers)."""
    num = lambda c: int(str(c).replace("seg", ""))
    odd = wide[[c for c in wide.columns if num(c) % 2 == 1]].mean(1)
    even = wide[[c for c in wide.columns if num(c) % 2 == 0]].mean(1)
    r = stats.spearmanr(odd, even, nan_policy="omit").correlation
    return {"r_half": float(r), "spearman_brown": float(2 * r / (1 + r)) if r > -1 else np.nan}


def listener_icc(long: pd.DataFrame, work: Path, rscript: str = "Rscript", seed: int | None = None) -> dict:
    """long: listener, segment, dr (single-segment gain, unscaled). Runs listener_stability.R (``seed``: bootstrap)."""
    work.mkdir(parents=True, exist_ok=True)
    f_in, f_out = work / "segment_gains.csv", work / "listener_icc.csv"
    long.assign(dr=long.dr * 1000).to_csv(f_in, index=False)
    subprocess.run([rscript, str(R_SCRIPT), str(f_in), str(f_out)] + ([str(seed)] if seed is not None else []), check=True, capture_output=True)
    r = pd.read_csv(f_out).iloc[0]
    return {"icc_listener": float(r.icc), "ci95": [float(r.lo), float(r.hi)]}
