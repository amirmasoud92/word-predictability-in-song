"""Shape of the surprisal response (exploratory; Fig. S2a).

Per listener and version, the full word-level model (acoustics [+ phonetic features] + word onsets + surprisal, with the
listener's ridge from the original word-level estimate) is fitted on all songs, and the surprisal kernel (23 lags x 64
channels, z-scored regressors) is kept. Because the surprisal regressor is correlated with word onsets, part of any
in-sample kernel is onset-response leakage; the surprisal-specific kernel is the true kernel minus the mean kernel over
10 within-song permutations of the surprisal values (timing kept). Group mean over the nine centro-parietal channels, a
sign-flip test per lag across listeners for a negative deflection with max-t correction over lags, and the paired
version contrasts, also averaged over 300-500 ms. Released annotation.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from readers import cantisong as c
from statistical_tests.inference import summarise
from statistical_tests.max_t import signflip_maxt
from stimulus_features import song_regressors as g

ROI = ["Cz", "CPz", "Pz", "C1", "C2", "CP1", "CP2", "P1", "P2"]


def stimuli(words_dir) -> dict:
    """Regressors of every song and version with the released annotation."""
    os.environ["CANTISONG_ANNOTATION"] = "released"
    words = {s: pd.read_csv(words_dir / f"{s}_words.csv") for s in c.song_ids()}
    return {(s, v): g.stimulus(s, v, words[s], False) for s in c.song_ids() for v in g.AUDIO}


def kernel(sub: int, version: str, data: dict, stims: dict, lam: float, seed: int, n_shuf: int = 10):
    """(true surprisal kernel, mean permuted-surprisal kernel), each lags x channels."""
    songs = [s for s in c.song_ids() if (version, s) in data and sub in data[(version, s)] and data[(version, s)][sub][1].any()]
    if len(songs) < 6:
        return None
    a = int(c.START_S * g.FS)
    rng = np.random.default_rng([seed, sub, list(g.AUDIO).index(version), 2])
    parts = []
    for s in songs:
        st = stims[(s, version)]
        Y, v = data[(version, s)][sub]
        T = min(Y.shape[1], st["acoustic"].shape[1] - a)
        names = [k for k in ("acoustic", "phonetic", "word") if k in st]
        Xo = np.hstack([g.lagged(st[k][:, a:a + T], T) for k in names])[v[:T]]
        Yv = Y[:, :T][:, v[:T]].T
        xs = lambda raw: g.lagged(g.bz(raw[None])[:, a:a + T], T)[v[:T]]
        shuf = []
        for _ in range(n_shuf):
            raw = np.zeros_like(st["raw_surprisal"])
            raw[st["word_idx"]] = rng.permutation(st["raw_surprisal"][st["word_idx"]])
            shuf.append(xs(raw))
        parts.append((Xo, Yv, g.lagged(st["surprisal"][:, a:a + T], T)[v[:T]], shuf))
    Goo = sum(Xo.T @ Xo for Xo, _, _, _ in parts)
    Co = sum(Xo.T @ Yv for Xo, Yv, _, _ in parts)

    def solve(which):
        Gos = sum(Xo.T @ which(i) for i, (Xo, _, _, _) in enumerate(parts))
        Gss = sum(which(i).T @ which(i) for i in range(len(parts)))
        Cs = sum(which(i).T @ Yv for i, (_, Yv, _, _) in enumerate(parts))
        G = np.block([[Goo, Gos], [Gos.T, Gss]])
        e = np.trace(G) / G.shape[0]
        W = np.linalg.solve(G + lam * e * np.eye(G.shape[0]), np.vstack([Co, Cs]))
        return W[-g.N_LAGS:]
    true = solve(lambda i: parts[i][2])
    sh = np.mean([solve(lambda i, k=k: parts[i][3][k]) for k in range(n_shuf)], axis=0)
    return true, sh


def kernel_statistics(K: dict, subs_v: dict, labels: list[str]) -> dict:
    """K: version -> listeners x lags x channels (surprisal-specific kernels); subs_v: version -> listener order."""
    roi = [labels.index(ch) for ch in ROI if ch in labels]
    lags_ms = np.arange(g.N_LAGS) * g.LAG_STEP * 1000 / g.FS
    res = {"roi": [labels[i] for i in roi], "lags_ms": lags_ms.tolist()}
    win = (lags_ms >= 300) & (lags_ms <= 500)
    for a_, b_ in (("song", "melody"), ("speech", "melody"), ("song", "speech")):
        common = sorted(set(subs_v[a_]) & set(subs_v[b_]))
        da = K[a_][[subs_v[a_].index(x) for x in common]][:, :, roi].mean(2)
        db = K[b_][[subs_v[b_].index(x) for x in common]][:, :, roi].mean(2)
        _, _, pf = signflip_maxt(-(da - db), alternative="greater")
        sw = summarise(-(da - db)[:, win].mean(1), "greater")
        res[f"contrast_{a_}_minus_{b_}"] = {"lags_fwe_sig_more_negative": lags_ms[pf < 0.05].tolist(),
                                            "window_300_500_more_negative": {k: sw[k] for k in ("n", "mean", "dz", "dz_ci95", "p_signflip")}}
    for v, k in K.items():
        roi_t = k[:, :, roi].mean(2)
        _, _, p_fwe = signflip_maxt(-roi_t, alternative="greater")
        res[v] = {"n": int(k.shape[0]), "roi_mean": roi_t.mean(0).tolist(), "p_fwe_negative": p_fwe.tolist(),
                  "lags_fwe_sig": lags_ms[p_fwe < 0.05].tolist()}
    return res
