"""Word-level gain in CantiSong with 10 permutations (the original estimates of tests G1, G2 and G2b).

Per listener and version: leave-one-song-out ridge (lags 0-688 ms in 31 ms steps) on acoustics (34 rows), phonetic
features (19 rows; spoken and sung versions) and word onsets, with the ridge chosen on this base model and then fixed;
the full model adds surprisal-weighted word onsets. Delta r = r(full) - mean r over 10 within-song permutations of the
surprisal values (timing kept), with r on held-out songs within 0-800 ms after each word, averaged over channels. Hummed
versions take the sung word timing and surprisal (the hummed control). Run with CANTISONG_ANNOTATION="released" for
the released annotation of song 09 and "corrected" for the corrected one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from readers import cantisong as c
from statistical_tests.inference import summarise
from stimulus_features import song_regressors as g
from stimulus_features.song_regressors import AUDIO, N_SHUF, WORD_WIN, bz, fit_predict, lagged, r_mean


def listener_version(sub: int, version: str, data: dict, stims: dict, seed: int) -> dict:
    """Delta r of one listener and version (at least 6 usable songs)."""
    songs = [s for s in c.song_ids() if (version, s) in data and sub in data[(version, s)] and data[(version, s)][sub][1].any()]
    if len(songs) < 6:
        return {}
    a = int(c.START_S * g.FS)
    rng = np.random.default_rng([seed, sub, list(AUDIO).index(version)])
    blocks, wmasks = [], []
    for s in songs:
        st = stims[(s, version)]
        Y, v = data[(version, s)][sub]
        T = min(Y.shape[1], st["acoustic"].shape[1] - a)
        Y, v = Y[:, :T], v[:T]
        X = {k: lagged(st[k][:, a:a + T], T) for k in ("acoustic", "phonetic", "word", "surprisal") if k in st}
        m = np.zeros(T, bool)
        for i in st["word_idx"]:
            lo = i - a
            if lo + WORD_WIN > 0 and lo < T:
                m[max(lo, 0): min(lo + WORD_WIN, T)] = True
        blocks.append({"X": X, "Y": Y, "v": v, "st": st, "a": a, "T": T})
        wmasks.append(m & v)
    base = [k for k in ("acoustic", "phonetic", "word") if k in blocks[0]["X"]]
    _, lam = fit_predict(blocks, base, None)
    r_full = r_mean(blocks, fit_predict(blocks, base + ["surprisal"], lam)[0], wmasks)
    r_sh = []
    for _ in range(N_SHUF):
        for bl in blocks:
            st = bl["st"]
            raw = np.zeros_like(st["raw_surprisal"])
            raw[st["word_idx"]] = rng.permutation(st["raw_surprisal"][st["word_idx"]])
            bl["X"]["surprisal_shuf"] = lagged(bz(raw[None])[:, bl["a"]:bl["a"] + bl["T"]], bl["T"])
        r_sh.append(r_mean(blocks, fit_predict(blocks, base + ["surprisal_shuf"], lam)[0], wmasks))
    return {"listener": sub, "version": version, "n_songs": len(songs), "r_full": r_full, "r_shuf": float(np.mean(r_sh)),
            "delta_r": r_full - float(np.mean(r_sh)), "lambda": lam}


def song_word_level(paths, seed: int, n_jobs: int = 6) -> tuple[dict, pd.DataFrame]:
    """G1 (spoken > 0), G2 (sung > 0), G2b (sung - hummed > 0), one-sided, and the hummed control (two-sided)."""
    data = c.load_all(paths.derived_root / "cantisong_1to8", "1-8")
    words = {s: g.load_words(s) for s in c.song_ids()}
    stims = {(s, v): g.stimulus(s, v, words[s], False) for s in c.song_ids() for v in AUDIO}
    subs = sorted({sub for d in data.values() for sub in d})
    own = lambda sub: {k: {sub: d[sub]} for k, d in data.items() if sub in d}
    rows = Parallel(n_jobs=n_jobs)(delayed(listener_version)(sub, v, own(sub), stims, seed) for sub in subs for v in AUDIO)
    df = pd.DataFrame([r for r in rows if r])
    d = df.pivot(index="listener", columns="version", values="delta_r")
    pick = lambda x, alt="greater": {k: v for k, v in summarise(np.asarray(x, float), alt).items()
                                     if k in ("n", "mean", "dz", "dz_ci95", "p_signflip", "n_positive")}
    both = d[["song", "melody"]].dropna()
    res = {"G1_spoken": pick(d["speech"].dropna()), "G2_sung": pick(d["song"].dropna()),
           "G2b_sung_minus_hummed": pick(both["song"] - both["melody"]), "hummed_control": pick(d["melody"].dropna(), "two-sided")}
    return res, df
