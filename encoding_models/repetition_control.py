"""Lyric-repetition control (exploratory; Fig. S2b).

In songs, repeated lines (choruses) repeat melody and words together, and repeated words are predictable. Two impulse
regressors at word onsets are added to the base of the original word-level model: line repeated (the normalised lyric
line occurred earlier in the song) and word repeated (the word occurred earlier). Delta r of surprisal is then computed as
in the original estimate: ridge chosen without surprisal, 10 within-song permutations of the surprisal values (timing and
repetition regressors kept), accuracy on held-out songs within 0-800 ms after word onsets. Released annotation.
One-sided tests over listeners (two-sided for the hummed control), and sung - hummed.
"""
from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd

from readers import cantisong as c
from statistical_tests.inference import summarise
from stimulus_features import song_regressors as g

WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'’]*(?:-[A-Za-z0-9'’]+)*")


def repetition(song: str) -> tuple[np.ndarray, np.ndarray]:
    """Per word (in the order of the word table): line repeated, word repeated (0/1)."""
    text = (c.ROOT / "Stimuli" / "lyrics" / f"{song}.txt").read_text(encoding="utf-8").replace("’", "'")
    starts = np.cumsum([0] + [len(ln) + 1 for ln in text.split("\n")])[:-1]
    lines = text.split("\n")
    norm = [re.sub(r"[^a-z ]", "", ln.lower()).strip() for ln in lines]
    seen_line, seen_word, rl, rw = set(), set(), [], []
    last_line = -1
    for m in WORD.finditer(text):
        li = int(np.searchsorted(starts, m.start(), side="right") - 1)
        if li != last_line and last_line >= 0:
            seen_line.add(norm[last_line])
        last_line = li
        rl.append(float(norm[li] in seen_line and norm[li] != ""))
        w = m.group(0).lower()
        rw.append(float(w in seen_word))
        seen_word.add(w)
    return np.array(rl), np.array(rw)


def stim_with_repetition(song: str, version: str, words: pd.DataFrame) -> dict:
    os.environ["CANTISONG_ANNOTATION"] = "released"
    st = g.stimulus(song, version, words, False)
    rl, rw = repetition(song)
    col = g.TIMING[version]
    w = words.dropna(subset=[f"onset_{col}"])
    keep_idx = w.index.to_numpy()
    idx = (w[f"onset_{col}"].to_numpy() * g.FS).round().astype(int)
    n = st["acoustic"].shape[1]
    ok = idx < n
    R = np.zeros((2, n))
    np.add.at(R[0], idx[ok], rl[keep_idx][ok])
    np.add.at(R[1], idx[ok], rw[keep_idx][ok])
    st["repeat"] = g.bz(R)
    return st


def listener_version(sub: int, version: str, data: dict, stims: dict, seed: int) -> dict:
    songs = [s for s in c.song_ids() if (version, s) in data and sub in data[(version, s)] and data[(version, s)][sub][1].any()]
    if len(songs) < 6:
        return {}
    a = int(c.START_S * g.FS)
    rng = np.random.default_rng([seed, sub, list(g.AUDIO).index(version), 3])
    blocks, wmasks = [], []
    for s in songs:
        st = stims[(s, version)]
        Y, v = data[(version, s)][sub]
        T = min(Y.shape[1], st["acoustic"].shape[1] - a)
        X = {k: g.lagged(st[k][:, a:a + T], T) for k in ("acoustic", "phonetic", "word", "repeat", "surprisal") if k in st}
        m = np.zeros(T, bool)
        for i in st["word_idx"]:
            lo = i - a
            if lo + g.WORD_WIN > 0 and lo < T:
                m[max(lo, 0): min(lo + g.WORD_WIN, T)] = True
        blocks.append({"X": X, "Y": Y[:, :T], "v": v[:T], "st": st, "a": a, "T": T})
        wmasks.append(m & v[:T])
    base = [k for k in ("acoustic", "phonetic", "word", "repeat") if k in blocks[0]["X"]]
    _, lam = g.fit_predict(blocks, base, None)
    r_full = g.r_mean(blocks, g.fit_predict(blocks, base + ["surprisal"], lam)[0], wmasks)
    r_sh = []
    for _ in range(g.N_SHUF):
        for bl in blocks:
            st = bl["st"]
            raw = np.zeros_like(st["raw_surprisal"])
            raw[st["word_idx"]] = rng.permutation(st["raw_surprisal"][st["word_idx"]])
            bl["X"]["surprisal_shuf"] = g.lagged(g.bz(raw[None])[:, bl["a"]:bl["a"] + bl["T"]], bl["T"])
        r_sh.append(g.r_mean(blocks, g.fit_predict(blocks, base + ["surprisal_shuf"], lam)[0], wmasks))
    return {"listener": sub, "version": version, "delta_r_rep": r_full - float(np.mean(r_sh))}


def summary(df: pd.DataFrame, words: dict) -> dict:
    """df: listeners x versions (delta_r_rep). Repetition rates and their correlation with surprisal, and the tests."""
    rep = {s: repetition(s) for s in c.song_ids()}
    surp = np.concatenate([words[s].surprisal_bits.to_numpy() for s in c.song_ids()])
    rl = np.concatenate([rep[s][0] for s in c.song_ids()])
    rw = np.concatenate([rep[s][1] for s in c.song_ids()])
    pick = lambda x, alt="greater": {k: v for k, v in summarise(np.asarray(x, float), alt).items()
                                     if k in ("n", "mean", "dz", "dz_ci95", "p_signflip", "n_positive")}
    return {"words": int(surp.size), "frac_line_repeated": float(rl.mean()), "frac_word_repeated": float(rw.mean()),
            "corr_surprisal_line_repeated": float(np.corrcoef(surp, rl)[0, 1]),
            "corr_surprisal_word_repeated": float(np.corrcoef(surp, rw)[0, 1]),
            "spoken": pick(df["speech"].dropna()), "sung": pick(df["song"].dropna()),
            "hummed": pick(df["melody"].dropna(), "two-sided"),
            "sung_minus_hummed": pick((df["song"] - df["melody"]).dropna())}
