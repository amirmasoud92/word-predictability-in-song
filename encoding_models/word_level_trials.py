"""Trials, regressors and the within-unit permutation of surprisal for the word-level measure (CantiSong and Alice).

A trial is one song (CantiSong) or one audio segment (Alice): its regressors (band-passed and z-scored per trial), the
EEG, the validity mask of the cleaned 2 s epochs, the word onsets and the raw surprisal impulses. The permuted versions
of a regressor reassign its values among the words of each trial, with the word timing kept, and are band-passed and
z-scored like the true regressor.
"""
from __future__ import annotations

import os
import zlib

import numpy as np
import pandas as pd

from readers import alice as h, cantisong as c
from stimulus_features import song_regressors as g

N_LAGS, WIN_S, STREAM = 23, 0.8, 6                      # lags 0-688 ms; words evaluated 0-800 ms after onset
WINDOWS = {"W1_0-188ms": (0, 6), "W2_219-406ms": (7, 13), "W3_438-688ms": (14, 22)}   # lag windows (lag indices)
CANTI_ROI = ["Cz", "CPz", "Pz", "C1", "C2", "CP1", "CP2", "P1", "P2"]                 # centro-parietal region


def rng_for(seed: int, key: str, sub_stream: int = 0) -> np.random.Generator:
    """Random stream of one job (dataset|version|listener) and sub-stream."""
    return np.random.default_rng([seed, zlib.crc32(key.encode()), STREAM, sub_stream])


def perm_controls(trials: list[dict], raw_key: str, rng, n: int) -> list[dict]:
    """n permuted versions of a regressor: values permuted among the words of each trial, timing kept, then band-passed."""
    ctrls = []
    for _ in range(n):
        cd = {}
        for j, tr in enumerate(trials):
            raw = np.zeros_like(tr[raw_key])
            raw[tr["idx"]] = rng.permutation(tr[raw_key][tr["idx"]])
            cd[j] = g.bz(raw[None])[:, tr["a"]:tr["a"] + tr["T"]]
        ctrls.append(cd)
    return ctrls


def canti_trials(specs: list[dict], a: int) -> list[dict]:
    trials = []
    for sp in specs:
        st, Y, v = sp["st"], sp["Y"], sp["v"]
        T = min(Y.shape[1], st["acoustic"].shape[1] - a)
        F = {k: st[k][:, a:a + T] for k in ("acoustic", "phonetic", "word", "surprisal") if k in st}
        if sp.get("ent_raw") is not None:
            F["entropy"] = g.bz(sp["ent_raw"][None])[:, a:a + T]
        trials.append({"F": F, "Y": Y[:, :T], "v": v[:T], "T": T, "a": a, "words": [i - a for i in st["word_idx"]],
                       "idx": np.asarray(st["word_idx"]), "raw_s": st["raw_surprisal"], "raw_e": sp.get("ent_raw"), "song": sp["song"]})
    return trials


def alice_trials(specs: list[dict], a: int) -> list[dict]:
    trials = []
    for sp in specs:
        st, se, Y, v = sp["st"], sp["se"], sp["Y"], sp["v"]
        T = min(Y.shape[1], st["acoustic"].shape[1] - a)
        F = {k: st[k][:, a:a + T] for k in ("acoustic", "word", "surprisal")}
        F["entropy"] = se["surprisal"][:, a:a + T]
        trials.append({"F": F, "Y": Y[:, :T], "v": v[:T], "T": T, "a": a, "words": [i - a for i in st["word_idx"]],
                       "idx": np.asarray(st["word_idx"]), "raw_s": st["raw_surprisal"], "raw_e": se["raw_surprisal"], "seg": sp["seg"]})
    return trials


def canti_job_list(paths, released: bool = False) -> list[tuple]:
    """One job per CantiSong listener and version (at least 6 usable songs), with surprisal and lexical entropy.
    ``released`` uses the released annotation of song 09 instead of the corrected one, without entropy."""
    os.environ["CANTISONG_ANNOTATION"] = "released" if released else "corrected"
    data = c.load_all(paths.derived_root / "cantisong_1to8", "1-8")
    words = {s: g.load_words(s) for s in c.song_ids()}
    ent = {s: pd.read_csv(paths.derived_root / "cantisong_words" / f"{s}_entropy.csv") for s in c.song_ids()}
    a = int(c.START_S * 64)
    jobs = []
    for v in g.AUDIO:
        st = {s: g.stimulus(s, v, words[s], False) for s in c.song_ids()}
        er = {}
        for s in c.song_ids():
            we = words[s].copy()
            assert (we.idx.to_numpy() == ent[s].idx.to_numpy()).all()
            we["surprisal_bits"] = ent[s].entropy_bits.to_numpy()
            _, er[s], idx = g.lexical(we, v, st[s]["acoustic"].shape[1], False)
            assert idx == st[s]["word_idx"]
        subs = sorted({sub for (vv, s), d in data.items() if vv == v for sub in d})
        for sub in subs:
            songs = [s for s in c.song_ids() if (v, s) in data and sub in data[(v, s)] and data[(v, s)][sub][1].any()]
            if len(songs) < 6:
                continue
            specs = [{"st": st[s], "ent_raw": None if released else er[s], "Y": data[(v, s)][sub][0], "v": data[(v, s)][sub][1], "song": s}
                     for s in songs]
            jobs.append(("canti", f"canti|{v}|{sub}", specs, a))
    return jobs


def alice_job_list(paths) -> list[tuple]:
    """One job per Alice listener (at least 8 usable segments), with GPT-2 surprisal and lexical entropy."""
    words = pd.read_csv(paths.derived_root / "alice" / "words_gpt2_entropy.csv")
    st = {s: h.seg_stim(s, words, "gpt2_surprisal_bits") for s in h.SEGMENTS}
    se = {s: h.seg_stim(s, words, "gpt2_entropy_bits") for s in h.SEGMENTS}
    data8, _ = h.load_all(paths.derived_root / "alice", "1-8")
    a = int(h.START_S * 64)
    jobs = []
    for sub in sorted(data8):
        segs = [s for s in h.SEGMENTS if s in data8[sub] and data8[sub][s][1].any()]
        if len(segs) < h.MIN_SEGMENTS:
            continue
        jobs.append(("alice", f"alice|speech|{sub}", [{"st": st[s], "se": se[s], "Y": data8[sub][s][0], "v": data8[sub][s][1], "seg": s}
                                                      for s in segs], a))
    return jobs
