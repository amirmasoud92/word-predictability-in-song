"""Stimulus regressors per chapter and stream at 64 Hz for the MEG attention dataset.

Per chapter of each recording (LP: book 1 attended, book 4 ignored; HP: book 3 attended, book 2 ignored), per stream:
* acoustic (34 rows): broadband envelope and its onsets, 16-band log-spaced spectrogram and its onsets;
* word onsets (impulses at the aligned onsets) and four controls (impulses scaled by Zipf frequency, function word,
  duration and peak envelope, each z-scored over the chapter's words);
* raw surprisal values (gerpt2, gerpt2-large) with the onset indices, from which true and permuted regressors are built;
* text-matched keep flags: within each recording, the words of both streams are kept with probability
  min(f_att, f_ign) / f_own (gerpt2 surprisal histogram over the recording, 1-bit bins); then, per chapter, the stream
  with more kept words is thinned at random to the other's count (one seeded draw).
Length = the MEG chapter at 64 Hz (ceiling of the 1000 Hz length x 8/125); regressors are band-passed when they are used.
"""
from __future__ import annotations

import math
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.signal as ss
from scipy.io import wavfile

from common.settings import dataset_root, load_analysis, load_paths
from stimulus_features import speech_envelope as tk

AUDIO = dataset_root("meg_attention") / "Audio"
TAB = load_paths().results_root / "meg_attention"
OUT = load_paths().derived_root / "meg_attention" / "features"
FILES = {"LP": {"attended": 1, "ignored": 4}, "HP": {"attended": 3, "ignored": 2}}
FS = 64


def rng(key: str) -> np.random.Generator:
    return np.random.default_rng([7, zlib.crc32(key.encode())])


def wav_path(book: int, k: int) -> Path:
    sub = "attended_speech" if book in (1, 3) else "ignored_speech"
    return AUDIO / sub / f"audiobook{book}_{k}.wav"


def acoustic(path: Path, n: int) -> np.ndarray:
    fs, x = wavfile.read(path)
    x = np.asarray(x, float)
    if x.dtype.kind != "f" or np.abs(x).max() > 1.5:
        x = x / 32768.0
    x = x.mean(1) if x.ndim > 1 else x
    e1k = tk.broadband_envelope(x, fs, load_analysis().section("envelope"))
    env = ss.resample_poly(e1k, 64, 1000)
    y = ss.resample_poly(x, 16000, fs)
    f, _, Z = ss.stft(y, 16000, nperseg=512, noverlap=512 - 250, boundary=None, padded=False)
    P = np.abs(Z) ** 2
    edges = 700 * (10 ** (np.linspace(np.log10(1 + 100 / 700), np.log10(1 + 7000 / 700), 18)) - 1)
    bands = np.array([P[(f >= lo) & (f < hi)].sum(0) for lo, hi in zip(edges[:-2], edges[2:])]) ** 0.3
    m = min(env.size, bands.shape[1], n)
    A = np.zeros((18, n))
    A[0, :m], A[1:17, :m] = env[:m], bands[:, :m]
    on = np.maximum(np.diff(A, axis=-1, prepend=A[:, :1]), 0)
    return np.vstack([A[:1], on[:1], A[1:17], on[1:17]])


def zs(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    x = np.where(np.isfinite(x), x, np.nanmean(x))
    sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else np.zeros_like(x)


def impulses(idx: np.ndarray, vals: list[np.ndarray], n: int) -> np.ndarray:
    R = np.zeros((len(vals), n))
    ok = (idx >= 0) & (idx < n)
    for k, v in enumerate(vals):
        np.add.at(R[k], idx[ok], np.asarray(v, float)[ok])
    return R


def matched_flags(w: pd.DataFrame) -> pd.Series:
    keep = pd.Series(False, index=w.index)
    for f, books in FILES.items():
        A = w[w.book == books["attended"]]
        I = w[w.book == books["ignored"]]
        edges = np.arange(0, max(A.surprisal_gerpt2.max(), I.surprisal_gerpt2.max()) + 2, 1.0)
        ha, _ = np.histogram(A.surprisal_gerpt2, edges, density=True)
        hi, _ = np.histogram(I.surprisal_gerpt2, edges, density=True)
        m = np.minimum(ha, hi)
        g = rng(f"p7|matched|{f}")
        for D, h in ((A, ha), (I, hi)):
            b = np.clip(np.digitize(D.surprisal_gerpt2, edges) - 1, 0, len(h) - 1)
            p = np.where(h[b] > 0, m[b] / np.where(h[b] > 0, h[b], 1), 0)
            keep.loc[D.index] = g.random(len(D)) < p
        for k in range(1, 6):
            ia = A.index[(A.chapter == k) & keep.loc[A.index]]
            ii = I.index[(I.chapter == k) & keep.loc[I.index]]
            big, small = (ia, ii) if len(ia) > len(ii) else (ii, ia)
            drop = g.choice(big, len(big) - len(small), replace=False)
            keep.loc[drop] = False
    return keep


def chapter_regressors():
    OUT.mkdir(parents=True, exist_ok=True)
    w = pd.read_csv(TAB / "word_features.csv")
    w = w[w.aligned].copy()
    w["keep_matched"] = matched_flags(w)
    a = pd.read_csv(TAB / "chapter_audio.csv")
    counts = []
    for f, books in FILES.items():
        for k in range(1, 6):
            n1000 = int(a[(a.book == f"audiobook{books['attended']}") & (a.chapter == k)].n1000_ceil.iloc[0])
            n = math.ceil(n1000 * 8 / 125)
            arrays = {"n": np.array(n)}
            for s, book in books.items():
                g = w[(w.book == book) & (w.chapter == k)].sort_values("word_index")
                idx = np.round(g.onset.to_numpy() * FS).astype(int)
                arrays[f"{s}_acoustic_raw"] = acoustic(wav_path(book, k), n)
                arrays[f"{s}_word_raw"] = impulses(idx, [np.ones(len(g))], n)
                arrays[f"{s}_controls_raw"] = impulses(idx, [zs(g.zipf), zs(g.function.astype(float)), zs(g.duration), zs(g.peak_env)], n)
                arrays[f"{s}_onset_idx"] = idx
                arrays[f"{s}_surprisal"] = g.surprisal_gerpt2.to_numpy()
                arrays[f"{s}_surprisal_large"] = g.surprisal_gerpt2_large.to_numpy()
                arrays[f"{s}_keep_matched"] = g.keep_matched.to_numpy()
                counts.append({"file": f, "chapter": k, "stream": s, "book": book, "n_words": len(g),
                               "n_kept_matched": int(g.keep_matched.sum()), "n_samples_64": n})
            np.savez(OUT / f"{f}_{k}.npz", **arrays)
    c = pd.DataFrame(counts)
    c.to_csv(TAB / "matched_counts.csv", index=False)
