"""Regressors of the lexical, acoustic and note-level controls.

Words (CantiSong and Alice): word-onset impulses scaled by Zipf frequency (SUBTLEX-US), function or content word (Snowball
stop-word list), word duration, peak envelope within the word (64 Hz broadband envelope of the presented audio) and, for
sung and hummed versions, note duration (the aligned note sounding at the word onset, else the nearest note onset within
150 ms, else the song mean). Each value is z-scored within the song or segment over its words; the rows are then
band-passed and z-scored like every regressor.
Notes (melodic surprisal): note-onset impulses scaled by the absolute interval, the pitch height and the note duration,
z-scored within the song.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from readers import alice as h
from stimulus_features import english_lexicon as X, song_regressors as g


def zs(x: np.ndarray) -> np.ndarray:
    """z-score with missing values set to the mean."""
    x = np.asarray(x, float)
    x = np.where(np.isfinite(x), x, np.nanmean(x))
    sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else np.zeros_like(x)


def impulse_rows(idx: np.ndarray, vals: list[np.ndarray], n: int) -> np.ndarray:
    R = np.zeros((len(vals), n))
    ok = (idx >= 0) & (idx < n)
    for k, v in enumerate(vals):
        np.add.at(R[k], idx[ok], v[ok])
    return R


def note_duration_at(t: np.ndarray, notes: pd.DataFrame) -> np.ndarray:
    """Duration of the note sounding at each time, else of the nearest note onset within 150 ms, else NaN."""
    on, off = notes.onset_audio.to_numpy(), notes.offset_audio.to_numpy()
    out = np.full(t.size, np.nan)
    for i, x in enumerate(t):
        sounding = np.where((on <= x) & (x < off))[0]
        if sounding.size:
            out[i] = off[sounding[0]] - on[sounding[0]]
        else:
            j = int(np.argmin(np.abs(on - x)))
            if abs(on[j] - x) <= 0.15:
                out[i] = off[j] - on[j]
    return out


def canti_controls(song: str, version: str, words: pd.DataFrame, notes: pd.DataFrame, n: int) -> dict:
    """Raw control impulses (rows) for the words of song_regressors.lexical (same filter and order) and the Zipf value
    of each included word."""
    col = g.TIMING[version]
    w = words.dropna(subset=[f"onset_{col}"])
    idx = (w[f"onset_{col}"].to_numpy() * g.FS).round().astype(int)
    keep = idx < n
    w, idx = w[keep], idx[keep]
    env = g.acoustic(song, version)[0]
    on_s, off_s = w[f"onset_{col}"].to_numpy(), w[f"offset_{col}"].to_numpy()
    dur = off_s - on_s
    peak = np.array([env[int(round(a * g.FS)): max(int(round(b * g.FS)), int(round(a * g.FS)) + 1)].max()
                     if int(round(a * g.FS)) < env.size else np.nan for a, b in zip(on_s, off_s)])
    zipf = np.array([X.zipf(x)[0] for x in w.word])
    func = np.array([float(X.is_function(x)) for x in w.word])
    vals = [zs(zipf), zs(func), zs(dur), zs(peak)]
    if version in ("song", "melody"):
        vals.append(zs(note_duration_at(on_s, notes)))
    return {"raw": impulse_rows(idx, vals, n), "zipf": zipf, "idx": idx}


def alice_controls(seg: int, words: pd.DataFrame, n: int) -> dict:
    """The same controls for one Alice segment (no note duration)."""
    w = words[words.Segment == seg]
    idx = (w.onset.to_numpy() * 64).round().astype(int)
    keep = (idx >= 0) & (idx < n)
    w, idx = w[keep], idx[keep]
    env = h.acoustic(seg)[0]
    on_s, off_s = w.onset.to_numpy(), w.offset.to_numpy()
    peak = np.array([env[int(round(a * 64)): max(int(round(b * 64)), int(round(a * 64)) + 1)].max() for a, b in zip(on_s, off_s)])
    zipf = np.array([X.zipf(x)[0] for x in w.Word])
    func = np.array([float(X.is_function(x)) for x in w.Word])
    return {"raw": impulse_rows(idx, [zs(zipf), zs(func), zs(off_s - on_s), zs(peak)], n), "zipf": zipf, "idx": idx}


def note_controls(feats: dict, notes: dict, songs: list[str]) -> dict:
    """Per song: |interval|, pitch height and note duration as note-onset impulse rows (the notes of the melodic
    surprisal regressor)."""
    out = {}
    for s in songs:
        ff = feats[(s, "song")]
        nt = notes[s].dropna(subset=["melodic_surprisal_bits"])
        vals = [zs(np.abs(nt.interval.to_numpy())), zs(nt.pitch.to_numpy()), zs((nt.offset_audio - nt.onset_audio).to_numpy())]
        assert np.array_equal((nt.onset_audio.to_numpy() * 64).round().astype(int), ff["note_idx"])
        out[s] = impulse_rows(ff["note_idx"], vals, ff["n"])
    return out
