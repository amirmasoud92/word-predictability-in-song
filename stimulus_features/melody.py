"""Note timings and melodic surprisal of the CantiSong melodies, and the stimulus-only audit of two note sources.

* Note timing: the composed MIDI note sequence is aligned to the audio timeline by dynamic time warping (50 ms frames) of
  MIDI pitch to the pYIN f0 contour (65-1000 Hz, 10 ms frames) of the hummed version (resynthesised on the sung
  timeline), after removing the median pitch of each. Cost: pitch distance in semitones folded to the nearest octave;
  unvoiced frames against rests cost 0, voiced frames against rests 3 semitones.
* Melodic surprisal (the IDyOM principle; Pearce, 2005): prediction by partial matching with escape method C, orders 0-4,
  over chromatic pitch intervals (alphabet -24..+24); a short-term model updated within the song and a long-term model
  trained on the other 17 songs, combined by an entropy-weighted geometric mean (bias 1).
* Notes segmented directly from the f0 of the hummed version (used only in the audit): pYIN f0 (10 ms), median filtered
  (50 ms), semitone-quantised after removing the song's tuning offset; a note starts at a voicing onset or where the
  quantised pitch changes and the new pitch persists >= 60 ms; notes shorter than 80 ms are merged into the preceding
  note.
* Audit of the two sources (Fig. S4a, b): frame-wise agreement of each source's pitch with the hummed f0 after a
  song-wise transposition (+/- 0.5 semitone), similarity of the interval sequences, and the share of sung words within
  150 ms of a note onset, raw and minus the mean over 50 random circular shifts (2-10 s) of the word onsets.

Requires librosa and mido.
"""
from __future__ import annotations

import difflib
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.signal as ss

HOP = 0.05
ALPHABET = list(range(-24, 25))
F0_HOP, PAIR_S, N_SHIFT = 0.01, 0.15, 50


# ---------------------------------------------------------------- composed notes aligned to the audio
def midi_notes(path: Path) -> pd.DataFrame:
    import mido
    t, notes, on = 0.0, [], {}
    for msg in mido.MidiFile(path):
        t += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            on[msg.note] = t
        elif msg.type in ("note_off", "note_on") and msg.note in on:
            notes.append((on.pop(msg.note), t, msg.note))
    df = pd.DataFrame(sorted(notes), columns=["onset", "offset", "pitch"])
    return df.drop_duplicates("onset").reset_index(drop=True)                    # melody: one note per onset


def f0_track(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """pYIN f0 in semitones, medians over 50 ms frames, and the frame times."""
    import librosa
    y, sr = librosa.load(path, sr=16000, mono=True)
    f0, voiced, _ = librosa.pyin(y, fmin=65, fmax=1000, sr=sr, hop_length=int(0.01 * sr), frame_length=1024)
    semi = np.where(voiced, librosa.hz_to_midi(np.where(voiced, f0, 440.0)), np.nan)
    step = int(HOP / 0.01)
    n = len(semi) // step
    frames = np.array([np.nanmedian(semi[i * step:(i + 1) * step]) if np.isfinite(semi[i * step:(i + 1) * step]).any() else np.nan
                       for i in range(n)])
    return frames, np.arange(n) * HOP


def midi_frames(notes: pd.DataFrame) -> np.ndarray:
    n = int(np.ceil(notes.offset.max() / HOP)) + 1
    fr = np.full(n, np.nan)
    for r in notes.itertuples():
        fr[int(r.onset / HOP): max(int(r.offset / HOP), int(r.onset / HOP) + 1)] = r.pitch
    return fr


def align(notes: pd.DataFrame, audio_f0: np.ndarray, audio_t: np.ndarray) -> pd.DataFrame:
    """Onset and offset of each MIDI note on the audio timeline (onset_audio, offset_audio)."""
    import librosa
    mf = midi_frames(notes)
    a = audio_f0 - np.nanmedian(audio_f0)
    m = mf - np.nanmedian(mf)
    A, M = a[:, None], m[None, :]
    d = np.abs(A - M)
    d = np.minimum(d % 12, 12 - d % 12)
    both_nan = np.isnan(A) & np.isnan(M)
    one_nan = np.isnan(A) ^ np.isnan(M)
    C = np.where(both_nan, 0.0, np.where(one_nan, 3.0, d))
    _, wp = librosa.sequence.dtw(C=C, subseq=False, backtrack=True)
    wp = wp[::-1]
    m2a = {}
    for ia, im in wp:
        m2a.setdefault(im, ia)
    out = notes.copy()
    out["onset_audio"] = [audio_t[m2a.get(int(r.onset / HOP), 0)] for r in notes.itertuples()]
    out["offset_audio"] = [audio_t[m2a.get(min(int(r.offset / HOP), len(mf) - 1), 0)] for r in notes.itertuples()]
    return out


# ---------------------------------------------------------------- melodic surprisal
class PPM:
    """PPM-C over a sequence of symbols, orders 0..max_order, with interpolated escape (Moffat method C)."""

    def __init__(self, max_order: int = 4):
        self.k = max_order
        self.counts = defaultdict(lambda: defaultdict(int))

    def update(self, seq: list[int], i: int):
        for n in range(0, self.k + 1):
            if i - n < 0:
                break
            self.counts[tuple(seq[i - n:i])][seq[i]] += 1

    def dist(self, seq: list[int], i: int) -> np.ndarray:
        p = np.full(len(ALPHABET), 1.0 / len(ALPHABET))
        for n in range(0, min(self.k, i) + 1):
            ctx = tuple(seq[i - n:i])
            c = self.counts.get(ctx)
            if not c:
                continue
            T, U = sum(c.values()), len(c)
            probs = np.array([c.get(s, 0) for s in ALPHABET], float) / (T + U)
            p = probs + (U / (T + U)) * p
        return p / p.sum()


def entropy(p):
    return float(-(p * np.log2(p + 1e-300)).sum())


def melodic_surprisal(target: list[int], corpus: list[list[int]]) -> list[float]:
    """Surprisal (bits) of each interval of ``target`` under the short-term model of the song and the long-term model of
    ``corpus`` (the other songs), combined by entropy weighting."""
    ltm = PPM()
    for seq in corpus:
        for i in range(len(seq)):
            ltm.update(seq, i)
    stm = PPM()
    out = []
    hmax = np.log2(len(ALPHABET))
    for i, sym in enumerate(target):
        ps = [stm.dist(target, i), ltm.dist(target, i)]
        w = [(entropy(p) / hmax) ** -1.0 for p in ps]
        comb = np.exp(sum(wi * np.log(p + 1e-300) for wi, p in zip(w, ps)) / sum(w))
        comb /= comb.sum()
        out.append(float(-np.log2(comb[ALPHABET.index(sym)])))
        stm.update(target, i)
    return out


def intervals_of(pitch: np.ndarray) -> list[int]:
    return [int(np.clip(v, -24, 24)) for v in np.diff(np.asarray(pitch))]


def note_tables(stim: Path, out: Path):
    """<song>_notes.csv per song: MIDI onset, offset and pitch, the aligned onset_audio and offset_audio, the melodic
    surprisal of each note (none for the first) and its interval."""
    out.mkdir(parents=True, exist_ok=True)
    songs = sorted(p.stem for p in (stim / "MIDI").glob("*.mid"))
    notes = {s: midi_notes(stim / "MIDI" / f"{s}.mid") for s in songs}
    ints = {s: intervals_of(n.pitch.to_numpy()) for s, n in notes.items()}
    for s in songs:
        f0, t = f0_track(stim / "sounds" / f"{s}_melody.wav")
        al = align(notes[s], f0, t)
        al["melodic_surprisal_bits"] = [np.nan] + melodic_surprisal(ints[s], [ints[q] for q in songs if q != s])
        al["interval"] = [np.nan] + ints[s]
        al.to_csv(out / f"{s}_notes.csv", index=False)


# ---------------------------------------------------------------- notes segmented from f0 (audit only)
def segment_f0(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    """Notes segmented from the hummed f0, and the f0 in semitones at 10 ms."""
    import librosa
    y, sr = librosa.load(path, sr=16000, mono=True)
    f0, voiced, _ = librosa.pyin(y, fmin=65, fmax=1000, sr=sr, hop_length=int(F0_HOP * sr), frame_length=1024)
    semi = np.where(voiced, librosa.hz_to_midi(np.where(voiced, f0, 440.0)), np.nan)
    sm = semi.copy()
    ok = np.isfinite(semi)
    sm[ok] = ss.medfilt(np.nan_to_num(semi), 5)[ok]
    tuning = np.nanmedian(sm - np.round(sm))
    q = np.where(ok, np.round(sm - tuning), np.nan)
    min_hold, min_len = int(0.06 / F0_HOP), int(0.08 / F0_HOP)
    notes, start, cur = [], None, np.nan
    i = 0
    while i < len(q):
        if not np.isfinite(q[i]):
            if start is not None:
                notes.append([start, i, cur])
                start = None
            i += 1
            continue
        if start is None:
            start, cur = i, q[i]
        elif q[i] != cur and np.all(q[i:i + min_hold] == q[i]):
            notes.append([start, i, cur])
            start, cur = i, q[i]
        i += 1
    if start is not None:
        notes.append([start, len(q), cur])
    merged = []
    for n in notes:
        if merged and n[1] - n[0] < min_len and n[0] == merged[-1][1]:
            merged[-1][1] = n[1]
        elif n[1] - n[0] >= min_len:
            merged.append(n)
    df = pd.DataFrame([{"onset_audio": a * F0_HOP, "offset_audio": b * F0_HOP, "pitch": int(p)} for a, b, p in merged])
    return df, semi


def f0_note_tables(stim: Path, out: Path):
    """<song>_notes.csv (f0-segmented notes with melodic surprisal) and <song>_f0_semitones_10ms.npy per song."""
    out.mkdir(parents=True, exist_ok=True)
    songs = sorted(p.stem for p in (stim / "MIDI").glob("*.mid"))
    notes = {}
    for s in songs:
        notes[s], semi = segment_f0(stim / "sounds" / f"{s}_melody.wav")
        np.save(out / f"{s}_f0_semitones_10ms.npy", semi.astype(np.float32))
    ints = {s: intervals_of(n.pitch.to_numpy()) for s, n in notes.items()}
    for s in songs:
        n = notes[s]
        n["melodic_surprisal_bits"] = [np.nan] + melodic_surprisal(ints[s], [ints[q] for q in songs if q != s])
        n["interval"] = [np.nan] + ints[s]
        n.to_csv(out / f"{s}_notes.csv", index=False)


# ---------------------------------------------------------------- audit of the two note sources
def _labels(notes: pd.DataFrame, n: int) -> np.ndarray:
    t = np.arange(n) * F0_HOP
    lab = np.full(n, np.nan)
    for _, r in notes.iterrows():
        lab[(t >= r.onset_audio) & (t < r.offset_audio)] = r.pitch
    return lab


def _agree(semi: np.ndarray, lab: np.ndarray) -> tuple[float, float]:
    ok = np.isfinite(lab) & np.isfinite(semi)
    tr = np.median(semi[ok] - lab[ok])
    return float(np.mean(np.abs(semi[ok] - lab[ok] - tr) <= 0.5)), float(ok.sum() / np.isfinite(semi).sum())


def _intervals(pitch: np.ndarray) -> list[int]:
    return list(np.clip(np.diff(pitch), -24, 24))


def _near(onsets: np.ndarray, words: np.ndarray) -> float:
    return float(np.mean([np.min(np.abs(onsets - v)) <= PAIR_S for v in words]))


def note_source_audit(midi_dir: Path, f0_dir: Path, words_dir: Path, seed: int | None = None) -> pd.DataFrame:
    """Per song: pitch agreement and coverage of both sources, interval-sequence similarity, and the coincidence of sung
    word onsets (released annotation) with note onsets, raw and above chance (``seed``: the chance-level shifts)."""
    songs = sorted(p.name.split("_")[0] for p in midi_dir.glob("*_notes.csv"))
    rng = np.random.default_rng(seed)
    rows = []
    for s in songs:
        semi = np.load(f0_dir / f"{s}_f0_semitones_10ms.npy").astype(float)
        dtw = pd.read_csv(midi_dir / f"{s}_notes.csv")
        f0n = pd.read_csv(f0_dir / f"{s}_notes.csv")
        w = pd.read_csv(words_dir / f"{s}_words.csv").onset_song.dropna().to_numpy()

        def coincidence(on: np.ndarray) -> tuple[float, float]:
            obs, lo, T = _near(on, w), w.min(), max(on.max(), w.max())
            null = np.mean([_near(on, (w - lo + rng.uniform(2, 10)) % (T - lo) + lo) for _ in range(N_SHIFT)])
            return obs, obs - null

        a1, c1 = _agree(semi, _labels(dtw, len(semi)))
        a2, c2 = _agree(semi, _labels(f0n, len(semi)))
        sm = difflib.SequenceMatcher(None, _intervals(dtw.pitch.to_numpy()), _intervals(f0n.pitch.to_numpy()), autojunk=False).ratio()
        o1, k1 = coincidence(dtw.onset_audio.to_numpy())
        o2, k2 = coincidence(f0n.onset_audio.to_numpy())
        rows.append({"song": s, "n_midi": len(dtw), "n_f0": len(f0n), "dtw_pitch_agree": a1, "dtw_cover": c1,
                     "f0_pitch_agree": a2, "f0_cover": c2, "interval_seq_match": sm, "dtw_words": o1, "dtw_words_cc": k1,
                     "f0_words": o2, "f0_words_cc": k2})
    return pd.DataFrame(rows)
