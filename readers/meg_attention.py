"""MEG attention dataset (Riegel et al., 2024; Schüller et al., 2023, 2025): listener chapters and per-chapter regressors in
the trial format of the encoding models.

Each listener has two recordings of five chapters: LP (book 1 attended, book 4 ignored; lower-pitched talker attended)
and HP (book 3 attended, book 2 ignored; higher-pitched talker attended). Preprocessed chapters and per-chapter features
are read from the derived-data folder (preprocessing/meg_attention.py, stimulus_features/meg_attention_regressors.py)."""
from __future__ import annotations

import json
import math
import re
import zlib

import numpy as np
import pandas as pd

from common.settings import dataset_root, load_paths
from preprocessing.cnd_recordings import band_64

DATA = dataset_root("meg_attention")               # the release folder with Audio/ and one folder per listener
DER = load_paths().derived_root / "meg_attention"
TAB = load_paths().results_root / "meg_attention"
FILES = {"LP": {"attended": 1, "ignored": 4}, "HP": {"attended": 3, "ignored": 2}}
CHAPTERS = [(f, k) for f in FILES for k in range(1, 6)]
FS, N_LAGS, WIN_S = 64, 23, 0.8
STREAMS = ("attended", "ignored")


def chapter_audio() -> pd.DataFrame:
    """Length of each chapter's audio in samples at the MEG rate (1000 Hz, ceiling); the chapter boundaries of the
    recordings are the cumulative lengths. Written to chapter_audio.csv."""
    from scipy.io import wavfile
    recs = []
    for sub in ("attended_speech", "ignored_speech"):
        for f in sorted((DATA / "Audio" / sub).glob("*.wav")):
            sr, y = wavfile.read(f)
            m = re.fullmatch(r"(audiobook\d)_(\d)", f.stem)
            recs.append({"book": m.group(1), "chapter": int(m.group(2)), "frames": y.shape[0], "samplerate": sr,
                         "n1000_ceil": math.ceil(y.shape[0] * 1000 / sr)})
    d = pd.DataFrame(recs)
    TAB.mkdir(parents=True, exist_ok=True)
    d.to_csv(TAB / "chapter_audio.csv", index=False)
    return d


def _num(s: str):
    s = str(s).strip()
    if s in ("", "-", "N/A", "not answered", "None"):
        return np.nan
    return float(s.replace(",", "."))


def participants() -> pd.DataFrame:
    """Comprehension (proportion of 30 questions answered correctly), from the participant overview of the release."""
    import pdfplumber
    pdf_path = next(f for f in (DATA / "Participants_Overview.pdf", DATA.parent / "Participants_Overview.pdf") if f.exists())
    with pdfplumber.open(pdf_path) as pdf:
        rows = [r for t in pdf.pages[0].extract_tables() for r in t]
    cols = ["participant", "age", "sex", "starting_age", "years_training", "instruments_past", "instruments_now",
            "training_hours_week", "classification", "correct_frau_ella", "correct_den_hund", "zenodo_link"]
    recs = []
    for r in rows:
        r = [("" if c is None else str(c).replace("\n", " ").strip()) for c in r]
        if r and re.fullmatch(r"\d+", r[0] or ""):
            recs.append(dict(zip(cols, r + [""] * (len(cols) - len(r)))))
    d = pd.DataFrame(recs)
    d["participant"] = d.participant.astype(int)
    d = d.sort_values("participant").reset_index(drop=True)
    for c in ("correct_frau_ella", "correct_den_hund"):
        d[c] = d[c].map(_num)
    d["correct_total"] = d.correct_frau_ella + d.correct_den_hund
    d["comprehension"] = d.correct_total / 30.0
    return d[["participant", "correct_total", "comprehension"]]


def rng(key: str) -> np.random.Generator:
    return np.random.default_rng([7, zlib.crc32(key.encode())])


def bz(F: np.ndarray) -> np.ndarray:
    """1-8 Hz on the 64 Hz grid (band_64), then z-scored per row over the chapter."""
    B = band_64(np.atleast_2d(F), FS, 1.0, 8.0)
    return (B - B.mean(1, keepdims=True)) / (B.std(1, keepdims=True) + 1e-12)


def qc(p: int) -> dict:
    return json.loads((TAB / "qc" / f"P{p:02d}.json").read_text(encoding="utf-8"))


def kept_chapters(p: int) -> list[tuple[str, int]]:
    q = qc(p)
    return [(c["file"], c["chapter"]) for c in q["chapters"] if c["kept"]]


def included(p: int) -> bool:
    return bool(qc(p)["included"])


def features(f: str, k: int) -> dict:
    with np.load(DER / "features" / f"{f}_{k}.npz") as z:
        return {key: z[key] for key in z.files}


def surprisal_impulse(st: dict, stream: str, values: np.ndarray, keep: np.ndarray | None, n: int) -> np.ndarray:
    """Raw impulse train (1 x n) at the stream's onsets with the given values; words outside `keep` get 0."""
    idx = st[f"{stream}_onset_idx"]
    v = np.asarray(values, float).copy()
    if keep is not None:
        v = np.where(keep, v, 0.0)
    R = np.zeros((1, n))
    ok = (idx >= 0) & (idx < n)
    np.add.at(R[0], idx[ok], v[ok])
    return R


def listener_trials(p: int, band: str = "d18") -> list[dict]:
    """Kept chapters of a listener: {'file', 'chapter', 'Y' (channels x T, float64), 'v', 'T', 'F': {}, 'st'}."""
    out = []
    with np.load(DER / "meg" / f"P{p:02d}.npz") as z:
        for f, k in kept_chapters(p):
            Y = z[f"{f}_{k}_{band}"].astype(np.float64)
            v = z[f"{f}_{k}_valid"]
            st = features(f, k)
            T = min(Y.shape[1], int(st["n"]))
            out.append({"file": f, "chapter": k, "Y": Y[:, :T], "v": v[:T].copy(), "T": T, "F": {}, "st": st})
    return out


def add_base(trials: list[dict]) -> list[str]:
    """Base regressors (both streams): acoustic (34), word onsets (1), controls (4), band-passed and z-scored."""
    names = []
    for s in STREAMS:
        for part in ("acoustic", "word", "controls"):
            nm = f"{s}_{part}"
            for tr in trials:
                tr["F"][nm] = bz(tr["st"][f"{s}_{part}_raw"][:, : tr["T"]])
            names.append(nm)
    return names


def word_masks(trials: list[dict], stream: str, keep_key: str | None = None) -> list[np.ndarray]:
    """0-800 ms after each word of the stream (optionally kept words only), within the valid samples."""
    w = int(round(WIN_S * FS))
    out = []
    for tr in trials:
        idx = tr["st"][f"{stream}_onset_idx"]
        if keep_key is not None:
            idx = idx[tr["st"][f"{stream}_{keep_key}"]]
        m = np.zeros(tr["T"], bool)
        for i in idx:
            if i + w > 0 and i < tr["T"]:
                m[max(i, 0): min(i + w, tr["T"])] = True
        out.append(m & tr["v"])
    return out
