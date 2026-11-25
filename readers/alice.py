"""Alice release (Brennan & Hale, 2019): EEG preparation, the 12 audio segments, word regressors and comprehension scores.

The EEG pipeline is that of the CantiSong data (readers/cantisong.py), with a 0.5-40 Hz band before the ICA, the
release's electrode positions and the anterior axis taken from the channels that follow the vertical EOG.
"""
from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import scipy.io.wavfile as wavfile
import scipy.signal as ss

from common.settings import dataset_root, load_analysis, rng as make_rng
from preprocessing import cnd_recordings as cnd, signal_operations as pp
from preprocessing.epoch_rejection import reject
from preprocessing.epochs import Trial
from stimulus_features import song_regressors as g, speech_envelope as tk

ROOT = dataset_root("alice")
FS = 128.0
SEGMENTS = range(1, 13)
START_S, END_TRIM_S, MAX_BAD, MIN_SEGMENTS = 1.0, 1.0, 1 / 3, 8
SURPRISAL_COLS = {"gpt2": "gpt2_surprisal_bits", "rnn": "RNN", "ngram": "NGRAM", "cfg": "CFG"}


def wav_path(seg: int) -> Path:
    return ROOT / "audio" / f"DownTheRabbitHoleFinal_SoundFile{seg}.wav"


def seg_durations() -> dict[int, float]:
    out = {}
    for s in SEGMENTS:
        fs, x = wavfile.read(wav_path(s))
        out[s] = len(x) / fs
    return out


def load_raw(sub: str) -> tuple[np.ndarray, list[str], np.ndarray, dict[int, int]]:
    """EEG (channels x samples, µV, 128 Hz), channel labels, VEOG (µV), segment onset samples at 128 Hz."""
    raw = mne.io.read_raw_brainvision(ROOT / f"{sub}.vhdr", preload=True, verbose="error")
    eeg = [c for c in raw.ch_names if c.isdigit()]
    X = raw.get_data(picks=eeg) * 1e6
    veog = raw.get_data(picks=["VEOG"])[0] * 1e6
    fr = Fraction(int(FS), int(raw.info["sfreq"]))
    X = ss.resample_poly(X, fr.numerator, fr.denominator, axis=1)
    veog = ss.resample_poly(veog, fr.numerator, fr.denominator)
    ann = mne.events_from_annotations(raw, verbose="error")
    events, ids = ann
    onsets = {}
    import re
    for code, val in ids.items():
        m = re.search(r"(\d+)\s*$", code.split("/")[-1])      # "1" (S01-S20) or "S  1" (S21-S49)
        tag = m.group(1) if m else ""
        if tag.isdigit() and int(tag) in SEGMENTS:
            samp = events[events[:, 2] == val, 0]
            if samp.size:
                onsets[int(tag)] = int(round(samp[0] * FS / raw.info["sfreq"]))
    return X, eeg, veog, onsets


def montage_info(labels: list[str], axis_flip: dict) -> mne.Info:
    """Head-frame montage from the supplied sfp (cm on a sphere); anterior axis oriented to +y."""
    rows = [ln.split() for ln in (ROOT / "easycapM10-acti61_elec.sfp").read_text().splitlines() if ln.strip()]
    pos = {r[0]: np.array([float(v) for v in r[1:4]]) for r in rows}
    ant, sign = axis_flip["anterior_axis"], axis_flip["anterior_sign"]
    lat = 1 - ant if ant in (0, 1) else 0
    ch_pos = {}
    for c in labels:
        p = pos[c]
        ch_pos[c] = np.array([p[lat], sign * p[ant], p[2]]) / 100.0
    info = mne.create_info(labels, FS, "eeg")
    info.set_montage(mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame="head"))
    return info


def anterior_axis(subs: list[str]) -> dict:
    """Axis (x or y of the sfp) and sign along which channels correlate most with VEOG (blinks, 1-4 Hz)."""
    rows = [ln.split() for ln in (ROOT / "easycapM10-acti61_elec.sfp").read_text().splitlines() if ln.strip()]
    pos = {r[0]: np.array([float(v) for v in r[1:4]]) for r in rows}
    rs = []
    for s in subs:
        X, labels, veog, _ = load_raw(s)
        Xf = cnd.band_64(X - X.mean(0, keepdims=True), FS, 1.0, 4.0)
        vf = cnd.band_64(veog[None], FS, 1.0, 4.0)[0]
        rs.append({c: abs(np.corrcoef(Xf[i], vf)[0, 1]) for i, c in enumerate(labels)})
    r = pd.DataFrame(rs).mean()
    top = r.sort_values(ascending=False).index[:6]
    centroid = np.mean([pos[c] for c in top], axis=0)                    # blinks dominate the frontal channels
    ant = int(np.argmax(np.abs(centroid[:2])))
    best = {0: float(centroid[0]), 1: float(centroid[1])}
    return {"anterior_axis": ant, "anterior_sign": int(np.sign(centroid[ant])), "top6_centroid_xy_cm": best,
            "top_veog_channels": r.sort_values(ascending=False).index[:6].tolist()}


def prepare_one(sub: str, axis: dict, cfg2: dict, seed: int, cache: Path) -> str:
    target = cache / f"{sub}.npz"
    if target.exists():
        return target.name
    X, labels, veog, onsets = load_raw(sub)
    dur = seg_durations()
    info = montage_info(labels, axis)
    hp = pp.fir_kernel(0.5, FS, cnd.fir_order(FS, 1.0), True)
    lp = pp.fir_kernel(40.0, FS, cnd.fir_order(FS, 5.0), False)
    items = []
    for s in SEGMENTS:
        if s not in onsets:
            continue
        a = onsets[s]
        n = int(dur[s] * FS)
        if a + n > X.shape[1]:
            continue
        items.append((s, X[:, a:a + n]))
    ys = [pp.robust_detrend(x, 3, 3.0) for _, x in items]
    broad_raw = [pp.fir_filter(pp.fir_filter(y, hp), lp) for y in ys]
    a0 = int(START_S * FS)
    n_ep = [int((x.shape[1] / FS - START_S - END_TRIM_S) // 2) for _, x in items]
    wins = [(a0, a0 + n * int(2 * FS)) for n in n_ep]
    bads = pp.bad_channels([bb[:, s:e] for bb, (s, e) in zip(broad_raw, wins)], labels, 5.0)
    op = cnd.spatial_operator(info, bads)
    rank = int((np.linalg.svd(op, compute_uv=False) > 1e-8).sum())
    fit = np.concatenate([(op @ bb[:, s:e])[:, ::2] for bb, (s, e) in zip(broad_raw, wins)], axis=1)
    info64 = info.copy()
    with info64._unlock():
        info64["sfreq"] = 64.0
    proj, removed = cnd.ocular_ica(fit, info64, min(rank, 40), 4, 3.0, 3, int(make_rng(seed, "alice", sub).integers(2**31)))
    full = proj @ op
    trials = [Trial(subject=sub, condition="seg", excerpt=s, label=f"seg{s}", sfreq=FS, t0=START_S, data=(full @ bb)[:, s0:e0])
              for (s, _), bb, (s0, e0) in zip(items, broad_raw, wins)]
    res = reject(trials, labels, 2.0, seed=seed, max_bad_fraction=cfg2["max_bad_epoch_fraction"], method=cfg2["epoch_rejection"],
                 z_threshold=cfg2["faster_z"], channel_robust_z=cfg2["channel_robust_z"])
    arrays, meta = {}, []
    a64 = int(START_S * 64)
    for (s, _), y, t, n in zip(items, ys, trials, n_ep):
        ok = res.epoch_ok[t.key]
        excluded = bool((~ok).mean() > MAX_BAD)
        arrays[f"valid|{s}"] = np.repeat(ok & ~excluded, 128)
        for bname, (lo, hi) in (("1-8", (1.0, 8.0)),):
            arrays[f"{bname}|{s}"] = cnd.band_64(full @ y, FS, lo, hi)[:, a64:a64 + n * 128].astype(np.float32)
        meta.append({"segment": s, "reject_fraction": float((~ok).mean()), "excluded": excluded})
    np.savez(target, **arrays, meta=json.dumps({"bads": bads, "ica": [int(x) for x in removed], "trials": meta,
                                                 "labels": labels}))
    return target.name


def acoustic(seg: int) -> np.ndarray:
    fs, x = wavfile.read(wav_path(seg))
    x = np.asarray(x, float)
    x = x[:, 0] if x.ndim > 1 else x
    e1k = tk.broadband_envelope(x, fs, load_analysis().section("envelope"))
    env = ss.resample_poly(e1k, 64, 1000)
    y = ss.resample_poly(x, 16000, fs)
    f, _, Z = ss.stft(y, 16000, nperseg=512, noverlap=512 - 250, boundary=None, padded=False)
    P = np.abs(Z) ** 2
    edges = 700 * (10 ** (np.linspace(np.log10(1 + 100 / 700), np.log10(1 + 7000 / 700), 18)) - 1)
    bands = np.array([P[(f >= lo) & (f < hi)].sum(0) for lo, hi in zip(edges[:-2], edges[2:])]) ** 0.3
    n = min(env.size, bands.shape[1])
    on = lambda a: np.maximum(np.diff(a, axis=-1, prepend=a[..., :1]), 0)
    return np.vstack([env[:n], on(env[:n]), bands[:, :n], on(bands[:, :n])])


def seg_stim(seg: int, words: pd.DataFrame, col: str) -> dict:
    A = acoustic(seg)
    n = A.shape[1]
    w = words[words.Segment == seg]
    idx = (w.onset.to_numpy() * 64).round().astype(int)
    keep = (idx >= 0) & (idx < n)
    idx, s = idx[keep], w[col].to_numpy()[keep]
    on, su = np.zeros(n), np.zeros(n)
    np.add.at(on, idx, 1.0)
    np.add.at(su, idx, s)
    return {"acoustic": g.bz(A), "word": g.bz(on[None]), "surprisal": g.bz(su[None]), "word_idx": idx.tolist(), "raw_surprisal": su}

def load_all(cache: Path, band: str) -> tuple[dict, dict]:
    """{listener: {segment: (data, valid)}} for listeners with >= MIN_SEGMENTS usable segments; QC per listener."""
    data, qc = {}, {}
    for f in sorted(cache.glob("S*.npz")):
        z = np.load(f)
        m = json.loads(str(z["meta"]))
        segs = {s: (z[f"{band}|{s}"].astype(float), z[f"valid|{s}"].astype(bool)) for s in SEGMENTS
                if f"valid|{s}" in z.files and z[f"valid|{s}"].any()}
        qc[f.stem] = {"usable_segments": len(segs), "reject_fraction": float(np.mean([t["reject_fraction"] for t in m["trials"]])),
                      "bads": len(m["bads"]), "ica": len(m["ica"])}
        if len(segs) >= MIN_SEGMENTS:
            data[f.stem] = segs
    return data, qc


def scores() -> pd.Series:
    out = {}
    for ln in (ROOT / "comprehension-scores.txt").read_text().splitlines()[1:]:
        parts = ln.split()
        if parts and parts[0].startswith("S") and len(parts) > 1 and "/" in parts[1]:
            a, b = parts[1].split("/")
            out[parts[0]] = int(a) / int(b)
    return pd.Series(out)
