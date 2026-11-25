"""Speech-in-noise release (Crosse et al., 2016): EEG trials of the audio-only and audiovisual passages, their
preparation for the encoding models (the CantiSong pipeline on 128 channels) and the intelligibility ratings.
"""
from __future__ import annotations

import json
from pathlib import Path

import mne
import numpy as np
import scipy.io as sio

from common.settings import dataset_root, rng as make_rng
from preprocessing import cnd_recordings as cnd, signal_operations as pp
from preprocessing.epoch_rejection import reject
from preprocessing.epochs import Trial

ROOT = dataset_root("speech_in_noise")
FS = 128.0                          # the files' fs variable (512) is the acquisition rate
OFFSET = {"a": 0, "v": 15, "av": 30}  # file number = passage + offset
PASSAGES = range(1, 16)
SUBJECTS = range(1, 22)
BANDS = {"1-8": (1.0, 8.0)}
START_S, END_TRIM_S, MAX_BAD = 5.0, 2.0, 1 / 3
RATING_COND = {"av": 0, "a": 1}       # ratings: participants x passage x condition (1 = AV, 2 = A)


def load_trial(sub: int, cond: str, passage: int) -> np.ndarray:
    f = ROOT / "EEG" / f"sub{sub}" / f"sub{sub}_{cond}_{passage + OFFSET[cond]}.mat"
    return sio.loadmat(f, variable_names=["EEG"])["EEG"].T.astype(float)


def labels_128() -> list[str]:
    """Channel names of the 128-channel BioSemi cap (A1-D32), in the order of the release."""
    return mne.channels.make_standard_montage("biosemi128").ch_names


def ratings(rows: str = "1-21") -> np.ndarray:
    """participants (EEG sub1..sub21) x passage x {av, a}. 22 rating rows vs 21 EEG participants;
    rows 1-21 are assumed to be sub1-sub21 (sensitivity: rows 2-22)."""
    s = sio.loadmat(ROOT / "Behavioural Data" / "Intelligibilty Ratings.mat")["score"].astype(float)
    return s[:21] if rows == "1-21" else s[1:22]


def prepare_one(sub: int, labels: list[str], cfg2: dict, seed: int, cache: Path) -> str:
    target = cache / f"sub{sub:02d}.npz"
    if target.exists():
        return target.name
    items = [(c, p, load_trial(sub, c, p)) for c in OFFSET for p in PASSAGES]
    info = cnd.montage_info(labels, FS, "biosemi128")
    hp = pp.fir_kernel(0.5, FS, cnd.fir_order(FS, 1.0), True)
    lp = pp.fir_kernel(40.0, FS, cnd.fir_order(FS, 5.0), False)
    ys = [pp.robust_detrend(x, 3, 3.0) for _, _, x in items]
    broad_raw = [pp.fir_filter(pp.fir_filter(y, hp), lp) for y in ys]
    a = int(START_S * FS)
    n_ep = [int((x.shape[1] / FS - START_S - END_TRIM_S) // 2) for _, _, x in items]
    wins = [(a, a + n * int(2 * FS)) for n in n_ep]
    bads = pp.bad_channels([bb[:, s:e] for bb, (s, e) in zip(broad_raw, wins)], labels, 5.0)
    op = cnd.spatial_operator(info, bads)
    rank = int((np.linalg.svd(op, compute_uv=False) > 1e-8).sum())
    step = int(round(FS / 64.0))
    fit = np.concatenate([(op @ bb[:, s:e])[:, ::step] for bb, (s, e) in zip(broad_raw, wins)], axis=1)
    info64 = cnd.montage_info(labels, FS / step, "biosemi128")
    proj, removed = cnd.ocular_ica(fit, info64, min(rank, 40), 4, 3.0, 3, int(make_rng(seed, "sin", sub).integers(2**31)))
    full = proj @ op
    trials = [Trial(subject=sub, condition=c, excerpt=p, label=f"{c}_{p}", sfreq=FS, t0=START_S, data=(full @ bb)[:, s:e])
              for (c, p, _), bb, (s, e) in zip(items, broad_raw, wins)]
    res = reject(trials, labels, 2.0, seed=seed, max_bad_fraction=cfg2["max_bad_epoch_fraction"], method=cfg2["epoch_rejection"],
                 z_threshold=cfg2["faster_z"], channel_robust_z=cfg2["channel_robust_z"])
    arrays, meta = {}, []
    a64 = int(START_S * 64)
    for (c, p, _), y, t, n in zip(items, ys, trials, n_ep):
        ok = res.epoch_ok[t.key]
        excluded = bool((~ok).mean() > MAX_BAD)
        arrays[f"valid|{c}|{p}"] = np.repeat(ok & ~excluded, 128)
        for bname, (lo, hi) in BANDS.items():
            arrays[f"{bname}|{c}|{p}"] = cnd.band_64(full @ y, FS, lo, hi)[:, a64:a64 + n * 128].astype(np.float32)
        meta.append({"cond": c, "passage": p, "n_epochs": n, "reject_fraction": float((~ok).mean()), "excluded": excluded})
    np.savez(target, **arrays, meta=json.dumps({"bads": bads, "ica": [int(x) for x in removed], "trials": meta}))
    return target.name

def load_all(cache: Path, band: str) -> dict:
    """{(cond, passage): {subject: (data, valid)}} for trials that passed exclusion."""
    out = {}
    for f in sorted(cache.glob("sub*.npz")):
        sub = int(f.stem[3:])
        z = np.load(f)
        for c in OFFSET:
            for p in PASSAGES:
                v = z[f"valid|{c}|{p}"]
                if v.any():
                    out.setdefault((c, p), {})[sub] = (z[f"{band}|{c}|{p}"].astype(float), v.astype(bool))
    return out
