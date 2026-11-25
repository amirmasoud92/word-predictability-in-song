"""CantiSong release (Cantisani et al., 2023): song list, EEG trials and their preparation for the encoding models.

Per listener, all trials of the three versions together: robust third-order polynomial detrending, bad channels (robust z
of the log SD > 5) replaced by spherical-spline interpolation, average reference, at most three ocular components removed
from one extended-Infomax decomposition, FASTER and robust-channel rejection of 2 s epochs, and exclusion of trials that
lose more than a third of their epochs. The cleaned trials are band-passed to 1-8 Hz at 64 Hz; the first 5 s and the last
2 s of each trial are not used.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

from common.settings import dataset_root, rng as make_rng
from preprocessing import cnd_recordings as cnd, signal_operations as pp
from preprocessing.epoch_rejection import reject
from preprocessing.epochs import Trial

ROOT = dataset_root("cantisong")
VERSIONS = {"speech": "speech", "song": "song", "melody": "melody"}
BANDS = {"1-8": (1.0, 8.0)}
FS, START_S, END_TRIM_S, MAX_BAD = 64, 5.0, 2.0, 1 / 3


def song_ids() -> list[str]:
    return sorted(p.stem for p in (ROOT / "Stimuli" / "lyrics").glob("*.txt"))


def stim_lengths() -> dict[str, float]:
    t = pd.read_csv(ROOT / "Stimuli" / "sounds" / "stimuli_lengths.csv")
    return dict(zip(t.File.str.replace(".wav", "", regex=False), t.Length))


def load_version(sub: int, version: str):
    e = sio.loadmat(ROOT / "EEG" / version / f"pre_dataSub{sub}.mat", squeeze_me=True, struct_as_record=False)["eeg"]
    return [np.asarray(d, float) for d in np.atleast_1d(e.data)], [str(c.labels).split(" (")[0] for c in np.atleast_1d(e.chanlocs)]


def prepare_one(sub: int, cfg2: dict, seed: int, cache: Path) -> str:
    target = cache / f"sub{sub:02d}.npz"
    if target.exists():
        return target.name
    ids, lengths = song_ids(), stim_lengths()
    items, labels = [], None
    for version in VERSIONS:
        data, labels = load_version(sub, version)
        for k, x in enumerate(data):
            if x.size == 0:
                continue
            dur = min(x.shape[0] / FS, lengths[f"{ids[k]}_{version}"])
            n_ep = int((dur - START_S - END_TRIM_S) // 2)
            if n_ep < 3:
                continue
            items.append((version, ids[k], x[: int(dur * FS)].T * 1e6, n_ep))
    info = cnd.montage_info(labels, FS, "biosemi64")
    a = int(START_S * FS)
    xs = [pp.robust_detrend(x, 3, 3.0) for _, _, x, _ in items]
    bads = pp.bad_channels([x[:, a:a + n * 128] for x, (_, _, _, n) in zip(xs, items)], labels, 5.0)
    op = cnd.spatial_operator(info, bads)
    rank = int((np.linalg.svd(op, compute_uv=False) > 1e-8).sum())
    fit = np.concatenate([(op @ x)[:, a:a + n * 128] for x, (_, _, _, n) in zip(xs, items)], axis=1)
    proj, removed = cnd.ocular_ica(fit, info, min(rank, 40), 4, 3.0, 3, int(make_rng(seed, "canti", sub).integers(2**31)))
    full = proj @ op
    trials = [Trial(subject=sub, condition=v, excerpt=int(i), label=f"{v}_{i}", sfreq=FS, t0=START_S, data=(full @ x)[:, a:a + n * 128])
              for x, (v, i, _, n) in zip(xs, items)]
    res = reject(trials, labels, 2.0, seed=seed, max_bad_fraction=cfg2["max_bad_epoch_fraction"], method=cfg2["epoch_rejection"],
                 z_threshold=cfg2["faster_z"], channel_robust_z=cfg2["channel_robust_z"])
    arrays, meta = {}, []
    for x, t, (v, i, _, n) in zip(xs, trials, items):
        ok = res.epoch_ok[t.key]
        excluded = (~ok).mean() > MAX_BAD
        valid = np.repeat(ok & ~excluded, 128)
        for bname, (lo, hi) in BANDS.items():
            arrays[f"{bname}|{v}|{i}"] = cnd.band_64(full @ x, FS, lo, hi)[:, a:a + n * 128].astype(np.float32)
        arrays[f"valid|{v}|{i}"] = valid
        meta.append({"version": v, "song": i, "n_epochs": n, "reject_fraction": float((~ok).mean()), "excluded": bool(excluded)})
    np.savez(target, **arrays, meta=json.dumps({"bads": bads, "ica": [int(c) for c in removed], "trials": meta}))
    return target.name


def load_all(cache: Path, band: str) -> dict:
    """{(version, song): {subject: (data, valid)}}"""
    out = {}
    for f in sorted(cache.glob("sub*.npz")):
        sub = int(f.stem[3:])
        z = np.load(f)
        for key in z.files:
            if key.startswith(band + "|"):
                _, v, i = key.split("|")
                out.setdefault((v, i), {})[sub] = (z[key].astype(float), z[f"valid|{v}|{i}"])
    return out
