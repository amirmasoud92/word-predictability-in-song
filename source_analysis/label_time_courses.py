"""Desikan-Killiany label time courses of the preprocessed MEG of the attention dataset (LCMV beamformer; exploratory).

* Source space: fsaverage, 5 mm volume grid restricted to the 68 cortical Desikan-Killiany labels of aparc+aseg (one
  volume per label).
* Per listener: coregistration of fsaverage to the digitised head shape (estimated fiducials; ICP 6 iterations, nasion
  weight 2; head-shape points beyond 5 mm omitted; ICP 20 iterations, nasion weight 10); MEG forward model on the
  fsaverage boundary-element model (minimum distance 1 mm); LCMV beamformer from the empirical covariance of the 1-8 Hz,
  64 Hz data over valid samples, with its diagonal as the noise covariance (no empty-room recordings are released;
  regularisation 0.05, maximum-power orientation, unit-noise-gain weights, rank = MNE's estimate of the data rank).
* Per label, the time course is the first principal component of its source time series over valid samples, with the
  sign aligned to the label mean.
The fsaverage template is read from the MNE data folder (config/paths.yaml, mne_data).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from common.settings import dataset_root, load_paths
from readers import meg_attention as C

SUBJECTS_DIR = load_paths().mne_data
FS_DIR = SUBJECTS_DIR / "fsaverage"
DS = dataset_root("meg_attention")
OUT = load_paths().derived_root / "meg_attention" / "source"
QC = load_paths().results_root / "meg_attention" / "source_qc"
BEM = FS_DIR / "bem" / "fsaverage-5120-5120-5120-bem-sol.fif"
SCALE = 1e15


def dk_labels() -> list[str]:
    import mne
    names = mne.get_volume_labels_from_aseg(str(FS_DIR / "mri" / "aparc+aseg.mgz"))
    return [n for n in names if (n.startswith("ctx-lh-") or n.startswith("ctx-rh-")) and not n.endswith(("unknown", "corpuscallosum"))]


_SRC = {}


def source_space():
    """Built once per process (written to disk it would exceed the 2 GB FIFF limit)."""
    import mne
    if "src" not in _SRC:
        _SRC["src"] = mne.setup_volume_source_space("fsaverage", pos=5.0, mri=str(FS_DIR / "mri" / "aparc+aseg.mgz"),
                                                    bem=str(BEM), volume_label=dk_labels(), single_volume=False,
                                                    add_interpolator=False, subjects_dir=str(SUBJECTS_DIR), verbose="error")
    return _SRC["src"]


def coregister(info) -> tuple[object, dict]:
    from mne.coreg import Coregistration
    coreg = Coregistration(info, "fsaverage", str(SUBJECTS_DIR), fiducials="estimated")
    coreg.fit_fiducials(verbose="error")
    coreg.fit_icp(n_iterations=6, nasion_weight=2.0, verbose="error")
    coreg.omit_head_shape_points(distance=5.0 / 1000)
    coreg.fit_icp(n_iterations=20, nasion_weight=10.0, verbose="error")
    d = coreg.compute_dig_mri_distances() * 1e3
    return coreg.trans, {"dig_mri_mm_mean": float(d.mean()), "dig_mri_mm_max": float(d.max()), "n_dig_used": int(d.size)}


def label_time_courses(p: int) -> Path:
    import mne
    target = OUT / "labels" / f"P{p:02d}.npz"
    if target.exists():
        return target
    src = source_space()
    info = mne.io.read_raw_fif(DS / str(p) / "LP_speaker_attended" / "data_meg.fif", preload=False, verbose="error").info
    trans, cq = coregister(info)
    fwd = mne.make_forward_solution(info, trans, src, str(BEM), meg=True, eeg=False, mindist=1.0, verbose="error")
    trials = C.listener_trials(p)
    X = [tr["Y"][:, tr["v"]] / SCALE for tr in trials]
    n = sum(x.shape[1] for x in X)
    mu = sum(x.sum(1) for x in X) / n
    Cd = sum((x - mu[:, None]) @ (x - mu[:, None]).T for x in X) / (n - 1)
    names = info["ch_names"]
    data_cov = mne.Covariance(Cd, names, [], [], n - 1, verbose="error")
    noise_cov = mne.Covariance(np.diag(np.diag(Cd)), names, [], [], n - 1, verbose="error")
    rank = mne.compute_rank(data_cov, rank=None, info=info, verbose="error")
    filters = mne.beamformer.make_lcmv(info, fwd, data_cov, reg=0.05, noise_cov=noise_cov, pick_ori="max-power",
                                       weight_norm="unit-noise-gain", rank=rank, verbose="error")
    W = filters["weights"]
    if filters.get("whitener") is not None:
        W = W @ filters["whitener"]
    if filters.get("proj") is not None and np.ndim(filters["proj"]) == 2:
        W = W @ filters["proj"]
    fsrc = fwd["src"]
    sizes = [int(s["nuse"]) for s in fsrc]
    edges = np.concatenate([[0], np.cumsum(sizes)])
    labels = [s.get("seg_name", f"label{k}") for k, s in enumerate(fsrc)]
    S_valid = [W @ x for x in X]
    comps = []
    for k in range(len(fsrc)):
        a, b = edges[k], edges[k + 1]
        if b - a == 0:
            comps.append(None)
            continue
        Ck = sum(s[a:b] @ s[a:b].T for s in S_valid)
        w, V = np.linalg.eigh(Ck)
        u = V[:, -1]
        mean_tc = sum(float((u @ s[a:b]) @ s[a:b].mean(0)) for s in S_valid)
        comps.append(u if mean_tc >= 0 else -u)
    arrays = {"labels": np.array(labels), "n_sources": np.array(sizes)}
    for tr in trials:
        S = W @ (tr["Y"] / SCALE)
        L = np.zeros((len(fsrc), S.shape[1]), np.float32)
        for k, u in enumerate(comps):
            if u is not None:
                L[k] = (u @ S[edges[k]: edges[k + 1]]).astype(np.float32)
        arrays[f"{tr['file']}_{tr['chapter']}"] = L
    (OUT / "labels").mkdir(parents=True, exist_ok=True)
    QC.mkdir(parents=True, exist_ok=True)
    tmp = OUT / "labels" / f"P{p:02d}.tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, target)
    qc = {"participant": p, **cq, "data_rank": int(rank["mag"]), "n_sources": int(sum(sizes)), "n_labels": len(fsrc),
          "empty_labels": [lab for lab, s in zip(labels, sizes) if s == 0]}
    (QC / f"P{p:02d}.json").write_text(json.dumps(qc, indent=1), encoding="utf-8")
    return target
