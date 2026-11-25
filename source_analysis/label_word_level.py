"""Test A per Desikan-Killiany label (exploratory).

The model of the primary Test A analysis (gerpt2 surprisal of both streams with the same base regressors; ridge chosen
on the base model for the label data) with the 68 label time courses as outputs, each rescaled by its SD over the
listener's valid samples (which leaves r unchanged); 100 permutations per stream, through the sufficient-statistics path.
Stored per listener: per-label r over all chapters, over the odd and over the even chapters (positions in the
listener's chapter list), for the true regressor and each permutation.
"""
from __future__ import annotations

import gc
import os
from pathlib import Path

import numpy as np
import torch

from common.settings import load_paths
from encoding_models import gpu_encoding as WG, meg_attention_word_level as W7, sufficient_statistics as PE
from readers import meg_attention as C

LAB = load_paths().derived_root / "meg_attention" / "source" / "labels"
OUT = load_paths().derived_root / "meg_attention" / "source_word_level"
N_PERM, BATCH = 100, 16


def label_gain(p: int) -> Path:
    target = OUT / f"P{p:02d}.npz"
    if target.exists():
        return target
    trials = C.listener_trials(p)
    with np.load(LAB / f"P{p:02d}.npz") as z:
        labels = z["labels"]
        for tr in trials:
            tr["Y"] = z[f"{tr['file']}_{tr['chapter']}"].astype(np.float64)[:, : tr["T"]]
    sd = np.sqrt(sum((tr["Y"][:, tr["v"]] ** 2).sum(1) for tr in trials) / sum(int(tr["v"].sum()) for tr in trials))
    for tr in trials:
        tr["Y"] = tr["Y"] / np.where(sd > 0, sd, 1.0)[:, None]
    base = C.add_base(trials)
    pos = np.arange(len(trials))
    subsets = [pos, pos[0::2], pos[1::2]]
    ff = WG.GpuFolds(trials, base, C.N_LAGS)
    out = {"lam": np.array(ff.lam), "labels": labels, "chapters": np.array([f"{tr['file']}{tr['chapter']}" for tr in trials])}
    for stream in C.STREAMS:
        other = "ignored" if stream == "attended" else "attended"
        F_other = [W7.regressor(tr, other, "surprisal", None) for tr in trials]
        ms = PE.MaskStats(ff, C.word_masks(trials, stream))
        st = PE.SuffTest(ff, F_other, ms, C.N_LAGS)
        S_true = st.stats([[W7.regressor(tr, stream, "surprisal", None) for tr in trials]])
        S_perm, batch = [], []
        for ver in W7.permuted_versions(trials, stream, "surprisal", None, N_PERM, f"p8|srcperm|{p}|{stream}"):
            batch.append(ver)
            if len(batch) == BATCH:
                S_perm.append(st.stats(batch))
                batch = []
        if batch:
            S_perm.append(st.stats(batch))
        S_perm = np.concatenate(S_perm, 0)
        out[f"{stream}_rtrue"] = np.stack([PE.r_subset(S_true[0], s) for s in subsets])
        out[f"{stream}_rperm"] = np.stack([PE.r_subset(S_perm, s) for s in subsets], 1)
        del st, ms
        torch.cuda.empty_cache()
    del ff
    gc.collect()
    torch.cuda.empty_cache()
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / f"P{p:02d}.tmp.npz"
    np.savez(tmp, **out)
    os.replace(tmp, target)
    return target
