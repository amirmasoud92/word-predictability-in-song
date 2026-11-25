"""Test A: word-level gain of the attended and the ignored stream per listener (MEG attention dataset).

The ridge parameter is chosen on the base model (both streams' acoustic, word-onset and control regressors) and then
fixed. For each variant and stream, the true surprisal regressor and its permuted versions (values shuffled over the
stream's words within each chapter; text-matched variant: over the kept words only) are tested with the other stream's
true surprisal in the model, through the exact sufficient-statistics path (encoding_models/sufficient_statistics.py).
Accuracy is the correlation within 0-800 ms after the stream's words, pooled over chapters and averaged over channels.
Stored per listener: per-channel r over all chapters, over the LP-recording chapters and over the HP-recording chapters,
for the true regressor and every permutation.

Variants: primary (gerpt2, all words, 500 permutations), matched (gerpt2, text-matched words, 500), large (gerpt2-large,
all words, 100). Random streams are keyed by listener, stream and variant.
"""
from __future__ import annotations

import gc
import os
from pathlib import Path

import numpy as np
import torch

from encoding_models import gpu_encoding as WG, sufficient_statistics as PE
from readers import meg_attention as C

OUT = C.DER / "A"
VARIANTS = {"primary": ("surprisal", None, 500), "matched": ("surprisal", "keep_matched", 500), "large": ("surprisal_large", None, 100)}
BATCH = 16


def regressor(tr: dict, stream: str, valkey: str, keepkey: str | None, values=None) -> np.ndarray:
    st = tr["st"]
    v = st[f"{stream}_{valkey}"] if values is None else values
    keep = st[f"{stream}_{keepkey}"] if keepkey else None
    return C.bz(C.surprisal_impulse(st, stream, v, keep, tr["T"]))


def permuted_versions(trials, stream, valkey, keepkey, n, key):
    g = C.rng(key)
    for _ in range(n):
        ver = []
        for tr in trials:
            vals = tr["st"][f"{stream}_{valkey}"]
            if keepkey:
                keep = tr["st"][f"{stream}_{keepkey}"]
                v2 = np.zeros_like(vals)
                v2[keep] = g.permutation(vals[keep])
            else:
                v2 = g.permutation(vals)
            ver.append(regressor(tr, stream, valkey, keepkey, v2))
        yield ver


def listener_gain(p: int) -> Path:
    target = OUT / f"P{p:02d}.npz"
    if target.exists():
        return target
    OUT.mkdir(parents=True, exist_ok=True)
    if not C.included(p):
        np.savez(target, excluded=np.array(True))
        return target
    trials = C.listener_trials(p)
    base = C.add_base(trials)
    files = np.array([tr["file"] for tr in trials])
    subsets = [np.arange(len(trials)), np.flatnonzero(files == "LP"), np.flatnonzero(files == "HP")]
    ff = WG.GpuFolds(trials, base, C.N_LAGS)
    out = {"lam": np.array(ff.lam), "chapters": np.array([f"{tr['file']}{tr['chapter']}" for tr in trials])}
    for variant, (valkey, keepkey, n_perm) in VARIANTS.items():
        for stream in C.STREAMS:
            other = "ignored" if stream == "attended" else "attended"
            F_other = [regressor(tr, other, valkey, keepkey) for tr in trials]
            ms = PE.MaskStats(ff, C.word_masks(trials, stream, keepkey))
            st = PE.SuffTest(ff, F_other, ms, C.N_LAGS)
            S_true = st.stats([[regressor(tr, stream, valkey, keepkey) for tr in trials]])
            S_perm, batch = [], []
            for ver in permuted_versions(trials, stream, valkey, keepkey, n_perm, f"p7|perm|{p}|{stream}|{variant}"):
                batch.append(ver)
                if len(batch) == BATCH:
                    S_perm.append(st.stats(batch))
                    batch = []
            if batch:
                S_perm.append(st.stats(batch))
            S_perm = np.concatenate(S_perm, 0)
            out[f"{variant}_{stream}_rtrue"] = np.stack([PE.r_subset(S_true[0], s) for s in subsets])
            out[f"{variant}_{stream}_rperm"] = np.stack([PE.r_subset(S_perm, s) for s in subsets], 1)
            out[f"{variant}_{stream}_nmask"] = np.array([float(S_true[0][s, 0, 0].sum()) for s in subsets])
            del st, ms
            torch.cuda.empty_cache()
    del ff
    gc.collect()
    torch.cuda.empty_cache()
    tmp = OUT / f"P{p:02d}.tmp.npz"
    np.savez(tmp, **out)
    os.replace(tmp, target)
    return target
