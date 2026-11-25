"""Lexical, acoustic and note-level controls of the word- and note-level gains (tests 1a-1c, 1e, 1f).

Words: the control regressors (stimulus_features/lexical_controls.py) are added to the base model (CantiSong: acoustics,
phonetic features, word onsets, corrected annotation of song 09; Alice: acoustics, word onsets). The ridge is reselected
on this base without surprisal; delta r uses the same permutation draws as the stabilised word-level estimate (same job
keys, 500 draws stored) and, per test, its number of permutations n. A further null permutes surprisal only within Zipf
tertiles of each song or segment (frequency-matched null).
Notes: melodic surprisal (L1) with |interval|, pitch height and note duration added to its base, on the draws of the
stabilised L1 estimate.
Tests are one-sided over listeners (two-sided for the hummed word control); the retained fraction is the ratio of the
group-mean delta r with and without the controls.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from encoding_models import gpu_encoding as WG, ridge_encoding as E, word_level_trials as P
from readers import cantisong as c
from statistical_tests.inference import summarise
from stimulus_features import song_regressors as g

N_PERM = 500


def tertile_controls(trials: list[dict], rng, n: int) -> list[dict]:
    """Surprisal permuted only within Zipf tertiles of each trial (song or segment)."""
    ctrls = []
    for _ in range(n):
        cd = {}
        for j, tr in enumerate(trials):
            raw = np.zeros_like(tr["raw_s"])
            vals = tr["raw_s"][tr["idx"]].copy()
            z = tr["zipf"]
            q = np.quantile(z, [1 / 3, 2 / 3])
            lab = np.digitize(z, q)
            for k in np.unique(lab):
                m = lab == k
                vals[m] = rng.permutation(vals[m])
            raw[tr["idx"]] = vals
            cd[j] = g.bz(raw[None])[:, tr["a"]:tr["a"] + tr["T"]]
        ctrls.append(cd)
    return ctrls


def word_job(kind: str, key: str, specs: list[dict], a: int, ctl: dict, seed: int) -> dict:
    """One listener (job key 'canti|version|listener' or 'alice|speech|listener'): surprisal beyond the controls, with
    the permutation null and the frequency-matched null; true and per-permutation accuracies."""
    trials = P.canti_trials(specs, a) if kind == "canti" else P.alice_trials(specs, a)
    for tr, sp in zip(trials, specs):
        cc = ctl[sp["song"] if kind == "canti" else sp["seg"]]
        tr["F"]["lexctl"] = g.bz(cc["raw"])[:, a:a + tr["T"]]
        assert np.array_equal(cc["idx"], tr["idx"]), "controls must align with the word onsets"
        tr["zipf"] = cc["zipf"]
    base = [k for k in ("acoustic", "phonetic", "word") if k in trials[0]["F"]] + ["lexctl"]
    masks = E.event_masks(trials, "words", P.WIN_S)
    rng = P.rng_for(seed, key)                                     # the draws of the stabilised word-level estimate
    ctrl = P.perm_controls(trials, "raw_s", rng, N_PERM)
    folds = WG.GpuFolds(trials, base, P.N_LAGS)
    det, _ = WG.delta_r_detail(trials, base, "surprisal", ctrl, masks, P.N_LAGS, folds=folds)
    del ctrl
    ter, _ = WG.delta_r_detail(trials, base, "surprisal", tertile_controls(trials, P.rng_for(seed, key, 1), N_PERM), masks, P.N_LAGS, folds=folds)
    return {"key": key, "lam": folds.lam, "true_pooled": det["true_pooled"], "null_pooled": det["null_pooled"],
            "true_tertile": ter["true_pooled"], "null_tertile": ter["null_pooled"], "per_trial": det["per_trial"]}


def melody_job(key: str, sub: int, version: str, data: dict, feats: dict, nctl: dict, seed: int) -> dict:
    """L1 with the note-level controls in the base (job key 'L|version|listener'); true and per-permutation accuracy."""
    from encoding_models.melodic_surprisal import impulses as imp
    songs = [s for s in c.song_ids() if (version, s) in data and sub in data[(version, s)] and data[(version, s)][sub][1].any()]
    if len(songs) < 6:
        return {}
    a = int(c.START_S * 64)
    rng = P.rng_for(seed, key)                                     # the draws of the stabilised L1 estimate
    trs = []
    for s in songs:
        ff = feats[(s, version)]
        Y, v = data[(version, s)][sub]
        T = min(Y.shape[1], ff["n"] - a)
        F = {"acoustic": ff["st"]["acoustic"][:, a:a + T], "note_on": ff["note_on"][:, a:a + T]}
        if version == "song":
            F |= {"phonetic": ff["st"]["phonetic"][:, a:a + T], "word": ff["st"]["word"][:, a:a + T],
                  "lex": g.bz(imp(ff["w_idx"], ff["lsur"], ff["n"])[None])[:, a:a + T]}
        F["notectl"] = g.bz(nctl[s])[:, a:a + T]
        F["mel"] = g.bz(imp(ff["note_idx"], ff["msur"], ff["n"])[None])[:, a:a + T]
        trs.append({"F": F, "Y": Y[:, :T], "v": v[:T], "T": T, "song": s, "notes": (ff["note_idx"] - a).tolist()})
    ctrls = [{j: g.bz(imp(feats[(tr["song"], version)]["note_idx"], rng.permutation(feats[(tr["song"], version)]["msur"]),
                          feats[(tr["song"], version)]["n"])[None])[:, a:a + tr["T"]] for j, tr in enumerate(trs)} for _ in range(N_PERM)]
    base = [k for k in ("acoustic", "note_on", "phonetic", "word", "lex", "notectl") if k in trs[0]["F"]]
    d, lam, nl = WG.delta_r(trs, base, "mel", ctrls, {"w": (E.event_masks(trs, "notes", 0.8), None)}, P.N_LAGS)
    return {"key": key, "listener": sub, "version": version, "lam": lam, "true": d["w"] + float(np.mean(nl["w"])), "null": nl["w"]}


def statistics(word_rows: list[dict], melody_rows: list[dict], n_used: dict, base: pd.DataFrame, base_l: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Controlled gains per listener with the n of the matching stabilised test (``n_used``: 'dataset|version' and
    'version' for L1), and the group tests. ``base`` and ``base_l`` are the stabilised word-level and L1 listener tables
    (columns dataset, version, listener, pooled; version, listener, L1_melodic)."""
    rows, mrows = [], []
    for m in word_rows:
        ds, v, sub = m["key"].split("|")
        n = n_used[f"{ds}|{v}"]
        rows.append({"dataset": ds, "version": v, "listener": sub, "controlled": m["true_pooled"] - float(m["null_pooled"][:n].mean()),
                     "tertile": m["true_tertile"] - float(m["null_tertile"][:n].mean())})
    for m in melody_rows:
        n = n_used[m["version"]]
        mrows.append({"listener": m["listener"], "version": m["version"], "L1_controlled": m["true"] - float(m["null"][:n].mean())})
    W, M = pd.DataFrame(rows), pd.DataFrame(mrows)
    pk = lambda x, alt: {k: v for k, v in summarise(np.asarray(x, float), alt).items() if k in ("n", "mean", "dz", "dz_ci95", "p_signflip", "n_positive")}
    res, pvals = {}, {}
    names = {("alice", "speech"): "1a_alice", ("canti", "speech"): "1b_spoken", ("canti", "song"): "1c_sung", ("canti", "melody"): "1d_hummed"}
    for (ds, v), nm in names.items():
        d = W[(W.dataset == ds) & (W.version == v)].merge(base[(base.dataset == ds) & (base.version == v)][["listener", "pooled"]], on="listener")
        alt = "two-sided" if v == "melody" else "greater"
        res[nm] = {"controlled": pk(d.controlled, alt), "tertile_null": pk(d.tertile, alt), "uncontrolled": pk(d.pooled, alt),
                   "retained_fraction": float(d.controlled.mean() / d.pooled.mean())}
        if v != "melody":
            pvals[nm] = res[nm]["controlled"]["p_signflip"]
    for v, nm in (("song", "1e_L1_sung"), ("melody", "1f_L1_hummed")):
        d = M[M.version == v].merge(base_l[base_l.version == v][["listener", "L1_melodic"]], on="listener")
        res[nm] = {"controlled": pk(d.L1_controlled, "greater"), "uncontrolled": pk(d.L1_melodic, "greater"),
                   "retained_fraction": float(d.L1_controlled.mean() / d.L1_melodic.mean())}
        pvals[nm] = res[nm]["controlled"]["p_signflip"]
    res["p_uncorrected"] = pvals
    return res, W, M
