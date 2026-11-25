"""The word-level measure in MEG-MASC per listener and session: the standard measure, Test 2 and its sensitivity
analyses, and the list words with their surprisal as heard.

Per usable session: ridge chosen on the base model (acoustic 34, phonetic 19, sentence and list word onsets, 4 controls)
and then fixed; each tested surprisal regressor, with the other surprisal regressors fixed, through the exact sufficient-statistics path
(encoding_models/sufficient_statistics.py; leave-one-sound-file-out folds); permutations within sound files among the
tested words; accuracy within 0-800 ms after the tested words. Stored: per-channel r for the true regressor and every
permutation.

* std (standard measure): sentence words, narrative GPT-2 surprisal; fixed: list words with the type-mean surprisal
  (100 permutations, extended in blocks of 100 up to 500 by the Monte Carlo rule, continuing the same seeded stream).
* std_medium (GPT-2 medium) and std_heard (context as heard): 100 permutations each.
* match (Test 2): 20 subsamples; per subsample M (drawn sentence occurrences of the list types, type-mean surprisal),
  L (list words, type-mean surprisal) and O (other sentence words, own surprisal); Delta r(M) with O and L fixed and
  Delta r(L) with M and O fixed; 20 permutations each. For each list token, a sentence occurrence of its type in the
  same story is drawn, with replacement only for types with more list tokens than sentence occurrences.
* unm_S, unm_L (sensitivity): all sentence occurrences of the list types with their own surprisal (S) against the list
  words (L), 100 permutations each.
* match_medium (sensitivity): 10 subsamples with GPT-2 medium, 20 permutations each.
* listctx (exploratory): list words with the surprisal as heard; fixed: sentence words (narrative), 100 permutations.
Random streams are keyed by listener, session, test and subsample.
"""
from __future__ import annotations

import gc
import os
from pathlib import Path

import numpy as np
import torch

from encoding_models import gpu_encoding as WG, sufficient_statistics as PE
from readers import meg_masc as C

OUT = C.DER / "word_level"
BASE = ["acoustic", "phonetic", "word_sent", "word_list", "controls"]
N_STD, N_VAR, N_DRAW, N_DRAW_MED, N_PERM_DRAW = 100, 100, 20, 10, 20


def base_trials(sub: str, ses: int) -> list[dict]:
    trials = C.listener_trials(sub, ses)
    for tr in trials:
        st, T = tr["st"], tr["T"]
        tr["F"] = {"acoustic": C.bz(st["acoustic_raw"][:, :T]), "phonetic": C.bz(st["phonetic_raw"][:, :T]),
                   "word_sent": C.bz(st["word_sent_raw"][:, :T]), "word_list": C.bz(st["word_list_raw"][:, :T]),
                   "controls": C.bz(st["controls_raw"][:, :T])}
    return trials


def reg(tr: dict, vals: np.ndarray) -> np.ndarray:
    """Band-passed, z-scored impulse train at the trial's word onsets with the given values (0 = word not in the set)."""
    return C.bz(C.impulse(tr["st"]["onset_idx"], np.nan_to_num(vals), tr["T"]))


class Runner:
    def __init__(self, ff: WG.GpuFolds, trials: list[dict], batch: int):
        self.ff, self.trials, self.batch = ff, trials, batch
        self.allt = np.arange(len(trials))

    def test(self, tested: list[np.ndarray], fixed: list[list[np.ndarray]], n_perm: int, key: str, n_skip: int = 0) -> dict:
        """tested: per trial, values over the trial's words (0 outside the tested set); fixed: per trial, a list of
        such value arrays. Permutations shuffle the tested values among the tested words within each trial."""
        trials = self.trials
        F_fixed = [np.vstack([reg(tr, v) for v in fx]) for tr, fx in zip(trials, fixed)]
        sel = [v != 0 for v in tested]
        masks = [C.word_mask(tr, tr["st"]["onset_idx"][s]) for tr, s in zip(trials, sel)]
        ms = PE.MaskStats(self.ff, masks)
        st = PE.SuffTest(self.ff, F_fixed, ms, C.N_LAGS)
        S_true = st.stats([[reg(tr, v) for tr, v in zip(trials, tested)]])
        g = C.rng(key)
        for _ in range(n_skip):                                      # continue the seeded stream after stored permutations
            for v, s in zip(tested, sel):
                g.permutation(v[np.flatnonzero(s)])
        S_perm, batch = [], []
        for _ in range(n_perm):
            ver = []
            for tr, v, s in zip(trials, tested, sel):
                v2 = v.copy()
                i = np.flatnonzero(s)
                v2[i] = g.permutation(v[i])
                ver.append(reg(tr, v2))
            batch.append(ver)
            if len(batch) == self.batch:
                S_perm.append(st.stats(batch))
                batch = []
        if batch:
            S_perm.append(st.stats(batch))
        out = {"rtrue": PE.r_subset(S_true[0], self.allt), "rperm": PE.r_subset(np.concatenate(S_perm, 0), self.allt),
               "nmask": float(S_true[0][:, 0, 0].sum()), "nwords": int(sum(s.sum() for s in sel))}
        del st, ms, S_true, S_perm
        torch.cuda.empty_cache()
        return out


def draws(trials: list[dict], tm_key: str, n: int, key: str) -> list[tuple[list[np.ndarray], list[np.ndarray], int]]:
    """Per subsample: per-trial counts of drawn sentence occurrences (M) and the per-trial list mask (L) of the list
    tokens that were matched; plus the number of draws made with replacement."""
    words = []                                                   # (trial, word index, story, type, is_list, tm)
    for j, tr in enumerate(trials):
        st = tr["st"]
        for i in range(len(st["onset_idx"])):
            words.append((j, i, tr["story"], int(st["type_id"][i]), bool(st["is_list"][i]), float(st[tm_key][i])))
    occ, toks = {}, {}
    for j, i, story, t, is_list, tm in words:
        if is_list and np.isfinite(tm):
            toks.setdefault((story, t), []).append((j, i))
        elif not is_list:
            occ.setdefault((story, t), []).append((j, i))
    out = []
    for d in range(n):
        g = C.rng(f"{key}|{d + 1}")
        M = [np.zeros(len(tr["st"]["onset_idx"])) for tr in trials]
        L = [np.zeros(len(tr["st"]["onset_idx"]), bool) for tr in trials]
        n_rep = 0
        for st_t, tk in sorted(toks.items()):
            oc = occ.get(st_t, [])
            if not oc:
                continue
            k = len(tk)
            pick = g.choice(len(oc), k, replace=k > len(oc))
            n_rep += int(k > len(oc))
            for (jt, it), p in zip(tk, pick):
                jo, io = oc[p]
                M[jo][io] += 1.0
                L[jt][it] = True
        out.append((M, L, n_rep))
    return out


def n_done(sub: str, ses: int, key: str) -> int:
    """Permutations already stored for one array (the base file and earlier extensions)."""
    n = 0
    for f in [OUT / f"sub-{sub}_ses-{ses}.npz"] + sorted(OUT.glob(f"sub-{sub}_ses-{ses}_ext*.npz")):
        with np.load(f) as z:
            if key in z.files:
                n += z[key].shape[-2]
    return n


def session_word_level(sub: str, ses: int, batch: int = 10, ext: dict[str, int] | None = None) -> Path | None:
    """All tests of one session, or (ext = {test: n}) n further permutations of the named tests, continuing each
    test's seeded stream after the permutations already stored (Monte Carlo rule); saved as _ext<k>.npz."""
    target = OUT / f"sub-{sub}_ses-{ses}.npz"
    if ext is None and target.exists():
        return target
    if ext is not None and not target.exists():
        return None
    if ext is not None:                                          # requested totals -> permutations still to compute
        ext = {k: v - n_done(sub, ses, f"{k}_M_rperm" if k.startswith("match") else f"{k}_rperm") for k, v in ext.items()}
        ext = {k: v for k, v in ext.items() if v > 0}
        if not ext:
            return target
    trials = base_trials(sub, ses)
    ff = WG.GpuFolds(trials, BASE, C.N_LAGS)
    R = Runner(ff, trials, batch)
    res = {"lam": np.array(ff.lam), "files": np.array([tr["key"] for tr in trials])}

    def pending(name):
        return ext is None or name in ext

    def n_perm(name, default):
        return default if ext is None else ext[name]

    def skip(key):
        return 0 if ext is None else n_done(sub, ses, key)

    def put(name, o):
        for k, v in o.items():
            if ext is None or k == "rperm":
                res[f"{name}_{k}"] = np.asarray(v)

    def vals(key, mask_fn):
        return [np.where(mask_fn(tr["st"]) & np.isfinite(tr["st"][key]), tr["st"][key], 0.0) for tr in trials]
    sent = lambda st: ~st["is_list"]
    lst = lambda st: st["is_list"]
    S, L = vals("s_gpt2", sent), vals("tm_gpt2", lst)
    tag = f"{sub}|{ses}"
    if pending("std"):
        put("std", R.test(S, [[l_] for l_ in L], n_perm("std", N_STD), f"9|perm|{tag}|std", skip("std_rperm")))
    if pending("std_medium"):
        put("std_medium", R.test(vals("s_gpt2_medium", sent), [[l_] for l_ in vals("tm_gpt2_medium", lst)], n_perm("std_medium", N_VAR),
                                 f"9|perm|{tag}|std_medium", skip("std_medium_rperm")))
    if pending("std_heard"):
        put("std_heard", R.test(vals("s_heard", sent), [[l_] for l_ in L], n_perm("std_heard", N_VAR), f"9|perm|{tag}|std_heard",
                                skip("std_heard_rperm")))
    if pending("listctx"):
        put("listctx", R.test(vals("s_heard", lst), [[s_] for s_ in S], n_perm("listctx", N_VAR), f"9|perm|{tag}|listctx", skip("listctx_rperm")))
    # unmatched sensitivity: every sentence occurrence of the list types (own surprisal) vs the list words
    Su = vals("s_gpt2", lambda st: ~st["is_list"] & st["list_type"])
    Ou = vals("s_gpt2", lambda st: ~st["is_list"] & ~st["list_type"])
    if pending("unm_S"):
        put("unm_S", R.test(Su, [[o, l_] for o, l_ in zip(Ou, L)], n_perm("unm_S", N_VAR), f"9|perm|{tag}|unm_S", skip("unm_S_rperm")))
    if pending("unm_L"):
        put("unm_L", R.test(L, [[s_, o] for s_, o in zip(Su, Ou)], n_perm("unm_L", N_VAR), f"9|perm|{tag}|unm_L", skip("unm_L_rperm")))
    # matched subsamples (Test 2 primary, GPT-2) and the GPT-2 medium sensitivity
    for name, tm_key, s_key, n in (("match", "tm_gpt2", "s_gpt2", N_DRAW), ("match_medium", "tm_gpt2_medium", "s_gpt2_medium", N_DRAW_MED)):
        if not pending(name):
            continue
        rec = {f"{part}_{k}": [] for part in ("M", "L") for k in ("rtrue", "rperm", "nmask", "nwords")}
        reps = []
        skM, skL = skip(f"{name}_M_rperm"), skip(f"{name}_L_rperm")
        for d, (Mc, Lm, n_rep) in enumerate(draws(trials, tm_key, n, f"9|draw|{tag}|{name}")):
            reps.append(n_rep)
            Mv = [c * np.nan_to_num(tr["st"][tm_key]) for c, tr in zip(Mc, trials)]
            Lv = [np.where(m, np.nan_to_num(tr["st"][tm_key]), 0.0) for m, tr in zip(Lm, trials)]
            Ov = [np.where(~tr["st"]["is_list"] & (c == 0) & np.isfinite(tr["st"][s_key]), tr["st"][s_key], 0.0) for c, tr in zip(Mc, trials)]
            oM = R.test(Mv, [[o, l_] for o, l_ in zip(Ov, Lv)], n_perm(name, N_PERM_DRAW), f"9|perm|{tag}|{name}|M|{d + 1}", skM)
            oL = R.test(Lv, [[m, o] for m, o in zip(Mv, Ov)], n_perm(name, N_PERM_DRAW), f"9|perm|{tag}|{name}|L|{d + 1}", skL)
            for part, o in (("M", oM), ("L", oL)):
                for k in ("rtrue", "rperm", "nmask", "nwords"):
                    rec[f"{part}_{k}"].append(o[k])
        for k, v in rec.items():
            if ext is None or k.endswith("rperm"):
                res[f"{name}_{k}"] = np.stack([np.asarray(x) for x in v])
        if ext is None:
            res[f"{name}_n_with_replacement"] = np.array(reps)
    del ff, R
    gc.collect()
    torch.cuda.empty_cache()
    OUT.mkdir(parents=True, exist_ok=True)
    if ext is not None:
        target = OUT / f"sub-{sub}_ses-{ses}_ext{len(list(OUT.glob(f'sub-{sub}_ses-{ses}_ext*.npz'))) + 1}.npz"
    tmp = OUT / f"sub-{sub}_ses-{ses}.tmp{os.getpid()}.npz"
    np.savez(tmp, **res)
    os.replace(tmp, target)
    return target
