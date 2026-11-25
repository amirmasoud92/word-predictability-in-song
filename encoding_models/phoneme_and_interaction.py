"""Word surprisal beyond phoneme surprisal, phoneme surprisal beyond word surprisal, and the continuous interaction of
lexical and melodic surprisal (CantiSong; Fig. S5).

Pipeline of the stabilised word-level estimate: 1-8 Hz EEG at 64 Hz, lags 0-688 ms, ridge chosen on each test's base
model and then fixed, leave-one-song-out, accuracy within 0-800 ms after the tested words averaged over channels, up to
500 permutations with the Monte Carlo rule, corrected annotation of song 09 (+2.88 s, from the annotation audit), GPU
engine. Random streams: default_rng([seed, CRC32(job key), 11, sub-stream]).

Word and phoneme surprisal (phone regressors: stimulus_features/phoneme_cohort.py). Base A: acoustics, phonetic features,
phone onsets (every annotated phone) and word onsets (hummed: no phonetic features).
  1a  phoneme surprisal beyond base A; values permuted among the valued phones of each song (one-sided; a pipeline check).
  1b  word surprisal beyond base A + phoneme surprisal + cohort entropy; values permuted among the words (one-sided).
  1c  phoneme surprisal beyond base A + word surprisal; values permuted among the phones of the same position class
      (word-initial; later) within each song, so that the null keeps the link between word-initial phones and words
      (one-sided).
  1d  the 1b and 1c models in the hummed versions with the sung timing (control; two-sided, BF01).
Interaction (sung words with a paired note, nearest note onset within 150 ms): word-onset impulses scaled by
z(lexical surprisal) x z(melodic surprisal of the paired note), both z-scored within the song over the paired words,
tested beyond acoustics, phonetic features, word onsets, lexical surprisal (all words), note onsets and melodic surprisal
(all notes); null: melodic values permuted among the paired words of each song, lexical values in place; accuracy 0-800
ms after the paired words. 3a: sung versions (two-sided, BF01, TOST at +-0.5, and the full-model kernel of the product
regressor at 219-406 ms in the nine centro-parietal channels); 3b: hummed versions with the sung word timing (control).
Song-level estimates: the per-song gain of each listener (channel mean of the per-song correlations, true minus mean
permuted), averaged over listeners.
"""
from __future__ import annotations

import os
import zlib

import numpy as np
import pandas as pd

from encoding_models import ridge_encoding as E
from encoding_models.melodic_surprisal import impulses
from encoding_models.word_level_gain import mc_rule
from readers import cantisong as c
from statistical_tests.inference import jzs_bf10, summarise, tost_one_sample
from stimulus_features import song_regressors as g

N_LAGS, WIN_S, N_MAX, STREAM, BATCH = 23, 0.8, 500, 11, 12
VERSIONS = ("speech", "song", "melody")
ROI = ["Cz", "CPz", "Pz", "C1", "C2", "CP1", "CP2", "P1", "P2"]


def rng_for(seed: int, key: str, sub: int = 0) -> np.random.Generator:
    return np.random.default_rng([seed, zlib.crc32(key.encode()), STREAM, sub])


def songs_of(data: dict, sub: int, V: str) -> list[str]:
    return [s for s in c.song_ids() if (V, s) in data and sub in data[(V, s)] and data[(V, s)][sub][1].any()]


def stimuli() -> tuple[dict, dict]:
    """Word tables and stimulus regressors of every song and version (corrected annotation)."""
    os.environ["CANTISONG_ANNOTATION"] = "corrected"
    words = {s: g.load_words(s) for s in c.song_ids()}
    return words, {(s, V): g.stimulus(s, V, words[s], False) for s in c.song_ids() for V in VERSIONS}


def word_trials(data: dict, stims: dict, phones: dict | None, sub: int, V: str) -> list[dict]:
    """Trials of one listener and version; with ``phones`` ((song, version) -> phoneme_cohort.phone_features), the phone
    onsets, phoneme surprisal and cohort entropy regressors."""
    a = int(c.START_S * 64)
    trials = []
    for s in songs_of(data, sub, V):
        st = stims[(s, V)]
        Y, v = data[(V, s)][sub]
        n = st["acoustic"].shape[1]
        T = min(Y.shape[1], n - a)
        F = {k: st[k][:, a:a + T] for k in ("acoustic", "phonetic", "word", "surprisal") if k in st}
        tr = {"F": F, "Y": Y[:, :T], "v": v[:T], "T": T, "a": a, "n": n, "song": s, "words": [i - a for i in st["word_idx"]],
              "idx": np.asarray(st["word_idx"]), "raw_s": st["raw_surprisal"]}
        if phones is not None:
            pf = phones[(s, V)]
            F["phone_on"] = g.bz(pf["phone_on"][None])[:, a:a + T]
            F["psur"] = g.bz(impulses(pf["pidx"], pf["psur"], n)[None])[:, a:a + T]
            F["pent"] = g.bz(impulses(pf["pidx"], pf["pent"], n)[None])[:, a:a + T]
            tr |= {"pidx": pf["pidx"], "psur": pf["psur"], "pcls": pf["pcls"]}
        trials.append(tr)
    return trials


def word_perms(trials, rng, n):
    """Word-surprisal values permuted among the words of each song (timing kept), band-passed."""
    for _ in range(n):
        cd = {}
        for j, tr in enumerate(trials):
            raw = np.zeros_like(tr["raw_s"])
            raw[tr["idx"]] = rng.permutation(tr["raw_s"][tr["idx"]])
            cd[j] = g.bz(raw[None])[:, tr["a"]:tr["a"] + tr["T"]]
        yield cd


def phone_perms(trials, rng, n, by_class: bool):
    """Phoneme-surprisal values permuted among the valued phones of each song, or within each position class."""
    for _ in range(n):
        cd = {}
        for j, tr in enumerate(trials):
            vals = tr["psur"].copy()
            if by_class:
                for cl in (0, 1):
                    m = tr["pcls"] == cl
                    vals[m] = rng.permutation(vals[m])
            else:
                vals = rng.permutation(vals)
            cd[j] = g.bz(impulses(tr["pidx"], vals, tr["n"])[None])[:, tr["a"]:tr["a"] + tr["T"]]
        yield cd


def run_one(trials, base, test, controls, mask_key: str) -> dict:
    """True and per-permutation channel-mean accuracy (pooled over songs, and per song)."""
    from encoding_models import gpu_encoding as WG
    folds = WG.GpuFolds(trials, base, N_LAGS)
    ms = WG.MaskSet(folds, E.event_masks(trials, mask_key, WIN_S))
    true, ctrl = WG.run_test(folds, trials, test, controls, {"m": ms}, N_LAGS, batch=BATCH)
    t, cc = true["m"], ctrl["m"]
    return {"true": float(t["pooled"].mean()), "null": cc["pooled"].mean(1), "true_trial": t["per"].mean(1),
            "null_trial": cc["per"].mean(2), "lam": folds.lam, "units": [tr["song"] for tr in trials]}


def word_phone_job(data, stims, phones, sub: int, V: str, seed: int, n: int = N_MAX) -> dict:
    """Tests 1a-1c (spoken, sung) or 1d (hummed) for one listener (job key 'a1|version|listener')."""
    trials = word_trials(data, stims, phones, sub, V)
    if len(trials) < 6:
        return {}
    key = f"a1|{V}|{sub}"
    A_ = ["acoustic", "phonetic", "phone_on", "word"] if V != "melody" else ["acoustic", "phone_on", "word"]
    res = {}
    if V != "melody":
        res["1a"] = run_one(trials, A_, "psur", phone_perms(trials, rng_for(seed, key, 1), n, False), "words")
    res["1b"] = run_one(trials, A_ + ["psur", "pent"], "surprisal", word_perms(trials, rng_for(seed, key, 2), n), "words")
    res["1c"] = run_one(trials, A_ + ["surprisal"], "psur", phone_perms(trials, rng_for(seed, key, 3), n, True), "words")
    return {"key": key, "listener": sub, "version": V, "tests": res}


# ---------------------------------------------------------------- interaction
def interaction_trials(data, feats: dict, sub: int, V: str) -> list[dict]:
    """feats: (song, version) -> melodic_surprisal.song_features for 'song' and 'melody'."""
    a = int(c.START_S * 64)
    trials = []
    for s in songs_of(data, sub, V):
        fs, fv = feats[(s, "song")], feats[(s, V)]
        st, n = fv["st"], fv["n"]
        Y, v = data[(V, s)][sub]
        T = min(Y.shape[1], n - a)
        paired = np.isfinite(fs["w_mel"])
        wi, ls, wm = fs["w_idx"][paired], fs["lsur"][paired], fs["w_mel"][paired]
        zl, zm = (ls - ls.mean()) / ls.std(), (wm - wm.mean()) / wm.std()
        F = {"acoustic": st["acoustic"][:, a:a + T], "word": st["word"][:, a:a + T], "note_on": fv["note_on"][:, a:a + T],
             "lex": g.bz(impulses(fs["w_idx"], fs["lsur"], n)[None])[:, a:a + T],
             "mel": g.bz(impulses(fv["note_idx"], fv["msur"], n)[None])[:, a:a + T],
             "prod": g.bz(impulses(wi, zl * zm, n)[None])[:, a:a + T]}
        if V == "song":
            F["phonetic"] = st["phonetic"][:, a:a + T]
        trials.append({"F": F, "Y": Y[:, :T], "v": v[:T], "T": T, "a": a, "n": n, "song": s, "pwords": (wi - a).tolist(),
                       "wi": wi, "zl": zl, "zm": zm, "n_words": int(len(paired)), "n_paired": int(paired.sum())})
    return trials


def prod_perms(trials, rng, n):
    for _ in range(n):
        yield {j: g.bz(impulses(tr["wi"], tr["zl"] * rng.permutation(tr["zm"]), tr["n"])[None])[:, tr["a"]:tr["a"] + tr["T"]]
               for j, tr in enumerate(trials)}


def full_weights(trials, names, lam) -> np.ndarray:
    """Weights of the model fitted on all songs (fixed ridge, scaled by the mean eigenvalue); the last block (23 lags x
    channels)."""
    G, C = 0.0, 0.0
    for tr in trials:
        X = E._design(tr, names, N_LAGS, None)[tr["v"]]
        G = G + X.T @ X
        C = C + X.T @ tr["Y"][:, tr["v"]].T
    p = G.shape[0]
    W = np.linalg.solve(G + lam * np.trace(G) / p * np.eye(p), C)
    return W[-N_LAGS:]


def interaction_job(data, feats, sub: int, V: str, seed: int, n: int = N_MAX) -> dict:
    """3a (sung) or 3b (hummed) for one listener (job key 'a3|version|listener'), with the full-model kernel."""
    trials = interaction_trials(data, feats, sub, V)
    if len(trials) < 6:
        return {}
    key = f"a3|{V}|{sub}"
    base = ["acoustic", "phonetic", "word", "lex", "note_on", "mel"] if V == "song" else ["acoustic", "word", "lex", "note_on", "mel"]
    r = run_one(trials, base, "prod", prod_perms(trials, rng_for(seed, key, 1), n), "pwords")
    r["kernel"] = full_weights(trials, base + ["prod"], float(r["lam"]))
    return {"key": key, "listener": sub, "version": V, "tests": {"3": r},
            "n_words": [tr["n_words"] for tr in trials], "n_paired": [tr["n_paired"] for tr in trials]}


# ---------------------------------------------------------------- statistics
SIDE = {"1a_speech": "greater", "1a_song": "greater", "1b_speech": "greater", "1b_song": "greater", "1c_speech": "greater",
        "1c_song": "greater", "1d_word": "two-sided", "1d_phone": "two-sided", "3a": "two-sided", "3b": "two-sided"}


def test_name(job: dict, t: str) -> str:
    if t == "3":
        return "3a" if job["version"] == "song" else "3b"
    if job["version"] == "melody":
        return "1d_word" if t == "1b" else "1d_phone"
    return f"{t}_{job['version']}"


def statistics(jobs: list[dict], roi: np.ndarray) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Monte Carlo rule per test, listener-level tests (BF01 for the two-sided tests, TOST for 3a), song-level
    estimates for 1b and 3a, and the 3a/3b kernel at 219-406 ms in the centro-parietal channels (``roi`` indices)."""
    tests = {}
    for job in jobs:
        for t, d in job["tests"].items():
            e = tests.setdefault(test_name(job, t), {"listeners": [], "true": [], "null": [], "true_trial": [], "null_trial": [], "units": [], "kernel": []})
            e["listeners"].append(job["listener"])
            for k in ("true", "null", "true_trial", "null_trial", "units"):
                e[k].append(d[k])
            if "kernel" in d:
                e["kernel"].append(d["kernel"])
    res, rows, song_rows = {}, [], []
    for name in sorted(tests):
        e = tests[name]
        tv = np.array(e["true"])
        nmin = min(len(x) for x in e["null"])
        nv = np.array([x[:nmin] for x in e["null"]])
        rule = mc_rule(tv, nv)
        nu = rule["n_used"]
        x = tv - nv[:, :nu].mean(1)
        s = summarise(x, SIDE[name])
        r = {k: s[k] for k in ("n", "mean", "dz", "dz_ci95", "p_signflip", "n_positive")} | {"alternative": SIDE[name], "n_perm_used": nu,
                                                                                              "mc_ratio": rule["ratio"], "mc_rule_met": rule["met"]}
        if SIDE[name] == "two-sided":
            r["bf01_two_sided"] = float(1.0 / jzs_bf10(x, alternative="two-sided"))
        if name == "3a":
            r["tost_0.5_p"] = tost_one_sample(x, 0.5).p
        res[name] = r
        rows += [{"test": name, "listener": L, "delta_r": float(v)} for L, v in zip(e["listeners"], x)]
        if name in ("1b_speech", "1b_song", "3a"):
            per = {}
            for units, tt, nt in zip(e["units"], e["true_trial"], e["null_trial"]):
                for u, a_, b_ in zip(units, tt, nt[:nu].mean(0)):
                    if np.isfinite(a_ - b_):
                        per.setdefault(u, []).append(float(a_ - b_))
            songs = sorted(per)
            xs = np.array([np.mean(per[u]) for u in songs])
            ss = summarise(xs, SIDE[name])
            res[f"{name}_songs"] = {k: ss[k] for k in ("n", "mean", "dz", "dz_ci95", "p_signflip", "n_positive")} | {"alternative": SIDE[name]}
            song_rows += [{"test": name, "song": u, "delta_r": float(v), "n_listeners": len(per[u])} for u, v in zip(songs, xs)]
        if e["kernel"]:
            ks = np.array(e["kernel"])
            w2 = ks[:, 7:14][:, :, roi].mean((1, 2))
            sk = summarise(w2, "two-sided")
            res[f"{name}_kernel_219_406ms_roi"] = {k: sk[k] for k in ("n", "mean", "dz", "dz_ci95", "p_signflip", "n_positive")} | \
                {"sign": "positive" if w2.mean() > 0 else "negative"}
    return res, pd.DataFrame(rows).pivot(index="listener", columns="test", values="delta_r"), pd.DataFrame(song_rows)
