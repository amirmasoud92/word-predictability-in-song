"""Melodic surprisal and the interference between melodic and lexical predictability (CantiSong).

L1  melodic surprisal (note-onset impulses scaled by surprisal) against permutations of the values among the notes of
    each song; prediction accuracy 0-800 ms after each note onset; base: acoustics and note onsets, plus, for the sung
    versions, phonetic features, word onsets and lexical (GPT-2) surprisal. Sung and hummed versions.
L2  sung versions: each word is paired with the note whose onset is nearest to the word onset, within 150 ms. Words are
    split at the song's median melodic surprisal of their paired notes (at or below: expected note; above: unexpected
    note); words without a note onset within 150 ms are in neither half. Lexical surprisal enters as two regressors, one
    per half, each tested against permutations within its half, in a model that also contains note onsets and melodic
    surprisal; accuracy 0-800 ms after the tested half's words. L2 = delta r(expected) - delta r(unexpected), one-sided.
L3  the mirror image: notes split at the song's median lexical surprisal of the nearest word within 150 ms, melodic
    surprisal as two regressors, in a model that also contains word onsets and lexical surprisal.

The ridge is chosen on the L1 base model and fixed for L2 and L3. Original estimates: 10 permutations, CPU engine;
stabilised estimates: up to 500 permutations stored per listener, GPU engine, Monte Carlo rule. Both use the corrected
annotation of song 09 (+2.88 s, from the annotation audit) and the composed notes aligned by dynamic time warping.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from encoding_models import ridge_encoding as E, word_level_trials as P
from encoding_models.word_level_gain import mc_rule
from readers import cantisong as c
from statistical_tests.inference import cohens_dz, dz_ci, jzs_bf10, sesoi_resource_based, sign_flip_test, summarise, tost_one_sample
from stimulus_features import song_regressors as g

N_LAGS, N_PERM, PAIR_S, BATCH = 23, 10, 0.15, 12


def impulses(idx: np.ndarray, vals: np.ndarray, n: int) -> np.ndarray:
    R = np.zeros(n)
    ok = (idx >= 0) & (idx < n)
    np.add.at(R, idx[ok], vals[ok])
    return R


def song_features(s: str, version: str, words: pd.DataFrame, notes: pd.DataFrame) -> dict:
    """Stimulus regressors of one song and version, the note onsets and melodic surprisal, and (sung) the word-note
    pairing: the melodic surprisal of each word's nearest note and the lexical surprisal of each note's nearest word,
    within 150 ms (NaN otherwise)."""
    st = g.stimulus(s, version, words, False)
    n = st["acoustic"].shape[1]
    nt = notes.dropna(subset=["melodic_surprisal_bits"])
    n_idx = (nt.onset_audio.to_numpy() * 64).round().astype(int)
    msur = nt.melodic_surprisal_bits.to_numpy()
    out = {"st": st, "n": n, "note_idx": n_idx, "msur": msur, "note_on": g.bz(impulses(n_idx, np.ones_like(msur), n)[None])}
    if version == "song":
        w = words.dropna(subset=["onset_song"])
        w_idx = (w.onset_song.to_numpy() * 64).round().astype(int)
        lsur = w.surprisal_bits.to_numpy()
        w_on, n_on = w.onset_song.to_numpy(), nt.onset_audio.to_numpy()
        w_mel = np.array([msur[np.argmin(np.abs(n_on - x))] if np.min(np.abs(n_on - x)) <= PAIR_S else np.nan for x in w_on])
        n_lex = np.array([lsur[np.argmin(np.abs(w_on - x))] if np.min(np.abs(w_on - x)) <= PAIR_S else np.nan for x in n_on])
        out |= {"w_idx": w_idx, "lsur": lsur, "w_mel": w_mel, "n_lex": n_lex}
    return out


def inputs(paths, notes_dir: str = "cantisong_notes") -> tuple[dict, dict, list]:
    """EEG (1-8 Hz), features of every song in the sung and hummed versions, and the listeners."""
    os.environ["CANTISONG_ANNOTATION"] = "corrected"
    data = c.load_all(paths.derived_root / "cantisong_1to8", "1-8")
    words = {s: g.load_words(s) for s in c.song_ids()}
    notes = {s: pd.read_csv(paths.derived_root / notes_dir / f"{s}_notes.csv") for s in c.song_ids()}
    feats = {(s, v): song_features(s, v, words[s], notes[s]) for s in c.song_ids() for v in ("song", "melody")}
    subs = sorted({sub for d in data.values() for sub in d})
    return data, feats, subs


def _tests(sub: int, version: str, data: dict, feats: dict, rng, n_perm: int, run) -> bool:
    """L1, then (sung) L2 and L3 for one listener and version. ``run(name, trials, base, tested, controls, masks, lam)``
    estimates one test and returns the ridge; controls are drawn from ``rng`` in the order of the tests."""
    songs = [s for s in c.song_ids() if (version, s) in data and sub in data[(version, s)] and data[(version, s)][sub][1].any()]
    if len(songs) < 6:
        return False
    a = int(c.START_S * 64)

    def trials_with(extra: dict) -> list[dict]:
        trs = []
        for s in songs:
            f = feats[(s, version)]
            Y, v = data[(version, s)][sub]
            T = min(Y.shape[1], f["n"] - a)
            F = {"acoustic": f["st"]["acoustic"][:, a:a + T], "note_on": f["note_on"][:, a:a + T]}
            if version == "song":
                F |= {"phonetic": f["st"]["phonetic"][:, a:a + T], "word": f["st"]["word"][:, a:a + T]}
            for k, fn in extra.items():
                F[k] = fn(f)[:, a:a + T]
            trs.append({"F": F, "Y": Y[:, :T], "v": v[:T], "T": T, "song": s,
                        "notes": (f["note_idx"] - a).tolist(), "words": ((f["w_idx"] - a).tolist() if version == "song" else [])})
        return trs

    def ctrls(trs, maker):
        return [{j: maker(feats[(tr["song"], version)])[:, a:a + tr["T"]] for j, tr in enumerate(trs)} for _ in range(n_perm)]

    mel = lambda f: g.bz(impulses(f["note_idx"], f["msur"], f["n"])[None])
    mel_perm = lambda f: g.bz(impulses(f["note_idx"], rng.permutation(f["msur"]), f["n"])[None])
    extra = {"lex": lambda f: g.bz(impulses(f["w_idx"], f["lsur"], f["n"])[None])} if version == "song" else {}
    trs = trials_with(extra | {"mel": mel})
    base = [k for k in ("acoustic", "note_on", "phonetic", "word", "lex") if k in trs[0]["F"]]
    lam = run("L1_melodic", trs, base, "mel", ctrls(trs, mel_perm), E.event_masks(trs, "notes", 0.8), None)
    if version != "song":
        return True

    def split_w(f, half):
        med = np.nanmedian(f["w_mel"])
        sel = (f["w_mel"] <= med) if half == "low" else (f["w_mel"] > med)
        return sel & np.isfinite(f["w_mel"])
    lex_h = lambda half, perm=False: (lambda f: g.bz(impulses(f["w_idx"][split_w(f, half)],
                                                             (rng.permutation(f["lsur"][split_w(f, half)]) if perm else f["lsur"][split_w(f, half)]), f["n"])[None]))
    for half, other in (("low", "high"), ("high", "low")):
        trs = trials_with({"mel": mel, f"lex_{half}": lex_h(half), f"lex_{other}": lex_h(other)})
        for tr in trs:
            f = feats[(tr["song"], version)]
            tr["half_words"] = (f["w_idx"][split_w(f, half)] - a).tolist()
        base2 = ["acoustic", "note_on", "phonetic", "word", "mel", f"lex_{other}"]
        run(f"L2_lex_{half}", trs, base2, f"lex_{half}", ctrls(trs, lex_h(half, True)), E.event_masks(trs, "half_words", 0.8), lam)

    def split_n(f, half):
        med = np.nanmedian(f["n_lex"])
        sel = (f["n_lex"] <= med) if half == "low" else (f["n_lex"] > med)
        return sel & np.isfinite(f["n_lex"])
    mel_h = lambda half, perm=False: (lambda f: g.bz(impulses(f["note_idx"][split_n(f, half)],
                                                             (rng.permutation(f["msur"][split_n(f, half)]) if perm else f["msur"][split_n(f, half)]), f["n"])[None]))
    for half, other in (("low", "high"), ("high", "low")):
        trs = trials_with({"lex": lambda f: g.bz(impulses(f["w_idx"], f["lsur"], f["n"])[None]),
                           f"mel_{half}": mel_h(half), f"mel_{other}": mel_h(other)})
        for tr in trs:
            f = feats[(tr["song"], version)]
            tr["half_notes"] = (f["note_idx"][split_n(f, half)] - a).tolist()
        base3 = ["acoustic", "note_on", "phonetic", "word", "lex", f"mel_{other}"]
        run(f"L3_mel_{half}", trs, base3, f"mel_{half}", ctrls(trs, mel_h(half, True)), E.event_masks(trs, "half_notes", 0.8), lam)
    return True


def original_listener(sub: int, version: str, data: dict, feats: dict, seed: int) -> dict:
    """Original estimates: 10 permutations, CPU engine."""
    rng = np.random.default_rng([seed, sub, 21 if version == "song" else 22])
    out = {"listener": sub, "version": version}

    def run(name, trs, base, tested, cl, masks, lam):
        d, lam_ = E.delta_r(trs, base, tested, cl, {"w": (masks, None)}, N_LAGS, lam)
        out[name] = d["w"]
        return lam_
    return out if _tests(sub, version, data, feats, rng, N_PERM, run) else {}


def stabilised_listener(sub: int, version: str, data: dict, feats: dict, seed: int, n: int) -> dict:
    """Stabilised estimates: n permutations per test on the GPU engine; true and per-permutation accuracies stored."""
    from encoding_models import gpu_encoding as WG
    key = f"L|{version}|{sub}"
    rng = P.rng_for(seed, key)
    out = {"key": key, "listener": sub, "version": version}

    def run(name, trs, base, tested, cl, masks, lam):
        d, lam_, nl = WG.delta_r(trs, base, tested, cl, {"w": (masks, None)}, N_LAGS, lam, batch=BATCH)
        out[f"null_{name}"] = nl["w"]
        out[f"true_{name}"] = float(d["w"] + float(np.mean(nl["w"])))
        return lam_
    return out if _tests(sub, version, data, feats, rng, n, run) else {}


TESTS = ["L1_melodic", "L2_lex_low", "L2_lex_high", "L3_mel_low", "L3_mel_high"]


def pick(x, alt="greater"):
    return {k: v for k, v in summarise(np.asarray(x, float), alt).items() if k in ("n", "mean", "dz", "dz_ci95", "p_signflip", "n_positive")}


def summary(df: pd.DataFrame) -> dict:
    """Group tests over listeners (one-sided): L1 sung and hummed, each half of L2 and L3, and the two contrasts."""
    sung, hum = df[df.version == "song"], df[df.version == "melody"]
    return {"L1_melodic_sung": pick(sung.L1_melodic), "L1_melodic_hummed": pick(hum.L1_melodic),
            "L2_lex_low": pick(sung.L2_lex_low), "L2_lex_high": pick(sung.L2_lex_high),
            "L2_low_minus_high": pick(sung.L2_lex_low - sung.L2_lex_high),
            "L3_mel_low": pick(sung.L3_mel_low), "L3_mel_high": pick(sung.L3_mel_high),
            "L3_low_minus_high": pick(sung.L3_mel_low - sung.L3_mel_high)}


def stabilised_table(rows: list[dict]) -> tuple[pd.DataFrame, dict]:
    """Per listener and test: true accuracy minus the mean of the first n permutations, n from the Monte Carlo rule
    applied per version and test over listeners. Returns the listener table and the rule per version and test."""
    mc, vals = {}, {}
    for version in sorted({r["version"] for r in rows}):
        rr = [r for r in rows if r["version"] == version]
        for t in TESTS:
            have = [r for r in rr if f"null_{t}" in r]
            if not have:
                continue
            tv = np.array([r[f"true_{t}"] for r in have])
            n_min = min(len(r[f"null_{t}"]) for r in have)
            nv = np.array([r[f"null_{t}"][:n_min] for r in have])
            rule = mc_rule(tv, nv)
            mc[f"{version}|{t}"] = rule
            for r, x in zip(have, tv - nv[:, :rule["n_used"]].mean(1)):
                vals.setdefault(r["key"], {"listener": r["listener"], "version": version})[t] = float(x)
    return pd.DataFrame(list(vals.values())), mc


def bounds(x: np.ndarray) -> dict:
    """Evidence about the size of a contrast: d_z with 90 % and 95 % CIs, one-sided BF+0, two-sided BF01, TOST at
    d_z = 0.5 and at the d_z detectable with 80 % power."""
    x = np.asarray(x, float)
    sesoi = sesoi_resource_based(x.size, 0.8, 0.05, "larger")
    t5, ts = tost_one_sample(x, 0.5), tost_one_sample(x, sesoi)
    return {"n": int(x.size), "dz": cohens_dz(x), "dz_ci90": list(dz_ci(x, 0.90)), "dz_ci95": list(dz_ci(x, 0.95)),
            "bf_plus0": float(jzs_bf10(x, alternative="greater")), "bf_01": float(1 / jzs_bf10(x, alternative="two-sided")),
            "tost_0.5_p": t5.p, "sesoi_80pct_power": sesoi, "tost_sesoi_p": ts.p}


def interference_bounds(df: pd.DataFrame) -> dict:
    """Bounds for L2 and L3 (original estimates), the ratio of the L2 halves (unexpected / expected, ratio of mean delta
    r), and L1 sung - hummed (two-sided)."""
    sung, hum = df[df.version == "song"].set_index("listener"), df[df.version == "melody"].set_index("listener")
    res = {"L2_low_minus_high": bounds(sung.L2_lex_low - sung.L2_lex_high),
           "L3_low_minus_high": bounds(sung.L3_mel_low - sung.L3_mel_high),
           "L2_ratio_high_over_low": float(sung.L2_lex_high.mean() / sung.L2_lex_low.mean())}
    both = sung[["L1_melodic"]].join(hum[["L1_melodic"]], lsuffix="_sung", rsuffix="_hummed").dropna()
    d = (both.L1_melodic_sung - both.L1_melodic_hummed).to_numpy()
    res["L1_sung_minus_hummed"] = {"n": int(d.size), "mean": float(d.mean()), "dz": cohens_dz(d), "dz_ci95": list(dz_ci(d)),
                                   "p_two_sided": float(sign_flip_test(d, "two-sided")[0])}
    return res


def write(out: Path, df: pd.DataFrame, name: str):
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{name}_listeners.csv", index=False)
    (out / f"{name}.json").write_text(json.dumps(summary(df), indent=2, default=float))


def stabilised_l1_by_song(sub: int, version: str, data: dict, feats: dict, seed: int, n: int) -> dict:
    """L1 with per-song detail on the stabilised random stream ('L|version|listener'; L1 is its first test) and its
    first n permutations (n of the Monte Carlo rule for L1): the listener value and the gain of each song (Fig. S4c)."""
    from encoding_models import gpu_encoding as WG
    key = f"L|{version}|{sub}"
    songs = [s for s in c.song_ids() if (version, s) in data and sub in data[(version, s)] and data[(version, s)][sub][1].any()]
    if len(songs) < 6:
        return {}
    a = int(c.START_S * 64)
    rng = P.rng_for(seed, key)
    trs = []
    for s in songs:
        ff = feats[(s, version)]
        Y, v = data[(version, s)][sub]
        T = min(Y.shape[1], ff["n"] - a)
        F = {"acoustic": ff["st"]["acoustic"][:, a:a + T], "note_on": ff["note_on"][:, a:a + T]}
        if version == "song":
            F |= {"phonetic": ff["st"]["phonetic"][:, a:a + T], "word": ff["st"]["word"][:, a:a + T],
                  "lex": g.bz(impulses(ff["w_idx"], ff["lsur"], ff["n"])[None])[:, a:a + T]}
        F["mel"] = g.bz(impulses(ff["note_idx"], ff["msur"], ff["n"])[None])[:, a:a + T]
        trs.append({"F": F, "Y": Y[:, :T], "v": v[:T], "T": T, "song": s, "notes": (ff["note_idx"] - a).tolist()})
    ctrls = [{j: g.bz(impulses(feats[(tr["song"], version)]["note_idx"], rng.permutation(feats[(tr["song"], version)]["msur"]),
                               feats[(tr["song"], version)]["n"])[None])[:, a:a + tr["T"]] for j, tr in enumerate(trs)} for _ in range(n)]
    base = [k for k in ("acoustic", "note_on", "phonetic", "word", "lex") if k in trs[0]["F"]]
    det, lam = WG.delta_r_detail(trs, base, "mel", ctrls, E.event_masks(trs, "notes", 0.8), N_LAGS)
    return {"key": key, "listener": sub, "version": version, "n": n, "lam": lam, "pooled": det["pooled"], "songs": songs,
            "per_song": [float(x) for x in det["per_trial"]]}
