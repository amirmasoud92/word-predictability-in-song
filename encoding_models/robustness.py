"""Robustness of the word-level gain (spoken, sung and hummed lyrics, Alice) to one-factor-at-a-time variants (Fig. 2d).

Variants of the primary specification (words evaluated 0-800 ms, all channels, lags 0-688 ms, GPT-2 small): window
0-600 ms; window 0-1000 ms; nine centro-parietal channels (CantiSong: Cz, CPz, Pz, C1, C2, CP1, CP2, P1, P2; Alice: the
nine channels nearest the midline electrode just behind the vertex, channel 4 of the release); lags 0-900 ms; GPT-2
medium surprisal. Window and channel variants reuse the fits of the primary specification. Per test, one-sided sign
flips over listeners with max-t correction over the six specifications.

Original estimates: 10 permutations, CPU engine, released annotation. Stabilised estimates: the window and channel
variants come with the stabilised word-level estimate (encoding_models/word_level_gain.py); the long lags and GPT-2
medium are computed here with up to 500 permutations on their own random sub-streams (3 and 4), corrected annotation of
song 09 (+2.88 s, from the annotation audit).
"""
from __future__ import annotations

import json
import os
import zlib

import numpy as np
import pandas as pd

from encoding_models import ridge_encoding as E, word_level_trials as P
from readers import alice as h, cantisong as c
from statistical_tests.max_t import signflip_maxt
from stimulus_features import song_regressors as g

N_SHUF, LAGS_REG, LAGS_LONG, BATCH = 10, 23, 29, 12
CANTI_ROI = ["Cz", "CPz", "Pz", "C1", "C2", "CP1", "CP2", "P1", "P2"]
COLS = ["primary", "window600", "window1000", "roi", "lags900", "gpt2_medium"]


def alice_roi(paths) -> np.ndarray:
    """Indices of the nine Alice channels nearest channel 4 (release electrode positions)."""
    rows = [ln.split() for ln in (h.ROOT / "easycapM10-acti61_elec.sfp").read_text().splitlines() if ln.strip()]
    pos = {r[0]: np.array([float(x) for x in r[1:4]]) for r in rows}
    labels = json.loads(str(np.load(paths.derived_root / "alice" / "S01.npz")["meta"]))["labels"]
    return np.argsort([np.linalg.norm(pos[ch] - pos["4"]) for ch in labels])[:9]


def canti_roi() -> np.ndarray:
    labels = c.load_version(1, "song")[1]
    return np.array([labels.index(ch) for ch in CANTI_ROI if ch in labels])


# ---------------------------------------------------------------- original estimates
def _build(trial_specs: list[dict], a: int) -> list[dict]:
    out = []
    for sp in trial_specs:
        st, Y, v = sp["st"], sp["Y"], sp["v"]
        T = min(Y.shape[1], st["acoustic"].shape[1] - a)
        F = {k: st[k][:, a:a + T] for k in ("acoustic", "phonetic", "word", "surprisal") if k in st}
        out.append({"F": F, "Y": Y[:, :T], "v": v[:T], "T": T, "words": [i - a for i in st["word_idx"]],
                    "raw": st["raw_surprisal"], "idx": st["word_idx"], "a": a})
    return out


def _controls(trials: list[dict], rng) -> list[dict]:
    ctrls = []
    for _ in range(N_SHUF):
        cdict = {}
        for j, tr in enumerate(trials):
            raw = np.zeros_like(tr["raw"])
            raw[tr["idx"]] = rng.permutation(tr["raw"][tr["idx"]])
            cdict[j] = g.bz(raw[None])[:, tr["a"]:tr["a"] + tr["T"]]
        ctrls.append(cdict)
    return ctrls


def original_listener(key: str, specs_small: list[dict], specs_medium: list[dict], a: int, roi, seed: int) -> dict:
    """One listener (key 'dataset|version|listener'): the six specifications with 10 permutations each."""
    rng = np.random.default_rng([seed, zlib.crc32(key.encode()), 11])
    tr = _build(specs_small, a)
    base = [k for k in ("acoustic", "phonetic", "word") if k in tr[0]["F"]]
    variants = {"primary": (E.event_masks(tr, "words", 0.8), None), "window600": (E.event_masks(tr, "words", 0.6), None),
                "window1000": (E.event_masks(tr, "words", 1.0), None), "roi": (E.event_masks(tr, "words", 0.8), roi)}
    out, _ = E.delta_r(tr, base, "surprisal", _controls(tr, rng), variants, LAGS_REG)
    reg_only = {"primary": variants["primary"]}
    out["lags900"] = E.delta_r(tr, base, "surprisal", _controls(tr, rng), reg_only, LAGS_LONG)[0]["primary"]
    trm = _build(specs_medium, a)
    out["gpt2_medium"] = E.delta_r(trm, base, "surprisal", _controls(trm, rng),
                                   {"primary": (E.event_masks(trm, "words", 0.8), None)}, LAGS_REG)[0]["primary"]
    return {"key": key, **{k: float(v) for k, v in out.items()}}


def job_list(paths, corrected: bool) -> list[tuple]:
    """(key, trial specs with GPT-2 small surprisal, with GPT-2 medium surprisal, first sample, channel subset) for every
    CantiSong listener and version and every Alice listener. ``corrected`` selects the annotation of song 09 (original
    estimates: released; stabilised estimates: corrected)."""
    os.environ["CANTISONG_ANNOTATION"] = "corrected" if corrected else "released"
    data = c.load_all(paths.derived_root / "cantisong_1to8", "1-8")
    words = {s: (g.load_words(s) if corrected else pd.read_csv(paths.derived_root / "cantisong_words" / f"{s}_words.csv"))
             for s in c.song_ids()}
    wmed = {}
    for s in c.song_ids():
        m = pd.read_csv(paths.derived_root / "cantisong_models" / f"{s}_words_medium.csv")
        if corrected:
            assert len(m) == len(words[s]) and (m.word.to_numpy() == words[s].word.to_numpy()).all()
            w = words[s].copy()
            w["surprisal_bits"] = m["surprisal_medium_bits"].to_numpy()
        else:
            w = m
            w["surprisal_bits"] = w["surprisal_medium_bits"]
        wmed[s] = w
    roi = canti_roi()
    a = int(c.START_S * 64)
    jobs = []
    for v in g.AUDIO:
        st_s = {s: g.stimulus(s, v, words[s], False) for s in c.song_ids()}
        st_m = {s: g.stimulus(s, v, wmed[s], False) for s in c.song_ids()}
        subs = sorted({sub for (vv, s), d in data.items() if vv == v for sub in d})
        for sub in subs:
            songs = [s for s in c.song_ids() if (v, s) in data and sub in data[(v, s)] and data[(v, s)][sub][1].any()]
            if len(songs) < 6:
                continue
            mk = lambda st: [{"st": st[s], "Y": data[(v, s)][sub][0], "v": data[(v, s)][sub][1], "song": s} for s in songs]
            jobs.append((f"canti|{v}|{sub}", mk(st_s), mk(st_m), a, roi))
    words_a = pd.read_csv(paths.derived_root / "alice" / ("words_gpt2_entropy.csv" if corrected else "words_gpt2.csv"))
    wmed_a = pd.read_csv(paths.derived_root / "alice" / "words_gpt2_medium.csv")
    st_s = {s: h.seg_stim(s, words_a, "gpt2_surprisal_bits") for s in h.SEGMENTS}
    st_m = {s: h.seg_stim(s, wmed_a, "gpt2_medium_surprisal_bits") for s in h.SEGMENTS}
    data8, _ = h.load_all(paths.derived_root / "alice", "1-8")
    roi_a = alice_roi(paths)
    a_a = int(h.START_S * 64)
    for sub in sorted(data8):
        segs = [s for s in h.SEGMENTS if s in data8[sub] and (not corrected or data8[sub][s][1].any())]
        if corrected and len(segs) < h.MIN_SEGMENTS:
            continue
        mk = lambda st: [{"st": st[s], "Y": data8[sub][s][0], "v": data8[sub][s][1], "song": s} for s in segs]
        jobs.append((f"alice|speech|{sub}", mk(st_s), mk(st_m), a_a, roi_a))
    return jobs


# ---------------------------------------------------------------- stabilised estimates (long lags and GPT-2 medium)
def stabilised_listener(key: str, specs_s: list[dict], specs_m: list[dict], a: int, seed: int, n: int) -> dict:
    """Lags 0-900 ms (sub-stream 3) and GPT-2 medium (sub-stream 4): true and per-permutation channel-mean accuracies."""
    from encoding_models import gpu_encoding as WG
    tr = P.canti_trials(specs_s, a)
    base = [k for k in ("acoustic", "phonetic", "word") if k in tr[0]["F"]]
    out = {"key": key}
    d, lam29, nl = WG.delta_r(tr, base, "surprisal", P.perm_controls(tr, "raw_s", P.rng_for(seed, key, 3), n),
                              {"r": (E.event_masks(tr, "words", 0.8), None)}, 29, batch=BATCH)
    out["null_lags900"], out["true_lags900"] = nl["r"], float(d["r"] + float(np.mean(nl["r"])))
    trm = P.canti_trials(specs_m, a)
    d, lam, nl = WG.delta_r(trm, base, "surprisal", P.perm_controls(trm, "raw_s", P.rng_for(seed, key, 4), n),
                            {"r": (E.event_masks(trm, "words", 0.8), None)}, P.N_LAGS, batch=BATCH)
    out["null_gpt2_medium"], out["true_gpt2_medium"] = nl["r"], float(d["r"] + float(np.mean(nl["r"])))
    out["lam"], out["lam_lags900"] = lam, lam29
    return out


def max_t_table(df: pd.DataFrame) -> dict:
    """Per dataset and version: mean, d_z, uncorrected and max-t corrected one-sided p and positive listeners of each of
    the six specifications (columns COLS of ``df``, one row per listener)."""
    res = {}
    for (ds, v), grp in df.groupby(["dataset", "version"]):
        M = grp[COLS].to_numpy(float)
        _, p_unc, p_fwe = signflip_maxt(M, alternative="greater")
        res[f"{ds}|{v}"] = {c_: {"mean": float(M[:, j].mean()), "dz": float(M[:, j].mean() / M[:, j].std(ddof=1)),
                                 "p_unc": float(p_unc[j]), "p_fwe": float(p_fwe[j]), "n_positive": int((M[:, j] > 0).sum())}
                            for j, c_ in enumerate(COLS)} | {"n": int(len(grp))}
    return res


def stabilised_table(rows: list[dict], core: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Long-lag and GPT-2 medium gains per listener (true minus the mean of the first n permutations, n from the Monte
    Carlo rule per dataset, version and variant), merged with the primary, window and channel variants of the stabilised
    word-level estimate (``core``: dataset, version, listener, pooled, window600, window1000, roi)."""
    from encoding_models.word_level_gain import mc_rule
    mc, table = {}, {}
    groups = sorted({"|".join(r["key"].split("|")[:2]) for r in rows})
    for gname in groups:
        rr = [r for r in rows if r["key"].startswith(gname + "|")]
        for t in ("lags900", "gpt2_medium"):
            tv = np.array([r[f"true_{t}"] for r in rr])
            n_min = min(len(r[f"null_{t}"]) for r in rr)
            nv = np.array([r[f"null_{t}"][:n_min] for r in rr])
            rule = mc_rule(tv, nv)
            mc[f"{gname}|{t}"] = rule
            for r, x in zip(rr, tv - nv[:, :rule["n_used"]].mean(1)):
                ds, v, sub = r["key"].split("|")
                table.setdefault(r["key"], {"dataset": ds, "version": v, "listener": sub})[t] = float(x)
    L = pd.DataFrame(list(table.values()))
    M = L.merge(core.astype({"listener": str}), on=["dataset", "version", "listener"], how="left").rename(columns={"pooled": "primary"})
    return M, mc
