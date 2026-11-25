"""Generalisation over songs, timing and topography of the word-level gain, and lexical entropy (CantiSong; Alice entropy).

Per listener and version, the original word-level model (corrected annotation of song 09, +2.88 s, from the annotation
audit; 10 permutations; CPU engine) with per-song, per-channel and sufficient-statistic detail:
* songs as the sample: per-song gains averaged over listeners, exact sign flips over songs, sung - hummed paired over
  songs;
* leave-one-song-out: the listener-level gain recomputed without each song from the sufficient statistics of the held-out
  predictions (no refit), then d_z over the listeners who heard the song left out;
* lag windows: the surprisal regressor restricted to lags 0-188, 219-406 or 438-688 ms, the base model keeping its lags
  (same permutation draws);
* topography: Pearson correlation between the group-mean per-channel maps of sung and spoken lyrics with a bootstrap over
  listeners, Spearman-Brown split-half reliability of each map (odd and even listeners), the noise ceiling, and a
  topographic analysis of variance on global-field-power-normalised maps (10,000 within-listener label flips);
* lexical entropy beyond surprisal (entropy regressor tested against its own permutations, surprisal in the base), Holm
  over Alice, spoken and sung.
The leave-one-song-out range is also computed on the stabilised estimate (sufficient statistics of its first n draws).
"""
from __future__ import annotations

import json
import os
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from encoding_models import ridge_encoding as E
from readers import alice as h, cantisong as c
from statistical_tests.inference import holm, summarise
from stimulus_features import song_regressors as g

N_LAGS, N_PERM, WIN_S = 23, 10, 0.8
WINDOWS = {"W1_0-188ms": (0, 6), "W2_219-406ms": (7, 13), "W3_438-688ms": (14, 22)}


def r_from_suff(S: np.ndarray) -> np.ndarray:
    """Pooled per-channel correlation from summed sufficient statistics (n, sum Y, sum P, sum Y^2, sum P^2, sum YP)."""
    n, sy, sp, syy, spp, syp = S
    return (syp - sy * sp / n) / np.sqrt((syy - sy * sy / n) * (spp - sp * sp / n) + 1e-300)


def perm_controls(trials: list[dict], raw_key: str, rng) -> list[dict]:
    ctrls = []
    for _ in range(N_PERM):
        cd = {}
        for j, tr in enumerate(trials):
            raw = np.zeros_like(tr[raw_key])
            raw[tr["idx"]] = rng.permutation(tr[raw_key][tr["idx"]])
            cd[j] = g.bz(raw[None])[:, tr["a"]:tr["a"] + tr["T"]]
        ctrls.append(cd)
    return ctrls


def canti_job(key: str, specs: list[dict], a: int, seed: int) -> dict:
    """One CantiSong listener and version (key 'canti|version|listener')."""
    rng = np.random.default_rng([seed, zlib.crc32(key.encode()), 55])
    trials = []
    for sp in specs:
        st, Y, v = sp["st"], sp["Y"], sp["v"]
        T = min(Y.shape[1], st["acoustic"].shape[1] - a)
        F = {k: st[k][:, a:a + T] for k in ("acoustic", "phonetic", "word", "surprisal") if k in st}
        F["entropy"] = g.bz(sp["ent_raw"][None])[:, a:a + T]
        trials.append({"F": F, "Y": Y[:, :T], "v": v[:T], "T": T, "a": a, "words": [i - a for i in st["word_idx"]],
                       "idx": np.asarray(st["word_idx"]), "raw_s": st["raw_surprisal"], "raw_e": sp["ent_raw"], "song": sp["song"]})
    base = [k for k in ("acoustic", "phonetic", "word") if k in trials[0]["F"]]
    masks = E.event_masks(trials, "words", WIN_S)
    ctrl_s = perm_controls(trials, "raw_s", rng)
    det, lam = E.delta_r_detail(trials, base, "surprisal", ctrl_s, masks, N_LAGS)
    win = {w: E.delta_r_detail(trials, base, "surprisal", ctrl_s, masks, N_LAGS, lam, lag_spec={"surprisal": kk})[0]
           for w, kk in WINDOWS.items()}
    ctrl_e = perm_controls(trials, "raw_e", rng)
    ent, lam_e = E.delta_r_detail(trials, base + ["surprisal"], "entropy", ctrl_e, masks, N_LAGS)
    return {"key": key, "songs": [tr["song"] for tr in trials], "lam": lam, "lam_entropy": lam_e, "pooled": det["pooled"],
            "entropy": ent["pooled"], **{w: d["pooled"] for w, d in win.items()}, "per_trial": det["per_trial"],
            "per_channel": det["per_channel"], "suff_true": det["suff_true"], "suff_ctrl": det["suff_ctrl"],
            **{f"per_channel_{w}": d["per_channel"] for w, d in win.items()}}


def alice_job(key: str, specs: list[dict], a: int, seed: int) -> dict:
    """One Alice listener: lexical entropy beyond surprisal."""
    rng = np.random.default_rng([seed, zlib.crc32(key.encode()), 56])
    trials = []
    for sp in specs:
        st, se, Y, v = sp["st"], sp["se"], sp["Y"], sp["v"]
        T = min(Y.shape[1], st["acoustic"].shape[1] - a)
        F = {k: st[k][:, a:a + T] for k in ("acoustic", "word", "surprisal")}
        F["entropy"] = se["surprisal"][:, a:a + T]
        trials.append({"F": F, "Y": Y[:, :T], "v": v[:T], "T": T, "a": a, "words": [i - a for i in st["word_idx"]],
                       "idx": np.asarray(st["word_idx"]), "raw_e": se["raw_surprisal"]})
    masks = E.event_masks(trials, "words", WIN_S)
    ent, lam = E.delta_r_detail(trials, ["acoustic", "word", "surprisal"], "entropy", perm_controls(trials, "raw_e", rng), masks, N_LAGS)
    return {"key": key, "lam_entropy": lam, "entropy": ent["pooled"]}


def canti_jobs(paths) -> list[tuple]:
    """(key, trial specs, first sample) per CantiSong listener and version, with surprisal and entropy."""
    os.environ["CANTISONG_ANNOTATION"] = "corrected"
    data = c.load_all(paths.derived_root / "cantisong_1to8", "1-8")
    words = {s: g.load_words(s) for s in c.song_ids()}
    ent = {s: pd.read_csv(paths.derived_root / "cantisong_words" / f"{s}_entropy.csv") for s in c.song_ids()}
    a = int(c.START_S * 64)
    jobs = []
    for v in g.AUDIO:
        st = {s: g.stimulus(s, v, words[s], False) for s in c.song_ids()}
        er = {}
        for s in c.song_ids():
            we = words[s].copy()
            assert (we.idx.to_numpy() == ent[s].idx.to_numpy()).all()
            we["surprisal_bits"] = ent[s].entropy_bits.to_numpy()
            _, er[s], idx = g.lexical(we, v, st[s]["acoustic"].shape[1], False)
            assert idx == st[s]["word_idx"]
        subs = sorted({sub for (vv, s), d in data.items() if vv == v for sub in d})
        for sub in subs:
            songs = [s for s in c.song_ids() if (v, s) in data and sub in data[(v, s)] and data[(v, s)][sub][1].any()]
            if len(songs) < 6:
                continue
            specs = [{"st": st[s], "ent_raw": er[s], "Y": data[(v, s)][sub][0], "v": data[(v, s)][sub][1], "song": s} for s in songs]
            jobs.append((f"canti|{v}|{sub}", specs, a))
    return jobs


def alice_jobs(paths) -> list[tuple]:
    words = pd.read_csv(paths.derived_root / "alice" / "words_gpt2_entropy.csv")
    st = {s: h.seg_stim(s, words, "gpt2_surprisal_bits") for s in h.SEGMENTS}
    se = {s: h.seg_stim(s, words, "gpt2_entropy_bits") for s in h.SEGMENTS}
    data8, _ = h.load_all(paths.derived_root / "alice", "1-8")
    a = int(h.START_S * 64)
    jobs = []
    for sub in sorted(data8):
        segs = [s for s in h.SEGMENTS if s in data8[sub] and data8[sub][s][1].any()]
        if len(segs) < h.MIN_SEGMENTS:
            continue
        jobs.append((f"alice|speech|{sub}", [{"st": st[s], "se": se[s], "Y": data8[sub][s][0], "v": data8[sub][s][1]} for s in segs], a))
    return jobs


# ---------------------------------------------------------------- statistics
def pick(x, alt="greater"):
    return {k: v for k, v in summarise(np.asarray(x, float), alt).items() if k in ("n", "mean", "dz", "dz_ci95", "p_signflip", "n_positive")}


def loso(suff_true: np.ndarray, suff_ctrl: np.ndarray) -> list[float]:
    """Listener-level delta r with each of the listener's songs left out (trials x 6 x channels; controls first)."""
    out = []
    for j in range(suff_true.shape[0]):
        keep = np.arange(suff_true.shape[0]) != j
        rt = r_from_suff(suff_true[keep].sum(0))
        rc = np.mean([r_from_suff(x[keep].sum(0)) for x in suff_ctrl], axis=0)
        out.append(float(np.mean(rt - rc)))
    return out


def loso_range(LO: pd.DataFrame) -> dict:
    """LO: listener, version, left_out, dr. Per version, d_z over listeners for each left-out song: range and largest p."""
    s4 = {}
    for v in ("speech", "song"):
        per = [pick(grp.set_index("listener").dr) for _, grp in LO[LO.version == v].groupby("left_out")]
        s4[v] = {"n_songs": len(per), "dz_min": min(p["dz"] for p in per), "dz_max": max(p["dz"] for p in per),
                 "p_max": max(p["p_signflip"] for p in per), "n_p_below_05": int(sum(p["p_signflip"] < 0.05 for p in per))}
    return s4


def stabilised_loso(cache: Path, n_used: dict) -> pd.DataFrame:
    """Leave-one-song-out of the stabilised estimate from its stored sufficient statistics (first n draws; n_used:
    version -> n of the Monte Carlo rule)."""
    lo = []
    for v in ("speech", "song"):
        n = n_used[v]
        for fz in sorted(cache.glob(f"canti_{v}_*.npz")):
            with np.load(fz) as z:
                m = json.loads(str(z["meta"]))
                st, sc = z["S_suff_true"], z["S_suff_ctrl"][:n]
            for s, d in zip(m["units"], loso(st, sc)):
                lo.append({"listener": int(m["key"].split("|")[2]), "version": v, "left_out": s, "dr": d})
    return pd.DataFrame(lo)


def tanova(maps_a: np.ndarray, maps_b: np.ndarray, rng, n: int = 10_000) -> tuple[float, float]:
    """Global dissimilarity of the GFP-normalised mean maps and its within-listener label-flip p."""
    norm = lambda m: (m - m.mean()) / m.std()
    diss = lambda A, B: float(np.sqrt(np.mean((norm(A.mean(0)) - norm(B.mean(0))) ** 2)))
    obs = diss(maps_a, maps_b)
    null = np.empty(n)
    for i in range(n):
        flip = rng.random(len(maps_a)) < 0.5
        A = np.where(flip[:, None], maps_b, maps_a)
        B = np.where(flip[:, None], maps_a, maps_b)
        null[i] = diss(A, B)
    return obs, float((1 + (null >= obs).sum()) / (1 + n))


def statistics(canti: list[dict], alice: list[dict], seed: int) -> tuple[dict, dict]:
    """canti/alice: outputs of canti_job and alice_job. Returns the results and the tables (listeners, listener x song,
    song means, leave-one-song-out, Alice entropy, group-mean topographies)."""
    rng = np.random.default_rng([seed, 57])
    rows, song_rows, maps, loso_rows = [], [], {}, []
    for m in canti:
        _, v, sub = m["key"].split("|")
        rows.append({"listener": int(sub), "version": v, **{k: m[k] for k in ("pooled", "entropy", *WINDOWS)}})
        song_rows += [{"listener": int(sub), "version": v, "song": s, "dr": float(d)} for s, d in zip(m["songs"], m["per_trial"])]
        maps[(int(sub), v)] = m["per_channel"]
        loso_rows += [{"listener": int(sub), "version": v, "left_out": s, "dr": d} for s, d in zip(m["songs"], loso(m["suff_true"], m["suff_ctrl"]))]
    L, S, LO = pd.DataFrame(rows), pd.DataFrame(song_rows), pd.DataFrame(loso_rows)
    A = pd.DataFrame([{"listener": m["key"].split("|")[2], "entropy": m["entropy"]} for m in alice])
    res = {"listener_level": {v: pick(L[L.version == v].pooled, "greater" if v != "melody" else "two-sided") for v in g.AUDIO}}
    item = S.groupby(["song", "version"]).dr.mean().unstack().rename_axis(index="song_id", columns=None)
    res["songs_spoken"] = pick(item["speech"].dropna())
    res["songs_sung"] = pick(item["song"].dropna())
    res["songs_hummed"] = pick(item["melody"].dropna(), "two-sided")
    both = item[["song", "melody"]].dropna()
    res["songs_sung_minus_hummed"] = pick(both["song"] - both["melody"])
    res["leave_one_song_out"] = loso_range(LO)
    subs = sorted({s for s, v in maps if v == "speech"} & {s for s, v in maps if v == "song"})
    Ms, Mp = np.array([maps[(s, "song")] for s in subs]), np.array([maps[(s, "speech")] for s in subs])
    r_obs = float(np.corrcoef(Ms.mean(0), Mp.mean(0))[0, 1])
    boot = []
    for _ in range(5000):
        ix = rng.integers(0, len(subs), len(subs))
        boot.append(np.corrcoef(Ms[ix].mean(0), Mp[ix].mean(0))[0, 1])
    sb = lambda r: 2 * r / (1 + r)
    rel = {nm: sb(float(np.corrcoef(M[0::2].mean(0), M[1::2].mean(0))[0, 1])) for nm, M in (("sung", Ms), ("spoken", Mp))}
    ceiling = float(np.sqrt(max(rel["sung"], 0) * max(rel["spoken"], 0)))
    diss, p_t = tanova(Ms, Mp, rng)
    res["topography"] = {"n": len(subs), "r_sung_spoken": r_obs, "ci95_bootstrap": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                         "split_half_reliability_sb": rel, "noise_ceiling": ceiling}
    res["tanova"] = {"dissimilarity": diss, "p_perm": p_t}
    labels = c.load_version(1, "song")[1]
    topo = pd.DataFrame({"channel": labels[:Ms.shape[1]], "sung": Ms.mean(0), "spoken": Mp.mean(0),
                         "hummed": np.array([maps[(s, "melody")] for s in sorted({s for s, v in maps if v == "melody"})]).mean(0)})
    t3 = {}
    for v in ("speech", "song", "melody"):
        d = L[L.version == v]
        t3[v] = {w: pick(d[w], "greater" if v != "melody" else "two-sided") for w in WINDOWS}
        if v != "melody":
            t3[v]["holm"] = holm({w: t3[v][w]["p_signflip"] for w in WINDOWS})
    wide = L.pivot(index="listener", columns="version")
    t3["sung_minus_spoken"] = {w: pick((wide[w]["song"] - wide[w]["speech"]).dropna(), "two-sided") for w in WINDOWS}
    t3["sung_minus_spoken"]["holm"] = holm({w: t3["sung_minus_spoken"][w]["p_signflip"] for w in WINDOWS})
    res["lag_windows"] = t3
    res["entropy_alice"] = pick(A.entropy)
    res["entropy_spoken"] = pick(L[L.version == "speech"].entropy)
    res["entropy_sung"] = pick(L[L.version == "song"].entropy)
    res["entropy_holm"] = holm({"alice": res["entropy_alice"]["p_signflip"], "spoken": res["entropy_spoken"]["p_signflip"],
                                "sung": res["entropy_sung"]["p_signflip"]})
    res["entropy_hummed"] = pick(L[L.version == "melody"].entropy, "two-sided")
    res["entropy_sung_minus_hummed"] = pick((wide["entropy"]["song"] - wide["entropy"]["melody"]).dropna())
    return res, {"listeners": L, "listener_song": S, "song_means": item, "leave_one_song_out": LO, "alice_entropy": A, "topographies": topo}
