"""Word-level gain with up to 500 permutations and the Monte Carlo rule (CantiSong and Alice).

Per listener (and version), the ridge is chosen on the base model (acoustics, phonetic features where available, word
onsets) and fixed; the surprisal regressor is tested against up to 500 permuted versions (values permuted among the words
of each song or segment). Accuracy is the Pearson r between held-out prediction and EEG, pooled over songs or segments
within 0-800 ms after each word, per channel, then averaged over channels; delta r = r(true) - mean r(permuted).
The same predictions also give the 0-600 and 0-1000 ms windows and the nine-channel centro-parietal region; the lag
windows restrict the surprisal regressor to lags 0-188, 219-406 or 438-688 ms; lexical entropy is tested beyond surprisal
with draws taken after the surprisal draws. The number of permutations used is the first n in 100, 200, ..., 500 at
which the largest listener's Monte Carlo standard error of the null mean is at most 10 % of the between-listener SD of
delta r (else 500).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from encoding_models import gpu_encoding as WG, ridge_encoding as E, word_level_trials as P
from statistical_tests.inference import summarise

N_MAX, BLOCK, M_START, BATCH = 500, 100, 50, 12
MC_FRACTION = 0.10


def _save(f: Path, meta: dict, arrays: dict):
    tmp = f.with_suffix(".tmp.npz")
    np.savez(tmp, meta=json.dumps(meta, default=float), **arrays)
    os.replace(tmp, f)


def word_job(kind: str, key: str, specs: list[dict], a: int, roi, seed: int, cache: Path, n: int, released: bool = False) -> str:
    """Surprisal (with per-trial and per-channel detail), window and region variants, lag windows and entropy, n
    permutations each, for one listener (and version). ``released``: the released annotation, surprisal only."""
    f = cache / (key.replace("|", "_") + ".npz")
    if f.exists():
        return f.name
    t0 = time.time()
    rng = P.rng_for(seed, key)
    trials = P.canti_trials(specs, a) if kind == "canti" else P.alice_trials(specs, a)
    base = [k for k in ("acoustic", "phonetic", "word") if k in trials[0]["F"]]
    m8 = E.event_masks(trials, "words", P.WIN_S)
    folds = WG.GpuFolds(trials, base, P.N_LAGS)
    ms = {"m8": WG.MaskSet(folds, m8)}
    if not released:
        ms |= {"m6": WG.MaskSet(folds, E.event_masks(trials, "words", 0.6)), "m10": WG.MaskSet(folds, E.event_masks(trials, "words", 1.0))}
    ctrl_s = P.perm_controls(trials, "raw_s", rng, n)
    true, ctrl = WG.run_test(folds, trials, "surprisal", ctrl_s, ms, P.N_LAGS, batch=BATCH, suff_for=("m8",))
    out = {"S_true_ch": true["m8"]["pooled"], "S_true_per": true["m8"]["per"], "S_ctrl_ch": ctrl["m8"]["pooled"],
           "S_ctrl_per": ctrl["m8"]["per"], "S_suff_true": true["m8"]["suff"], "S_suff_ctrl": ctrl["m8"]["suff"]}
    meta = {"key": key, "units": [tr.get("song", tr.get("seg")) for tr in trials], "lam": folds.lam, "n": n,
            "roi": [int(x) for x in roi] if roi is not None else None}
    if not released:
        for mk in ("m6", "m10"):
            out[f"R_{mk}_true_ch"], out[f"R_{mk}_ctrl_ch"] = true[mk]["pooled"], ctrl[mk]["pooled"]
        for w, kk in P.WINDOWS.items():
            t, cc = WG.run_test(folds, trials, "surprisal", ctrl_s, {"m8": ms["m8"]}, P.N_LAGS, lag_spec={"surprisal": kk}, batch=BATCH)
            out[f"W_{w}_true_ch"], out[f"W_{w}_ctrl_ch"] = t["m8"]["pooled"], cc["m8"]["pooled"]
    del folds, ms, ctrl_s
    if not released and trials[0].get("raw_e") is not None:
        ctrl_e = P.perm_controls(trials, "raw_e", rng, n)                 # drawn after the surprisal draws
        fe = WG.GpuFolds(trials, base + ["surprisal"], P.N_LAGS)
        t, cc = WG.run_test(fe, trials, "entropy", ctrl_e, {"m8": WG.MaskSet(fe, m8)}, P.N_LAGS, batch=BATCH)
        out["E_true_ch"], out["E_ctrl_ch"] = t["m8"]["pooled"], cc["m8"]["pooled"]
        meta["lam_entropy"] = fe.lam
        del fe, ctrl_e
    meta["seconds"] = time.time() - t0
    _save(f, meta, out)
    return f.name


def canti_roi() -> np.ndarray:
    """Channel indices of the CantiSong centro-parietal region."""
    from readers import cantisong as c
    labels = c.load_version(1, "song")[1]
    return np.array([labels.index(ch) for ch in P.CANTI_ROI if ch in labels])


def alice_roi(paths) -> np.ndarray:
    """The nine Alice channels nearest the midline electrode just behind the vertex (channel 4 of the cap)."""
    from readers import alice as h
    rows = [ln.split() for ln in (h.ROOT / "easycapM10-acti61_elec.sfp").read_text().splitlines() if ln.strip()]
    pos = {r[0]: np.array([float(x) for x in r[1:4]]) for r in rows}
    labels_a = json.loads(str(np.load(paths.derived_root / "alice" / "S01.npz")["meta"]))["labels"]
    return np.argsort([np.linalg.norm(pos[ch] - pos["4"]) for ch in labels_a])[:9]


def mc_rule(true: np.ndarray, nulls: np.ndarray, start: int = BLOCK) -> dict:
    """true: listeners; nulls: listeners x draws (channel-mean r per permuted version). The first n (start, start + 100,
    ..., 500) at which max_i SE_i(n) <= 10 % of SD_i(delta r_i(n)); else the largest available."""
    n_avail = nulls.shape[1]
    grid = sorted({min(x, n_avail) for x in list(range(start, N_MAX + 1, BLOCK)) + [N_MAX]})
    trace = []
    for n in grid:
        se = nulls[:, :n].std(1, ddof=1) / np.sqrt(n)
        dr = true - nulls[:, :n].mean(1)
        ratio = float(se.max() / dr.std(ddof=1))
        trace.append({"n": int(n), "ratio": ratio})
        if ratio <= MC_FRACTION:
            return {"n_used": int(n), "ratio": ratio, "met": True, "trace": trace}
    return {"n_used": int(grid[-1]), "ratio": trace[-1]["ratio"], "met": False, "trace": trace}


def pick(x, alt="greater"):
    return {k: v for k, v in summarise(np.asarray(x, float), alt).items() if k in ("n", "mean", "dz", "dz_ci95", "p_signflip", "n_positive")}


def load_jobs(cache: Path) -> list[tuple[dict, dict]]:
    out = []
    for f in sorted(cache.glob("*.npz")):
        if f.name.endswith(".tmp.npz"):
            continue
        with np.load(f, allow_pickle=False) as z:
            out.append((json.loads(str(z["meta"])), {k: z[k] for k in z.files if k != "meta"}))
    return out


def _test_from_channels(rows: list, tkey: str, ckey: str, chans=None) -> tuple[np.ndarray, np.ndarray]:
    tv, nv = [], []
    for m, z in rows:
        ix = slice(None) if chans is None else np.asarray(m["roi"])
        tv.append(float(np.mean(z[tkey][ix])))
        nv.append(z[ckey][:, ix].mean(1))
    return np.array(tv), np.array(nv)


def word_level_statistics(cache: Path, out: Path, name: str = "word_level", released: bool = False) -> dict:
    """Monte Carlo rule per test, delta r per listener, per-song gains and the tests over listeners and over songs.
    Spoken and sung versions and Alice are tested one-sided (> 0), the hummed control two-sided."""
    res, mc, rows_l, rows_s, groups = {}, {}, [], [], {}
    for m, z in load_jobs(cache):
        ds, v, sub = m["key"].split("|")
        groups.setdefault((ds, v), []).append((m, z))
    for (ds, v), rows in sorted(groups.items()):
        alt = "two-sided" if v == "melody" else "greater"
        tests = {"pooled": ("S_true_ch", "S_ctrl_ch", None)}
        if not released:
            tests |= {"window600": ("R_m6_true_ch", "R_m6_ctrl_ch", None), "window1000": ("R_m10_true_ch", "R_m10_ctrl_ch", None),
                      "roi": ("S_true_ch", "S_ctrl_ch", "roi")}
            tests |= {w: (f"W_{w}_true_ch", f"W_{w}_ctrl_ch", None) for w in P.WINDOWS}
            if "E_true_ch" in rows[0][1]:
                tests["entropy"] = ("E_true_ch", "E_ctrl_ch", None)
        vals = {}
        for t, (tk, ck, ch) in tests.items():
            tv, nv = _test_from_channels(rows, tk, ck, ch)
            rule = mc_rule(tv, nv)
            n = rule["n_used"]
            mc[f"{ds}|{v}|{t}"] = rule
            vals[t] = tv - nv[:, :n].mean(1)
            vals[f"mc_se_{t}"] = nv[:, :n].std(1, ddof=1) / np.sqrt(n)
        nS = mc[f"{ds}|{v}|pooled"]["n_used"]
        for i, (m, z) in enumerate(rows):
            rows_l.append({"dataset": ds, "version": v, "listener": m["key"].split("|")[2], "n_perm": nS, "mc_se": vals["mc_se_pooled"][i],
                           **{t: float(vals[t][i]) for t in tests}})
            if ds == "canti":
                per = np.nanmean(z["S_true_per"] - z["S_ctrl_per"][:nS].mean(0), axis=1)
                rows_s += [{"listener": int(m["key"].split("|")[2]), "version": v, "song": s, "dr": float(d)} for s, d in zip(m["units"], per)]
        res[f"{ds}|{v}"] = {"listeners": pick(vals["pooled"], alt), **{t: pick(vals[t], alt) for t in tests if t != "pooled"}}
    L = pd.DataFrame(rows_l)
    L.to_csv(out / f"{name}_listeners.csv", index=False)
    S = pd.DataFrame(rows_s)
    S.to_csv(out / f"{name}_listener_song.csv", index=False)
    item = S.groupby(["song", "version"]).dr.mean().unstack()
    item.rename_axis("song_id").to_csv(out / f"{name}_song_means.csv")
    res["by_songs"] = {"spoken": pick(item["speech"].dropna()), "sung": pick(item["song"].dropna()),
                       "hummed": pick(item["melody"].dropna(), "two-sided"), "sung_minus_hummed": pick((item["song"] - item["melody"]).dropna())}
    wide = L[L.dataset == "canti"].pivot(index="listener", columns="version", values="pooled")
    res["canti|sung_minus_hummed"] = pick((wide["song"] - wide["melody"]).dropna())
    res["monte_carlo"] = mc
    (out / f"{name}.json").write_text(json.dumps(res, indent=2, default=float))
    return res
