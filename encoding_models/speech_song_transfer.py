"""Transfer of the surprisal response between the spoken and sung versions (CantiSong; additional analysis).

For each listener, target version and held-out song j: (i) the base model of the target version (acoustics, phonetic
features and word onsets; hummed: acoustics and word onsets; ridge chosen as usual and fixed) is fitted on the other
songs, and the residual EEG of song j is taken; (ii) in the source version, every song k other than j is residualised in
the same way with the base model fitted on the songs other than j and k, and the surprisal kernel (23 lags x channels)
is the ridge fit of these residuals on the lagged surprisal regressor (ridge: the source base model's value scaled by
the mean eigenvalue of the kernel's Gram matrix); (iii) the kernel is applied to the surprisal regressor of song j in
the target version, with the true values and with the values permuted among the song's words (timing kept). Accuracy
is the correlation of prediction and target residual within 0-800 ms after each word, pooled over the held-out songs,
averaged over channels; delta r = r(true) - mean r(permuted), with the Monte Carlo rule (up to 500 permutations).

Estimates: 2a spoken -> spoken and sung -> sung (pipeline check, one-sided), 2b spoken -> sung and 2c sung -> spoken
(one-sided), 2d spoken -> hummed with the sung timing (control, two-sided, BF01), and 2e the same transfers with the
time axis of the kernel stretched by the ratio of the mean word durations, sung / spoken for 2b and spoken / sung for 2c
(exploratory, two-sided). Transfer as a fraction of the within-version estimate: ratio of means with a percentile
bootstrap interval over listeners (5,000 resamples). Corrected annotation of song 09.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from encoding_models import ridge_encoding as E
from encoding_models.phoneme_and_interaction import rng_for, word_trials
from encoding_models.word_level_gain import mc_rule
from statistical_tests.inference import jzs_bf10, summarise
from stimulus_features import song_regressors as g

N_LAGS, WIN_S, N_MAX, CHUNK = 23, 0.8, 500, 50
VERSIONS = ("speech", "song", "melody")
SIDE = {"2a_speech": "greater", "2a_song": "greater", "2b": "greater", "2c": "greater", "2d": "two-sided", "2e_b": "two-sided",
        "2e_c": "two-sided"}


def duration_ratio(words: dict) -> float:
    d = pd.concat(words.values())
    return float((d.offset_song - d.onset_song).mean() / (d.offset_speech - d.onset_speech).mean())


def stretch(K: np.ndarray, r: float) -> np.ndarray:
    """Kernel (lags x channels, lags 0, 2, ..., 44 samples) on the time axis stretched by r, lags 0, 2, ... <= 44 r."""
    lags = np.arange(K.shape[0]) * 2.0
    M = int(np.floor((K.shape[0] - 1) * r + 1e-9))
    new = np.arange(M + 1) * 2.0 / r
    return np.stack([np.interp(new, lags, K[:, ch]) for ch in range(K.shape[1])], 1)


def transfer_job(data: dict, stims: dict, sub: int, seed: int, ratio: float, n: int = N_MAX) -> dict:
    """Within-version and transfer estimates of one listener, with the true and up to ``n`` permuted values of the
    target version (the same permutations for every kernel applied to that target)."""
    import torch
    from encoding_models import gpu_encoding as WG
    dev = WG.device()
    T_ = lambda x: torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float64, device=dev)
    info = {}
    for V in VERSIONS:
        trials = word_trials(data, stims, None, sub, V)
        if len(trials) < 6:
            continue
        base = ["acoustic", "phonetic", "word"] if V != "melody" else ["acoustic", "word"]
        lam = WG.GpuFolds(trials, base, N_LAGS).lam
        G, C, Q, SY, SS, Sm = {}, {}, {}, {}, {}, {}
        masks = E.event_masks(trials, "words", WIN_S)
        for tr, m in zip(trials, masks):
            s, v = tr["song"], tr["v"]
            Xb = T_(E._design(tr, base, N_LAGS, None))
            Xs = T_(E.lagged(tr["F"]["surprisal"], tr["T"], N_LAGS))
            Y = T_(tr["Y"].T)
            vt = torch.as_tensor(v, device=dev)
            G[s], C[s] = Xb[vt].T @ Xb[vt], Xb[vt].T @ Y[vt]
            Q[s], SY[s], SS[s] = Xs[vt].T @ Xb[vt], Xs[vt].T @ Y[vt], Xs[vt].T @ Xs[vt]
            Sm[s] = m
            del Xb, Xs, Y
        info[V] = {"trials": trials, "lam": lam, "base": base, "G": G, "C": C, "Q": Q, "SY": SY, "SS": SS, "masks": Sm,
                   "songs": [tr["song"] for tr in trials], "pb": next(iter(G.values())).shape[0]}
    torch.cuda.empty_cache()

    def solve_base(V, train):
        d = info[V]
        Gt = sum(d["G"][s] for s in train)
        Ct = sum(d["C"][s] for s in train)
        return torch.linalg.solve(Gt + d["lam"] * torch.trace(Gt) / d["pb"] * torch.eye(d["pb"], dtype=torch.float64, device=dev), Ct)

    kernels = {}

    def kernel(U, j):
        if (U, j) not in kernels:
            d = info[U]
            src = [k for k in d["songs"] if k != j]
            Gs, Cs = 0.0, 0.0
            for k in src:
                W = solve_base(U, [m for m in src if m != k])
                Gs = Gs + d["SS"][k]
                Cs = Cs + d["SY"][k] - d["Q"][k] @ W
            kernels[(U, j)] = torch.linalg.solve(Gs + d["lam"] * torch.trace(Gs) / N_LAGS * torch.eye(N_LAGS, dtype=torch.float64, device=dev), Cs)
        return kernels[(U, j)]

    combos = {"2a_speech": ("speech", "speech", 1.0), "2a_song": ("song", "song", 1.0), "2b": ("speech", "song", 1.0),
              "2c": ("song", "speech", 1.0), "2d": ("speech", "melody", 1.0), "2e_b": ("speech", "song", ratio),
              "2e_c": ("song", "speech", 1.0 / ratio)}
    res = {}
    for V in ("speech", "song", "melody"):
        if V not in info:
            continue
        d = info[V]
        mine = {k: cb for k, cb in combos.items() if cb[1] == V and cb[0] in info}
        if not mine:
            continue
        R = {}
        for tr in d["trials"]:
            s = tr["song"]
            W = solve_base(V, [m for m in d["songs"] if m != s])
            mt = torch.as_tensor(d["masks"][s], device=dev)
            Xb = T_(E._design(tr, d["base"], N_LAGS, None))[mt]
            R[s] = T_(tr["Y"].T)[mt] - Xb @ W                                     # residual EEG on the masked samples
            del Xb
        K = {}
        for name, (U, _, r) in mine.items():
            K[name] = {}
            for s in d["songs"]:
                k_ = kernel(U, s)
                K[name][s] = k_ if r == 1.0 else T_(stretch(k_.cpu().numpy(), r))
        trials = d["trials"]
        rng = rng_for(seed, f"a2|{V}|{sub}", 0)

        def stats_for(preds_by_song):
            """preds_by_song: song -> K x Nm x ch. Pooled r (K x ch) and per-song r (K x songs x ch)."""
            sy = sum(R[s].sum(0) for s in d["songs"])
            N = sum(R[s].shape[0] for s in d["songs"])
            ybar = sy / N
            Kd = next(iter(preds_by_song.values())).shape[0]
            pbar = sum(preds_by_song[s].sum(1) for s in d["songs"]) / N
            syy = sum(((R[s] - ybar) ** 2).sum(0) for s in d["songs"])
            spp = torch.zeros_like(pbar)
            syp = torch.zeros_like(pbar)
            per = []
            for s in d["songs"]:
                P = preds_by_song[s] - pbar[:, None, :]
                Yc = R[s] - ybar
                spp += (P * P).sum(1)
                syp += (Yc[None] * P).sum(1)
                if R[s].shape[0] >= 64:
                    Pt = preds_by_song[s] - preds_by_song[s].mean(1, keepdim=True)
                    Yt = R[s] - R[s].mean(0, keepdim=True)
                    per.append((Yt[None] * Pt).sum(1) / torch.sqrt((Yt * Yt).sum(0)[None] * (Pt * Pt).sum(1) + 1e-300))
                else:
                    per.append(torch.full((Kd, R[s].shape[1]), float("nan"), dtype=torch.float64, device=dev))
            pooled = (syp / torch.sqrt(syy[None] * spp + 1e-300)).mean(1)
            return pooled.cpu().numpy(), torch.stack(per, 1).mean(2).cpu().numpy()

        def predict(raws, name):
            """raws: song -> K x n (unfiltered impulses); predictions on the masked samples, K x Nm x ch."""
            out = {}
            for tr in trials:
                s = tr["song"]
                Fz = T_(g.bz(raws[s])[:, tr["a"]:tr["a"] + tr["T"]])[:, None, :]
                k_ = K[name][s]
                X = WG.lagged(Fz, k_.shape[0])[:, torch.as_tensor(d["masks"][s], device=dev)]
                out[s] = X @ k_
            return out

        true_raw = {tr["song"]: tr["raw_s"][None] for tr in trials}
        acc = {name: {"true": None, "null": [], "null_trial": []} for name in mine}
        for name in mine:
            p, pt = stats_for(predict(true_raw, name))
            acc[name]["true"], acc[name]["true_trial"] = p[0], pt[0]
        done = 0
        while done < n:
            k_ = min(CHUNK, n - done)
            raws = {tr["song"]: np.zeros((k_, tr["raw_s"].size)) for tr in trials}
            for i in range(k_):
                for tr in trials:
                    raws[tr["song"]][i, tr["idx"]] = rng.permutation(tr["raw_s"][tr["idx"]])
            for name in mine:
                p, pt = stats_for(predict(raws, name))
                acc[name]["null"].append(p)
                acc[name]["null_trial"].append(pt)
            done += k_
        for name in mine:
            res[name] = {"true": np.array(acc[name]["true"]), "null": np.concatenate(acc[name]["null"]),
                         "true_trial": acc[name]["true_trial"], "null_trial": np.concatenate(acc[name]["null_trial"]),
                         "lam_source": np.array(info[mine[name][0]]["lam"]), "lam_target": np.array(d["lam"]),
                         "kernel_lags": np.array(next(iter(K[name].values())).shape[0])}
            res[name]["units"] = [int(s) for s in d["songs"]]
        del R, K
        torch.cuda.empty_cache()
    return {"key": f"a2|{sub}", "listener": sub, "ratio": ratio, "lam": {V: info[V]["lam"] for V in info}, "tests": res}


def statistics(jobs: list[dict], seed: int) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Monte Carlo rule per estimate, tests over listeners, the song-level estimate of 2b, the pipeline check and the
    transfer fractions."""
    tests = {}
    for job in jobs:
        for t, d in job["tests"].items():
            e = tests.setdefault(t, {"listeners": [], "true": [], "null": [], "true_trial": [], "null_trial": [], "units": []})
            e["listeners"].append(job["listener"])
            for k in ("true", "null", "true_trial", "null_trial", "units"):
                e[k].append(d[k])
    res, rows, song_rows = {}, [], []
    for name in sorted(tests):
        e = tests[name]
        tv = np.array([float(x) for x in e["true"]])
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
        res[name] = r
        rows += [{"test": name, "listener": L, "delta_r": float(v)} for L, v in zip(e["listeners"], x)]
        if name == "2b":
            per = {}
            for units, tt, nt in zip(e["units"], e["true_trial"], e["null_trial"]):
                for u, a_, b_ in zip(units, tt, nt[:nu].mean(0)):
                    if np.isfinite(a_ - b_):
                        per.setdefault(u, []).append(float(a_ - b_))
            songs = sorted(per)
            xs = np.array([np.mean(per[u]) for u in songs])
            ss = summarise(xs, SIDE[name])
            res["2b_songs"] = {k: ss[k] for k in ("n", "mean", "dz", "dz_ci95", "p_signflip", "n_positive")} | {"alternative": SIDE[name]}
            song_rows += [{"test": name, "song": f"{u:02d}", "delta_r": float(v), "n_listeners": len(per[u])} for u, v in zip(songs, xs)]
    L = pd.DataFrame(rows).pivot(index="listener", columns="test", values="delta_r")
    res["pipeline_check_passed"] = all(res[k]["p_signflip"] < 0.05 and res[k]["mean"] > 0 for k in ("2a_speech", "2a_song"))
    gb = np.random.default_rng(seed)
    for t, w in (("2b", "2a_song"), ("2c", "2a_speech")):
        both = L[[t, w]].dropna()
        boot = [both[t].iloc[ix].mean() / both[w].iloc[ix].mean() for ix in (gb.integers(0, len(both), len(both)) for _ in range(5000))]
        res[f"fraction_{t}_of_{w}"] = {"ratio": float(both[t].mean() / both[w].mean()),
                                       "ci95_bootstrap": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))], "n": int(len(both))}
    return res, L, pd.DataFrame(song_rows)


def within_version_agreement(L: pd.DataFrame, standard: dict[str, pd.Series]) -> dict:
    """Spearman correlation over listeners between the within-version transfer estimates (2a) and the standard
    word-level gains (``standard``: version -> gain per listener)."""
    out = {}
    for v, nm in (("speech", "2a_speech"), ("song", "2a_song")):
        both = pd.concat([L[nm], standard[v]], axis=1, keys=["residual", "standard"]).dropna()
        out[nm] = float(both.corr(method="spearman").iloc[0, 1])
    return out
