"""Word-level gain in the Alice data (test TH1) and its association with comprehension across listeners (test TH2).

Per listener: leave-one-segment-out ridge (lags 0-688 ms) on acoustics (34 rows) and word onsets, ridge chosen on this
base model and fixed; the full model adds GPT-2 surprisal-weighted word onsets (released word onsets; surprisal over
sliding 1024-token windows with a 512-token stride). Delta r = r(full) - mean r over 10 within-segment permutations of
the surprisal values, r within 0-800 ms after each word, averaged over channels. TH1: delta r > 0, one-sided sign-flip
test over listeners (10^4 random flips). TH2: Spearman rho between comprehension (proportion of the eight quiz items
answered correctly) and delta r, one-sided permutation test of the ranks (10^5 permutations), Fisher-z 95 % CI.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

from readers import alice as h
from statistical_tests.inference import summarise
from stimulus_features import song_regressors as g

SURPRISAL_COL = "gpt2_surprisal_bits"


def listener_delta_r(sub: str, data: dict, stims: dict, seed: int) -> dict:
    segs = [s for s in h.SEGMENTS if s in data and data[s][1].any()]
    if len(segs) < h.MIN_SEGMENTS:
        return {}
    a = int(h.START_S * 64)
    rng = np.random.default_rng([seed, int(sub[1:]), 8])
    blocks, masks = [], []
    for s in segs:
        st = stims[s]
        Y, v = data[s]
        T = min(Y.shape[1], st["acoustic"].shape[1] - a)
        X = {k: g.lagged(st[k][:, a:a + T], T) for k in ("acoustic", "word", "surprisal")}
        m = np.zeros(T, bool)
        for i in st["word_idx"]:
            lo = i - a
            if lo + g.WORD_WIN > 0 and lo < T:
                m[max(lo, 0): min(lo + g.WORD_WIN, T)] = True
        blocks.append({"X": X, "Y": Y[:, :T], "v": v[:T], "st": st, "a": a, "T": T})
        masks.append(m & v[:T])
    _, lam = g.fit_predict(blocks, ["acoustic", "word"], None)
    r_full = g.r_mean(blocks, g.fit_predict(blocks, ["acoustic", "word", "surprisal"], lam)[0], masks)
    r_sh = []
    for _ in range(g.N_SHUF):
        for bl in blocks:
            st = bl["st"]
            raw = np.zeros_like(st["raw_surprisal"])
            raw[st["word_idx"]] = rng.permutation(st["raw_surprisal"][st["word_idx"]])
            bl["X"]["surprisal_shuf"] = g.lagged(g.bz(raw[None])[:, bl["a"]:bl["a"] + bl["T"]], bl["T"])
        r_sh.append(g.r_mean(blocks, g.fit_predict(blocks, ["acoustic", "word", "surprisal_shuf"], lam)[0], masks))
    return {"listener": sub, "delta_r": r_full - float(np.mean(r_sh))}


def spearman_perm(x, y, n=100000, seed=None) -> tuple[float, float]:
    """Spearman rho and its one-sided (rho > 0) permutation p, (1 + #(null >= rho)) / (1 + n)."""
    r = stats.spearmanr(x, y).correlation
    g_ = np.random.default_rng(seed)
    ry, rx = stats.rankdata(y), stats.rankdata(x)
    null = np.array([np.corrcoef(rx, g_.permutation(ry))[0, 1] for _ in range(n)])
    return float(r), float((1 + (null >= r).sum()) / (1 + n))


def fisher_ci(r: float, n: int, level: float = 0.95) -> list[float]:
    z, q = np.arctanh(r), stats.norm.ppf(0.5 + level / 2)
    return [float(np.tanh(z - q / np.sqrt(n - 3))), float(np.tanh(z + q / np.sqrt(n - 3)))]


def alice_tests(paths, seed: int, n_jobs: int = 6) -> tuple[dict, pd.DataFrame]:
    words = pd.read_csv(paths.derived_root / "alice" / "words_gpt2.csv")
    stims = {s: h.seg_stim(s, words, SURPRISAL_COL) for s in h.SEGMENTS}
    data8, qc = h.load_all(paths.derived_root / "alice", "1-8")
    rows = Parallel(n_jobs=n_jobs)(delayed(listener_delta_r)(sub, data8[sub], stims, seed) for sub in sorted(data8))
    df = pd.DataFrame([r for r in rows if r]).set_index("listener")
    df["comprehension"] = h.scores().reindex(df.index)
    res = {"TH1": {k: v for k, v in summarise(df.delta_r.to_numpy(float), "greater").items()
                   if k in ("n", "mean", "dz", "dz_ci95", "p_signflip", "n_positive")}}
    d = df.dropna(subset=["comprehension"])
    rho, p = spearman_perm(d.comprehension, d.delta_r, seed=seed)
    res["TH2"] = {"n": int(len(d)), "rho": rho, "ci95_fisher": fisher_ci(rho, len(d)), "p_one_sided": p}
    return res, df


def comprehension_test_on(delta_r: pd.Series, seed: int) -> dict:
    """TH2 on another estimate of delta r (e.g. the estimate with up to 500 permutations), indexed by listener."""
    d = pd.DataFrame({"delta_r": delta_r, "comprehension": h.scores().reindex(delta_r.index)}).dropna()
    rho, p = spearman_perm(d.comprehension, d.delta_r, seed=seed)
    return {"n": int(len(d)), "rho": rho, "ci95_fisher": fisher_ci(rho, len(d)), "p_one_sided": p}
