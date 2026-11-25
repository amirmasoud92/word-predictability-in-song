"""Language- and audio-model representations of the songs in the CantiSong EEG (1-8 Hz, leave-one-song-out, lags
0-688 ms).

Base model: acoustics (34 rows), phonetic features (spoken and sung versions) and word onsets; ridge chosen on the base.
* GPT-2 layers 0, 4, 8 and 12 (16 principal components, word-onset impulses): delta r against permutations of the word
  vectors among the words of each song (timing kept), word-locked 0-800 ms, all channels. Test M1: mean over the four
  layers > 0 (one-sided; sung and spoken; the hummed version, two-sided, is the control).
* HuBERT and AST layers 3, 6, 9 and 12 (exploratory; 16 principal components, resampled to 64 Hz): delta r against the
  same model's features of the next song (cyclic) of the same version, on all valid samples.
* Layer-profile similarity of sung and spoken versions (Spearman over layers); sung - hummed per audio layer (Holm within
  each model); paired two-sided contrasts of the layer means between versions.
Original estimates: 5 permutations, released annotation, CPU engine. Stabilised estimates: up to 500 permutations per
layer (first 50, then blocks of 100), corrected annotation of song 09 (+2.88 s, from the annotation audit), GPU engine with
the exact impulse-basis path for the GPT-2 block.
"""
from __future__ import annotations

import itertools
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from encoding_models import ridge_encoding as E, word_level_trials as P
from encoding_models.word_level_gain import mc_rule
from readers import cantisong as c
from statistical_tests.inference import cohens_dz, dz_ci, holm, sign_flip_test, summarise
from stimulus_features import song_regressors as g

GPT_LAYERS, AUDIO_LAYERS, N_LAGS, N_PERM = (0, 4, 8, 12), (3, 6, 9, 12), 23, 5
M_START, BATCH, BATCH_M = 50, 12, 4


def gpt_regressor(vecs: np.ndarray, idx: list[int], n: int) -> np.ndarray:
    R = np.zeros((vecs.shape[1], n))
    for k, i in enumerate(idx):
        if 0 <= i < n:
            R[:, i] += vecs[k]
    return g.bz(R)


def audio_regressor(npz: Path, n: int) -> np.ndarray:
    z = np.load(npz)
    x, rate = z["x"], float(z["rate"])
    t_src = np.arange(x.shape[0]) / rate
    t_dst = np.arange(n) / 64.0
    R = np.stack([np.interp(t_dst, t_src, x[:, k], left=0.0, right=0.0) for k in range(x.shape[1])])
    return g.bz(R)


def _trials(sub: int, version: str, data: dict, words: dict, feat_dir: Path, audio: bool) -> list[dict]:
    songs = [s for s in c.song_ids() if (version, s) in data and sub in data[(version, s)] and data[(version, s)][sub][1].any()]
    if len(songs) < 6:
        return []
    a = int(c.START_S * 64)
    all_songs = c.song_ids()
    trials = []
    for s in songs:
        st = g.stimulus(s, version, words[s], False)
        Y, v = data[(version, s)][sub]
        n_full = st["acoustic"].shape[1]
        T = min(Y.shape[1], n_full - a)
        F = {k: st[k][:, a:a + T] for k in ("acoustic", "phonetic", "word") if k in st}
        col = g.TIMING[version]
        w = words[s].dropna(subset=[f"onset_{col}"])
        idx_full = (w[f"onset_{col}"].to_numpy() * 64).round().astype(int)
        keep = w.index.to_numpy()
        for L in GPT_LAYERS:
            F[f"gpt{L}"] = gpt_regressor(np.load(feat_dir / f"gpt2_L{L}_{s}.npy")[keep], idx_full.tolist(), n_full)[:, a:a + T]
        if audio:
            nxt = all_songs[(all_songs.index(s) + 1) % len(all_songs)]
            for m in ("hubert", "ast"):
                for L in AUDIO_LAYERS:
                    F[f"{m}{L}"] = audio_regressor(feat_dir / f"{m}_L{L}_{s}_{version}.npz", n_full)[:, a:a + T]
                    F[f"{m}{L}_mis"] = audio_regressor(feat_dir / f"{m}_L{L}_{nxt}_{version}.npz", n_full)[:, a:a + T]
        trials.append({"F": F, "Y": Y[:, :T], "v": v[:T], "T": T, "words": [i - a for i in idx_full], "idx": idx_full,
                       "keep": keep, "song": s, "n_full": n_full, "a": a})
    return trials


# ---------------------------------------------------------------- original estimates
def original_listener(sub: int, version: str, data: dict, feat_dir: Path, words_dir: Path, seed: int) -> dict:
    """One listener and version: 5 permutations per GPT-2 layer, mismatched-song controls for the audio models; released
    word timing (the word tables as aligned, without the song-09 correction)."""
    os.environ["CANTISONG_ANNOTATION"] = "released"
    words = {s: pd.read_csv(words_dir / f"{s}_words.csv") for s in c.song_ids()}
    trials = _trials(sub, version, data, words, feat_dir, True)
    if not trials:
        return {}
    rng = np.random.default_rng([seed, sub, list(g.AUDIO).index(version), 13])
    base = [k for k in ("acoustic", "phonetic", "word") if k in trials[0]["F"]]
    _, lam = E.fit_predict(trials, base, None, N_LAGS)
    wmask = E.event_masks(trials, "words", 0.8)
    out = {"listener": sub, "version": version, "lambda": lam}
    for L in GPT_LAYERS:
        ctrls = []
        for _ in range(N_PERM):
            cd = {}
            for j, tr in enumerate(trials):
                vecs = np.load(feat_dir / f"gpt2_L{L}_{tr['song']}.npy")[tr["keep"]]
                cd[j] = gpt_regressor(vecs[rng.permutation(len(vecs))], tr["idx"].tolist(), tr["n_full"])[:, tr["a"]:tr["a"] + tr["T"]]
            ctrls.append(cd)
        d, _ = E.delta_r(trials, base, f"gpt{L}", ctrls, {"w": (wmask, None)}, N_LAGS, lam)
        out[f"gpt{L}"] = d["w"]
    for m in ("hubert", "ast"):
        for L in AUDIO_LAYERS:
            ctrl = [{j: tr["F"][f"{m}{L}_mis"] for j, tr in enumerate(trials)}]
            d, _ = E.delta_r(trials, base, f"{m}{L}", ctrl, {"all": (None, None)}, N_LAGS, lam)
            out[f"{m}{L}"] = d["all"]
    return out


def pick(x, alt="greater"):
    return {k: v for k, v in summarise(np.asarray(x, float), alt).items() if k in ("n", "mean", "dz", "dz_ci95", "p_signflip", "n_positive")}


def summary(df: pd.DataFrame) -> dict:
    """M1 per version (layer mean; one-sided for sung and spoken, two-sided for hummed), each layer, Holm over the two
    M1 tests, the layer-profile similarity of sung and spoken versions, and sung - hummed per audio layer (Holm)."""
    res = {}
    gcols = [f"gpt{L}" for L in GPT_LAYERS]
    for v in g.AUDIO:
        dv = df[df.version == v]
        res[f"M1_gpt_mean_{v}"] = pick(dv[gcols].mean(1), "greater" if v != "melody" else "two-sided")
        res[f"gpt_layers_{v}"] = {c_: pick(dv[c_]) for c_ in gcols}
        for m in ("hubert", "ast"):
            if f"{m}{AUDIO_LAYERS[0]}" in dv:
                res[f"{m}_layers_{v}"] = {f"L{L}": pick(dv[f"{m}{L}"]) for L in AUDIO_LAYERS}
    res["M1_holm"] = holm({v: res[f"M1_gpt_mean_{v}"]["p_signflip"] for v in ("speech", "song")})
    prof = df.groupby("version")[[c_ for c_ in df.columns if c_.startswith(("gpt", "hubert", "ast"))]].mean()
    res["M2_profile_similarity_sung_spoken"] = {m: float(stats.spearmanr(prof.loc["song", [c_ for c_ in prof.columns if c_.startswith(m)]],
                                                                        prof.loc["speech", [c_ for c_ in prof.columns if c_.startswith(m)]]).correlation)
                                                for m in ("gpt", "hubert", "ast")}
    wide = df.pivot(index="listener", columns="version")
    for m in ("hubert", "ast"):
        ps = {}
        for L in AUDIO_LAYERS:
            both = wide[f"{m}{L}"][["song", "melody"]].dropna()
            s_ = pick(both["song"] - both["melody"])
            res[f"M3_{m}_L{L}_sung_minus_hummed"] = s_
            ps[f"L{L}"] = s_["p_signflip"]
        res[f"M3_{m}_holm"] = holm(ps)
    return res


FAMILIES = {"gpt_all": ["gpt0", "gpt4", "gpt8", "gpt12"], "gpt_contextual": ["gpt4", "gpt8"],
            "hubert": ["hubert3", "hubert6", "hubert9", "hubert12"], "ast": ["ast3", "ast6", "ast9", "ast12"]}
PAIRS = (("song", "melody"), ("speech", "song"), ("speech", "melody"))


def contrasts(df: pd.DataFrame, families: dict = FAMILIES) -> dict:
    """Exploratory paired contrasts between versions of the layer-mean delta r (two-sided sign-flip over listeners who
    heard both versions)."""
    res = {}
    for fam, cols in families.items():
        if not all(c_ in df for c_ in cols):
            continue
        wide = df.assign(v=df[cols].mean(axis=1)).pivot(index="listener", columns="version", values="v")
        for a, b in PAIRS:
            both = wide[[a, b]].dropna()
            d = (both[a] - both[b]).to_numpy()
            res[f"{fam}|{a}-{b}"] = {"n": int(d.size), "mean": float(d.mean()), "dz": cohens_dz(d), "dz_ci95": list(dz_ci(d)),
                                     "p_two_sided": float(sign_flip_test(d, "two-sided")[0]), "n_positive": int((d > 0).sum())}
    return res


# ---------------------------------------------------------------- stabilised estimates (GPU)
class ImpulseBlock:
    """Exact fast path for a tested block of impulse rows that are band-passed and z-scored per row over the whole trial,
    when each version reassigns the row values among fixed onsets (GPT-2 vectors permuted over words). With f_w the
    band-passed unit impulse at onset w and g_w = f_w - mean(f_w), a version with values c (words x rows) is
    R_r = sum_w a_wr g_w, a_wr = c_wr / (sigma_r + 1e-12), sigma_r^2 = c_r' Cov c_r, Cov = g g' / n (linearity of the
    filter; population SD). So B_j = Xa_j[v]' Xs_j[v] = sum_w a_w Q_j[w] and C_j similarly, with Q_j = U_j XaUv_j' Lg_j and
    LY_j = Lg_j' Yv_j precomputed once (Lg_j: lagged g, valid rows); D_j and the prediction use Xs_j built from R."""

    def __init__(self, folds, onsets: list[np.ndarray], n_full: list[int], a: int, n_lags: int, bandpass):
        from encoding_models import gpu_encoding as WG
        self.f, self.n_lags = folds, n_lags
        self.keep, self.Cov, self.Gc, self.Q, self.LY, self.W = [], [], [], [], [], []
        for tr, idx, n, fd, Yv, v in zip(folds.trials, onsets, n_full, folds.folds, folds.Yv, folds.v):
            idx = np.asarray(idx)
            keep = (idx >= 0) & (idx < n)
            W = int(keep.sum())
            imp = np.zeros((W, n))
            imp[np.arange(W), idx[keep]] = 1.0
            G = bandpass(imp)
            G = G - G.mean(1, keepdims=True)
            Gt = folds._t(G)
            self.Cov.append(Gt @ Gt.T / n)
            Gc = Gt[:, a:a + tr["T"]].contiguous()
            Lg = WG.lagged(Gc[None], n_lags)[0][v]
            self.Q.append(fd["U"] @ (fd["XaUv"].T @ Lg))
            self.LY.append(Lg.T @ Yv)
            self.Gc.append(Gc)
            self.keep.append(keep)
            self.W.append(W)
            del Gt, Lg

    def predict_batch(self, C_batch: list[list[np.ndarray]]):
        """C_batch: K versions, each a list over trials of values (words with onsets inside the trial x rows)."""
        import torch
        from encoding_models import gpu_encoding as WG
        f, nl = self.f, self.n_lags
        K = len(C_batch)
        Xs, B, D, Cs = [], [], [], []
        for j, v in enumerate(f.v):
            C = f._t(np.stack([C_batch[k][j] for k in range(K)]))
            r = C.shape[2]
            sig = torch.sqrt(torch.einsum("kwr,wv,kvr->kr", C, self.Cov[j], C))
            A = C / (sig[:, None, :] + 1e-12)
            R = torch.einsum("kwr,wt->krt", A, self.Gc[j])
            X = WG.lagged(R, nl)[:, v].contiguous()
            Xs.append(X)
            B.append(torch.einsum("pwl,kwr->kprl", self.Q[j].reshape(f.pa, self.W[j], nl), A).reshape(K, f.pa, r * nl))
            Cs.append(torch.einsum("wlc,kwr->krlc", self.LY[j].reshape(self.W[j], nl, f.nch), A).reshape(K, r * nl, f.nch))
            D.append(torch.bmm(X.transpose(1, 2), X))
        return f.solve_blocks(Xs, B, D, Cs)


def run_test_impulse(block: ImpulseBlock, c_true: list[np.ndarray], c_versions, msets: dict, batch: int = 4) -> tuple[dict, dict]:
    """True and per-version statistics of an ImpulseBlock for each mask set (c_versions consumed in order)."""
    p = block.predict_batch([c_true])
    true = {k: {kk: vv[0] for kk, vv in ms.stats(p).items()} for k, ms in msets.items()}
    acc = {k: {} for k in msets}
    it = iter(c_versions)
    while True:
        chunk = list(itertools.islice(it, batch))
        if not chunk:
            break
        p = block.predict_batch(chunk)
        for k, ms in msets.items():
            for kk, vv in ms.stats(p).items():
                acc[k].setdefault(kk, []).append(vv)
        del p
    return true, {k: {kk: np.concatenate(vv) for kk, vv in d.items()} for k, d in acc.items()}


def stabilised_listener(sub: int, version: str, data: dict, feat_dir: Path, seed: int, n: int, n0: int = 0,
                        audio: bool = True) -> dict:
    """One listener and version (job key 'M|version|listener'): per GPT-2 layer, the true channel-mean accuracy and the
    accuracies of permutations n0..n-1 (one random sub-stream per layer, 10 + layer, so that blocks of permutations can be
    added); with ``audio``, the mismatched-song gains of HuBERT and AST. Corrected annotation of song 09."""
    from encoding_models import gpu_encoding as WG
    os.environ["CANTISONG_ANNOTATION"] = "corrected"
    words = {s: g.load_words(s) for s in c.song_ids()}
    trials = _trials(sub, version, data, words, feat_dir, audio)
    if not trials:
        return {}
    key = f"M|{version}|{sub}"
    base = [k for k in ("acoustic", "phonetic", "word") if k in trials[0]["F"]]
    wmask = E.event_masks(trials, "words", 0.8)
    folds = WG.GpuFolds(trials, base, N_LAGS)
    ms = WG.MaskSet(folds, wmask)
    block = ImpulseBlock(folds, [tr["idx"] for tr in trials], [tr["n_full"] for tr in trials], int(c.START_S * 64), N_LAGS,
                         lambda X: g.cnd.band_64(X, g.FS, *g.BAND))
    vecs = {(L, tr["song"]): np.load(feat_dir / f"gpt2_L{L}_{tr['song']}.npy")[tr["keep"]] for L in GPT_LAYERS for tr in trials}
    out = {"key": key, "listener": sub, "version": version, "lambda": folds.lam}
    for L in GPT_LAYERS:
        rng = P.rng_for(seed, key, 10 + L)

        def cgen():
            for i in range(n):
                cs = [vecs[(L, tr["song"])][rng.permutation(len(tr["keep"]))][block.keep[j]] for j, tr in enumerate(trials)]
                if i >= n0:
                    yield cs
        c_true = [vecs[(L, tr["song"])][block.keep[j]] for j, tr in enumerate(trials)]
        t, cc = run_test_impulse(block, c_true, cgen(), {"w": ms}, batch=BATCH_M)
        tval, nul = WG.channel_mean(t["w"], cc["w"])
        out[f"true_gpt{L}"], out[f"null_gpt{L}"] = tval, nul
    if audio:
        for m in ("hubert", "ast"):
            for L in AUDIO_LAYERS:
                t, cc = WG.run_test(folds, trials, f"{m}{L}", [{j: tr["F"][f"{m}{L}_mis"] for j, tr in enumerate(trials)}],
                                    {"w": folds.all_valid}, N_LAGS, batch=BATCH)
                tval, nul = WG.channel_mean(t["w"], cc["w"])
                out[f"{m}{L}"] = tval - float(np.mean(nul))
    return out


def stabilised_table(rows: list[dict]) -> tuple[pd.DataFrame, dict]:
    """Per listener and layer: true minus the mean of the first n permutations, n from the Monte Carlo rule (start 50,
    then blocks of 100) per version and layer; the audio-model gains are carried over."""
    mc, table = {}, []
    for version in sorted({r["version"] for r in rows}):
        rr = [r for r in rows if r["version"] == version]
        vals = {}
        for L in GPT_LAYERS:
            t = f"gpt{L}"
            tv = np.array([r[f"true_{t}"] for r in rr])
            n_min = min(len(r[f"null_{t}"]) for r in rr)
            nv = np.array([r[f"null_{t}"][:n_min] for r in rr])
            rule = mc_rule(tv, nv, start=M_START)
            mc[f"{version}|{t}"] = rule
            vals[t] = tv - nv[:, :rule["n_used"]].mean(1)
        for i, r in enumerate(rr):
            row = {"listener": r["listener"], "version": version, **{t: float(vals[t][i]) for t in vals}}
            row |= {k: r[k] for k in r if k.startswith(("hubert", "ast"))}
            table.append(row)
    return pd.DataFrame(table), mc
