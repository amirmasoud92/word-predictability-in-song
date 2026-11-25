"""Per-label statistics of the source analysis (exploratory).

Per listener and label: Delta r = r(true) - mean r over the 100 permutations, for the attended and the ignored stream;
attended - ignored per label, one-sided sign-flip test over listeners (10^5 flips), Benjamini-Hochberg false-discovery
rate over the 68 labels. Lateralisation of the attended effect: L, R = mean attended Delta r over the superior and middle
temporal labels of the left and right hemisphere; LI = (L - R)/(|L| + |R|) and L - R, two-sided sign-flip tests; the
same on attended - ignored; split-half reliability from odd versus even chapters (Spearman across listeners,
Spearman-Brown).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

from common.settings import load_paths
from readers import meg_attention as C
from statistical_tests.inference import dz_ci, sign_flip_test

SRC = load_paths().derived_root / "meg_attention" / "source_word_level"
T8 = load_paths().results_root / "meg_attention"
TEMPORAL = ("superiortemporal", "middletemporal")


def bh(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    n = p.size
    o = np.argsort(p)
    q = np.empty(n)
    q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1.0)


def test(x, alternative, key) -> dict:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    p, _ = sign_flip_test(x, alternative, n_perm=100_000, rng=C.rng(f"p8|src|{key}"))
    lo, hi = dz_ci(x)
    return {"n": int(x.size), "mean": float(x.mean()), "dz": float(x.mean() / x.std(ddof=1)), "dz_ci95": [float(lo), float(hi)],
            "p": float(p), "alternative": alternative, "n_positive": int((x > 0).sum())}


def label_statistics() -> dict:
    files = sorted(SRC.glob("P*.npz"))
    labels = None
    dA, dI = {}, {}
    for f in files:
        with np.load(f) as z:
            labels = list(z["labels"])
            p = int(f.stem[1:])
            dA[p] = z["attended_rtrue"] - z["attended_rperm"].mean(0)          # subsets x labels
            dI[p] = z["ignored_rtrue"] - z["ignored_rperm"].mean(0)
    ps = sorted(dA)
    A = np.stack([dA[p] for p in ps])                                          # listeners x subsets x labels
    I = np.stack([dI[p] for p in ps])
    rows = []
    for k, lab in enumerate(labels):
        c = test(A[:, 0, k] - I[:, 0, k], "greater", f"contrast|{lab}")
        a = test(A[:, 0, k], "greater", f"att|{lab}")
        rows.append({"label": lab, "contrast_mean": c["mean"], "contrast_dz": c["dz"], "contrast_p": c["p"],
                     "attended_mean": a["mean"], "attended_dz": a["dz"], "attended_p": a["p"],
                     "ignored_mean": float(I[:, 0, k].mean())})
    d = pd.DataFrame(rows)
    d["contrast_q_bh"] = bh(d.contrast_p)
    d["attended_q_bh"] = bh(d.attended_p)
    T8.mkdir(parents=True, exist_ok=True)
    d.to_csv(T8 / "source_labels.csv", index=False)
    idx = {h: [labels.index(f"ctx-{h}-{t}") for t in TEMPORAL] for h in ("lh", "rh")}

    def lat(M, s):
        L, R = M[:, s, idx["lh"]].mean(1), M[:, s, idx["rh"]].mean(1)
        return L, R, (L - R) / (np.abs(L) + np.abs(R) + 1e-300), L - R

    res = {"n_listeners": len(ps), "n_labels": len(labels),
           "contrast_labels_q05": d.loc[d.contrast_q_bh < 0.05, "label"].tolist(),
           "attended_labels_q05": d.loc[d.attended_q_bh < 0.05, "label"].tolist(),
           "top_contrast": d.sort_values("contrast_dz", ascending=False).head(8)[["label", "contrast_dz", "contrast_q_bh"]].to_dict("records")}
    for name, M in (("attended", A), ("contrast", A - I)):
        L, R, LI, D = lat(M, 0)
        Lo, Ro, LIo, Do = lat(M, 1)
        Le, Re, LIe, De = lat(M, 2)
        r_li = stats.spearmanr(LIo, LIe).correlation
        r_d = stats.spearmanr(Do, De).correlation
        res[f"lateralisation_{name}"] = {
            "L_mean": float(L.mean()), "R_mean": float(R.mean()),
            "LI": test(LI, "two-sided", f"LI|{name}"), "L_minus_R": test(D, "two-sided", f"LmR|{name}"),
            "split_half_LI": {"spearman": float(r_li), "spearman_brown": float(2 * r_li / (1 + r_li))},
            "split_half_L_minus_R": {"spearman": float(r_d), "spearman_brown": float(2 * r_d / (1 + r_d))}}
    (T8 / "source.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res
