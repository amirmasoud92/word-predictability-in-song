"""Sung against spoken lyrics: the paired contrast of the word-level gain and its equivalence test (Q1), and the speech
rate of the two versions.

Paired difference sung - spoken over listeners (and over songs): d_z with its 95 % CI, two-sided sign-flip p, TOST at
d_z = +-0.5 and at the d_z detectable with 80 % power, and the two-sided JZS BF01. Speech rate: words per minute of each
song and version from the word timing (corrected annotation of song 09, +2.88 s, from the annotation audit).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from readers import cantisong as c
from statistical_tests.inference import jzs_bf10, sesoi_resource_based, summarise, tost_one_sample
from stimulus_features import song_regressors as g


def paired(x: np.ndarray) -> dict:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    s = summarise(x, "two-sided")
    sesoi = sesoi_resource_based(x.size, 0.8, 0.05, "larger")
    return {"n": int(x.size), "mean": float(x.mean()), "dz": s["dz"], "dz_ci95": s["dz_ci95"], "p_two_sided": s["p_signflip"],
            "tost_0.5_p": tost_one_sample(x, 0.5).p, "sesoi_80pct_power": sesoi, "tost_sesoi_p": tost_one_sample(x, sesoi).p,
            "bf01_two_sided": float(1 / jzs_bf10(x, alternative="two-sided"))}


def sung_minus_spoken(L: pd.DataFrame, S: pd.DataFrame | None = None) -> dict:
    """L: listener, version, pooled (listener-level gains); S: listener, version, song, dr (per-song gains)."""
    wide = L.pivot(index="listener", columns="version", values="pooled")
    out = {"by_listeners": paired((wide["song"] - wide["speech"]).dropna())}
    if S is not None:
        item = S.groupby(["song", "version"]).dr.mean().unstack()
        out["by_songs"] = paired((item["song"] - item["speech"]).dropna())
    return out


def words_per_minute() -> dict:
    os.environ["CANTISONG_ANNOTATION"] = "corrected"
    rows = []
    for s in c.song_ids():
        w = g.load_words(s)
        for v, col in (("speech", "speech"), ("song", "song")):
            d = w.dropna(subset=[f"onset_{col}"])
            dur = float(d[f"offset_{col}"].max() - d[f"onset_{col}"].min())
            rows.append({"song": s, "version": v, "n_words": len(d), "minutes": dur / 60, "wpm": len(d) / (dur / 60)})
    wide = pd.DataFrame(rows).pivot(index="song", columns="version", values="wpm")
    return {"spoken": {"mean": float(wide["speech"].mean()), "range": [float(wide["speech"].min()), float(wide["speech"].max())]},
            "sung": {"mean": float(wide["song"].mean()), "range": [float(wide["song"].min()), float(wide["song"].max())]},
            "ratio_sung_over_spoken_mean": float((wide["song"] / wide["speech"]).mean()),
            "paired_difference": paired((wide["song"] - wide["speech"]).to_numpy())}
