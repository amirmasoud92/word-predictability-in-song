"""Listeners and songs as crossed random samples: mixed models and min F' for the sung - hummed contrast of the per-song
word-level gain (CantiSong).

* Model without random slopes: dr ~ 0 + version + (1 | listener) + (1 | song), contrasts per version and sung - hummed
  (one-sided for the predicted effects), Satterthwaite t; sensitivity model with (1 | listener:song).
* Model with random slopes (test 5): dr ~ condition + (condition | listener) + (condition | song) + (1 | listener:song)
  with condition +-0.5, and the singular-fit fallbacks of crossed_mixed_model_slopes.R; one-sided p from the
  Satterthwaite t.
* min F' (Clark, 1973) combining the paired t tests of sung - hummed over listeners (means over songs) and over songs
  (means over listeners): min F' = F1 F2 / (F1 + F2) with denominator df (F1 + F2)^2 / (F1^2 / df2 + F2^2 / df1); its p
  is one-sided (sung > hummed), from t = sqrt(min F'), and only when both t statistics are positive.
The R scripts run with lme4 and lmerTest; the Rscript executable is taken from the RSCRIPT environment variable, else
from the PATH.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

HERE = Path(__file__).parent


def _rscript() -> str:
    return os.environ.get("RSCRIPT") or shutil.which("Rscript") or "Rscript"


def intercepts_model(S: pd.DataFrame, work: Path) -> dict:
    """S: listener, version, song, dr (per-song gains of the three versions)."""
    work.mkdir(parents=True, exist_ok=True)
    f_in, f_fx, f_sens, f_vc = (work / n for n in ("crossed_input.csv", "crossed_fixed.csv", "crossed_fixed_sensitivity.csv", "crossed_varcomp.csv"))
    S.to_csv(f_in, index=False)
    subprocess.run([_rscript(), str(HERE / "crossed_mixed_model_intercepts.R"), str(f_in), str(f_fx), str(f_sens), str(f_vc)],
                   check=True, capture_output=True)
    res = {}
    fx = pd.read_csv(f_fx)
    fx["p_one_sided"] = st.t.sf(fx.t, fx.df)
    fx.loc[fx.term == "hummed", "p_one_sided"] = np.nan
    fx["p_two_sided"] = 2 * st.t.sf(np.abs(fx.t), fx.df)
    res["fixed"] = {r.term: {k: float(getattr(r, k)) for k in ("estimate", "se", "df", "t", "p_one_sided", "p_two_sided")} for r in fx.itertuples()}
    res["variance_components"] = pd.read_csv(f_vc).set_index("grp").vcov.to_dict()
    fs = pd.read_csv(f_sens)
    fs["p_one_sided"] = st.t.sf(fs.t, fs.df)
    fs.loc[fs.term == "hummed", "p_one_sided"] = np.nan
    res["sensitivity_listener_song"] = {r.term: {k: float(getattr(r, k)) for k in ("estimate", "se", "df", "t", "p_one_sided")} for r in fs.itertuples()}
    return res


def min_f_prime(d: pd.DataFrame) -> dict:
    """d: listener, song, version (song or melody), dr."""
    by_l = d.pivot_table(index="listener", columns="version", values="dr", aggfunc="mean").dropna()
    by_s = d.pivot_table(index="song", columns="version", values="dr", aggfunc="mean").dropna()
    t1 = st.ttest_1samp(by_l["song"] - by_l["melody"], 0)
    t2 = st.ttest_1samp(by_s["song"] - by_s["melody"], 0)
    F1, F2, df1, df2 = t1.statistic ** 2, t2.statistic ** 2, len(by_l) - 1, len(by_s) - 1
    minF = F1 * F2 / (F1 + F2)
    df_den = (F1 + F2) ** 2 / (F1 ** 2 / df2 + F2 ** 2 / df1)
    both_pos = t1.statistic > 0 and t2.statistic > 0
    return {"F1_listeners": float(F1), "df1": df1, "F2_songs": float(F2), "df2": df2, "min_F": float(minF), "df": [1, float(df_den)],
            "p_one_sided": float(st.t.sf(np.sqrt(minF), df_den)) if both_pos else None}


def slopes_model(S: pd.DataFrame, work: Path) -> dict:
    """Test 5 and min F' on the per-song gains S (listener, version, song, dr)."""
    work.mkdir(parents=True, exist_ok=True)
    d = S[S.version.isin(["song", "melody"])].copy()
    d["condition"] = np.where(d.version == "song", 0.5, -0.5)
    f_in, f_fx, f_vc = work / "crossed_slopes_input.csv", work / "crossed_slopes_fixed.csv", work / "crossed_slopes_vc.csv"
    d.to_csv(f_in, index=False)
    r = subprocess.run([_rscript(), str(HERE / "crossed_mixed_model_slopes.R"), str(f_in), str(f_fx), str(f_vc)], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr[-2000:])
    fx = pd.read_csv(f_fx).iloc[0]
    res = {"model": fx.model, "estimate": float(fx.estimate), "se": float(fx.se), "df": float(fx.df), "t": float(fx.t),
           "p_one_sided": float(st.t.sf(fx.t, fx.df)), "singular": bool(fx.singular),
           "variance_components": pd.read_csv(f_vc).fillna("").to_dict("records")}
    res["min_F_prime"] = min_f_prime(d)
    return res
