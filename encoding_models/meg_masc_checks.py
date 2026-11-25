"""Pipeline checks of MEG-MASC.

C1, envelope tracking: per listener and session, a forward model with the envelope and its onsets (1-8 Hz, lags 0-688 ms,
leave-one-sound-file-out ridge chosen on this model) predicts the held-out MEG; r over valid samples, mean over channels.
C2, word timing: the model with acoustic, phonetic, word-onset, control and surprisal regressors, with the true word
regressors and with those of the next sound file of the same story (cyclic; truncated or extended cyclically to the
file's length); r over valid samples (each model with its own ridge choice), mean over channels.
"""
from __future__ import annotations

import gc

import numpy as np
import pandas as pd
import torch

from encoding_models import gpu_encoding as WG
from readers import meg_masc as C


def heldout_r(ff: WG.GpuFolds) -> float:
    preds = [(fd["XaUv"] @ (fd["UtCa"] / (fd["e"] + ff.lam * fd["e"].mean())[:, None])).T[None] for fd in ff.folds]
    return float(ff.all_valid.stats(preds)["pooled"][0].mean())


def word_rows(st: dict, n: int) -> np.ndarray:
    """Word onsets (sentence, list), controls and narrative surprisal of the sentence words, raw (7 x n)."""
    s = np.where(~st["is_list"] & np.isfinite(st["s_gpt2"]), st["s_gpt2"], 0.0)
    S = C.impulse(st["onset_idx"], s, n)
    return np.vstack([st["word_sent_raw"][:, :n], st["word_list_raw"][:, :n], st["controls_raw"][:, :n], S])


def fit_r(trials, names) -> float:
    ff = WG.GpuFolds(trials, names, C.N_LAGS)
    r = heldout_r(ff)
    del ff
    gc.collect()
    torch.cuda.empty_cache()
    return r


def cyclic(F: np.ndarray, n: int) -> np.ndarray:
    reps = int(np.ceil(n / F.shape[1]))
    return np.tile(F, (1, reps))[:, :n]


def session_checks(sub: str, ses: int) -> dict:
    trials = C.listener_trials(sub, ses)
    for tr in trials:
        st, T = tr["st"], tr["T"]
        tr["F"]["env"] = C.bz(st["acoustic_raw"][:2, :T])
    out = {"listener": sub, "session": ses, "n_files": len(trials), "c1_r": fit_r(trials, ["env"])}
    keys = [tr["key"] for tr in trials]
    for tr in trials:
        st, T = tr["st"], tr["T"]
        tr["F"] = {"acoustic": C.bz(st["acoustic_raw"][:, :T]), "phonetic": C.bz(st["phonetic_raw"][:, :T])}
        tr["F"]["words_true"] = C.bz(word_rows(st, T))
        story_keys = C.sound_keys(tr["story"])
        nxt = story_keys[(story_keys.index(tr["key"]) + 1) % len(story_keys)]
        so = C.features(nxt)
        tr["F"]["words_mis"] = C.bz(cyclic(word_rows(so, int(so["n"])), T))
    out["c2_r_true"] = fit_r(trials, ["acoustic", "phonetic", "words_true"])
    out["c2_r_mismatched"] = fit_r(trials, ["acoustic", "phonetic", "words_mis"])
    out["c2_diff"] = out["c2_r_true"] - out["c2_r_mismatched"]
    out["files"] = ";".join(keys)
    return out



def checks_table() -> pd.DataFrame:
    """C1 and C2 for every usable session, written to checks_listeners.csv."""
    t = C.inclusion()
    rows = [session_checks(r.listener, int(r.session)) for _, r in t[t.usable].iterrows()]
    d = pd.DataFrame(rows)
    d.to_csv(C.TAB / "checks_listeners.csv", index=False)
    return d
