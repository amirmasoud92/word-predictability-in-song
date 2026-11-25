"""Pipeline checks of the MEG attention dataset: envelope tracking (positive control) and the timing gate.

Per included listener and stream: a forward model with the stream's envelope and envelope onsets (1-8 Hz, lags 0-688 ms,
leave-one-chapter-out ridge, ridge chosen per model from the grid), r over all valid samples averaged over channels.
Mismatched versions s = 1-4: chapter k of a recording takes the same stream's envelope rows from chapter k + s (cyclic
over the book's five chapters), extended cyclically or truncated to chapter k's length, then band-passed and z-scored.
Positive control: attended > ignored envelope tracking. Timing gate: true > next chapter (s = 1) for the ignored stream
(and, as a check, the attended stream); a listener is flagged when the true r is not above the mean of the four
mismatched versions (statistical_tests/meg_attention_tests.py).
"""
from __future__ import annotations

import numpy as np

from encoding_models import ridge_encoding as E
from readers import meg_attention as C


def envelope_rows(f: str, k: int, stream: str, T: int, shift: int) -> np.ndarray:
    k2 = (k - 1 + shift) % 5 + 1
    raw = C.features(f, k2)[f"{stream}_acoustic_raw"][:2]
    idx = np.arange(T) % raw.shape[1]
    return C.bz(raw[:, idx])


def listener_tracking(p: int) -> dict:
    if not C.included(p):
        return {"listener": p, "included": False}
    trials = C.listener_trials(p)
    row = {"listener": p, "included": True, "n_chapters": len(trials)}
    masks = [tr["v"] for tr in trials]
    for stream in C.STREAMS:
        for s in range(5):
            for tr in trials:
                tr["F"]["env"] = envelope_rows(tr["file"], tr["chapter"], stream, tr["T"], s)
            preds, lam = E.fit_predict(trials, ["env"], None, C.N_LAGS)
            tag = "true" if s == 0 else f"mis{s}"
            row[f"r_{stream}_{tag}"] = E.r_mean(trials, preds, masks, None)
            row[f"lam_{stream}_{tag}"] = lam
    return row


def checks_table(n_jobs: int = 4):
    """Envelope tracking of every listener (true and mismatched chapters), written to checks_listeners.csv."""
    import pandas as pd
    from joblib import Parallel, delayed
    rows = Parallel(n_jobs=n_jobs)(delayed(listener_tracking)(p) for p in range(1, 59))
    d = pd.DataFrame(rows)
    C.TAB.mkdir(parents=True, exist_ok=True)
    d.to_csv(C.TAB / "checks_listeners.csv", index=False)
    return d
