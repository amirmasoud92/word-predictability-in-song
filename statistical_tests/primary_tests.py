"""Multiplicity across the 13 primary tests of the paper, and across the seven tests of the additional analyses.

The tests are G1, G2, G2b (CantiSong), TH1, TH2 (Alice), 2b (speech in noise), A (MEG attention), 2 (MEG-MASC), L1 sung,
L1 hummed, L2 (melody) and M1 spoken, M1 sung (language-model representations). All are one-sided and use the estimates
with which they were run (10 permutations per listener; 5 for M1; the MEG tests and 2b under their own permutation
counts). Their p values are adjusted with Holm's step-down procedure (family-wise error rate) and with the
Benjamini-Hochberg procedure (false-discovery rate). The seven tests of the additional analyses (word surprisal beyond
phoneme surprisal and the reverse, in the spoken and the sung versions; transfer from the spoken to the sung versions and
the reverse; the continuous interaction test) are Holm-corrected as one family.
"""
from __future__ import annotations

PRIMARY = ("G1", "G2", "G2b", "TH1", "TH2", "2b", "A", "2", "L1 sung", "L1 hummed", "L2", "M1 spoken", "M1 sung")
ADDITIONAL = ("1b_speech", "1b_song", "1c_speech", "1c_song", "2b", "2c", "3a")


def holm(p: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p values."""
    names = sorted(p, key=p.get)
    m, run, out = len(names), 0.0, {}
    for i, k in enumerate(names):
        run = max(run, min(1.0, (m - i) * p[k]))
        out[k] = run
    return out


def benjamini_hochberg(p: dict[str, float]) -> dict[str, float]:
    """Benjamini-Hochberg adjusted p values."""
    names = sorted(p, key=p.get)
    m, run, out = len(names), 1.0, {}
    for i in range(m - 1, -1, -1):
        run = min(run, p[names[i]] * m / (i + 1))
        out[names[i]] = run
    return out


def adjust_primary(p: dict[str, float]) -> dict[str, dict[str, float]]:
    """p: the uncorrected one-sided p value of each of the 13 primary tests (keys as in PRIMARY)."""
    assert set(p) == set(PRIMARY), sorted(set(PRIMARY) ^ set(p))
    h, b = holm(p), benjamini_hochberg(p)
    return {k: {"p": p[k], "p_holm": h[k], "p_bh": b[k]} for k in PRIMARY}


def adjust_additional(p: dict[str, float]) -> dict[str, float]:
    """p: the uncorrected p value of each of the seven additional tests (keys as in ADDITIONAL); Holm over the seven."""
    assert set(p) == set(ADDITIONAL), sorted(set(ADDITIONAL) ^ set(p))
    return holm(p)
