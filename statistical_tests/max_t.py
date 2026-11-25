"""Sign-flip max-t correction over several statistics measured on the same units (Nichols & Holmes, 2002).

One-sample t statistics of each column of a units x specifications matrix; the same sign flip is applied to all columns
of a unit, preserving the dependence between specifications (or lags). The family-wise p of a column is the share of
flips whose largest statistic over columns reaches the observed one. Flips are exhaustive up to 18 units, otherwise
Monte Carlo.
"""
from __future__ import annotations

import numpy as np


def signflip_maxt(S: np.ndarray, n_perm: int | None = None, seed: int | None = None, alternative: str = "greater",
                  exact_max_n: int = 18, chunk: int = 4096) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """S: units x specifications (no missing values). Returns (t, uncorrected p, family-wise corrected p)."""
    S = np.asarray(S, float)
    n, V = S.shape
    sumsq = (S ** 2).sum(0)

    def tstat(means):
        var = (sumsq - n * means ** 2) / (n - 1)
        return means / np.sqrt(np.maximum(var, 1e-300) / n)

    t_obs = tstat(S.mean(0))
    sign = {"greater": 1.0, "less": -1.0, "two-sided": None}[alternative]
    stat = (lambda t: sign * t) if sign is not None else np.abs
    obs = stat(t_obs)
    if n <= exact_max_n:
        codes = np.arange(2 ** n, dtype=np.int64)
        flips_iter = (1 - 2 * ((codes[i:i + chunk, None] >> np.arange(n)) & 1) for i in range(0, codes.size, chunk))
        total = codes.size
    else:
        g = np.random.default_rng(seed)
        total = n_perm or 20000
        flips_iter = (g.choice([-1.0, 1.0], size=(min(chunk, total - i), n)) for i in range(0, total, chunk))
    ge_unc = np.zeros(V)
    max_null = []
    for F in flips_iter:
        T = stat(tstat(F.astype(float) @ S / n))
        ge_unc += (T >= obs[None, :] - 1e-12).sum(0)
        max_null.append(T.max(1))
    max_null = np.concatenate(max_null)
    p_unc = ge_unc / total
    p_fwe = (max_null[:, None] >= obs[None, :] - 1e-12).mean(0)
    return t_obs, p_unc, p_fwe
