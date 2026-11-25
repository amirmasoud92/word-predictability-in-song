"""Figure S2a: the surprisal-specific response kernel (true minus permuted surprisal; mean over the nine centro-parietal
channels) after word onset for spoken, sung and hummed versions (lines, means; bands, 95% CI over 20 listeners); squares
mark lags with a family-wise corrected negative deflection (max-t); shaded, 300-500 ms."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1]))
import style as S  # noqa: E402

VERSIONS = ("speech", "song", "melody")


def main():
    K = pd.read_csv(HERE / "kernels.csv")
    tests = pd.read_csv(HERE / "kernel_tests.csv")
    fig = S.figure(4.60, 2.30)
    a = S.ax(fig, 0.72, 0.15, 3.75, 1.70)
    a.axvspan(300, 500, color=S.BAND, lw=0, zorder=-1)
    S.zero(a)
    for k, v in enumerate(VERSIONS):
        ramp, nm = S.VERSION[v]
        w = K[K.version == v].pivot(index="listener_row", columns="lag_ms", values="kernel") * 1e3
        lags = w.columns.to_numpy(float)
        m = w.mean(axis=0).to_numpy()
        h = np.array([S.mean_ci(w[c])[1] for c in w.columns])
        a.fill_between(lags, m - h, m + h, color=ramp.base, alpha=0.14, lw=0, zorder=1)
        a.plot(lags, m, color=ramp.base, lw=1.5, zorder=2 + k)
        t = tests[tests.version == v]
        sig = t.p_fwe_negative.to_numpy() < 0.05
        if sig.any():
            a.plot(t.lag_ms.to_numpy()[sig], np.full(sig.sum(), -2.75), marker="s", ms=4.2, color=ramp.base, ls="none", zorder=5)
        a.text(0.02 + 0.17 * k, 0.94, nm, transform=a.transAxes, ha="left", va="center", color=ramp.dark, fontweight="bold")
    a.set_xlim(0, 700)
    a.set_ylim(-3.0, 3.6)
    a.set_xlabel("lag after word onset (ms)")
    a.set_ylabel("kernel, true − permuted (× 10$^{3}$)")
    S.save(fig, HERE / "panel")


if __name__ == "__main__":
    main()
