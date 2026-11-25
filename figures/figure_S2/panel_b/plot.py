"""Figure S2b: word-level gain per listener with line and word repetition modelled (dots, listeners; bars, mean and 95% CI)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1]))
import style as S  # noqa: E402

VERSIONS = ("speech", "song", "melody")


def main():
    d = pd.read_csv(HERE / "listener_gains.csv")
    fig = S.figure(2.25, 2.30)
    b = S.ax(fig, 0.62, 0.15, 1.55, 1.70)
    S.zero(b)
    b.set_xlim(-0.55, 2.7)
    b.set_ylim(-1.7, 1.4)
    for x, v in enumerate(VERSIONS):
        ramp = S.VERSION[v][0]
        vals = np.asarray(d[v] * 1e3, float)
        S.dots(b, x - 0.10, vals, ramp.base, size=3.4, max_pt=11)
        S.crossbar(b, x + 0.36, vals, ramp.dark, w=0.18)
    S.name_ticks(b, [0.1, 1.1, 2.1], [S.VERSION[v][1] for v in VERSIONS], [S.VERSION[v][0].dark for v in VERSIONS])
    b.set_yticks([-1.5, -1, -0.5, 0, 0.5, 1], ["−1.5", "−1", "−0.5", "0", "0.5", "1"])
    b.set_ylabel(S.E3)
    S.save(fig, HERE / "panel")


if __name__ == "__main__":
    main()
