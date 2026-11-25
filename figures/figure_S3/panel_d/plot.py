"""Figure S3d: lexical entropy beyond surprisal, per listener (CantiSong versions and the Alice audiobook; dots, listeners;
bars, mean and 95% CI). Stabilised permutation null."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1]))
import style as S  # noqa: E402


def main():
    d = pd.read_csv(HERE / "listener_gains.csv")
    conds = [("spoken", S.WORDS), ("sung", S.SUNG), ("hummed", S.MELODY), ("audiobook", S.WORDS)]
    fig = S.figure(4.45, 2.10)
    a = S.ax(fig, 0.72, 0.12, 3.60, 1.50)
    S.zero(a)
    a.set_xlim(-0.55, 3.75)
    a.set_ylim(-0.75, 1.35)
    for x, (c, ramp) in enumerate(conds):
        vals = np.asarray(d[d.condition == c].delta_r * 1e3, float)
        S.dots(a, x - 0.10, vals, ramp.base, size=3.4, max_pt=13)
        S.crossbar(a, x + 0.36, vals, ramp.dark, w=0.18)
    S.name_ticks(a, [0.1, 1.1, 2.1, 3.1], [c for c, _ in conds], [S.WORDS.dark, S.SUNG.dark, S.MELODY.dark, S.WORDS.dark])
    a.set_yticks([-0.5, 0, 0.5, 1], ["−0.5", "0", "0.5", "1"])
    a.set_ylabel(f"entropy {S.E3}")
    S.save(fig, HERE / "panel")


if __name__ == "__main__":
    main()
