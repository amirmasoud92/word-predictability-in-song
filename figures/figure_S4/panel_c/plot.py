"""Figure S4c: song-mean melodic-surprisal gain for sung and hummed versions (dots, songs; bars, mean and 95% CI).
Stabilised permutation null."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1]))
import style as S  # noqa: E402


def main():
    d = pd.read_csv(HERE / "song_gains.csv", dtype={"song": str})
    fig = S.figure(1.40, 2.40)
    c = S.ax(fig, 0.40, 0.30, 0.95, 1.65)
    S.zero(c)
    c.set_xlim(-0.55, 1.7)
    c.set_ylim(-0.4, 1.45)
    for x, (col, ramp) in enumerate((("delta_r_sung", S.SUNG), ("delta_r_hummed", S.MELODY))):
        vals = np.asarray(d[col] * 1e3, float)
        S.dots(c, x - 0.10, vals, ramp.base, size=3.4, max_pt=8)
        S.crossbar(c, x + 0.36, vals, ramp.dark, w=0.18)
    S.name_ticks(c, [0.1, 1.1], ["sung", "hummed"], [S.SUNG.dark, S.MELODY.dark])
    c.set_yticks([0, 0.5, 1], ["0", "0.5", "1"])
    S.text(fig, 0.10, 0.18, f"note-level {S.E3}", color=S.INK)
    S.save(fig, HERE / "panel")


if __name__ == "__main__":
    main()
