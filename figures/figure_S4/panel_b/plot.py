"""Figure S4b: coincidence of word onsets with note onsets above chance for the two note sources (dots, songs; bars, mean
and 95% CI)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1]))
import style as S  # noqa: E402


def main():
    d = pd.read_csv(HERE / "word_note_coincidence.csv", dtype={"song": str})
    fig = S.figure(1.80, 2.30)
    b = S.ax(fig, 0.70, 0.15, 0.95, 1.65)
    S.zero(b)
    b.set_xlim(-0.55, 1.7)
    b.set_ylim(-0.1, 0.65)
    b.set_yticks([0, 0.2, 0.4, 0.6], ["0", "0.2", "0.4", "0.6"])
    for x, (col, ramp) in enumerate((("coincidence_above_chance_aligned_midi", S.MELODY), ("coincidence_above_chance_f0_segmentation", S.SLATE))):
        vals = np.asarray(d[col], float)
        S.dots(b, x - 0.10, vals, ramp.base, size=3.4, max_pt=8)
        S.crossbar(b, x + 0.36, vals, ramp.dark, w=0.18)
    S.name_ticks(b, [0.1, 1.1], ["MIDI", "f0"], [S.MELODY.dark, S.SLATE.dark])
    b.set_ylabel("word onsets at notes, above chance")
    S.save(fig, HERE / "panel")


if __name__ == "__main__":
    main()
