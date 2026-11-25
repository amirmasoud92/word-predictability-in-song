"""Figure 5d: melodic-surprisal gain of notes under expected and under unexpected words (split at the song's median
lexical surprisal of the nearest word; dots, listeners; bars, mean and 95% CI; above, d_z). Original permutation null."""
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1]))
import style as S  # noqa: E402


def main():
    d = pd.read_csv(HERE / "listener_gains.csv")
    dz = json.loads((HERE / "effect_sizes.json").read_text())
    cols = [d.delta_r_expected_word.to_numpy() * 1e3, d.delta_r_unexpected_word.to_numpy() * 1e3]
    fig = S.figure(2.35, 2.45)
    a = S.ax(fig, 0.72, 0.30, 1.50, 1.62)
    S.zero(a)
    a.set_xlim(-0.6, 1.62)
    a.set_ylim(-1.0, 2.5)
    for k, v in enumerate(cols):
        S.dots(a, k - 0.10, v, S.MELODY.base, size=3.5, max_pt=12)
        S.crossbar(a, k + 0.40, v, S.MELODY.dark, w=0.20)
    S.name_ticks(a, [0.05, 1.05], ["expected\nword", "unexpected\nword"], [S.INK, S.INK], bold=False)
    a.set_yticks([-1, 0, 1, 2], ["−1", "0", "1", "2"])
    a.set_ylabel(f"note-predictability gain, {S.E3}")
    for j, key in enumerate(("expected_word_dz", "unexpected_word_dz")):
        S.above(a, j + 0.05, S.dz_txt(dz[key]), color=S.INK2)
    S.save(fig, HERE / "panel")


if __name__ == "__main__":
    main()
