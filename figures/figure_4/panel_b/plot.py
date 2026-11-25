"""Figure 4b: word-level gain for sentence occurrences matched to the list words by type and for the list words, both
with the type-mean surprisal; means over 20 subsamples and over each listener's sessions (27 listeners)."""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1]))
import style as S  # noqa: E402


def main():
    d = pd.read_csv(HERE / "listener_gains.csv")
    cols = [d.delta_r_sentences.to_numpy() * 1e3, d.delta_r_lists.to_numpy() * 1e3]
    fig = S.figure(2.35, 2.25)
    a = S.ax(fig, 0.72, 0.15, 1.50, 1.62)
    S.zero(a)
    a.set_xlim(-0.6, 1.62)
    a.set_ylim(-1.15, 1.55)
    for k, v in enumerate(cols):
        S.dots(a, k - 0.10, v, S.WORDS.base, size=3.5, max_pt=12)
        S.crossbar(a, k + 0.40, v, S.WORDS.dark, w=0.20)
    S.name_ticks(a, [0.05, 1.05], ["sentences", "word lists"], [S.WORDS.dark, S.WORDS.dark], bold=False)
    a.set_yticks([-1, -0.5, 0, 0.5, 1, 1.5], ["−1", "−0.5", "0", "0.5", "1", "1.5"])
    a.set_ylabel(f"word-predictability gain, {S.E3}")
    S.save(fig, HERE / "panel")


if __name__ == "__main__":
    main()
