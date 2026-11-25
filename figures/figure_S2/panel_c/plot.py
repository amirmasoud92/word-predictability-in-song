"""Figure S2c: offset between each released phone annotation and its audio (stimulus-only cross-correlation) for the sung
and the spoken version of each song; the flagged offset is written out."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1]))
import style as S  # noqa: E402


def main():
    d = pd.read_csv(HERE / "annotation_offsets.csv", dtype={"song": str})
    songs = d.song.tolist()
    fig = S.figure(5.05, 2.15)
    c = S.ax(fig, 0.72, 0.12, 4.20, 1.55)
    x = np.arange(len(songs))
    S.zero(c)
    for dx, col, ramp in ((-0.17, "sung_offset_s", S.SUNG), (0.17, "spoken_offset_s", S.WORDS)):
        vals = d[col].to_numpy()
        c.vlines(x + dx, 0, vals, color=ramp.base, lw=1.6)
        c.plot(x + dx, vals, "o", ms=4.4, mfc=ramp.base, mec=S.PAPER, mew=0.4, ls="none", zorder=3)
    c.text(0.02, 0.94, "sung", transform=c.transAxes, ha="left", va="center", color=S.SUNG.dark, fontweight="bold")
    c.text(0.13, 0.94, "spoken", transform=c.transAxes, ha="left", va="center", color=S.WORDS.dark, fontweight="bold")
    for i in np.flatnonzero(d.sung_flagged.to_numpy()):
        v = d.sung_offset_s.iloc[i]
        c.text(i + 0.15, v, f"{v:.2f} s", ha="left", va="center", color=S.INK)
    c.set_xticks(x, songs)
    c.tick_params(axis="x", length=0)
    c.set_xlim(-0.7, len(songs) - 0.3)
    c.set_ylim(-0.15, 3.1)
    c.set_xlabel("song")
    c.set_ylabel("offset (s)")
    S.save(fig, HERE / "panel")


if __name__ == "__main__":
    main()
