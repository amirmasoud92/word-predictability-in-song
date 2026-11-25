"""Figure S4a: stimulus-only audit of the two note sources per song: agreement of each source's pitch with the hummed f0
(filled, composed MIDI aligned by dynamic time warping; open, notes segmented from f0)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1]))
import style as S  # noqa: E402


def main():
    au = pd.read_csv(HERE / "note_source_audit.csv", dtype={"song": str})
    fig = S.figure(3.90, 2.35)
    a = S.ax(fig, 0.72, 0.15, 3.05, 1.65)
    x = np.arange(len(au))
    for xx, r in zip(x, au.itertuples()):
        a.plot([xx, xx], [r.pitch_agreement_f0_segmentation, r.pitch_agreement_aligned_midi], color=S.SLATE.light, lw=1.2, zorder=1)
    a.plot(x, au.pitch_agreement_aligned_midi, "o", ms=4.8, mfc=S.MELODY.base, mec=S.PAPER, mew=0.4, ls="none", zorder=3)
    a.plot(x, au.pitch_agreement_f0_segmentation, "o", ms=4.4, mfc=S.PAPER, mec=S.SLATE.base, mew=1.0, ls="none", zorder=3)
    a.text(0.02, 0.07, "aligned MIDI", transform=a.transAxes, ha="left", va="center", color=S.MELODY.dark, fontweight="bold")
    a.text(0.30, 0.07, "segmented from f0", transform=a.transAxes, ha="left", va="center", color=S.SLATE.dark, fontweight="bold")
    a.set_xticks(x[::2], au.song.str.zfill(2)[::2])
    a.tick_params(axis="x", length=0)
    a.set_xlim(-0.7, len(au) - 0.3)
    a.set_ylim(0.5, 1.0)
    a.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0], ["0.5", "0.6", "0.7", "0.8", "0.9", "1"])
    a.set_xlabel("song")
    a.set_ylabel("pitch agreement with hummed f0")
    S.save(fig, HERE / "panel")


if __name__ == "__main__":
    main()
