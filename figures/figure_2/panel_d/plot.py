"""Figure 2d. Robustness of the word-level effect: d_z of the primary analysis, five analysis variants and two
controls, for spoken, sung and hummed lyrics and the Alice audiobook. Filled, p < 0.05 (variants: family-wise
corrected by max-t over the six specifications; controls: sign-flip test); open, not significant. Stabilised
permutation null.

Data: specifications.csv (one row per condition and specification: d_z, p, type of p, filled).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1]))
import style as S  # noqa: E402

T = pd.read_csv(HERE / "specifications.csv")
cols = [("spoken", S.WORDS), ("sung", S.SUNG), ("hummed", S.MELODY), ("audiobook", S.WORDS)]
labels = list(dict.fromkeys(T.specification))
ys = np.r_[np.arange(6), 6.6, 7.6]
x_lab, x, y_top, cw, gap, rh = 1.42, 1.50, 0.25, 0.56, 0.10, 0.185
h = rh * (ys[-1] + 1)
fig = S.figure(4.10, 2.30)

for k, (nm, ramp) in enumerate(cols):
    a = S.ax(fig, x + k * (cw + gap), y_top, cw, h)
    a.axvline(0, color=S.SLATE.mid, lw=0.6, zorder=0)
    for lab, y in zip(labels, ys):
        r = T[(T.condition == nm) & (T.specification == lab)].iloc[0]
        col = ramp.base if r.filled else S.SLATE.mid
        a.plot([0, r.dz], [y, y], color=col, lw=1.3, solid_capstyle="butt", zorder=2)
        a.plot(r.dz, y, "o", ms=5.0, mfc=col if r.filled else S.PAPER, mec=col, mew=1.1, zorder=3)
    a.set_ylim(ys[-1] + 0.55, -0.55)
    a.set_xlim(-0.08, 1.25)
    a.set_xticks([0, 1], ["0", "1"])
    a.set_yticks([])
    a.spines["left"].set_visible(False)
    a.set_title(nm, color=ramp.dark, fontweight="bold", fontsize=8.0, pad=5)
    if k == 0:
        tr = a.get_yaxis_transform()
        for lab, y in zip(labels, ys):
            a.text((x_lab - x) / cw, y, lab, transform=tr, ha="right", va="center", color=S.INK,
                   fontweight="bold" if lab == "primary analysis" else "normal")
S.text(fig, x + 2 * (cw + gap) - gap / 2, y_top + h + 0.34, S.DZ, ha="center", color=S.INK)

S.save(fig, HERE / "panel")
