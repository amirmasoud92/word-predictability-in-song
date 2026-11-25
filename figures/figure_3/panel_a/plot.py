"""Figure 3a. An excerpt of the MEG attention paradigm (chapter 1, 15-24 s, of the recording in which the
lower-pitched talker was attended): amplitude envelopes of the attended and the ignored stream (shading) and the
words of each stream, with tick height proportional to their surprisal (clipped at 22 bits); below, the layout of
the 248 magnetometers (nose up).

Data: envelopes.csv (envelopes at 200 Hz, normalised to their 99.5th percentile over the chapter), words.csv (word
onsets and surprisal of each stream), sensors.csv (2-D magnetometer positions, head radius 1).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1]))
import style as S  # noqa: E402

env = pd.read_csv(HERE / "envelopes.csv")
words = pd.read_csv(HERE / "words.csv")
xy = pd.read_csv(HERE / "sensors.csv")[["x", "y"]].to_numpy()

t0, t1, smax = 15.0, 24.0, 22.0
X0, X1 = 0.92, 2.52
tx = lambda t: X0 + (np.asarray(t) - t0) / (t1 - t0) * (X1 - X0)
fig = S.figure(2.90, 2.70)
x0, y0, W, H = 0.10, 0.26, 2.75, 2.40
c = S.ax(fig, x0, y0, W, H)
c.set_xlim(0, W)
c.set_ylim(H, 0)
c.axis("off")

for nm, ramp, yb in (("attended", S.WORDS, 0.62), ("ignored", S.SLATE, 1.34)):
    c.fill_between(tx(env.time_s), yb, yb - 0.19 * np.clip(env[nm], 0, 1.2), color=ramp.light, lw=0)
    w = words[words.stream == nm]
    c.vlines(tx(w.onset_s), yb - 0.25, yb - 0.25 - 0.24 * np.clip(w.surprisal_bits, 0, smax) / smax, color=ramp.base, lw=0.9)
    c.plot([X0, X1], [yb, yb], color=S.RULE, lw=0.5)
    c.text(X0 - 0.10, yb - 0.14, nm, ha="right", va="baseline", fontweight="bold", color=ramp.dark)

# 2-s scale bar
c.plot([tx(t1 - 2), tx(t1)], [1.48, 1.48], color=S.INK, lw=1.0, solid_capstyle="butt")
c.text((tx(t1 - 2) + tx(t1)) / 2, 1.48 + 0.14, "2 s", ha="center", va="baseline", color=S.INK2)

# both streams reach the listener, whose MEG is recorded
xm = (X0 + X1) / 2 - 0.10
c.plot([X0 + 0.10, X0 + 0.10, xm - 0.42], [1.42, 1.86, 1.86], color=S.SLATE.mid, lw=0.7)
c.add_patch(FancyArrowPatch((xm - 0.42, 1.86), (xm - 0.34, 1.86), arrowstyle="-|>", mutation_scale=6, color=S.SLATE.mid, lw=0.7,
                            shrinkA=0, shrinkB=0, zorder=2))
h = S.ax(fig, x0 + xm - 0.33, y0 + 1.52, 0.66, 0.68)
h.scatter(xy[:, 0], xy[:, 1], s=0.7, color=S.SLATE.base, linewidths=0, zorder=3)
S.head_outline(h, color=S.INK2, lw=0.7)
h.set_xlim(-1.20, 1.20)
h.set_ylim(-1.14, 1.24)
h.set_aspect("equal")
h.axis("off")
c.text(xm + 0.42, 1.90, "MEG", ha="left", va="baseline", color=S.INK2)

S.save(fig, HERE / "panel")
