"""Figure 1a. The first line of CantiSong song 01 in its three versions: amplitude envelope (shading), words (blue bars
over their duration, written out for the sung version) and notes (gold bars; height, pitch). Sung and hummed versions
share one timeline; the hummed version has no words (dotted boxes mark where they would be). The spoken version has
its own, faster timeline, aligned here at the first word. Right, what each version contains (filled, present; open,
absent).

Data: envelopes.csv (broadband amplitude envelope at 200 Hz, normalised to its 99.5th percentile over the song; the
spoken times are shifted to align its first word with the sung one), words.csv (word onsets and offsets in seconds,
spoken times shifted in the same way), notes.csv (note onsets, offsets and MIDI pitch).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Rectangle

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1]))
import style as S  # noqa: E402

env = pd.read_csv(HERE / "envelopes.csv")
words = pd.read_csv(HERE / "words.csv")
notes = pd.read_csv(HERE / "notes.csv")


def stagger(xs, labels, char_in=0.058, pad=0.05):
    """Row (0 or 1) for each label at x (inches) so that neighbouring labels on one row do not overlap."""
    rows, last = [], [-1e9, -1e9]
    for x, lab in zip(xs, labels):
        r = 0 if x >= last[0] + pad else (1 if x >= last[1] + pad else 0)
        rows.append(r)
        last[r] = x + len(lab) * char_in
    return rows


t0, t1 = 3.3, 18.5
X0, X1 = 1.02, 5.62
tx = lambda t: X0 + (np.asarray(t) - t0) / (t1 - t0) * (X1 - X0)
fig = S.figure(7.20, 2.50)
c = S.ax(fig, 0, 0, 7.20, 2.50)
c.set_xlim(0, 7.20)
c.set_ylim(2.50, 0)
c.axis("off")

rows = {"speech": 0.62, "song": 1.50, "melody": 2.34}                    # envelope baselines (inches)
pmin, pmax = notes.midi_pitch.min(), notes.midi_pitch.max()
for v, yb in rows.items():
    ramp, name = S.VERSION[v]
    e = env[env.version == v]
    c.fill_between(tx(e.time_s), yb, yb - 0.17 * np.clip(e.envelope, 0, 1.15), color=ramp.light, lw=0, zorder=1)
    c.plot([X0, X1], [yb, yb], color=S.RULE, lw=0.5, zorder=0)
    c.text(X0 - 0.14, yb - 0.20, name, ha="right", va="center", fontsize=9.0, fontweight="bold", color=ramp.dark)
    yw = yb - 0.26 - (0.30 if v != "speech" else 0.0)                    # the lane of the words
    if v != "speech":
        for r in notes.itertuples():
            a_, b_ = max(r.onset_s, t0), min(r.offset_s, t1)
            yy = yb - 0.24 - 0.22 * (r.midi_pitch - pmin) / (pmax - pmin)
            c.add_patch(Rectangle((tx(a_), yy - 0.017), tx(b_) - tx(a_) - 0.006, 0.034, color=S.MELODY.base, lw=0, zorder=2))
    on = (words.onset_sung_s if v != "speech" else words.onset_spoken_s).to_numpy()
    off = (words.offset_sung_s if v != "speech" else words.offset_spoken_s).to_numpy()
    for a_, b_ in zip(on, off):
        if v == "melody":
            c.add_patch(Rectangle((tx(a_) + 0.01, yw - 0.035), tx(b_) - tx(a_) - 0.02, 0.07, fc="none", ec=S.SLATE.mid, lw=0.5,
                                  ls=(0, (1.5, 1.5)), zorder=3))
        else:
            c.add_patch(FancyBboxPatch((tx(a_) + 0.01, yw - 0.035), tx(b_) - tx(a_) - 0.02, 0.07, boxstyle="round,pad=0,rounding_size=0.02",
                                       fc=S.WORDS.base, ec="none", lw=0.6, zorder=3))
    if v == "song":
        for a_, w_, rr in zip(on, words.word, stagger([tx(a_) for a_ in on], list(words.word))):
            c.text(tx(a_) + 0.01, yw - 0.07 - 0.15 * rr, w_, ha="left", va="baseline", color=S.INK)

# 2-s scale bar
y = rows["speech"] - 0.22
c.plot([tx(t1 - 2), tx(t1)], [y, y], color=S.INK, lw=1.0, solid_capstyle="butt")
c.text((tx(t1 - 2) + tx(t1)) / 2, y + 0.14, "2 s", ha="center", va="baseline", color=S.INK2)

# what each version contains
cols = [("words", 6.12, S.WORDS), ("melody", 6.72, S.MELODY)]
has = {"speech": (1, 0), "song": (1, 1), "melody": (0, 1)}
for nm, x, ramp in cols:
    c.text(x, 0.14, nm, ha="center", va="baseline", color=ramp.dark, fontweight="bold")
for v, yb in rows.items():
    for (nm, x, ramp), h in zip(cols, has[v]):
        c.plot(x, yb - 0.20, "o", ms=7.0, mfc=ramp.base if h else S.PAPER, mec=ramp.base if h else S.SLATE.mid, mew=0.9, zorder=3)

S.save(fig, HERE / "panel")
