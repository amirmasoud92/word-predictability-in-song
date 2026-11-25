"""Figure 5a: words and notes of song 01 (sung, 18-26 s) on one timeline. Words are shaded by lexical (GPT-2) surprisal and
notes (height, pitch) by melodic surprisal. A line joins each word to its paired note (nearest note onset within 150 ms);
the circle is filled when that note's melodic surprisal is above the song's median (unexpected note)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import FancyBboxPatch, Rectangle

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1]))
import style as S  # noqa: E402

MAP_LEX = LinearSegmentedColormap.from_list("lex_seq", ["#E3F1F8", "#BDE7EF", "#76C4DD", "#3B95CA", "#2365A8", "#1D3C79"])
MAP_MELODY = LinearSegmentedColormap.from_list("melody_seq", ["#FBF6EC", "#F1DFBC", "#E6BD77", "#D0953C", "#AB691C", "#7A4217", "#48220F"])


def canvas(fig, x, y_top, w, h):
    """Axes whose data units are inches, y increasing downwards."""
    a = S.ax(fig, x, y_top, w, h)
    a.set_xlim(0, w)
    a.set_ylim(h, 0)
    a.axis("off")
    return a


def colour_scale(fig, x, y_top, w, h, cmap, norm, ticks, label):
    c = S.ax(fig, x, y_top, w, h)
    g = np.linspace(norm.vmin, norm.vmax, 256)
    c.imshow(cmap(norm(g))[None], aspect="auto", extent=(norm.vmin, norm.vmax, 0, 1))
    for s in c.spines.values():
        s.set_visible(False)
    c.set_yticks([])
    c.set_xticks(ticks, [S.num(t, "g") for t in ticks])
    c.tick_params(axis="x", length=2.0, pad=1.5, colors=S.INK2)
    c.set_xlabel(label, color=S.INK2, labelpad=1.5)
    return c


def main():
    words = pd.read_csv(HERE / "words.csv")
    notes = pd.read_csv(HERE / "notes.csv")
    med = json.loads((HERE / "song_median.json").read_text())["median_melodic_surprisal_bits_of_paired_notes"]
    t0, t1 = 18.35, 26.40
    X0, X1 = 0.92, 4.30
    tx = lambda t: X0 + (np.asarray(t) - t0) / (t1 - t0) * (X1 - X0)
    fig = S.figure(4.80, 2.62)
    c = canvas(fig, 0.0, 0.02, 4.45, 2.05)
    nw, nn = Normalize(0, 20), Normalize(0, 12)
    WB0, WB1 = 0.34, 0.54
    py = lambda p: 1.56 - (p - 48) / 11 * 0.62
    for _, r in words.iterrows():
        a_, b_ = tx(r.onset), tx(r.offset)
        c.add_patch(FancyBboxPatch((a_ + 0.008, WB0), b_ - a_ - 0.016, WB1 - WB0, boxstyle="round,pad=0,rounding_size=0.03",
                                   fc=MAP_LEX(nw(r.lexical_surprisal_bits)), ec="none", lw=0.6, zorder=1))
        c.text((a_ + b_) / 2, WB0 - 0.07, r.word, ha="center", va="baseline", color=S.INK)
        if not np.isfinite(r.paired_note_onset):
            continue
        un = r.paired_note_melodic_surprisal_bits > med
        xn = tx(r.paired_note_onset) + 0.02
        c.plot([xn, xn], [WB1 + 0.03, py(r.paired_note_pitch) - 0.045], color=S.SLATE.mid, lw=0.6, zorder=1)
        c.plot(xn, WB1 + 0.07, "o", ms=4.4, mfc=S.MELODY.dark if un else S.PAPER, mec=S.MELODY.dark, mew=0.9, zorder=3)
    for _, r in notes.iterrows():
        a_, b_ = tx(r.onset), tx(min(r.offset, t1))
        c.add_patch(Rectangle((a_ + 0.004, py(r.pitch) - 0.035), max(b_ - a_ - 0.008, 0.014), 0.07,
                              color=MAP_MELODY(nn(r.melodic_surprisal_bits)), lw=0, zorder=2))
    c.text(X0 - 0.12, (WB0 + WB1) / 2, "words", ha="right", va="center", fontweight="bold", color=S.WORDS.dark)
    c.text(X0 - 0.12, 1.30, "notes", ha="right", va="center", fontweight="bold", color=S.MELODY.dark)
    c.plot([tx(t1 - 2), tx(t1)], [1.80, 1.80], color=S.INK, lw=1.0, solid_capstyle="butt")
    c.text((tx(t1 - 2) + tx(t1)) / 2, 1.94, "2 s", ha="center", va="baseline", color=S.INK2)
    colour_scale(fig, 0.92, 2.22, 1.0, 0.07, MAP_LEX, nw, [0, 20], "word surprisal (bits)")
    cb = colour_scale(fig, 2.22, 2.22, 1.0, 0.07, MAP_MELODY, nn, [0, 12], "note surprisal (bits)")
    cb.plot([med, med], [-0.5, 1.5], color=S.INK, lw=0.9, clip_on=False)
    k = canvas(fig, 3.44, 2.12, 1.3, 0.44)
    for j, (fill, lab) in enumerate(((True, "unexpected note"), (False, "expected note"))):
        k.plot(0.06, 0.10 + j * 0.19, "o", ms=4.4, mfc=S.MELODY.dark if fill else S.PAPER, mec=S.MELODY.dark, mew=0.9)
        k.text(0.17, 0.10 + j * 0.19, lab, ha="left", va="center", color=S.INK2)
    S.save(fig, HERE / "panel")


if __name__ == "__main__":
    main()
