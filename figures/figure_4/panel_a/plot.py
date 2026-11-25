"""Figure 4a: the first 44 s of the story LW1 with its first word list, and the word lists of the four stories to scale.

Each tick is a word; its height is the mean surprisal of the word's type over its sentence occurrences in the story
(GPT-2, bits). Blue: the list words and the last earlier sentence occurrence of the same type, joined by an arc.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path as MPath

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1]))
import style as S  # noqa: E402

STORIES = ["lw1", "cable_spool_fort", "easy_money", "The_Black_Willow"]


def canvas(fig, x, y_top, w, h):
    """Axes whose data units are inches, y increasing downwards."""
    a = S.ax(fig, x, y_top, w, h)
    a.set_xlim(0, w)
    a.set_ylim(h, 0)
    a.axis("off")
    return a


def scalebar(c, x0, x1, y, label):
    c.plot([x0, x1], [y, y], color=S.INK, lw=1.0, solid_capstyle="butt")
    c.text((x0 + x1) / 2, y + 0.14, label, ha="center", va="baseline", color=S.INK2)


def main():
    ex = pd.read_csv(HERE / "story_excerpt_words.csv").sort_values("order")
    lists = pd.read_csv(HERE / "word_lists_in_stories.csv")
    fig = S.figure(7.2, 1.80)
    c = canvas(fig, 0, 0.05, 7.2, 1.70)
    T1, X0, X1 = 44.0, 0.34, 7.02
    tx = lambda t: X0 + np.asarray(t) / T1 * (X1 - X0)
    lst = ex[ex.condition == "word_list"]
    sen = ex[ex.condition != "word_list"]
    yb, hs = 0.92, 0.30 / 20                                             # baseline; inches per bit
    l0, l1 = lst.start.min() - 0.35, (lst.start + lst.duration).max() + 0.35
    c.add_patch(Rectangle((tx(l0), yb - 0.40), tx(l1) - tx(l0), 0.40, color=S.BAND, lw=0, zorder=0))
    c.plot([X0, X1], [yb, yb], color=S.RULE, lw=0.5, zorder=1)
    linked = {}
    for _, r in lst.iterrows():
        prev = sen[(sen.type_id == r.type_id) & (sen.start < r.start)]
        if len(prev) and np.isfinite(r.type_mean_surprisal_bits):
            linked[prev.index[-1]] = r
    for i, r in ex.iterrows():
        s = r.type_mean_surprisal_bits
        if not np.isfinite(s):
            continue
        hot = r.condition == "word_list" or i in linked
        c.plot([tx(r.start)] * 2, [yb, yb - hs * s], color=S.WORDS.base if hot else S.SLATE.mid, lw=1.2 if hot else 0.7,
               solid_capstyle="butt", zorder=3 if hot else 2)
    for i, r in linked.items():
        a_, b_ = tx(ex.loc[i].start), tx(r.start)
        top = yb - 0.44 - 0.36 * (b_ - a_) / (X1 - X0)
        c.add_patch(PathPatch(MPath([(a_, yb - hs * ex.loc[i].type_mean_surprisal_bits - 0.03), (a_, top), (b_, top), (b_, yb - 0.41)],
                                    [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4]), fc="none", ec=S.WORDS.mid, lw=0.6, zorder=1))
        if len(ex.loc[i].word) > 3:
            c.text(a_, yb + 0.15, ex.loc[i].word, ha="center", va="baseline", color=S.WORDS.dark)
    c.text((tx(l0) + tx(l1)) / 2, yb + 0.15, "word list", ha="center", va="baseline", color=S.WORDS.dark, fontweight="bold")
    scalebar(c, tx(0), tx(5), yb + 0.44, "5 s")
    # the word lists in the four stories of a session, to scale
    B0, B1, yy = 2.90, 7.02, 1.42
    mins = lists.groupby("story").story_minutes.first()
    scale = (B1 - B0 - 0.10 * (len(STORIES) - 1)) / sum(mins[s] for s in STORIES)
    x = B0
    for story in STORIES:
        L = mins[story] * scale
        c.add_patch(Rectangle((x, yy - 0.03), L, 0.06, color=S.SLATE.light, lw=0, zorder=1))
        for _, r in lists[lists.story == story].iterrows():
            c.add_patch(Rectangle((x + r.list_start_fraction * L, yy - 0.055), max((r.list_end_fraction - r.list_start_fraction) * L, 0.012),
                                  0.11, color=S.WORDS.base, lw=0, zorder=2))
        x += L + 0.10
    c.text(B0 - 0.10, yy + 0.03, f"four stories, {len(lists)} word lists", ha="right", va="baseline", color=S.INK2)
    S.save(fig, HERE / "panel")


if __name__ == "__main__":
    main()
