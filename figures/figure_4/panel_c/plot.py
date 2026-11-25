"""Figure 4c: each kind of word against zero, and the primary statistic (sentences − lists for the matched words) with its
sensitivity analyses. d_z with noncentral-t 95% CIs; filled, the CI excludes zero; shaded, |d_z| < 0.5."""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1]))
import style as S  # noqa: E402


def forest(a, rows, note_col=None, note_fmt=None, xlim=(-1.0, 2.4)):
    """Effect sizes on a shared d_z axis; header rows name a group of estimates."""
    S.band(a, horizontal=True)
    S.zero(a, horizontal=True)
    tf = a.get_yaxis_transform()
    for k, r in rows.iterrows():
        if r.kind == "head":
            a.text(-0.04, k, r.label, transform=tf, ha="right", va="center", color=S.INK3)
            continue
        bold = bool(r.primary)
        S.effect(a, k, r.dz, (r.ci_low, r.ci_high), S.WORDS.dark if bold else S.WORDS.base, ms=6.0 if bold else 5.2)
        a.text(-0.04, k, r.label, transform=tf, ha="right", va="center", color=S.INK, fontweight="bold" if bold else "normal")
        if bold and note_col:
            a.text(r.ci_high + 0.12, k, note_fmt(r[note_col]), ha="left", va="center", color=S.INK2, linespacing=1.25)
    a.set_ylim(len(rows) - 0.4, -0.6)
    a.set_xlim(*xlim)
    a.set_xticks([-1, -0.5, 0, 0.5, 1, 2], ["−1", "", "0", "", "1", "2"])
    a.spines["left"].set_visible(False)
    a.tick_params(left=False, labelleft=False)
    a.set_yticks([])
    a.set_xlabel(S.DZ)


def main():
    rows = pd.read_csv(HERE / "effect_sizes.csv")
    fig = S.figure(4.47, 2.30)
    a = S.ax(fig, 1.99, 0.15, 2.33, 0.195 * len(rows))
    forest(a, rows, "p", S.p_txt)
    S.save(fig, HERE / "panel")


if __name__ == "__main__":
    main()
