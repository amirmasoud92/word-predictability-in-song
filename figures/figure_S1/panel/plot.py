"""Figure S1: the primary tests and their secondary estimates. Effect sizes (d_z, or Spearman's rho where marked) with 95%
CIs, grouped by level; arrows mark intervals extending beyond the axis; filled, the interval excludes zero; shaded,
|d_z| < 0.5. Right, the unit and its number. Original estimates."""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1]))
import style as S  # noqa: E402

COLOUR = {"words": S.WORDS, "sung": S.SUNG, "melody": S.MELODY}


def main():
    rows = pd.read_csv(HERE / "primary_tests.csv", keep_default_na=False, na_values=[""])
    XMIN, XMAX, rh = -1.5, 3.5, 0.178
    n = len(rows)
    fig = S.figure(7.2, rh * n + 0.85)
    a = S.ax(fig, 3.30, 0.20, 2.75, rh * n)
    S.band(a, horizontal=True)
    S.zero(a, horizontal=True)
    tr = a.get_yaxis_transform()
    xl = (0.12 - 3.30) / 2.75
    for i, r in rows.iterrows():
        ramp = COLOUR[r.colour]
        if r.kind == "group":
            a.text(xl, i, r.label, transform=tr, ha="left", va="center", color=ramp.dark, fontweight="bold")
            continue
        lo, hi = r.ci_low, r.ci_high
        col = ramp.base
        clo, chi = max(lo, XMIN), min(hi, XMAX)
        a.plot([clo, chi], [i, i], color=col, lw=1.2, solid_capstyle="butt", zorder=2)
        for edge, clipped, mk in ((clo, lo < XMIN, "<"), (chi, hi > XMAX, ">")):
            if clipped:
                a.plot(edge, i, marker=mk, ms=4.5, color=col, mew=0, zorder=3)
        sig = lo > 0 or hi < 0
        a.plot(r.dz, i, marker="o", ms=5.2, mfc=col if sig else S.PAPER, mec=col, mew=1.1, zorder=4)
        a.text(xl + 0.06, i, r.label, transform=tr, ha="left", va="center", color=S.INK)
        a.text(1.03, i, r.unit, transform=tr, ha="left", va="center", color=S.INK2)
    a.set_xlim(XMIN, XMAX)
    a.set_ylim(n - 0.4, -0.6)
    a.set_yticks([])
    a.spines["left"].set_visible(False)
    a.set_xticks([-1, 0, 1, 2, 3], ["−1", "0", "1", "2", "3"])
    a.set_xlabel(f"{S.DZ} (or ρ)")
    S.save(fig, HERE / "panel")


if __name__ == "__main__":
    main()
