"""Figure S4d: word-level gain by layer of GPT-2 word representations (word-locked, 16 principal components per layer;
layer 0 is the non-contextual input embedding), mean and 95% CI over 20 listeners, for spoken, sung and hummed versions.
Tested against 5 within-song permutations of the word vectors."""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1]))
import style as S  # noqa: E402

VERSIONS = ("speech", "song", "melody")
LAYERS = (0, 4, 8, 12)


def main():
    ml = pd.read_csv(HERE / "listener_gains.csv")
    fig = S.figure(2.45, 2.10)
    a = S.ax(fig, 0.62, 0.15, 1.72, 1.55)
    S.zero(a)
    for i, v in enumerate(VERSIONS):
        ramp, nm = S.VERSION[v]
        d = ml[ml.version == v]
        xs = [k + (i - 1) * 0.2 for k in range(len(LAYERS))]
        mh = [S.mean_ci(d[f"layer_{L}"] * 1e3) for L in LAYERS]
        a.plot(xs, [m for m, _ in mh], color=ramp.mid, lw=0.9, zorder=1)
        for xx, (m, h) in zip(xs, mh):
            a.plot([xx, xx], [m - h, m + h], color=ramp.base, lw=1.2, solid_capstyle="butt", zorder=2)
            a.plot(xx, m, "o", ms=5.0, mfc=ramp.base, mec=S.PAPER, mew=0.5, zorder=3)
        a.text(0.98, 0.95 - 0.11 * i, nm, transform=a.transAxes, ha="right", va="center", color=ramp.dark, fontweight="bold")
    a.set_xticks(range(len(LAYERS)), [str(L) for L in LAYERS])
    a.tick_params(axis="x", length=0)
    a.spines["bottom"].set_visible(False)
    a.set_xlim(-0.5, len(LAYERS) - 0.5)
    a.set_xlabel("layer")
    a.set_ylabel(S.E3)
    S.save(fig, HERE / "panel")


if __name__ == "__main__":
    main()
