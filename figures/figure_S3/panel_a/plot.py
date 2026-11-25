"""Figure S3a: word-level gain with the surprisal regressor restricted to one of three lag windows (mean and 95% CI over
listeners), for spoken, sung and hummed versions. Original estimates."""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1]))
import style as S  # noqa: E402

VERSIONS = ("speech", "song", "melody")
WINDOWS = ("delta_r_0_188ms", "delta_r_219_406ms", "delta_r_438_688ms")


def main():
    L = pd.read_csv(HERE / "listener_gains.csv")
    fig = S.figure(3.70, 2.35)
    a = S.ax(fig, 0.72, 0.15, 2.85, 1.65)
    S.zero(a)
    for k, v in enumerate(VERSIONS):
        ramp, nm = S.VERSION[v]
        for j, w in enumerate(WINDOWS):
            m, h = S.mean_ci(L[L.version == v][w] * 1e3)
            xx = j + (k - 1) * 0.22
            a.plot([xx, xx], [m - h, m + h], color=ramp.base, lw=1.2, solid_capstyle="butt")
            a.plot(xx, m, "o", ms=5.2, mfc=ramp.base, mec=S.PAPER, mew=0.5, zorder=3)
        a.text(0.03 + 0.20 * k, 0.07, nm, transform=a.transAxes, ha="left", va="center", color=ramp.dark, fontweight="bold")
    a.set_xticks(range(3), ["0–188", "219–406", "438–688"])
    a.tick_params(axis="x", length=0)
    a.spines["bottom"].set_visible(False)
    a.set_xlim(-0.5, 2.5)
    a.set_ylim(-0.16, 0.24)
    a.set_yticks([-0.1, 0, 0.1, 0.2], ["−0.1", "0", "0.1", "0.2"])
    a.set_xlabel("lags of the surprisal regressor (ms)")
    a.set_ylabel(S.E3)
    S.save(fig, HERE / "panel")


if __name__ == "__main__":
    main()
