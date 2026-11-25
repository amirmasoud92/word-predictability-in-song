"""Figure S2d: sung word-level gain per listener in the primary test (original estimate) with the released and with the
corrected annotation timing (dots, listeners, joined by lines; bars, mean and 95% CI)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1]))
import style as S  # noqa: E402


def main():
    both = pd.read_csv(HERE / "listener_gains.csv")[["delta_r_released", "delta_r_corrected"]] * 1e3
    fig = S.figure(1.95, 2.15)
    d = S.ax(fig, 0.62, 0.12, 1.25, 1.55)
    S.zero(d)
    d.set_xlim(-0.6, 1.62)
    d.set_ylim(-0.35, 0.9)
    d.set_yticks([-0.25, 0, 0.25, 0.5, 0.75], ["−0.25", "0", "0.25", "0.5", "0.75"])
    for u, v in zip(both.delta_r_released, both.delta_r_corrected):
        d.plot([-0.10, 0.90], [u, v], color=S.SLATE.light, lw=0.6, zorder=1)
    for k, col in enumerate(("delta_r_released", "delta_r_corrected")):
        d.plot(np.full(len(both), k - 0.10), both[col], "o", ms=3.8, mfc=S.SUNG.base, mec=S.PAPER, mew=0.4, ls="none", zorder=3)
        S.crossbar(d, k + 0.36, both[col], S.SUNG.dark, w=0.18)
    S.name_ticks(d, [0.1, 1.1], ["released", "corrected"], [S.INK, S.INK], bold=False)
    d.set_ylabel(f"sung {S.E3}")
    S.save(fig, HERE / "panel")


if __name__ == "__main__":
    main()
