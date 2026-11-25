"""Figure 5b: prediction gain from melodic surprisal per listener, sung and hummed (dots; bars, mean and 95% CI), and the
mean and 95% CI after note-level controls for interval size, pitch height and note duration (diamonds). Above, d_z without
the controls. Stabilised permutation null."""
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1]))
import style as S  # noqa: E402


def main():
    d = pd.read_csv(HERE / "listener_gains.csv")
    dz = json.loads((HERE / "effect_sizes.json").read_text())
    fig = S.figure(2.15, 2.75)
    a = S.ax(fig, 0.55, 0.35, 1.55, 1.90)
    S.zero(a)
    sets = [("song", "sung", S.SUNG, dz["sung_dz"]), ("melody", "hummed", S.MELODY, dz["hummed_dz"])]
    a.set_xlim(-0.55, 1.75)
    a.set_ylim(-1.0, 1.55)
    for k, (v, nm, ramp, d_z) in enumerate(sets):
        base = d[d.version == v].delta_r.to_numpy() * 1e3
        cont = d[d.version == v].delta_r_note_controls.to_numpy() * 1e3
        S.dots(a, k - 0.12, base, ramp.base, size=3.5, max_pt=11)
        S.crossbar(a, k + 0.30, base, ramp.dark, w=0.18)
        m, hh = S.mean_ci(cont)
        a.plot([k + 0.55, k + 0.55], [m - hh, m + hh], color=ramp.dark, lw=1.0, solid_capstyle="butt", zorder=5)
        a.plot(k + 0.55, m, marker="D", ms=4.6, mfc=S.PAPER, mec=ramp.dark, mew=1.1, zorder=6)
        S.above(a, k + 0.15, S.dz_txt(d_z), fontweight="bold")
    S.name_ticks(a, [0.15, 1.15], [s_[1] for s_ in sets], [s_[2].dark for s_ in sets])
    a.set_yticks([-1, -0.5, 0, 0.5, 1, 1.5], ["−1", "−0.5", "0", "0.5", "1", "1.5"])
    a.set_ylabel(f"note-predictability gain, {S.E3}")
    S.save(fig, HERE / "panel")


if __name__ == "__main__":
    main()
