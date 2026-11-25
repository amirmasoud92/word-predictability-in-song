"""Figure 2b. Word-predictability gain of the 48 Alice listeners (dots; bar, mean and 95% CI; above, d_z) and the
group-mean gain at each of the 60 electrodes (nose up; zero neutral, gains blue). Stabilised permutation null.

Data: listeners.csv (delta r per listener), sensor_map.csv (2-D electrode positions and group-mean delta r per
electrode), summary.json (d_z).
"""
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1]))
import style as S  # noqa: E402

al = pd.read_csv(HERE / "listeners.csv").delta_r.to_numpy() * 1e3
m = pd.read_csv(HERE / "sensor_map.csv")
st = json.loads((HERE / "summary.json").read_text(encoding="utf-8"))
fig = S.figure(2.25, 3.15)

a = S.ax(fig, 0.42, 0.55, 0.72, 2.10)
S.zero(a)
a.set_xlim(-0.62, 0.72)
a.set_ylim(-2.0, 3.3)
S.dots(a, -0.08, al, S.WORDS.base, size=3.4, max_pt=15)
S.crossbar(a, 0.52, al, S.WORDS.dark, w=0.20)
S.above(a, 0.05, S.dz_txt(st["dz"]), fontweight="bold")
S.name_ticks(a, [0.05], ["audiobook"], [S.WORDS.dark])
a.set_ylabel(S.E3)

d = 0.92
b = S.ax(fig, 1.20, 0.87, d, d * 1.02)
S.gain_map(b, m[["x", "y"]].to_numpy(), m.delta_r.to_numpy() * 1e3, 1.2, sensor_size=1.0)
S.gain_bar(fig, 1.34, 0.87 + d * 1.02 + 0.10, d - 0.28, 0.07, 0, 1.2, [0, 0.6, 1.2], label=S.E3)

S.save(fig, HERE / "panel")
