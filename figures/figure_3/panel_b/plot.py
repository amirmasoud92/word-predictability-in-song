"""Figure 3b. Left, the group-mean word-predictability gain of the attended and the ignored stream at each of the 248
magnetometers (58 listeners; nose up; one scale, zero neutral, gains blue, losses tan; drawn only within the sensor
array). Right, the gain of each listener for each stream (dots; bars, mean and 95% CI; above, d_z against zero); the
bracket gives the primary contrast, attended - ignored, with its d_z and its p value Holm-corrected over the 13
primary tests.

Data: sensor_maps.csv (2-D sensor positions and group-mean delta r per stream), listeners.csv (delta r per listener
and stream), summary.json (d_z and the corrected p).
"""
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1]))
import style as S  # noqa: E402

M = pd.read_csv(HERE / "sensor_maps.csv")
L = pd.read_csv(HERE / "listeners.csv")
st = json.loads((HERE / "summary.json").read_text(encoding="utf-8"))
fig = S.figure(4.30, 2.55)

# sensor maps
x, y_top, d, vmax = 0.08, 0.50, 1.0, 4.0
xy = M[["x", "y"]].to_numpy()
for k, (nm, ramp) in enumerate((("attended", S.WORDS), ("ignored", S.SLATE))):
    a = S.ax(fig, x + k * (d + 0.06), y_top, d, d * 1.02)
    S.gain_map(a, xy, M[nm].to_numpy() * 1e4, vmax, sensor_size=0.6, hull=True)
    S.text(fig, x + k * (d + 0.06) + d / 2, y_top - 0.06, nm, ha="center", color=ramp.dark, fontweight="bold")
S.gain_bar(fig, x + d - 0.56, y_top + d * 1.02 + 0.12, 1.24, 0.07, -1, 4, [-1, 0, 2, 4], label=S.E4)

# listeners
att, ign = L.attended.to_numpy() * 1e4, L.ignored.to_numpy() * 1e4
a = S.ax(fig, 2.70, 0.56, 1.46, 1.66)
S.zero(a)
a.set_xlim(-0.6, 1.62)
a.set_ylim(-3.3, 6.3)
for k, (v, ramp, key) in enumerate(((att, S.WORDS, "attended"), (ign, S.SLATE, "ignored"))):
    S.dots(a, k - 0.10, v, ramp.base if k == 0 else S.SLATE.mid, size=3.2, max_pt=13)
    S.crossbar(a, k + 0.40, v, ramp.dark, w=0.20)
    S.above(a, k + 0.05, S.dz_txt(st["dz"][key]), color=S.INK2)
tp = a.get_xaxis_transform()
a.plot([0.05, 0.05, 1.05, 1.05], [1.115, 1.14, 1.14, 1.115], transform=tp, color=S.INK, lw=0.7, clip_on=False)
c = st["attended_minus_ignored"]
a.text(0.55, 1.17, f"{S.dz_txt(c['dz'])},  {S.p_txt(c['p_holm_13_primary_tests'], 'Holm')}", transform=tp, ha="center",
       va="baseline", color=S.INK, fontweight="bold")
S.name_ticks(a, [0.05, 1.05], ["attended", "ignored"], [S.WORDS.dark, S.SLATE.dark])
a.set_yticks([-2, 0, 2, 4, 6], ["−2", "0", "2", "4", "6"])
a.set_ylabel(f"word-predictability gain, {S.E4}")

S.save(fig, HERE / "panel")
