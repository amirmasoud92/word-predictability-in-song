"""Figure 2c. Mean word-predictability gain of each of the 18 songs, hummed (gold) and sung (red), ordered by the
difference; red lines, songs in which sung exceeded hummed; grey lines, songs in which hummed exceeded sung; inset,
the count, d_z and one-sided sign-flip p over songs. Stabilised permutation null.

Data: songs.csv (mean delta r per song and version, over listeners), summary.json (test over songs).
"""
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1]))
import style as S  # noqa: E402

item = pd.read_csv(HERE / "songs.csv", dtype={"song": str}).set_index("song") * 1e3
st = json.loads((HERE / "summary.json").read_text(encoding="utf-8"))["sung_minus_hummed_over_songs"]
item = item.assign(d=item.sung - item.hummed).sort_values("d", ascending=False)
fig = S.figure(2.85, 2.45)

a = S.ax(fig, 0.30, 0.30, 2.40, 1.70)
S.zero(a, horizontal=True)
for k, (_, r) in enumerate(item.iterrows()):
    up = r.d > 0
    a.plot([r.hummed, r.sung], [k, k], color=S.SUNG.mid if up else S.SLATE.light, lw=1.6, solid_capstyle="butt", zorder=1)
    a.plot(r.hummed, k, "o", ms=4.6, mfc=S.MELODY.base, mec=S.PAPER, mew=0.5, zorder=3)
    a.plot(r.sung, k, "o", ms=4.6, mfc=S.SUNG.base, mec=S.PAPER, mew=0.5, zorder=3)
n_up = st["n_positive"]
a.axhline(n_up - 0.5, color=S.RULE, lw=0.6, zorder=0)
a.set_ylim(len(item) - 0.4, -0.7)
a.set_xlim(-1.2, 0.75)
a.set_xticks([-1, -0.5, 0, 0.5], ["−1", "−0.5", "0", "0.5"])
a.set_yticks([])
a.spines["left"].set_visible(False)
a.set_xlabel(f"song mean {S.E3}")
a.text(0.02, (n_up - 1) / 2 + 1.2, f"sung > hummed\nin {n_up} of {st['n']} songs\n{S.dz_txt(st['dz'])}\n{S.p_txt(st['p'])}",
       transform=a.get_yaxis_transform(), ha="left", va="center", color=S.INK, linespacing=1.3)
top = item.iloc[0]
a.text(top.hummed, -0.62, "hummed", ha="center", va="bottom", color=S.MELODY.dark, fontweight="bold")
a.text(top.sung, -0.62, "sung", ha="center", va="bottom", color=S.SUNG.dark, fontweight="bold")

S.save(fig, HERE / "panel")
