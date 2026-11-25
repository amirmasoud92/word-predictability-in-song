"""Figure 2a. Word-predictability gain of the same 20 CantiSong listeners for the spoken, sung and hummed versions
(dots, listeners; bars, mean and 95% CI; above, d_z), and each listener's paired differences, sung - spoken and
sung - hummed, with d_z and the sign-flip p. Stabilised permutation null.

Data: listeners.csv (delta r per listener and version), summary.json (d_z and p of the tests).
"""
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1]))
import style as S  # noqa: E402

L = pd.read_csv(HERE / "listeners.csv", index_col="listener") * 1e3
st = json.loads((HERE / "summary.json").read_text(encoding="utf-8"))
ticks = ([-1, -0.5, 0, 0.5, 1], ["−1", "−0.5", "0", "0.5", "1"])
fig = S.figure(4.85, 3.15)

# the three versions
a = S.ax(fig, 0.70, 0.55, 2.30, 2.10)
S.zero(a)
a.set_xlim(-0.55, 2.62)
a.set_ylim(-1.35, 1.25)
order = [("spoken", "speech"), ("sung", "song"), ("hummed", "melody")]
for k, (col, v) in enumerate(order):
    ramp, _ = S.VERSION[v]
    vals = L[col].to_numpy()
    S.dots(a, k - 0.08, vals, ramp.base, size=3.6, max_pt=13)
    S.crossbar(a, k + 0.33, vals, ramp.dark)
    S.above(a, k + 0.05, S.dz_txt(st["dz"][col]), fontweight="bold" if v != "melody" else "normal",
            color=S.INK if v != "melody" else S.INK2)
S.name_ticks(a, [0.05, 1.05, 2.05], [c for c, _ in order], [S.VERSION[v][0].dark for _, v in order])
a.set_yticks(*ticks)
a.set_ylabel(f"word-predictability gain, {S.E3}")

# paired differences of the same listeners
b = S.ax(fig, 3.52, 0.55, 1.20, 2.10)
S.zero(b)
b.set_xlim(-0.55, 1.62)
b.set_ylim(-1.35, 1.25)
for k, (col, key) in enumerate((("spoken", "sung_minus_spoken"), ("hummed", "sung_minus_hummed"))):
    d = (L["sung"] - L[col]).to_numpy()
    S.dots(b, k - 0.08, d, S.SLATE.mid, size=3.6, max_pt=13)
    S.crossbar(b, k + 0.33, d, S.INK)
    S.above(b, k + 0.05, S.dz_txt(st[key]["dz"]), dy=0.075, color=S.INK2)
    S.above(b, k + 0.05, S.p_txt(st[key]["p"]), color=S.INK2)
S.name_ticks(b, [0.05, 1.05], ["sung −\nspoken", "sung −\nhummed"], [S.INK, S.INK], bold=False)
b.set_yticks(*ticks)
b.set_ylabel(f"paired difference, {S.E3}")

S.save(fig, HERE / "panel")
