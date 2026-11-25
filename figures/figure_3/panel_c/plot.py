"""Figure 3c. The primary contrast, attended - ignored, and its sensitivity analyses in three groups (text, talker,
recording): d_z with 95% CI; filled, the CI excludes zero; vertical line, the primary estimate.

Data: contrasts.csv (group, analysis, d_z, 95% CI).
"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1]))
import style as S  # noqa: E402

T = pd.read_csv(HERE / "contrasts.csv", keep_default_na=False)
x_lab, x, y_top, w, rh = 0.34, 2.02, 0.10, 1.20, 0.215
ys, y = [], 0.0
for k, grp in enumerate(T.group):
    y += 1.9 if grp else (1.0 if k else 0.0)
    ys.append(y)
fig = S.figure(3.30, 2.70)

a = S.ax(fig, x, y_top, w, rh * (ys[-1] + 1.2))
S.zero(a, horizontal=True)
a.axvline(T.dz.iloc[0], color=S.WORDS.light, lw=1.0, zorder=0)
ty = a.get_yaxis_transform()
for i, (r, yy) in enumerate(zip(T.itertuples(), ys)):
    S.effect(a, yy, r.dz, (r.ci_low, r.ci_high), S.WORDS.dark if i == 0 else S.WORDS.base, ms=6.0 if i == 0 else 5.2)
    a.text(-0.06, yy, r.analysis, transform=ty, ha="right", va="center", color=S.INK, fontweight="bold" if i == 0 else "normal")
    if r.group:
        a.text((x_lab - x) / w, yy - 0.95, r.group, transform=ty, ha="left", va="center", color=S.INK3)
a.set_ylim(ys[-1] + 0.6, -0.6)
a.set_xlim(-0.1, 1.2)
a.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
a.spines["left"].set_visible(False)
a.tick_params(left=False, labelleft=False)
a.set_xlabel(f"attended − ignored, {S.DZ}")

S.save(fig, HERE / "panel")
