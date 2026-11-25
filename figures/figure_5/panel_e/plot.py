"""Figure 5e: interference estimates (expected − unexpected, d_z) for words and notes: thick bars, 90% CI (the criterion
of the equivalence test); thin bars, 95% CI; open, the CI includes zero; shaded, the ±0.5 equivalence bounds; above, the
two-sided Bayes factor BF01."""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1]))
import style as S  # noqa: E402


def main():
    d = pd.read_csv(HERE / "interference_estimates.csv").set_index("stream")
    fig = S.figure(2.20, 2.40)
    ex = S.ax(fig, 0.70, 0.30, 1.40, 1.62)
    S.band(ex)
    S.zero(ex)
    rows = [("words", S.WORDS), ("notes", S.MELODY)]
    for k, (nm, ramp) in enumerate(rows):
        r = d.loc[nm]
        S.effect(ex, k, r.dz, (r.ci95_low, r.ci95_high), ramp.dark, horizontal=False, ci_inner=(r.ci90_low, r.ci90_high), lw=1.0, ms=6.0)
        S.above(ex, k, S.bf(r.bf01), color=S.INK2)
    S.name_ticks(ex, [0, 1], [nm for nm, _ in rows], [ramp.dark for _, ramp in rows])
    ex.set_xlim(-0.6, 1.6)
    ex.set_ylim(-0.85, 0.85)
    ex.set_yticks([-0.5, 0, 0.5], ["−0.5", "0", "0.5"])
    ex.set_ylabel(f"expected − unexpected, {S.DZ}")
    S.save(fig, HERE / "panel")


if __name__ == "__main__":
    main()
