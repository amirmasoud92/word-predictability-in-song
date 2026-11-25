"""Figure S5b: phoneme surprisal beyond word surprisal (null within phone position class). Per-listener gains (dots; bars, mean and 95% CI) for spoken, sung and hummed versions (hummed EEG with the sung timing); above, d_z."""
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1]))
import style as S  # noqa: E402

CONDITIONS = ("spoken", "sung", "hummed")
VERSION_OF = {"spoken": "speech", "sung": "song", "hummed": "melody"}


def main():
    d = pd.read_csv(HERE / "listener_gains.csv")
    dz = json.loads((HERE / "effect_sizes.json").read_text())
    fig = S.figure(2.45, 2.95)
    a = S.ax(fig, 0.40, 0.40, 1.95, 2.10)
    S.zero(a, ls=(0, (3, 2)))
    a.set_xlim(-0.55, len(CONDITIONS) - 0.38)
    a.set_ylim(-1.35, 1.25)
    for k, c in enumerate(CONDITIONS):
        ramp = S.VERSION[VERSION_OF[c]][0]
        vals = d[c].dropna().to_numpy() * 1e3
        S.dots(a, k - 0.08, vals, ramp.base, size=3.6, max_pt=13)
        S.crossbar(a, k + 0.33, vals, ramp.dark)
        S.above(a, k + 0.05, S.dz_txt(dz[c]["dz"]), fontweight="normal", color=S.INK if c != "hummed" else S.INK2)
    S.name_ticks(a, [k + 0.05 for k in range(len(CONDITIONS))], list(CONDITIONS),
                 [S.VERSION[VERSION_OF[c]][0].dark for c in CONDITIONS])
    a.set_yticks([-1, -0.5, 0, 0.5, 1], ["−1", "−0.5", "0", "0.5", "1"])
    S.save(fig, HERE / "panel")


if __name__ == "__main__":
    main()
