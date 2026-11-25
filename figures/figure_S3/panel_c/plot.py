"""Figure S3c: group-mean word-level gain at each of the 64 channels, sung lyrics (nose up; zero neutral, gains blue,
losses tan; the same scale as panel b). Channel positions: azimuthal-equidistant projection of the standard 10-05
positions. Original estimates."""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1]))
import style as S  # noqa: E402


def main():
    d = pd.read_csv(HERE / "channel_map.csv")
    fig = S.figure(1.50, 1.95)
    m = S.ax(fig, 0.10, 0.05, 1.30, 1.33)
    S.gain_map(m, d[["x", "y"]].to_numpy(), d.delta_r.to_numpy() * 1e3, 0.9, sensor_size=1.2)
    S.gain_bar(fig, 0.20, 1.57, 1.10, 0.07, -0.3, 0.9, [-0.3, 0, 0.9], label=S.E3)
    S.save(fig, HERE / "panel")


if __name__ == "__main__":
    main()
