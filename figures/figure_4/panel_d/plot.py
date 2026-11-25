"""Figure 4d: the standard word-level measure (surprisal of all sentence words) in the two sessions. Left, the group-mean
gain at each of the 208 gradiometers in session 1 (27 listeners) and session 2 (22 listeners), on one scale (nose up).
Right, the 22 listeners with two sessions, session 1 against session 2, with the line of equality and Spearman's rho."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1]))
import style as S  # noqa: E402


def project(pos, radius=0.94):
    """Azimuthal-equidistant projection of sensor positions (x right, y anterior, z up) about the centre of the sphere
    fitted to them, scaled so that the outermost sensor lies at ``radius`` of the head circle."""
    p = np.asarray(pos, float)
    A = np.c_[2 * p, np.ones(len(p))]
    c = np.linalg.lstsq(A, (p ** 2).sum(1), rcond=None)[0][:3]
    v = p - c
    th = np.arccos(np.clip(v[:, 2] / np.linalg.norm(v, axis=1), -1, 1))
    ph = np.arctan2(v[:, 1], v[:, 0])
    xy = np.c_[th * np.cos(ph), th * np.sin(ph)]
    return xy / np.linalg.norm(xy, axis=1).max() * radius


def main():
    maps = pd.read_csv(HERE / "sensor_maps.csv")
    ses = pd.read_csv(HERE / "listener_sessions.csv")
    rho = json.loads((HERE / "session_correlation.json").read_text())
    xy = project(maps[["x", "y", "z"]].to_numpy())
    fig = S.figure(3.96, 1.75)
    d = 0.98
    for k, col in enumerate(("delta_r_session1", "delta_r_session2")):
        a = S.ax(fig, 0.12 + k * (d + 0.06), 0.22, d, d * 1.02)
        S.gain_map(a, xy, maps[col].to_numpy() * 1e3, 1.0, sensor_size=0.7, hull=True)
        S.text(fig, 0.12 + k * (d + 0.06) + d / 2, 0.16, f"session {k + 1}", ha="center", color=S.INK)
    S.gain_bar(fig, 0.12 + d + 0.03 - 0.50, 0.22 + d * 1.02 + 0.12, 1.0, 0.07, 0, 1.0, [0, 0.5, 1], label=S.E3)
    a = S.ax(fig, 2.74, 0.10, 1.12, 1.12)
    lim = (-0.2, 1.15)
    a.plot(lim, lim, color=S.SLATE.mid, lw=0.6, zorder=0)
    a.scatter(ses.delta_r_session1 * 1e3, ses.delta_r_session2 * 1e3, s=16, facecolor=S.WORDS.base, edgecolors=S.PAPER, linewidths=0.4, zorder=3)
    a.set_xlim(*lim)
    a.set_ylim(*lim)
    a.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
    a.set_yticks([0, 0.5, 1], ["0", "0.5", "1"])
    a.set_xlabel(f"session 1, {S.E3}")
    a.set_ylabel(f"session 2, {S.E3}")
    a.text(0.97, 0.06, f"ρ = {S.num(rho['spearman_rho'])}", transform=a.transAxes, ha="right", va="bottom", color=S.INK)
    S.save(fig, HERE / "panel")


if __name__ == "__main__":
    main()
