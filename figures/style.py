"""Shared drawing conventions of the figure panels: colours, type, marks and sensor maps.

Colours carry meaning: spoken words and the word-level measure in blue, sung versions in red, melody and hummed versions
in gold, and slate grey for everything else (the ignored talker, estimates whose interval includes zero, reference lines).
A listener, song or sensor is a dot; a mean is a bar with its 95% CI; an effect size is a point with its 95% CI, filled
when the interval excludes zero. Lettering is Arial, 8 pt.
"""
from __future__ import annotations

from collections import namedtuple
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Circle, Polygon
from scipy import stats

Ramp = namedtuple("Ramp", "tint light mid base dark")
WORDS = Ramp("#EBF5FF", "#C9E1FC", "#85A8D0", "#1F5C99", "#003565")
SUNG = Ramp("#FFEFED", "#FBD3CF", "#D78E88", "#B4413F", "#7D2021")
MELODY = Ramp("#FAF2E6", "#EFDBBD", "#D9B777", "#BF8A24", "#8D6204")
SLATE = Ramp("#F3F5F7", "#DDE2E8", "#9AA6B4", "#5B6673", "#373F4A")
INK, INK2, INK3 = "#161B22", "#4A525C", "#69717B"
RULE, BAND, PAPER = "#D3D8DE", "#EEF1F4", "#FFFFFF"
VERSION = {"speech": (WORDS, "spoken"), "song": (SUNG, "sung"), "melody": (MELODY, "hummed")}
DZ = r"$\mathit{d}_\mathrm{z}$"
E3, E4 = "Δr × 10$^{3}$", "Δr × 10$^{4}$"

_POS = ["#F4F4F2", "#BDE7EF", "#76C4DD", "#3B95CA", "#2365A8", "#1D3C79", "#101D4A"]
_NEG = ["#6B4A2B", "#9C7B55", "#C9B08C", "#E8DCC8"]
MAP_POS = LinearSegmentedColormap.from_list("gain_pos", _POS)
MAP_DIV = LinearSegmentedColormap.from_list(
    "gain_div", [(0.5 * k / len(_NEG), c) for k, c in enumerate(_NEG)] + [(0.5 + 0.5 * k / (len(_POS) - 1), c) for k, c in enumerate(_POS)])


def setup():
    t = 8.0
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
        "mathtext.sf": "Arial", "mathtext.default": "regular",
        "font.size": t, "axes.labelsize": t, "axes.titlesize": t, "xtick.labelsize": t, "ytick.labelsize": t, "legend.fontsize": t,
        "axes.linewidth": 0.6, "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.4, "ytick.major.size": 2.4,
        "xtick.major.pad": 2.0, "ytick.major.pad": 2.0, "axes.labelpad": 3.0,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
        "figure.facecolor": PAPER, "axes.facecolor": "none", "savefig.facecolor": PAPER, "legend.frameon": False,
        "lines.solid_capstyle": "round", "pdf.fonttype": 42, "ps.fonttype": 42, "path.simplify": False,
    })


def figure(w: float, h: float):
    """A figure of w x h inches."""
    setup()
    return plt.figure(figsize=(w, h))


def ax(fig, x, y_top, w, h, **kw):
    """Axes placed in inches from the figure's top-left corner."""
    fw, fh = fig.get_size_inches()
    return fig.add_axes([x / fw, (fh - y_top - h) / fh, w / fw, h / fh], **kw)


def text(fig, x, y_top, s, **kw):
    fw, fh = fig.get_size_inches()
    return fig.text(x / fw, (fh - y_top) / fh, s, **{"fontsize": 8.0, "va": "baseline", **kw})


def save(fig, path: Path):
    """Write the panel as PDF and PNG next to its script."""
    path = Path(path)
    fig.savefig(path.with_suffix(".pdf"), dpi=600, metadata={"CreationDate": None, "Creator": None, "Producer": None})
    fig.savefig(path.with_suffix(".png"), dpi=300, metadata={"Software": None})
    plt.close(fig)


# ---------------------------------------------------------------- numbers
def num(x, fmt=".2f"):
    return format(x, fmt).replace("-", "−")


def ci_txt(ci, fmt=".2f"):
    return f"[{num(ci[0], fmt)}, {num(ci[1], fmt)}]"


def dz_txt(d, label=True):
    return (f"{DZ} = " if label else "") + num(d)


def bf(v, kind="01"):
    return f"BF$_{{{kind}}}$ = {v:.1f}"


def p_txt(p, sub=None):
    """p with one significant digit in scientific notation below 0.001, otherwise two significant digits."""
    lab = r"$\mathit{p}$" if not sub else r"$\mathit{p}_\mathrm{" + sub + "}$"
    if p < 1e-3:
        e = int(np.floor(np.log10(p)))
        m = int(round(p / 10 ** e))
        if m == 10:
            m, e = 1, e + 1
        return f"{lab} = {m} × 10$^{{{e}}}$"
    if p < 0.01:
        return f"{lab} = {p:.4f}".rstrip("0")
    if p < 0.1:
        return f"{lab} = {p:.3f}"
    return f"{lab} = {p:.2f}"


def mean_ci(x, level=0.95):
    """Mean and half-width of its t-based confidence interval."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return x.mean(), stats.t.ppf(0.5 + level / 2, x.size - 1) * x.std(ddof=1) / np.sqrt(x.size)


# ---------------------------------------------------------------- marks
def zero(a, horizontal=False, **kw):
    kw = {"color": SLATE.mid, "lw": 0.6, "zorder": 0, **kw}
    (a.axvline if horizontal else a.axhline)(0, **kw)


def band(a, lo=-0.5, hi=0.5, horizontal=False, color=BAND, z=-1):
    (a.axvspan if horizontal else a.axhspan)(lo, hi, color=color, lw=0, zorder=z)


def name_ticks(a, xs, names, colors, axis="x", bold=True, pad=4):
    """Condition names as tick labels, each in its colour, without tick marks or an axis line."""
    (a.set_xticks if axis == "x" else a.set_yticks)(xs, names)
    for lab, c in zip(a.get_xticklabels() if axis == "x" else a.get_yticklabels(), colors):
        lab.set_color(c)
        if bold:
            lab.set_fontweight("bold")
    a.tick_params(axis=axis, length=0, pad=pad)
    a.spines["bottom" if axis == "x" else "left"].set_visible(False)


def above(a, x, s, dy=0.0, color=INK, **kw):
    """One line of statistics above the plotting area at data x."""
    return a.text(x, 1.03 + dy, s, transform=a.get_xaxis_transform(), ha="center", va="baseline", color=color, **kw)


def swarm_offsets(a, values, size_pt=3.2, gap_pt=0.25, max_pt=None):
    """Deterministic beeswarm: horizontal offsets (points) that keep markers of diameter size_pt from overlapping, given
    the axes' current y-scale. Points are placed in order of value, each at the smallest free offset."""
    v = np.asarray(values, float)
    y0, y1 = a.get_ylim()
    h_pt = a.get_window_extent().height * 72 / a.figure.dpi
    ypt = (v - y0) / (y1 - y0) * h_pt
    d = size_pt + gap_pt
    placed, off = [], np.zeros(v.size)
    for i in np.argsort(ypt):
        for c in [0.0] + [s * k * d * 0.5 for k in range(1, 400) for s in (1, -1)]:
            if max_pt is not None and abs(c) > max_pt:
                continue
            if all((c - px) ** 2 + (ypt[i] - py) ** 2 >= d ** 2 for px, py in placed if abs(ypt[i] - py) < d):
                off[i] = c
                placed.append((c, ypt[i]))
                break
    return off


def pts_to_data_x(a, pts):
    x0, x1 = a.get_xlim()
    w_pt = a.get_window_extent().width * 72 / a.figure.dpi
    return np.asarray(pts) / w_pt * (x1 - x0)


def dots(a, x, v, color, size=3.5, max_pt=None, z=3, ring=PAPER, alpha=1.0):
    """A beeswarm of units (listeners, songs) at position x; the axis limits must be set before the call."""
    v = np.asarray(v, float)
    off = pts_to_data_x(a, swarm_offsets(a, v, size, max_pt=max_pt))
    a.scatter(x + off, v, s=size ** 2, facecolor=color, edgecolors=ring, linewidths=0.4, zorder=z, alpha=alpha)
    return x + off


def crossbar(a, x, v=None, color=INK, w=0.22, m=None, ci=None, lw=2.0, z=5, horizontal=False):
    """The mean of the units (a bar) with its 95% CI (a whisker)."""
    if v is not None:
        m, h = mean_ci(v)
        ci = (m - h, m + h)
    if horizontal:
        a.plot(ci, [x, x], color=color, lw=1.0, solid_capstyle="butt", zorder=z)
        a.plot([m, m], [x - w / 2, x + w / 2], color=color, lw=lw, solid_capstyle="butt", zorder=z + 1)
    else:
        a.plot([x, x], ci, color=color, lw=1.0, solid_capstyle="butt", zorder=z)
        a.plot([x - w / 2, x + w / 2], [m, m], color=color, lw=lw, solid_capstyle="butt", zorder=z + 1)
    return m, ci


def effect(a, pos, dz, ci, color, horizontal=True, ms=5.4, lw=1.2, z=4, filled=None, marker="o", ci_inner=None):
    """An effect size with its 95% CI: filled when the interval excludes zero, open (in slate) when it does not."""
    sig = (ci[0] > 0 or ci[1] < 0) if filled is None else filled
    col = color if sig else SLATE.base
    seg = (lambda u, v: ([u, v], [pos, pos])) if horizontal else (lambda u, v: ([pos, pos], [u, v]))
    a.plot(*seg(*ci), color=col, lw=lw, solid_capstyle="butt", zorder=z)
    if ci_inner is not None:
        a.plot(*seg(*ci_inner), color=col, lw=lw * 2.6, solid_capstyle="butt", zorder=z)
    xy = (dz, pos) if horizontal else (pos, dz)
    a.plot(*xy, marker=marker, ms=ms, mfc=col if sig else PAPER, mec=col, mew=1.2, zorder=z + 1, ls="none")
    return sig


# ---------------------------------------------------------------- sensor maps
def head_outline(a, radius=1.0, color=INK2, lw=0.7, z=5):
    a.add_patch(Circle((0, 0), radius, fill=False, ec=color, lw=lw, zorder=z))
    a.add_patch(Polygon([(-0.11 * radius, 0.993 * radius), (0, 1.12 * radius), (0.11 * radius, 0.993 * radius)], closed=False,
                        fill=False, ec=color, lw=lw, zorder=z, joinstyle="round"))
    for s in (-1, 1):
        a.add_patch(Polygon([(s * 0.995 * radius, 0.13 * radius), (s * 1.06 * radius, 0.11 * radius), (s * 1.075 * radius, -0.12 * radius),
                             (s * 0.995 * radius, -0.17 * radius)], closed=False, fill=False, ec=color, lw=lw, zorder=z, joinstyle="round"))


def _interp_map(xy, v, res=260, radius=1.0):
    from scipy.interpolate import CloughTocher2DInterpolator, NearestNDInterpolator
    ang = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    rim = np.c_[np.cos(ang), np.sin(ang)] * radius * 1.04
    pts = np.r_[xy, rim]
    vals = np.r_[v, NearestNDInterpolator(xy, v)(rim)]
    g = np.linspace(-radius, radius, res)
    gx, gy = np.meshgrid(g, g)
    z = CloughTocher2DInterpolator(pts, vals)(gx, gy)
    z[gx ** 2 + gy ** 2 > radius ** 2] = np.nan
    return gx, gy, z


def gain_map(a, xy, v, vmax, sensors=True, sensor_size=0.8, hull=False, res=260):
    """A sensor map of a prediction gain (head radius 1, nose up): zero is neutral, gains are blue and losses tan, on one
    symmetric scale. With ``hull``, the map is drawn only within the (slightly enlarged) convex hull of the sensors."""
    from matplotlib.path import Path as MPath
    from scipy.spatial import ConvexHull
    xy, v = np.asarray(xy, float), np.asarray(v, float)
    gx, gy, z = _interp_map(xy, v, res)
    if hull:
        h = xy[ConvexHull(xy).vertices]
        c = h.mean(0)
        inside = MPath(c + (h - c) * 1.06).contains_points(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
        z[~inside] = np.nan
    a.imshow(np.ma.masked_invalid(z), origin="lower", extent=(-1, 1, -1, 1), cmap=MAP_DIV, norm=Normalize(-vmax, vmax),
             interpolation="bilinear", zorder=1).set_rasterized(True)
    if sensors:
        a.scatter(xy[:, 0], xy[:, 1], s=sensor_size, color=INK, alpha=0.5, linewidths=0, zorder=3)
    head_outline(a)
    a.set_xlim(-1.18, 1.18)
    a.set_ylim(-1.12, 1.22)
    a.set_aspect("equal")
    a.axis("off")


def gain_bar(fig, x, y_top, w, h, lo, hi, ticks, label=None, ticklabels=None):
    """The colour scale of gain_map between lo and hi."""
    c = ax(fig, x, y_top, w, h)
    g = np.linspace(lo, hi, 256)
    vmax = max(abs(lo), abs(hi))
    c.imshow(MAP_DIV(Normalize(-vmax, vmax)(g))[None], aspect="auto", extent=(lo, hi, 0, 1))
    for s in c.spines.values():
        s.set_visible(False)
    c.set_yticks([])
    c.set_xticks(ticks, ticklabels or [num(t, "g") for t in ticks])
    c.tick_params(axis="x", length=2.0, pad=1.5, colors=INK2)
    if label:
        c.set_xlabel(label, color=INK2, labelpad=1.5)
    return c
