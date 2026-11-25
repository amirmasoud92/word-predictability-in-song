"""Figure 3d. The attended - ignored contrast (d_z) in the 68 Desikan-Killiany labels on the inflated fsaverage
cortex, lateral and medial views (LCMV beamformer; exploratory). Labels with q >= 0.05 (false-discovery-rate
correction over 68 labels) are faded; the six labels with the largest effects are named.

Data: labels.csv (label, hemisphere, d_z, q). The cortical surface is the fsaverage template of FreeSurfer, read from
$SUBJECTS_DIR/fsaverage or obtained with mne.datasets.fetch_fsaverage().
"""
import os
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import Normalize

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1]))
import style as S  # noqa: E402

# screen x, screen y, towards the viewer (RAS: x right, y anterior, z superior)
VIEWS = {"left": (np.array([0, -1, 0]), np.array([0, 0, 1]), np.array([-1, 0, 0])),
         "right": (np.array([0, 1, 0]), np.array([0, 0, 1]), np.array([1, 0, 0])),
         "left_medial": (np.array([0, 1, 0]), np.array([0, 0, 1]), np.array([1, 0, 0])),
         "right_medial": (np.array([0, -1, 0]), np.array([0, 0, 1]), np.array([-1, 0, 0]))}
HEMI = {"left": "lh", "right": "rh", "left_medial": "lh", "right_medial": "rh"}
GYRUS, SULCUS = "#ECEDEF", "#D2D5D9"
NAMES = {"transversetemporal": "TTG", "middletemporal": "MTG", "bankssts": "STS", "supramarginal": "SMG", "parsopercularis": "IFG"}


def fsaverage_dir() -> Path:
    if os.environ.get("SUBJECTS_DIR"):
        return Path(os.environ["SUBJECTS_DIR"]) / "fsaverage"
    import mne
    return Path(mne.datasets.fetch_fsaverage())


_SURF = {}


def surface(fs: Path, hemi: str):
    """Inflated surface of one hemisphere (hemispheres moved apart by 6 % of their width), its triangles, the
    Desikan-Killiany label of each vertex ('' outside cortex) and the sulcal depth."""
    import nibabel as nib
    if hemi not in _SURF:
        rr, faces = nib.freesurfer.read_geometry(fs / "surf" / f"{hemi}.inflated")
        rr = rr.astype(float)
        rr[:, 0] += (-1 if hemi == "lh" else 1) * (rr[:, 0].max() - rr[:, 0].min()) * 0.06
        labels, _, names = nib.freesurfer.read_annot(fs / "label" / f"{hemi}.aparc.annot")
        names = [n.decode() if isinstance(n, bytes) else n for n in names]
        roi = np.array([f"{names[k]}_{hemi}" if k >= 0 and names[k] not in ("unknown", "corpuscallosum") else "" for k in labels], dtype=object)
        sulc = nib.freesurfer.read_morph_data(fs / "surf" / f"{hemi}.sulc")
        _SURF[hemi] = (rr, np.asarray(faces), roi, sulc)
    return _SURF[hemi]


def draw(a, fs, values, view, cmap, norm, faded, fade=0.78, light=(0.35, 0.55, 1.0), ambient=0.52):
    """Orthographic projection of the triangulated surface, painter's algorithm, per-face shading; label values in
    colour, faded labels blended towards the cortex, label borders in white."""
    sx, sy, sz = VIEWS[view]
    L = np.asarray(light, float) / np.linalg.norm(light)
    g_rgb, s_rgb = np.array(mpl.colors.to_rgb(GYRUS)), np.array(mpl.colors.to_rgb(SULCUS))
    rr, tris, roi, sulc = surface(fs, HEMI[view])
    P = np.c_[rr @ sx, rr @ sy, rr @ sz]
    T = P[tris]
    nrm = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
    front = nrm[:, 2] > 0
    fr = roi[tris]
    face_roi = np.where((fr[:, 0] == fr[:, 1]) | (fr[:, 0] == fr[:, 2]), fr[:, 0], fr[:, 1])
    s = (sulc[tris].mean(1) > 0).astype(float)[:, None]
    base = g_rgb * (1 - s) + s_rgb * s
    for r in set(face_roi) - {""}:
        if r not in values:
            continue
        m = face_roi == r
        c = np.array(cmap(norm(values[r]))[:3])
        cc = c[None, :] * (1 - fade) + base[m] * fade if r in faded else np.repeat(c[None, :], m.sum(), 0)
        base[m] = cc * (0.93 + 0.07 * (1 - s[m]))
    shade = ambient + (1 - ambient) * np.clip(nrm @ L, 0, 1)
    rgb = np.clip(base * shade[:, None], 0, 1)
    polys, cols, depth = T[front][:, :, :2], rgb[front], T[front][:, :, 2].mean(1)
    tf = tris[front]
    edges = []
    for i, j in ((0, 1), (1, 2), (2, 0)):
        ra, rb = roi[tf[:, i]], roi[tf[:, j]]
        m = (ra != rb) & ((ra != "") | (rb != ""))
        edges.append(np.stack([P[tf[m, i]][:, :2], P[tf[m, j]][:, :2]], 1))
    order = np.argsort(depth)
    pc = PolyCollection(polys[order], facecolors=cols[order], edgecolors=cols[order], linewidths=0.1, zorder=1)
    pc.set_rasterized(True)
    a.add_collection(pc)
    lc = LineCollection(np.concatenate(edges), colors=S.PAPER, linewidths=0.25, zorder=2, capstyle="round")
    lc.set_rasterized(True)
    a.add_collection(lc)
    xy = polys.reshape(-1, 2)
    a.set_xlim(xy[:, 0].min() - 2.0, xy[:, 0].max() + 2.0)
    a.set_ylim(xy[:, 1].min() - 2.0, xy[:, 1].max() + 2.0)
    a.set_aspect("equal")
    a.axis("off")


def label_xy(fs, label, hemi, view):
    rr, _, roi, _ = surface(fs, hemi)
    sx, sy, _ = VIEWS[view]
    p = rr[roi == f"{label}_{hemi}"].mean(0)
    return float(p @ sx), float(p @ sy)


lab = pd.read_csv(HERE / "labels.csv")
lab["key"] = lab.label + "_" + lab.hemisphere
vals = dict(zip(lab.key, lab.dz))
faded = set(lab.key[lab.q >= 0.05])
top = lab.sort_values("dz", ascending=False).head(6)
norm = Normalize(0, 0.65)
fs = fsaverage_dir()

fig = S.figure(3.80, 3.12)
x0, y_top = 0.04, 0.16
wl, hl, wm, hm = 1.78, 1.30, 1.16, 0.80
views = [("left", x0, y_top, wl, hl), ("right", x0 + wl + 0.12, y_top, wl, hl),
         ("left_medial", x0 + (wl - wm) / 2, y_top + hl + 0.42, wm, hm), ("right_medial", x0 + wl + 0.12 + (wl - wm) / 2, y_top + hl + 0.42, wm, hm)]
axes = {}
for view, x, y, w, h in views:
    a = S.ax(fig, x, y, w, h)
    draw(a, fs, vals, view, S.MAP_POS, norm, faded)
    axes[view] = a
S.text(fig, x0 + wl / 2, y_top - 0.02, "left", ha="center", color=S.INK2)
S.text(fig, x0 + wl + 0.12 + wl / 2, y_top - 0.02, "right", ha="center", color=S.INK2)

# the six labels with the largest effects, named below the lateral views
for view, hemi in (("left", "lh"), ("right", "rh")):
    a = axes[view]
    rows = [r.label for r in top.itertuples() if r.hemisphere == hemi and r.label in NAMES]
    pts = sorted(((nm, label_xy(fs, nm, hemi, view)) for nm in rows), key=lambda t: t[1][0])
    xa, xb = a.get_xlim()
    ya, yb = a.get_ylim()
    w_, h_ = xb - xa, yb - ya
    slots = np.linspace(xa + 0.22 * w_, xb - 0.22 * w_, len(pts)) if len(pts) > 1 else [pts[0][1][0]]
    yt = ya - 0.05 * h_
    for (nm, xy), sx in zip(pts, slots):
        a.plot([xy[0], xy[0], sx], [xy[1], ya + 0.04 * h_, yt + 0.01 * h_], color=S.INK, lw=0.5, clip_on=False, zorder=5,
               path_effects=[pe.withStroke(linewidth=1.4, foreground=S.PAPER)])
        a.plot(*xy, "o", ms=2.6, mfc=S.INK, mec=S.PAPER, mew=0.5, zorder=6, clip_on=False)
        a.text(sx, yt, NAMES[nm], ha="center", va="top", color=S.INK, clip_on=False)

# colour scale
c = S.ax(fig, x0 + wl + 0.06 - 0.60, y_top + hl + 0.42 + hm + 0.10, 1.20, 0.07)
g = np.linspace(norm.vmin, norm.vmax, 256)
c.imshow(S.MAP_POS(norm(g))[None], aspect="auto", extent=(norm.vmin, norm.vmax, 0, 1))
for sp in c.spines.values():
    sp.set_visible(False)
c.set_yticks([])
c.set_xticks([0, 0.3, 0.6], ["0", "0.3", "0.6"])
c.tick_params(axis="x", length=2.0, pad=1.5, colors=S.INK2)
c.set_xlabel(f"attended − ignored, {S.DZ}", color=S.INK2, labelpad=1.5)

S.save(fig, HERE / "panel")
