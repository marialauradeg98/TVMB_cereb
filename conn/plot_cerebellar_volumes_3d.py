"""
plot_cerebellar_volumes_3d.py
=============================

Interactive **3D volume rendering** of the cerebellar input/output pathway for
every lobule of the cerebellar cortex, built directly from the atlas

    RegionVolumeMapping_196.h5   (114x132x80 int grid; voxel value = region idx)
    VolumeAllen_196.h5           (origin + voxel_size, here 100 um isotropic)
    ConnKnox_196.h5              (196x196 weights, rows=source, cols=target)

For each cerebellar **cortical lobule** L the script highlights, as separate
3D isosurfaces drawn from the region-volume-mapping, the four anatomical stages
of its canonical circuit and prints each region's **volume**:

  1. L itself                                         (the lobule)
  2. the single most-connected **afferent** region    (top source into L)
  3. the region L **projects to most**  (top efferent; for a cortical lobule
     this is normally a deep cerebellar nucleus, DCN)
  4. IF that top-efferent region is a DCN, the region **the DCN projects to
     most** (the cerebellar OUTPUT target, e.g. thalamus / red nucleus / LRN)

so the picture reads  cortex  <-afferent | lobule | efferent->  DCN -> target.

Connectivity convention
-----------------------
``ConnKnox_196.h5`` stores the *builder* matrix with **rows = source,
cols = target** (W[i,j] = weight FROM i TO j; verified in the project notes).
So, for a region r:
    afferents into r  = column  W[:, r]   (who projects onto r)
    efferents from r  = row     W[r, :]   (where r projects)

Before ranking we zero the three cerebellum-internal artefact blocks
(cortex<->cortex, DCN<->DCN, DCN->cortex) but **keep cortex->DCN** (the real
Purkinje cortico-nuclear output). Without this, every lobule's top
afferent/efferent would be a neighbouring lobule (a voxel-adjacency artefact,
e.g. Culmen is Declive's biggest *raw* afferent). This is the same net zeroing
used by simulator/mouse_conn.tvb_cereb_preprocessing; we work on the raw rows=source
matrix (no transpose) so the afferent/efferent directions stay explicit.

Output
------
A single self-contained HTML (plotly) with a **dropdown** to pick the lobule.
Left panel: a 3D scene (no axes/grid) with the faint whole-mouse-brain mesh
always shown, the lobule's regions as coloured isosurfaces, and **curved
directional arrows** tracing the flow  afferent -> lobule -> efferent -> DCN
output. Right panel: the distribution of ALL connectome weights with the
selected chain's three weights marked on it, so you can see whether those
specific projections are strong or marginal. A CSV of the chain is also written.

Usage
-----
    python plot_cerebellar_volumes_3d.py                 # right hemisphere
    python plot_cerebellar_volumes_3d.py --hemi left
    python plot_cerebellar_volumes_3d.py --hemi both
    python plot_cerebellar_volumes_3d.py --datadir <dir> --outdir <dir>
"""

import os
import argparse
import numpy as np
import h5py
from scipy import ndimage
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Single source of truth for region classification / matrix preprocessing /
# normalization (imported TVB-free thanks to mouse_conn's lazy TVB import).
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from simulator.mouse_conn import MouseConnectivity


# ─────────────────────────────────────────────────────────────────────────────
# Region groups — reuse mouse_conn's canonical keyword lists (one definition).
# ─────────────────────────────────────────────────────────────────────────────
CEREB_CORTEX_KEYS = list(MouseConnectivity.CEREB_CORTEX_KW)
DCN_KEYS = list(MouseConnectivity.DCN_KW)

# How many top afferents / DCN-output targets to draw per lobule.
TOPN = 3

# Colours. The lobule and its (single) DCN efferent are fixed; the TOPN
# afferents and TOPN DCN-outputs get a weight-ordered shade ramp (rank 0 =
# strongest = darkest), and their arrows are drawn thicker for larger weights.
LOBULE_COLOR = "#D81B60"      # the lobule itself (magenta)
EFFERENT_COLOR = "#FB8C00"    # lobule -> DCN (orange, always one, a DCN)
AFF_SHADES = ["#0D47A1", "#1E88E5", "#90CAF9"]      # afferents, dark->light blue
DCNOUT_SHADES = ["#1B5E20", "#43A047", "#A5D6A7"]   # DCN outputs, dark->light green
CONTEXT_COLOR = "#BBBBBB"


def role_color(role, rank):
    if role == "lobule":
        return LOBULE_COLOR
    if role == "efferent":
        return EFFERENT_COLOR
    if role == "afferent":
        return AFF_SHADES[min(rank, len(AFF_SHADES) - 1)]
    return DCNOUT_SHADES[min(rank, len(DCNOUT_SHADES) - 1)]


def role_abbr(role, rank):
    if role == "afferent":
        return f"aff{rank + 1}"
    if role == "efferent":
        return "eff"
    return f"out{rank + 1}"


def short_label(name, n=18):
    """Hemisphere-stripped, comma-trimmed, truncated region label for tick text."""
    for p in ("Right ", "Left "):
        if name.startswith(p):
            name = name[len(p):]
    name = name.split(",")[0]
    return name if len(name) <= n else name[:n - 1] + "…"


# ─────────────────────────────────────────────────────────────────────────────
# Loading / classification
# ─────────────────────────────────────────────────────────────────────────────
def load_atlas(datadir):
    with h5py.File(os.path.join(datadir, "ConnKnox_216.h5"), "r") as h:
        labels = [l.decode() if isinstance(l, bytes) else l
                  for l in h["region_labels"][:]]
        W = h["weights"][:].astype(float)          # rows=source, cols=target
    np.fill_diagonal(W, 0.0)                        # no self-connections (always)
    with h5py.File(os.path.join(datadir, "RegionVolumeMapping_216.h5"), "r") as h:
        rvm = h["array_data"][:]
    with h5py.File(os.path.join(datadir, "VolumeAllen_216.h5"), "r") as h:
        voxel_size = h["voxel_size"][:]            # micrometres (here 100,100,100)
    return labels, W, rvm, voxel_size


def zero_internal_blocks(W, cortex, dcn):
    """Zero cerebellum-internal artefact blocks; keep cortex->DCN.
    Delegates to mouse_conn (single source of truth)."""
    return MouseConnectivity.zero_internal_blocks(W, cortex, dcn)


# ─────────────────────────────────────────────────────────────────────────────
# Chain per lobule
# ─────────────────────────────────────────────────────────────────────────────
def _topn(vec, self_idx, n):
    """Indices of the n largest strictly-positive entries of vec (self removed)."""
    vec = vec.copy()
    vec[self_idx] = 0.0
    order = np.argsort(vec)[::-1]
    return [int(i) for i in order[:n] if vec[i] > 0]


def build_chain(L, W, labels, dcn_set, topn=TOPN):
    """Structured chain for lobule L.

    Returns a dict with:
      lobule   : (idx, nan)
      afferents: up to ``topn`` (rank, idx, weight)  -- strongest sources INTO L
      efferent : (idx, weight) -- the single region L projects to most
      eff_is_dcn: bool -- whether that efferent is a deep cerebellar nucleus
      dcn_outs : up to ``topn`` (rank, idx, weight) -- strongest targets of the DCN
                 (empty if the efferent is not a DCN)
    """
    aff = W[:, L]                                    # column = sources into L
    afferents = [(r, i, float(aff[i]))
                 for r, i in enumerate(_topn(aff, L, topn))]

    eff = W[L, :].copy(); eff[L] = 0.0               # row = targets of L
    e = int(eff.argmax())
    efferent, eff_is_dcn, dcn_outs = None, False, []
    if eff[e] > 0:
        efferent = (e, float(eff[e]))
        eff_is_dcn = e in dcn_set
        if eff_is_dcn:
            do = W[e, :]
            dcn_outs = [(r, i, float(do[i]))
                        for r, i in enumerate(_topn(do, e, topn))]
    return dict(lobule=(L, np.nan), afferents=afferents, efferent=efferent,
                eff_is_dcn=eff_is_dcn, dcn_outs=dcn_outs)


def region_volume_mm3(rvm, idx, voxel_mm3):
    return int((rvm == idx).sum()) * voxel_mm3


# ─────────────────────────────────────────────────────────────────────────────
# 3D rendering helpers
# ─────────────────────────────────────────────────────────────────────────────
def isosurface_trace(rvm, idx, color, name, scale, opacity=0.6,
                     smooth=0.8, visible=True, maxpts=10000, legendgroup=None):
    """Isosurface of one region, cropped to its padded bounding box.

    Big convoluted regions (e.g. a whole cortical lobule) are first down-
    sampled so the grid stays under ``maxpts`` points: plotly's isosurface
    silently returns *no* polygons above ~1-2e4 grid points, which is why a
    large lobule such as the Ansiform would otherwise vanish.
    """
    mask = (rvm == idx)
    xs, ys, zs = np.where(mask)
    if xs.size == 0:
        return None
    pad = 2
    x0 = max(xs.min() - pad, 0); y0 = max(ys.min() - pad, 0); z0 = max(zs.min() - pad, 0)
    x1 = xs.max() + pad + 1; y1 = ys.max() + pad + 1; z1 = zs.max() + pad + 1
    sub = mask[x0:x1, y0:y1, z0:z1].astype(float)

    step = max(1, int(np.ceil((sub.size / maxpts) ** (1.0 / 3.0))))
    if step > 1:
        sub = ndimage.zoom(sub, 1.0 / step, order=1)
    sub = ndimage.gaussian_filter(sub, smooth)

    nx, ny, nz = sub.shape
    X, Y, Z = np.mgrid[0:nx, 0:ny, 0:nz]
    s = scale * step
    return go.Isosurface(
        x=(X.ravel() * s + x0 * scale),
        y=(Y.ravel() * s + y0 * scale),
        z=(Z.ravel() * s + z0 * scale),
        value=sub.ravel(), isomin=0.5, isomax=0.5, surface_count=1,
        colorscale=[[0, color], [1, color]], showscale=False,
        opacity=opacity, name=name, showlegend=True, visible=visible,
        legendgroup=legendgroup,
        caps=dict(x_show=False, y_show=False, z_show=False),
        hovertemplate=name + "<extra></extra>",
    )


def brain_mesh_trace(rvm, scale, step=2):
    """Faint grey surface of the WHOLE mouse brain (always shown as context)."""
    mask = (rvm != -1)[::step, ::step, ::step].astype(float)
    mask = ndimage.gaussian_filter(mask, 0.6)
    X, Y, Z = np.mgrid[0:mask.shape[0], 0:mask.shape[1], 0:mask.shape[2]]
    s = scale * step
    return go.Isosurface(
        x=(X.ravel() * s), y=(Y.ravel() * s), z=(Z.ravel() * s),
        value=mask.ravel(), isomin=0.5, isomax=0.5, surface_count=1,
        colorscale=[[0, CONTEXT_COLOR], [1, CONTEXT_COLOR]], showscale=False,
        opacity=0.06, name="mouse brain", showlegend=True, visible=True,
        caps=dict(x_show=False, y_show=False, z_show=False), hoverinfo="skip",
    )


def centroid(rvm, idx, scale):
    return np.argwhere(rvm == idx).mean(axis=0) * scale


def curved_arrow(p0, p1, color, name, visible, bend=0.16, n=24, head=0.45,
                 width=4, legendgroup=None):
    """A gently curved directed arrow p0 -> p1 as (line, cone) traces.

    The small cone sits at the very end of the path (the target region) with its
    tip pointing INTO that region, so the direction of the projection is clear.
    ``width`` (line) and ``head`` (cone) are scaled by weight by the caller.
    """
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    L = np.linalg.norm(d) + 1e-9
    ref = np.array([0.0, 0.0, 1.0])
    perp = np.cross(d, ref)
    if np.linalg.norm(perp) < 1e-6:
        perp = np.cross(d, np.array([0.0, 1.0, 0.0]))
    perp = perp / (np.linalg.norm(perp) + 1e-9)
    ctrl = (p0 + p1) / 2 + perp * bend * L
    t = np.linspace(0, 1, n)[:, None]
    pts = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * ctrl + t ** 2 * p1
    # tip exactly at the target; direction = last path segment (points into target)
    tip = p1
    tang = pts[-1] - pts[-2]
    tang = tang / (np.linalg.norm(tang) + 1e-9)
    line = go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="lines",
        line=dict(color=color, width=width), name=name, showlegend=False,
        legendgroup=legendgroup,
        visible=visible, hovertemplate=name + "<extra></extra>")
    cone = go.Cone(
        x=[tip[0]], y=[tip[1]], z=[tip[2]],
        u=[tang[0]], v=[tang[1]], w=[tang[2]], anchor="tip",
        sizemode="absolute", sizeref=head, showscale=False,
        legendgroup=legendgroup,
        colorscale=[[0, color], [1, color]], visible=visible, hoverinfo="skip")
    return line, cone


# ─────────────────────────────────────────────────────────────────────────────
# Weight-distribution panel
# ─────────────────────────────────────────────────────────────────────────────
def weight_marker(xval, w, ymax, color, role, visible):
    """A vertical line + labelled head marking one chain weight on the histogram.

    ``xval`` is the x-position on the distribution axis (raw weight, or its
    log10 when --logweights is used); ``w`` is always the raw weight for labels.
    """
    line = go.Scatter(x=[xval, xval], y=[0, ymax], mode="lines",
                      line=dict(color=color, width=3, dash="dash"),
                      showlegend=False, visible=visible,
                      hovertemplate=f"{role}  w={w:.4f}<extra></extra>")
    head = go.Scatter(x=[xval], y=[ymax * 1.02], mode="markers+text",
                      text=[f"{w:.3f}"], textposition="top center",
                      textfont=dict(color=color, size=11),
                      marker=dict(color=color, size=9, symbol="triangle-down"),
                      showlegend=False, visible=visible,
                      hovertemplate=f"{role}  w={w:.4f}<extra></extra>")
    return line, head


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datadir",
                    default=os.path.join(here, "conn_data", "Knox_conn_216"),
                    help="dir holding ConnKnox/RegionVolumeMapping/VolumeAllen h5")
    ap.add_argument("--outdir", default=os.path.join(here, "volume_report_216"))
    ap.add_argument("--hemi", choices=["right", "left", "both"], default="right")
    ap.add_argument("--logweights", action="store_true",
                    help="plot the weight distribution on a log10 axis "
                         "(default: raw weights)")
    ap.add_argument("--normalize", default="none",
                    choices=["none", "floor_log", "log", "spectral", "row"],
                    help="weight normalization applied before analysis "
                         "(floor_log = threshold noise floor + log -> ~Gaussian).")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    labels, W, rvm, voxel_size = load_atlas(args.datadir)
    voxel_mm3 = float(np.prod(voxel_size)) / 1e9     # um^3 -> mm^3
    scale = float(voxel_size[0]) / 1000.0            # voxel index -> mm

    if args.normalize != "none":
        W = MouseConnectivity.normalize(W, method=args.normalize)
        print(f"[plot] applied normalize='{args.normalize}' "
              f"(rho={MouseConnectivity.spectral_radius(W):.3f})")

    # region classification + cerebellum-internal zeroing via the single source
    cortex, dcn = MouseConnectivity.classify_cerebellum(labels)
    dcn_set = set(dcn.tolist())
    Wz = MouseConnectivity.zero_internal_blocks(W, cortex, dcn)

    def in_hemi(i):
        lb = labels[i].lower()
        if args.hemi == "right":
            return lb.startswith("right")
        if args.hemi == "left":
            return lb.startswith("left")
        return True

    lobules = sorted([L for L in cortex if in_hemi(L)], key=lambda i: labels[i])
    n_lobules = len(lobules)

    # Distribution of ALL connectome weights (drop the numerical-noise floor).
    # Raw weights by default; --logweights switches the x-axis to log10.
    w_all = W[W > 1e-6]
    x_all = np.log10(w_all) if args.logweights else w_all
    xpos = (lambda w: np.log10(w)) if args.logweights else (lambda w: w)
    hist_counts, _ = np.histogram(x_all, bins=50)
    ymax = float(hist_counts.max())

    # Total incoming / outgoing weight per region on the (zeroed) matrix, used to
    # express each chain weight as a % of the receiving region's total input
    # (and, in hover, of the source's total output).
    col_in = Wz.sum(axis=0)          # incoming total per region (sum over sources)
    row_out = Wz.sum(axis=1)         # outgoing total per region (sum over targets)

    # ── layout: 3D scene (left, full height) + two stacked right panels:
    #    top = weight distribution, bottom = chain weight as % of target input ──
    fig = make_subplots(
        rows=2, cols=2, column_widths=[0.62, 0.38], row_heights=[0.55, 0.45],
        vertical_spacing=0.12, horizontal_spacing=0.06,
        specs=[[{"type": "scene", "rowspan": 2}, {"type": "xy"}],
               [None, {"type": "xy"}]],
        subplot_titles=("", "Weight Distribution",
                        "chain weight as % of the receiving region's total input"))

    # Trace 0: whole-brain mesh (always visible). Trace 1: histogram (always).
    fig.add_trace(brain_mesh_trace(rvm, scale), row=1, col=1)
    fig.add_trace(go.Histogram(x=x_all, nbinsx=50, marker_color="#cfcfcf",
                               marker_line_color="#9a9a9a", marker_line_width=0.5,
                               name="all weights", showlegend=False),
                  row=1, col=2)

    csv_rows = []
    lobule_ranges = []
    buttons = []
    verify = []                                      # (lobule, efferent, is_dcn)
    # trace indices + original opacities of everything the hide/show toggle acts
    # on: the coloured region areas AND their arrows (lines + cones). The faint
    # whole-brain mesh (trace 0) is left on as a spatial reference.
    hide_idx = []
    hide_op = []

    def arrow_width(w, wmax):
        return 1.5 + 3.5 * (w / wmax) if wmax > 0 else 2.0

    def cone_head(w, wmax):
        return 0.16 + 0.16 * (w / wmax) if wmax > 0 else 0.22

    def add_arrow(line, cone):
        """Add an arrow's line+cone and register them with the hide toggle."""
        fig.add_trace(line, row=1, col=1)
        hide_idx.append(len(fig.data) - 1); hide_op.append(1.0)
        fig.add_trace(cone, row=1, col=1)
        hide_idx.append(len(fig.data) - 1); hide_op.append(1.0)

    for li, L in enumerate(lobules):
        ch = build_chain(L, Wz, labels, dcn_set)
        vis0 = (li == 0)
        start = len(fig.data)
        Lidx = ch["lobule"][0]
        lob_c = centroid(rvm, Lidx, scale)
        eff_idx = ch["efferent"][0] if ch["efferent"] else None
        verify.append((labels[L], labels[eff_idx] if eff_idx is not None else "-",
                       ch["eff_is_dcn"]))

        # (role, rank, idx, weight) list: lobule, 3 afferents, efferent DCN, 3 outputs
        items = [("lobule", 0, Lidx, np.nan)]
        items += [("afferent", r, i, w) for r, i, w in ch["afferents"]]
        if ch["efferent"] is not None:
            items.append(("efferent", 0, eff_idx, ch["efferent"][1]))
        items += [("dcn_out", r, i, w) for r, i, w in ch["dcn_outs"]]

        # region isosurfaces. Each area shares a legendgroup with the arrow that
        # connects it, so clicking its legend entry hides the area AND its arrow.
        for role, rank, idx, w in items:
            color = role_color(role, rank)
            tag = role if role in ("lobule", "efferent") else f"{role} #{rank+1}"
            op = 0.85 if role == "lobule" else 0.45
            lg = f"L{li}_{role}{rank}"
            tr = isosurface_trace(rvm, idx, color, f"[{tag}] {labels[idx]}", scale,
                                  opacity=op, visible=vis0, legendgroup=lg)
            if tr is not None:
                fig.add_trace(tr, row=1, col=1)
                hide_idx.append(len(fig.data) - 1)
                hide_op.append(op)
            csv_rows.append((labels[L], tag, labels[idx],
                             region_volume_mm3(rvm, idx, voxel_mm3),
                             "" if np.isnan(w) else f"{w:.6f}"))

        # directed arrows: colour + thickness ORDERED BY WEIGHT within each group
        aff_wmax = max((w for _, _, w in ch["afferents"]), default=0.0)
        do_wmax = max((w for _, _, w in ch["dcn_outs"]), default=0.0)

        for r, i, w in ch["afferents"]:                       # sources -> lobule
            color = role_color("afferent", r)
            line, cone = curved_arrow(
                centroid(rvm, i, scale), lob_c, color,
                f"{labels[i]} → {labels[Lidx]} (w={w:.3f})", vis0,
                width=arrow_width(w, aff_wmax), head=cone_head(w, aff_wmax),
                legendgroup=f"L{li}_afferent{r}")
            add_arrow(line, cone)

        if eff_idx is not None:                               # lobule -> DCN (one)
            w = ch["efferent"][1]
            line, cone = curved_arrow(
                lob_c, centroid(rvm, eff_idx, scale), EFFERENT_COLOR,
                f"{labels[Lidx]} → {labels[eff_idx]} (w={w:.3f})", vis0,
                width=3, head=0.26, legendgroup=f"L{li}_efferent0")
            add_arrow(line, cone)

        for r, i, w in ch["dcn_outs"]:                        # DCN -> outputs
            color = role_color("dcn_out", r)
            line, cone = curved_arrow(
                centroid(rvm, eff_idx, scale), centroid(rvm, i, scale), color,
                f"{labels[eff_idx]} → {labels[i]} (w={w:.3f})", vis0,
                width=arrow_width(w, do_wmax), head=cone_head(w, do_wmax),
                legendgroup=f"L{li}_dcn_out{r}")
            add_arrow(line, cone)

        # weight markers on the distribution
        for role, rank, idx, w in items:
            if np.isnan(w) or w <= 0:
                continue
            tag = role if role in ("lobule", "efferent") else f"{role}#{rank+1}"
            ln, hd = weight_marker(xpos(w), w, ymax, role_color(role, rank),
                                   tag, vis0)
            fig.add_trace(ln, row=1, col=2); fig.add_trace(hd, row=1, col=2)

        # percentage panel: each chain edge (src->tgt) as % of the TARGET's total
        # incoming weight (hover also gives % of the SOURCE's total output).
        pedges = [("afferent", r, i, Lidx, w) for r, i, w in ch["afferents"]]
        if eff_idx is not None:
            pedges.append(("efferent", 0, Lidx, eff_idx, ch["efferent"][1]))
        pedges += [("dcn_out", r, eff_idx, i, w) for r, i, w in ch["dcn_outs"]]
        ys, xs, cols_, txt, hov = [], [], [], [], []
        for role, rank, src, tgt, w in pedges:
            pin = 100.0 * w / col_in[tgt] if col_in[tgt] > 0 else 0.0
            pout = 100.0 * w / row_out[src] if row_out[src] > 0 else 0.0
            cp = src if role == "afferent" else tgt
            ys.append(f"{role_abbr(role, rank)}: {short_label(labels[cp])}")
            xs.append(pin); cols_.append(role_color(role, rank))
            txt.append(f"{pin:.1f}%")
            hov.append(f"{labels[src]} → {labels[tgt]}<br>weight {w:.4f}"
                       f"<br>{pin:.1f}% of {short_label(labels[tgt])} total input"
                       f"<br>{pout:.1f}% of {short_label(labels[src])} total output")
        fig.add_trace(go.Bar(y=ys, x=xs, orientation="h", marker_color=cols_,
                             text=txt, textposition="outside", cliponaxis=False,
                             hovertext=hov, hoverinfo="text", showlegend=False,
                             visible=vis0), row=2, col=2)

        lobule_ranges.append((start, len(fig.data)))

    # dropdown: traces 0 & 1 always on; one lobule block at a time
    total = len(fig.data)
    for li, (start, end) in enumerate(lobule_ranges):
        vis = [False] * total
        vis[0] = vis[1] = True
        for t in range(start, end):
            vis[t] = True
        buttons.append(dict(label=labels[lobules[li]], method="update",
                            args=[{"visible": vis},
                                  {"title": _title(labels[lobules[li]])}]))

    # Toggle to hide/show the coloured region areas AND their arrows together
    # (opacity restyle on those traces; the faint whole-brain mesh stays on as a
    # reference). NB: switching lobule re-shows that lobule's areas+arrows
    # (press "hide" again after switching).
    area_toggle = dict(
        type="buttons", direction="right", showactive=False,
        x=0.0, xanchor="left", y=1.07, yanchor="bottom", bgcolor="#f5f5f5",
        buttons=[
            dict(label="areas+arrows: show", method="restyle",
                 args=[{"opacity": hide_op}, hide_idx]),
            dict(label="areas+arrows: hide", method="restyle",
                 args=[{"opacity": [0] * len(hide_idx)}, hide_idx]),
        ])

    no_axis = dict(visible=False, showgrid=False, showbackground=False,
                   showticklabels=False, title="")
    fig.update_layout(
        title=_title(labels[lobules[0]]),
        updatemenus=[dict(buttons=buttons, direction="down", showactive=True,
                          x=0.0, xanchor="left", y=1.0, yanchor="bottom",
                          bgcolor="#f5f5f5"),
                     area_toggle],
        scene=dict(xaxis=no_axis, yaxis=no_axis, zaxis=no_axis,
                   aspectmode="data", bgcolor="white"),
        legend=dict(x=0.0, y=0.0, bgcolor="rgba(255,255,255,0.7)",
                    font=dict(size=11), groupclick="togglegroup"),
        margin=dict(l=0, r=0, t=70, b=0),
    )
    fig.update_xaxes(title_text=("log₁₀(weight)" if args.logweights else "weight"),
                     row=1, col=2)
    fig.update_yaxes(title_text="count", type="log", row=1, col=2)
    fig.update_xaxes(title_text="% of target's total input", ticksuffix="%",
                     row=2, col=2)
    fig.update_yaxes(autorange="reversed", automargin=True,
                     tickfont=dict(size=10), row=2, col=2)  # first edge on top

    html = os.path.join(args.outdir, f"cerebellar_volumes_3d_{args.hemi}.html")
    fig.write_html(html, include_plotlyjs="cdn")

    csv = os.path.join(args.outdir, f"cerebellar_volume_chain_{args.hemi}.csv")
    with open(csv, "w") as f:
        f.write("lobule,role,region,volume_mm3,weight\n")
        for lob, role, reg, vol, w in csv_rows:
            f.write(f"{lob},{role},{reg},{vol:.4f},{w}\n")

    # ── verification: is each lobule's #1 efferent really a DCN? ──
    n_bad = sum(1 for _, _, ok in verify if not ok)
    print(f"voxel = {voxel_mm3*1e6:.1f} x10^-6 mm^3 ;  {n_lobules} lobules "
          f"({args.hemi} hemisphere)")
    print("Verification — top efferent of each lobule is a DCN:")
    for lob, eff, ok in verify:
        print(f"  {'OK ' if ok else 'NOT'}  {lob:28s} -> {eff}")
    print(f"  => {n_lobules - n_bad}/{n_lobules} lobules project most strongly "
          f"to a DCN"
          + ("" if n_bad == 0 else f"  ({n_bad} do NOT — see 'NOT' rows above!)"))
    print(f"wrote {html}")
    print(f"wrote {csv}")


def _title(lob):
    return (f"<b>{lob}</b> — afferent → lobule → efferent → DCN output "
            f"(curved arrows = direction)")


if __name__ == "__main__":
    main()
