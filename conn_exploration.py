import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from collections import Counter
from scipy.ndimage import binary_erosion
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

BASE     = os.path.dirname(os.path.abspath(__file__))
VXC_SITE = os.path.join(BASE, "../../cereb_env/lib/python3.11/site-packages")
sys.path.insert(0, VXC_SITE)
sys.path.insert(0, BASE)

from voxcell import RegionMap
from simulator.mouse_conn import MouseConnectivity

CONNECTIVITY_FILE = os.path.join(BASE, "../../tvmb_data/My_conn_194/ConnAllen_194.h5")
RVM_FILE          = os.path.join(BASE, "../../tvmb_data/My_conn_194/RegionVolumeMapping_194.h5")
JSON_FILE         = os.path.join(BASE, "../connectivity_from_allen/1.json")
VOXEL_MM          = 0.025


EXCLUDE_LABELS = {
    "Right Hippocampo-amygdalar transition area",
    "Left Hippocampo-amygdalar transition area",
}

MACRO_LIST = [
    "Isocortex", "Olfactory areas", "Hippocampal formation",
    "Cortical subplate", "Striatum", "Pallidum", "Thalamus",
    "Hypothalamus", "Midbrain", "Cerebellum", "Brain stem", "Medulla",
]

MACRO_COLORS = {
    "Isocortex":             "#0057FF",
    "Olfactory areas":       "#FF007F",
    "Hippocampal formation": "#FF8C00",
    "Cortical subplate":     "#00C49A",
    "Striatum":              "#9B00E8",
    "Pallidum":              "#00B300",
    "Thalamus":              "#FF1A1A",
    "Hypothalamus":          "#FFD700",
    "Midbrain":              "#A0522D",
    "Cerebellum":            "#00D4FF",
    "Brain stem":            "#4CAF50",
    "Medulla":               "#FF5500",
    "Unknown":               "#BDBDBD",
}


SPINE_C    = "#333333"
LABEL_SIZE = 15
TITLE_SIZE = 17
TICK_SIZE  = 13
C_TEXT     = "#1a1a1a"
C_MEAN     = "#C0392B"
C_MEDIAN   = "#E67E22"

CMAP_W  = "Blues"
CMAP_TL = "YlGn"
C_W     = plt.cm.Blues(0.72)
C_TL    = plt.cm.YlGn(0.72)


def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(SPINE_C)
    ax.spines["bottom"].set_color(SPINE_C)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.tick_params(axis="both", labelsize=TICK_SIZE, color=SPINE_C)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))


def add_vlines_legend(ax, data, fmt=".4f", loc="upper right"):
    mean   = data.mean()
    median = np.median(data)
    ax.axvline(mean,   color=C_MEAN,   linewidth=1.8, linestyle="--", zorder=3)
    ax.axvline(median, color=C_MEDIAN, linewidth=1.8, linestyle=":",  zorder=3)
    elems = [
        Line2D([0], [0], color=C_MEAN,   linewidth=1.8, linestyle="--",
               label=f"Mean = {mean:{fmt}}"),
        Line2D([0], [0], color=C_MEDIAN, linewidth=1.8, linestyle=":",
               label=f"Median = {median:{fmt}}"),
    ]
    ax.legend(handles=elems, frameon=True, framealpha=0.9,
              edgecolor="#CCCCCC", fontsize=11, loc=loc)


def get_macro(label, region_map):
    clean = label.replace("Left ", "").replace("Right ", "")
    ids = region_map.find(clean, "name", with_descendants=False)
    if not ids:
        return "Unknown"
    rid = int(next(iter(ids)))
    pid = region_map.get(rid, "parent_structure_id")
    pname = region_map.get(pid, "name")
    for _ in range(20):
        if pname in MACRO_LIST:
            return pname
        pid   = region_map.get(pid, "parent_structure_id")
        pname = region_map.get(pid, "name")
    return "Unknown"


def brain_surface_points(rvm_path, downsample=3, voxel_mm=VOXEL_MM, n_max=8000):
    print("  Loading RegionVolumeMapping…")
    with h5py.File(rvm_path) as f:
        rvm = np.array(f["array_data"][()])
    rvm_ds   = rvm[::downsample, ::downsample, ::downsample]
    mask     = rvm_ds >= 0
    surface  = mask & ~binary_erosion(mask)
    surf_idx = np.argwhere(surface)
    step     = max(1, len(surf_idx) // n_max)
    pts      = surf_idx[::step] * (downsample * voxel_mm)
    print(f"  Brain surface: {len(pts)} points")
    return pts



def add_colorbar(im, ax, label, fmt="%.2f"):
    cb = plt.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label(label, fontsize=12)
    cb.ax.tick_params(labelsize=11)
    cb.locator   = ticker.MaxNLocator(nbins=5)
    cb.formatter = ticker.FormatStrFormatter(fmt)
    cb.update_ticks()
    return cb


def plot_conn_matrix(ax, data, title, cmap, cb_label, fmt="%.2f", vmin=None, vmax=None):
    """Plot a connectivity matrix (weights or tract lengths) on ax with standard labels."""
    im = ax.imshow(data, aspect="auto", cmap=cmap, interpolation="nearest",
                   vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=TITLE_SIZE, fontweight="bold", pad=12, color=C_TEXT)
    ax.set_xlabel("Target ROI", fontsize=LABEL_SIZE, labelpad=8)
    ax.set_ylabel("Source ROI", fontsize=LABEL_SIZE, labelpad=8)
    ax.tick_params(labelsize=TICK_SIZE)
    add_colorbar(im, ax, cb_label, fmt=fmt)


def plot_threshold_matrices(weights_log, percentiles, out_path, save_arrays=False):
    n_thr = len(percentiles)
    fig, axes = plt.subplots(1, n_thr, figsize=(5 * n_thr, 5))
    if n_thr == 1:
        axes = [axes]
    fig.patch.set_facecolor("white")
    for ax in axes:
        ax.set_facecolor("white")

    out_dir = os.path.dirname(out_path)
    for ax, pct in zip(axes, percentiles):
        w_thr   = MouseConnectivity.threshold(weights_log, pct)
        n_edges = int((w_thr > 0).sum())
        plot_conn_matrix(ax, w_thr, f"pct ≥ {pct}%\n({n_edges} edges)",
                         CMAP_W, "weight [0, 1]", vmin=0, vmax=1)
        if save_arrays:
            arr_path = os.path.join(out_dir, f"weights_thr{pct}pct.npy")
            np.save(arr_path, w_thr)
            print(f"Saved {arr_path}")

    fig.suptitle("Log-scaled weights — percentile thresholding",
                 fontsize=TITLE_SIZE + 1, fontweight="bold", color=C_TEXT)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_conn_matrices(weights_raw, weights_log, tract_lengths, n, out_path):
    weights_raw_norm = weights_raw / weights_raw.max()
    w_raw_nz = weights_raw_norm[weights_raw_norm > 0].ravel()
    w_log_nz = weights_log[weights_log > 0].ravel()
    tl_nz    = tract_lengths[tract_lengths > 0].ravel()

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.patch.set_facecolor("white")
    for ax in axes.flat:
        ax.set_facecolor("white")

    plot_conn_matrix(axes[0, 0], weights_raw_norm, "Weights matrix (raw, norm.)",
                     CMAP_W, "weight [0, 1]", vmin=0, vmax=1)

    axes[1, 0].hist(w_raw_nz, bins=80, color=C_W, edgecolor="none", alpha=0.85)
    add_vlines_legend(axes[1, 0], w_raw_nz, fmt=".4f")
    axes[1, 0].set_xlabel("Raw weight", fontsize=LABEL_SIZE, labelpad=8)
    axes[1, 0].set_ylabel("Count", fontsize=LABEL_SIZE, labelpad=8)
    axes[1, 0].set_title("Weights distribution (raw, non-zero)",
                          fontsize=TITLE_SIZE, fontweight="bold", pad=12)
    style_ax(axes[1, 0])

    plot_conn_matrix(axes[0, 1], weights_log, "Weights matrix (log-scaled)",
                     CMAP_W, "weight [0, 1]", vmin=0, vmax=1)

    axes[1, 1].hist(w_log_nz, bins=80, color=C_W, edgecolor="none", alpha=0.85)
    add_vlines_legend(axes[1, 1], w_log_nz, fmt=".2f", loc="upper left")
    axes[1, 1].set_xlabel("Log-scaled weight", fontsize=LABEL_SIZE, labelpad=8)
    axes[1, 1].set_ylabel("Count", fontsize=LABEL_SIZE, labelpad=8)
    axes[1, 1].set_title("Weights distribution (log-scaled, non-zero)",
                          fontsize=TITLE_SIZE, fontweight="bold", pad=12)
    style_ax(axes[1, 1])

    plot_conn_matrix(axes[0, 2], tract_lengths, "Tract lengths matrix (mm)",
                     CMAP_TL, "mm")

    axes[1, 2].hist(tl_nz, bins=80, color=C_TL, edgecolor="none", alpha=0.85)
    add_vlines_legend(axes[1, 2], tl_nz, fmt=".2f")
    axes[1, 2].set_xlabel("Tract length (mm)", fontsize=LABEL_SIZE, labelpad=8)
    axes[1, 2].set_ylabel("Count", fontsize=LABEL_SIZE, labelpad=8)
    axes[1, 2].set_title("Tract lengths distribution (non-zero, mm)",
                          fontsize=TITLE_SIZE, fontweight="bold", pad=12)
    style_ax(axes[1, 2])

    fig.suptitle(f"Structural connectivity — Allen {n} ROIs",
                 fontsize=20, fontweight="bold", color=C_TEXT)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_network_3d(weights_raw, centres, macros, surf_pts, out_path):
    n      = weights_raw.shape[0]
    w_flat = weights_raw[weights_raw > 0]
    thresh = np.percentile(w_flat, 90) if len(w_flat) else 0
    w_max  = weights_raw.max()

    colors_3d = [MACRO_COLORS.get(m, "#BDBDBD") for m in macros]

    fig = plt.figure(figsize=(14, 12))
    fig.patch.set_facecolor("white")
    ax3d = fig.add_subplot(111, projection="3d")
    ax3d.set_facecolor("white")

    ax3d.scatter(surf_pts[:, 0], surf_pts[:, 1], surf_pts[:, 2],
                 c="lightsteelblue", s=1.0, alpha=0.06, linewidths=0)

    x, y, z = centres[:, 0], centres[:, 1], centres[:, 2]
    for i in range(n):
        for j in range(i + 1, n):
            w = weights_raw[i, j]
            if w < thresh:
                continue
            alpha = float(np.clip(
                (w - thresh) / (w_max - thresh + 1e-9) * 0.55 + 0.08, 0.08, 0.65
            ))
            ax3d.plot([x[i], x[j]], [y[i], y[j]], [z[i], z[j]],
                      color="#999999", alpha=alpha, linewidth=0.7, zorder=1)

    ax3d.scatter(x, y, z, c=colors_3d, s=70, edgecolors="white", linewidths=0.5,
                 depthshade=False, zorder=5)

    ax3d.set_axis_off()
    for pane in [ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor((0, 0, 0, 0))

    ax3d.set_title("Structural Connectome — 3D Network",
                   fontsize=18, fontweight="bold", color=C_TEXT, pad=6)

    present = sorted(set(macros),
                     key=lambda m: MACRO_LIST.index(m) if m in MACRO_LIST else 99)
    handles = [mpatches.Patch(color=MACRO_COLORS.get(m, "#BDBDBD"), label=m)
               for m in present]
    leg = ax3d.legend(handles=handles, loc="upper left", frameon=True,
                      framealpha=0.9, edgecolor="#DDDDDD",
                      fontsize=10, title="Macroarea", title_fontsize=11,
                      labelcolor=C_TEXT, bbox_to_anchor=(0.0, 1.0),
                      handlelength=1.2, handleheight=1.2)
    leg.get_title().set_color(C_TEXT)

    ax3d.view_init(elev=90, azim=-90)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_macroarea_distribution(labels, macros, out_path):
    seen   = set()
    counts = Counter()
    for lb, macro in zip(labels, macros):
        clean = lb.replace("Left ", "").replace("Right ", "")
        if clean in seen:
            continue
        seen.add(clean)
        counts[macro] += 1

    counts_bi  = {m: counts.get(m, 0) * 2 for m in MACRO_LIST}
    macro_plot = sorted(
        [m for m in MACRO_LIST if counts_bi[m] > 0],
        key=lambda m: counts_bi[m],
    )
    vals       = [counts_bi[m] for m in macro_plot]
    bar_colors = [MACRO_COLORS[m] for m in macro_plot]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    y_pos = np.arange(len(macro_plot))
    ax.barh(y_pos, vals, height=0.55, color=bar_colors, edgecolor="none", zorder=2)

    for i, v in enumerate(vals):
        ax.text(v + 0.4, y_pos[i], str(v), ha="left", va="center",
                fontsize=10.5, color=C_TEXT, fontweight="bold")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(macro_plot, fontsize=12.5)
    ax.set_xlabel("Number of ROIs (bilateral)", fontsize=LABEL_SIZE, labelpad=8)
    ax.xaxis.set_tick_params(labelsize=TICK_SIZE)
    ax.set_xlim(0, max(vals) * 1.14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.spines["bottom"].set_color(SPINE_C)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("ROI distribution per Macroarea", fontsize=TITLE_SIZE,
                 fontweight="bold", pad=14, color=C_TEXT)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":

    print("Loading connectivity…")
    mc       = MouseConnectivity(CONNECTIVITY_FILE, voxel_size_mm=VOXEL_MM)
    _labels  = mc.region_labels
    _weights = mc.weights
    _centres = mc.preprocess_centres(mc.centres)
    _tracts  = mc.preprocess_tract_lengths(mc.tract_lengths)

    keep          = np.array([lb not in EXCLUDE_LABELS for lb in _labels])
    weights_raw   = _weights[np.ix_(keep, keep)]
    tract_lengths = _tracts[np.ix_(keep, keep)]
    centres       = _centres[keep]
    labels        = _labels[keep]
    n             = len(labels)
    print(f"  {n} regions after filtering")

    weights_log = MouseConnectivity.log_scale(weights_raw)

    print("Assigning macroareas…")
    region_map = RegionMap.load_json(JSON_FILE)
    macros     = [get_macro(lb, region_map) for lb in labels]

    THRESHOLDS = [20, 40, 60, 75, 95]

    plot_conn_matrices(
        weights_raw, weights_log, tract_lengths, n,
        out_path=os.path.join(BASE, "fig_conn_matrices.png"),
    )
    plot_network_3d(
        weights_raw, centres, macros,
        surf_pts=brain_surface_points(RVM_FILE),
        out_path=os.path.join(BASE, "fig_network_3d.png"),
    )
    plot_macroarea_distribution(
        labels, macros,
        out_path=os.path.join(BASE, "fig_regions_per_macroarea.png"),
    )
    plot_threshold_matrices(
        weights_log, THRESHOLDS,
        out_path=os.path.join(BASE, "fig_threshold_matrices.png"),
    )
