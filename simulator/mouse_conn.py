from __future__ import annotations

import os
from collections import Counter

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.ndimage import binary_erosion
from mpl_toolkits.mplot3d import Axes3D

DCN_KW = ("fastigial", "interposed", "dentate nucleus",
          "vestibulocerebellar")
CEREB_CORTEX_KW = ("lingula", "central lobule", "culmen", "declive",
                   "folium", "pyramus", "uvula", "nodulus", "simple lobule",
                   "ansiform", "crus", "paramedian", "copula",
                   "paraflocculus", "flocculus")

# Coarse macroarea classification
MACRO_RULES = [
    ("Cerebellar cortex", list(CEREB_CORTEX_KW)),
    ("Deep cereb. nuclei", list(DCN_KW)),
    ("Inferior olive", ["inferior olivary complex"]),
    ("Pons", ["pons,"]),
    ("Lateral reticular n.", ["lateral reticular nucleus"]),
    ("Trigeminal/Facial/Retic.", ["trigeminal", "facial motor",
                                  "reticular nucleus"]),
    ("Thalamus", ["thalamus,"]),
    ("Midbrain", ["midbrain,"]),
    ("Hypothalamus", ["hypothalamic", "periventricular"]),
    ("Striatum/Pallidum", ["striatum", "pallidum", "septal",
                           "amygdalar nuclei"]),
    ("Hippocampal formation",
     ["ammon", "dentate gyrus", "subiculum", "entorhinal",
      "parasubiculum", "postsubiculum", "presubiculum"]),
    ("Olfactory", ["olfactory", "piriform", "taenia", "endopiriform",
                   "lateral olfactory", "peduncular"]),
    ("Isocortex", []),  # fallback bucket, assigned last
]


def tvb_cereb_preprocessing(mc, zero_internal=True, verbose=True):
    """Make a freshly-loaded ``MouseConnectivity`` ready for the TVB cerebellar
    co-simulation. Operates on the RAW weights (before any log transform) and
    MUTATES ``mc.weights`` in place, returning ``mc``. Call it right after
    constructing the MouseConnectivity instance and before get_tvb_connectivity().

    Steps:

    0. ALWAYS zero the diagonal (no self-connections), independent of
       ``zero_internal``.

    1. (if ``zero_internal``) zero the cerebellum-INTERNAL links that are voxel
       adjacency artefacts and/or already modelled by the NEST/BSB network,
       while KEEPING the physiologically relevant cortico-nuclear output:
           - DCN -> cerebellar cortex   (nucleo-cortical)       ZEROED
           - cerebellar cortex <-> cortex                       ZEROED
           - DCN <-> DCN                                         ZEROED
           - cerebellar cortex -> DCN   (Purkinje -> nucleus)   KEPT

    2. TRANSPOSE source<->target. The Knox/Allen builders (conn_data/
       knox_voxel_conn.py, AllenConnectomeBuilder.py) store weights as
       ``rows=source, cols=target``. TVB's coupling instead consumes
       ``weights[target, source]`` (node i's input = sum_j weights[i,j]*state_j,
       verified in tvb/simulator/coupling.py). Without this transpose the
       simulated directionality is reversed. After this call ``mc.weights`` has
       ROWS = TARGET, as TVB expects.
    """
    w = MouseConnectivity.preprocess_matrix(
        mc.weights, mc.region_labels, zero_internal=zero_internal,
        transpose=True, do_log=False, verbose=verbose)
    mc.set_weights(w)
    return mc


class MouseConnectivity:
    """
    Loads mouse structural connectivity from an Allen Atlas HDF5 file
    and exposes it as a TVB Connectivity object.

    Parameters
    ----------
    path : str
        Path to the HDF5 connectivity file (weights, tract_lengths,
        region_labels, centres).
    speed : float
        Conduction speed in m/s (default: 4.0, Rabuffo 2021).
    weight_mode : str
        How to preprocess weights before building the TVB connectivity:
        - "raw"       : zero diagonal only (default)
        - "log_scale" : log-transform + normalize to [0, 1]
    voxel_size_mm : float or None
        Voxel size in mm. If provided, both tract lengths and centres are
        converted from voxel units to mm. Use 0.025 for the Allen 25-µm
        atlas. Default: None (no conversion).
    exclude_labels : iterable of str or None
        Region labels to discard right after loading. Matching rows/cols
        are removed from weights and tract_lengths, and the corresponding
        entries are removed from region_labels and centres. Default: None.
    region_map : voxcell.RegionMap or None
        If provided, each (post-exclusion) region is automatically
        classified into a macroarea (see MACRO_LIST) and an Allen
        ontology acronym, exposed as self.macros/self.acronyms, aligned
        with self.region_labels. Default: None (both stay None).
    """

    WEIGHT_MODES = ("raw", "log_scale")

    # Canonical keyword lists / macro rules, exposed as class attributes so other
    # scripts can reuse the single definition (the module globals of the same
    # name are the source; see top of file).
    DCN_KW = DCN_KW
    CEREB_CORTEX_KW = CEREB_CORTEX_KW
    MACRO_RULES = MACRO_RULES

    MACRO_LIST = [
        "Isocortex", "Olfactory areas", "Hippocampal formation",
        "Cortical subplate", "Striatum", "Pallidum", "Thalamus",
        "Hypothalamus", "Midbrain", "Pons", "Medulla",
        "Cerebellum", "Brain stem",
    ]

    def __init__(
            self,
            path: str,
            speed: float = 4.0,
            weight_mode: str = "raw",
            voxel_size_mm: float = None,
            exclude_labels=None,
            region_map=None,
    ):
        if weight_mode not in self.WEIGHT_MODES:
            raise ValueError(f"weight_mode must be one of {self.WEIGHT_MODES}")
        self.path = path
        self.speed = speed
        self.weight_mode = weight_mode
        self.voxel_size_mm = voxel_size_mm
        self.exclude_labels = set(exclude_labels) if exclude_labels else set()
        self.region_map = region_map
        self.connectivity = None
        self.delays = None
        self._load()
        self._apply_exclusions()
        self._assign_macroareas()

    # ── raw loading ───────────────────────────────────────────────────────────

    def _load(self):
        with h5py.File(self.path, "r") as f:
            self.region_labels = np.array(f["region_labels"][()]).astype("<U128")
            self.centres       = np.array(f["centres"][()])
            self.weights       = np.array(f["weights"][()],       dtype=np.float64)
            self.tract_lengths = np.array(f["tract_lengths"][()], dtype=np.float64)
        np.fill_diagonal(self.weights, 0.0)

    def _apply_exclusions(self):
        """Drop regions in self.exclude_labels from all loaded arrays."""
        if not self.exclude_labels:
            return
        keep = np.array([lb not in self.exclude_labels for lb in self.region_labels])
        self.region_labels = self.region_labels[keep]
        self.centres       = self.centres[keep]
        self.weights       = self.weights[np.ix_(keep, keep)]
        self.tract_lengths = self.tract_lengths[np.ix_(keep, keep)]

    def _assign_macroareas(self):
        """Classify each (post-exclusion) region into a macroarea/acronym using region_map."""
        if self.region_map is None:
            self.macros = None
            self.acronyms = None
            return
        self.macros = np.array(
            [self.get_macro(lb, self.region_map) for lb in self.region_labels]
        )
        self.acronyms = np.array(
            [self.get_acronym(lb, self.region_map) for lb in self.region_labels]
        )
    def set_delays(self):
        self.delays = self.tract_lengths / self.speed

    @staticmethod
    def get_macro(label: str, region_map) -> str:
        """Walk up the region_map hierarchy from label to its macroarea in MACRO_LIST."""
        clean = label.replace("Left ", "").replace("Right ", "")
        ids = region_map.find(clean, "name", with_descendants=False)
        if not ids:
            return "Unknown"
        rid = int(next(iter(ids)))
        pid = region_map.get(rid, "parent_structure_id")
        pname = region_map.get(pid, "name")
        for _ in range(20):
            if pname in MouseConnectivity.MACRO_LIST:
                return pname
            pid   = region_map.get(pid, "parent_structure_id")
            pname = region_map.get(pid, "name")
        return "Unknown"

    @staticmethod
    def get_acronym(label: str, region_map) -> str:
        """Look up the Allen ontology acronym for a '<Hemisphere> <Region name>' label."""
        clean = label.replace("Left ", "").replace("Right ", "")
        ids = region_map.find(clean, "name", with_descendants=False)
        if not ids:
            return label
        rid = int(next(iter(ids)))
        return region_map.get(rid, "acronym")

    # ── preprocessing ─────────────────────────────────────────────────────────

    @staticmethod
    def threshold(w_in: np.ndarray, pct: float) -> np.ndarray:
        """Zero weights below the pct-th percentile of non-zero values."""
        nz = w_in[w_in > 0]
        if len(nz) == 0:
            return w_in.copy()
        thr = np.percentile(nz, pct)
        out = w_in.copy()
        out[out < thr] = 0.0
        return out

    @staticmethod
    def log_scale(w_in: np.ndarray) -> np.ndarray:
        """
        Log-transform weights and normalize to [0, 1]:
          1. NaN → 0
          2. Divide by min positive  → min_positive = 1
          3. Multiply by e           → min_positive = e
          4. log(w) for positives    → min_positive = 1
          5. Zeros stay 0
          6. Divide by max           → normalize to [0, 1]
        """
        w = w_in.copy()
        w[np.isnan(w)] = 0.0
        w0         = w <= 0
        w_positive = w > 0
        w         /= w[w_positive].min()
        w         *= np.exp(1)
        w[w_positive] = np.log(w[w_positive])
        w[w0]     = 0.0
        w         /= w.max()
        return w

    # ── cerebellar region classification (pure-numpy, label-based) ──────────────

    @staticmethod
    def idx_of_any(labels, keys) -> np.ndarray:
        """Indices whose (case-insensitive) label contains any of ``keys``."""
        kl = [k.lower() for k in keys]
        return np.array([i for i, l in enumerate(labels)
                         if any(k in str(l).lower() for k in kl)], dtype=int)

    @staticmethod
    def classify_cerebellum(labels):
        """(cortex_idx, dcn_idx) for the Knox labels — the single source of truth
        for the cortex/DCN split used across all scripts (case-insensitive)."""
        return (MouseConnectivity.idx_of_any(labels, CEREB_CORTEX_KW),
                MouseConnectivity.idx_of_any(labels, DCN_KW))

    @staticmethod
    def classify_macro(label: str) -> str:
        """Coarse macroarea of one label by keyword (MACRO_RULES, no region_map)."""
        ll = str(label).lower()
        for name, keys in MACRO_RULES:
            if keys and any(k in ll for k in keys):
                return name
        return "Isocortex"

    @staticmethod
    def macros_of(labels) -> np.ndarray:
        return np.array([MouseConnectivity.classify_macro(l) for l in labels])

    # ── matrix preprocessing (pure-numpy, single source of truth) ───────────────

    @staticmethod
    def zero_diagonal(W: np.ndarray) -> np.ndarray:
        W = np.asarray(W, dtype=float).copy()
        np.fill_diagonal(W, 0.0)
        return W

    @staticmethod
    def zero_internal_blocks(W: np.ndarray, cortex, dcn) -> np.ndarray:
        """Zero cerebellum-INTERNAL blocks on a rows=SOURCE matrix, keeping the
        cortico-nuclear output (cortex->DCN). Returns a copy."""
        W = np.asarray(W, dtype=float).copy()
        if len(cortex) and len(dcn):
            W[np.ix_(dcn, cortex)] = 0.0         # DCN -> cortex      (zeroed)
            W[np.ix_(cortex, cortex)] = 0.0      # cortex <-> cortex  (zeroed)
            W[np.ix_(dcn, dcn)] = 0.0            # DCN <-> DCN        (zeroed)
            # cortex -> DCN (W[cortex, dcn]) intentionally KEPT
        return W

    @staticmethod
    def preprocess_matrix(W, labels, zero_internal=True, transpose=True,
                          do_log=False, verbose=False) -> np.ndarray:
        """TVB-ready cerebellar matrix from a RAW rows=SOURCE matrix. Single
        source of truth shared by tvb_cereb_preprocessing(), explore_conn and
        plot_cerebellar_volumes_3d.

          0. ALWAYS zero the diagonal (no self-connections).
          1. (zero_internal) zero DCN->cortex, cortex<->cortex, DCN<->DCN
             (KEEP cortex->DCN).
          2. (transpose) source->target so rows=TARGET (TVB's coupling consumes
             weights[target, source]).
          3. (do_log) log_scale to [0, 1].
        """
        cortex, dcn = MouseConnectivity.classify_cerebellum(labels)
        w = MouseConnectivity.zero_diagonal(W)
        n_dcn_ctx = 0
        if zero_internal:
            n_dcn_ctx = (int(np.count_nonzero(w[np.ix_(dcn, cortex)]))
                         if len(cortex) and len(dcn) else 0)
            w = MouseConnectivity.zero_internal_blocks(w, cortex, dcn)
        if transpose:
            w = w.T                               # rows = TARGET
        if do_log:
            w = MouseConnectivity.log_scale(w)
        if verbose:
            print(f"[preprocess_matrix] {len(cortex)} cortex, {len(dcn)} nuclei; "
                  f"zeroed diagonal" +
                  (f" + {n_dcn_ctx} DCN->cortex + cortex<->cortex + DCN<->DCN "
                   f"(kept cortex->DCN)" if zero_internal else "") +
                  f"; transpose={transpose} "
                  f"(rows={'TARGET' if transpose else 'SOURCE'}); log={do_log}")
        return w

    # ── weight normalization for simulation (see the stability discussion) ──────

    @staticmethod
    def threshold_abs(W: np.ndarray, floor: float) -> np.ndarray:
        """Zero every weight <= ``floor`` (drops the regression noise floor)."""
        W = np.asarray(W, dtype=float).copy()
        W[W <= floor] = 0.0
        return W

    @staticmethod
    def spectral_radius(W: np.ndarray) -> float:
        return float(np.max(np.abs(np.linalg.eigvals(np.asarray(W, dtype=float)))))

    @staticmethod
    def normalize(W, method="floor_log", floor=1e-4):
        """Weight normalization for TVB/Montbrio (see the stability analysis).

          - "none"      : unchanged.
          - "floor_log" : zero the noise floor (<= ``floor``) THEN log_scale to
                          [0,1]. Fixes the "log makes every weight look the same"
                          problem (that collapse comes from the 1e-23 regression
                          floor stretching the log range) and yields a near-
                          Gaussian, non-negative, bounded weight distribution.
          - "log"       : plain log_scale (no thresholding).
          - "spectral"  : W / spectral_radius(W)  -> spectral radius 1
                          (controls the critical G directly).
          - "row"       : row-normalize in-strength -> every node's total input 1.
        """
        W = np.asarray(W, dtype=float)
        if method == "none":
            return W.copy()
        if method == "floor_log":
            return MouseConnectivity.log_scale(
                MouseConnectivity.threshold_abs(W, floor))
        if method == "log":
            return MouseConnectivity.log_scale(W)
        if method == "spectral":
            r = MouseConnectivity.spectral_radius(W)
            return W / r if r > 0 else W.copy()
        if method == "row":
            s = W.sum(axis=1, keepdims=True)
            return np.divide(W, s, out=np.zeros_like(W), where=s > 0)
        raise ValueError(f"unknown normalize method: {method}")

    def preprocess_tract_lengths(self, tl: np.ndarray) -> np.ndarray:
        """Convert tract lengths from voxel units to mm if voxel_size_mm is set.
        For confimation about the expected ranges in mm see """
        if self.voxel_size_mm is not None:
            return tl * self.voxel_size_mm
        return tl

    def preprocess_centres(self, centres: np.ndarray) -> np.ndarray:
        """Convert centres from voxel units to mm if voxel_size_mm is set."""
        if self.voxel_size_mm is not None:
            return centres * self.voxel_size_mm
        return centres

    def set_weights(self, weights: np.ndarray):
        """
        Replace self.weights with an externally provided/already-loaded
        weights matrix (e.g. a matrix produced and saved by a previous
        run), instead of re-reading it from the original HDF5 file.
        """
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != self.weights.shape:
            raise ValueError(
                f"weights shape {weights.shape} does not match "
                f"current shape {self.weights.shape}"
            )
        self.weights = weights

    def threshold_weights(self, pct: float) -> np.ndarray:
        """
        Apply percentile threshold (see threshold()) to self.weights,
        update self.weights in place, and if a TVB Connectivity object
        has already been built via get_tvb_connectivity(), refresh its
        weights too (re-running configure()).
        """
        self.weights = self.threshold(self.weights, pct)
        if self.connectivity is not None:
            self.connectivity.weights = self.weights.copy()
            self.connectivity.configure()
        return self.weights

    # ── TVB connectivity ──────────────────────────────────────────────────────

    def get_tvb_connectivity(self):
        """
        Build and return a configured TVB Connectivity object applying
        the weight_mode and tract_length preprocessing selected at init.
        The result is cached on self.connectivity so that subsequent
        calls to threshold_weights() can update it in place.
        """
        from tvb.datatypes import connectivity   # lazy: keeps module TVB-free
        w  = self.log_scale(self.weights) if self.weight_mode == "log_scale" \
            else self.weights.copy()
        tl = self.preprocess_tract_lengths(self.tract_lengths)
        c  = self.preprocess_centres(self.centres)

        conn = connectivity.Connectivity()
        conn.weights       = w
        conn.centres       = c
        conn.region_labels = self.region_labels
        conn.tract_lengths = tl
        conn.speed         = np.r_[self.speed]
        conn.configure()
        self.connectivity = conn
        return conn

    # ── cerebellar indices ────────────────────────────────────────────────────

    def get_cereb_indices(self, cereb_labels: np.ndarray) -> np.ndarray:
        """Return indices of cerebellar regions within region_labels."""
        cereb_set = set(cereb_labels)
        return np.array([
            i for i, r in enumerate(self.region_labels)
            if r in cereb_set
        ])


class PlotConn:
    """
    Plotting helpers for MouseConnectivity data: connectivity matrices,
    weight/tract-length distributions, the 3D connectome and the
    macroarea breakdown.
    """

    MACRO_LIST = MouseConnectivity.MACRO_LIST

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
        "Pons":                  "#8B4513",
        "Medulla":               "#FF5500",
        "Cerebellum":            "#00D4FF",
        "Brain stem":            "#4CAF50",
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

    @property
    def C_W(self):
        return plt.cm.Blues(0.72)

    @property
    def C_TL(self):
        return plt.cm.YlGn(0.72)

    # ── styling helpers ───────────────────────────────────────────────────────

    def style_ax(self, ax):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(self.SPINE_C)
        ax.spines["bottom"].set_color(self.SPINE_C)
        ax.spines["left"].set_linewidth(0.6)
        ax.spines["bottom"].set_linewidth(0.6)
        ax.tick_params(axis="both", labelsize=self.TICK_SIZE, color=self.SPINE_C)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    def add_vlines_legend(self, ax, data, fmt=".4f", loc="upper right"):
        mean   = data.mean()
        median = np.median(data)
        ax.axvline(mean,   color=self.C_MEAN,   linewidth=1.8, linestyle="--", zorder=3)
        ax.axvline(median, color=self.C_MEDIAN, linewidth=1.8, linestyle=":",  zorder=3)
        elems = [
            Line2D([0], [0], color=self.C_MEAN,   linewidth=1.8, linestyle="--",
                   label=f"Mean = {mean:{fmt}}"),
            Line2D([0], [0], color=self.C_MEDIAN, linewidth=1.8, linestyle=":",
                   label=f"Median = {median:{fmt}}"),
        ]
        ax.legend(handles=elems, frameon=True, framealpha=0.9,
                  edgecolor="#CCCCCC", fontsize=11, loc=loc)

    def add_colorbar(self, im, ax, label, fmt="%.2f"):
        cb = plt.colorbar(im, ax=ax, shrink=0.85)
        cb.set_label(label, fontsize=12)
        cb.ax.tick_params(labelsize=11)
        cb.locator   = ticker.MaxNLocator(nbins=5)
        cb.formatter = ticker.FormatStrFormatter(fmt)
        cb.update_ticks()
        return cb

    @staticmethod
    def brain_surface_points(rvm_path, downsample=3, voxel_mm=0.025, n_max=8000):
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

    # ── matrix plots ──────────────────────────────────────────────────────────

    def plot_conn_matrix(self, ax, data, title, cmap, cb_label, fmt="%.2f", vmin=None, vmax=None):
        """Plot a connectivity matrix (weights or tract lengths) on ax with standard labels."""
        im = ax.imshow(data, aspect="auto", cmap=cmap, interpolation="nearest",
                       vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=self.TITLE_SIZE, fontweight="bold", pad=12, color=self.C_TEXT)
        ax.set_xlabel("Target ROI", fontsize=self.LABEL_SIZE, labelpad=8)
        ax.set_ylabel("Source ROI", fontsize=self.LABEL_SIZE, labelpad=8)
        ax.tick_params(labelsize=self.TICK_SIZE)
        self.add_colorbar(im, ax, cb_label, fmt=fmt)

    def plot_matrix(self, data, title, cmap, cb_label, out_path=None, fmt="%.2f", vmin=None, vmax=None, save=False):
        """Plot a single connectivity matrix (weights or tract lengths); save it if `save=True`."""
        fig, ax = plt.subplots(figsize=(7, 6))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        self.plot_conn_matrix(ax, data, title, cmap, cb_label, fmt=fmt, vmin=vmin, vmax=vmax)
        fig.tight_layout()
        plt.show()
        if save:
            if out_path is None:
                raise ValueError("out_path is required when save=True")
            fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            print(f"Saved {out_path}")
        return fig, ax

    def plot_histogram(self, data, title, xlabel, out_path=None, color=None, fmt=".2f", loc="upper right", save=False):
        """Plot the distribution of the non-zero values of a single matrix; save it if `save=True`."""
        nz = data[data > 0].ravel()
        fig, ax = plt.subplots(figsize=(7, 5))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        ax.hist(nz, bins=80, color=color or self.C_W, edgecolor="none", alpha=0.85)
        self.add_vlines_legend(ax, nz, fmt=fmt, loc=loc)
        ax.set_xlabel(xlabel, fontsize=self.LABEL_SIZE, labelpad=8)
        ax.set_ylabel("Count", fontsize=self.LABEL_SIZE, labelpad=8)
        ax.set_title(title, fontsize=self.TITLE_SIZE, fontweight="bold", pad=12)
        self.style_ax(ax)
        fig.tight_layout()
        plt.show()
        if save:
            if out_path is None:
                raise ValueError("out_path is required when save=True")
            fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            print(f"Saved {out_path}")
        return fig, ax

    def plot_network_3d(self, weights_raw, centres, macros, surf_pts, out_path=None, save=False):
        n      = weights_raw.shape[0]
        w_flat = weights_raw[weights_raw > 0]
        thresh = np.percentile(w_flat, 90) if len(w_flat) else 0
        w_max  = weights_raw.max()

        colors_3d = [self.MACRO_COLORS.get(m, "#BDBDBD") for m in macros]

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
                       fontsize=18, fontweight="bold", color=self.C_TEXT, pad=6)

        present = sorted(set(macros),
                         key=lambda m: self.MACRO_LIST.index(m) if m in self.MACRO_LIST else 99)
        handles = [mpatches.Patch(color=self.MACRO_COLORS.get(m, "#BDBDBD"), label=m)
                   for m in present]
        leg = ax3d.legend(handles=handles, loc="upper left", frameon=True,
                          framealpha=0.9, edgecolor="#DDDDDD",
                          fontsize=10, title="Macroarea", title_fontsize=11,
                          labelcolor=self.C_TEXT, bbox_to_anchor=(0.0, 1.0),
                          handlelength=1.2, handleheight=1.2)
        leg.get_title().set_color(self.C_TEXT)

        ax3d.view_init(elev=90, azim=-90)

        fig.tight_layout()
        plt.show()
        if save:
            if out_path is None:
                raise ValueError("out_path is required when save=True")
            fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            print(f"Saved {out_path}")
        return fig, ax3d

    def plot_macroarea_distribution(self, labels, macros, out_path=None, save=False):
        seen   = set()
        counts = Counter()
        for lb, macro in zip(labels, macros):
            clean = lb.replace("Left ", "").replace("Right ", "")
            if clean in seen:
                continue
            seen.add(clean)
            counts[macro] += 1

        counts_bi  = {m: counts.get(m, 0) * 2 for m in self.MACRO_LIST}
        macro_plot = sorted(
            [m for m in self.MACRO_LIST if counts_bi[m] > 0],
            key=lambda m: counts_bi[m],
        )
        vals       = [counts_bi[m] for m in macro_plot]
        bar_colors = [self.MACRO_COLORS[m] for m in macro_plot]

        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        y_pos = np.arange(len(macro_plot))
        ax.barh(y_pos, vals, height=0.55, color=bar_colors, edgecolor="none", zorder=2)

        for i, v in enumerate(vals):
            ax.text(v + 0.4, y_pos[i], str(v), ha="left", va="center",
                    fontsize=10.5, color=self.C_TEXT, fontweight="bold")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(macro_plot, fontsize=12.5)
        ax.set_xlabel("Number of ROIs (bilateral)", fontsize=self.LABEL_SIZE, labelpad=8)
        ax.xaxis.set_tick_params(labelsize=self.TICK_SIZE)
        ax.set_xlim(0, max(vals) * 1.14)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_linewidth(0.8)
        ax.spines["bottom"].set_color(self.SPINE_C)
        ax.tick_params(axis="y", length=0)
        ax.xaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
        ax.set_title("ROI distribution per Macroarea", fontsize=self.TITLE_SIZE,
                     fontweight="bold", pad=14, color=self.C_TEXT)

        fig.tight_layout()
        plt.show()
        if save:
            if out_path is None:
                raise ValueError("out_path is required when save=True")
            fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            print(f"Saved {out_path}")
        return fig, ax