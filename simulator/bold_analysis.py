import os
import warnings
from copy import deepcopy
from typing import List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
import tqdm
import vbi
from vbi.feature_extraction.calc_features import calc_features as _vbi_calc_features
from vbi.feature_extraction.features_settings import load_json as _vbi_load_json
from voxcell import RegionMap

from simulator.mouse_conn import MouseConnectivity

VBI_FEATURES_JSON = os.path.join(os.path.dirname(vbi.__file__), "feature_extraction", "features.json")


class BoldTimeSeries:
    """
    Generic loader/container for a BOLD time series stored in a .mat file.

    Works for both simulated and empirical recordings: the .mat file only
    needs to hold a single (n_timepoints, n_regions) array. Region labels
    are derived from the ACRO/NAME/ID_ALLEN columns of an atlas spreadsheet
    (e.g. emp_data/macro_atlas_n_172_rois_excel_final.xlsx) and harmonized
    against the Allen ontology (e.g. conn_data/1.json, loaded via
    voxcell.RegionMap) so they follow the same "<Hemisphere> <Region name>"
    convention used by MouseConnectivity.region_labels (simulator/mouse_conn.py).
    Each region is also classified into a macroarea (see
    MouseConnectivity.MACRO_LIST/get_macro) and exposed as self.macros,
    aligned with self.region_labels, the same way MouseConnectivity does
    for the structural connectivity.

    Parameters
    ----------
    path : str
        Path to the .mat file.
    tr_ms : float
        Repetition time / dt of the recording, in milliseconds.
    atlas_xlsx : str
        Path to the atlas spreadsheet with columns ID_ALLEN, ACRO, NAME
        and "Position in .mat file" (1-indexed column of the first
        hemisphere block).
    region_map_json : str
        Path to the Allen ontology JSON used to harmonize region names.
    duration_s : float or None
        Total recording duration in seconds. If None (default), it is
        derived from the number of timepoints in the data and tr_ms.
    var_name : str or None
        Name of the array variable inside the .mat file. If None
        (default), the loader uses the only non-metadata ("__"-prefixed)
        key found in the file.
    first_hemisphere : {"Right", "Left"}
        Hemisphere assigned to the first half of the ROI columns; the
        second half gets the other hemisphere, in the same ROI order
        (see emp_data/Data_to_share/README.md). Default "Right", matching
        the Right-then-Left ordering of ConnAllen.h5's region_labels.
        The atlas/README do not state this explicitly: override it if it
        turns out to be reversed for your data.
    """

    def __init__(
        self,
        path: str,
        tr_ms: float,
        atlas_xlsx: str,
        region_map_json: str,
        duration_s: Optional[float] = None,
        var_name: Optional[str] = None,
        first_hemisphere: str = "Right",
    ):
        if first_hemisphere not in ("Right", "Left"):
            raise ValueError('first_hemisphere must be "Right" or "Left"')

        self.path = path
        self.name = os.path.splitext(os.path.basename(path))[0]
        self.tr_ms = tr_ms

        self.ts = self._load_array(path, var_name)
        self.n_timepoints, self.n_regions = self.ts.shape
        self.duration_s = (
            duration_s if duration_s is not None
            else self.n_timepoints * tr_ms / 1000.0
        )

        region_map = RegionMap.load_json(region_map_json)
        self.region_labels = self._build_region_labels(
            atlas_xlsx, region_map, first_hemisphere,
        )
        if len(self.region_labels) != self.n_regions:
            raise ValueError(
                f"{atlas_xlsx} yields {len(self.region_labels)} region "
                f"labels but {path} has {self.n_regions} columns"
            )
        self.macros = [
            MouseConnectivity.get_macro(lab, region_map) for lab in self.region_labels
        ]

    @staticmethod
    def _load_array(path: str, var_name: Optional[str]) -> np.ndarray:
        mat = sio.loadmat(path)
        keys = [k for k in mat if not k.startswith("__")]
        if var_name is not None:
            if var_name not in keys:
                raise KeyError(f"'{var_name}' not found in {path} (available: {keys})")
            key = var_name
        elif len(keys) == 1:
            key = keys[0]
        else:
            raise ValueError(
                f"{path} has multiple candidate variables {keys}; pass var_name explicitly"
            )
        return np.asarray(mat[key], dtype=np.float64)

    @staticmethod
    def _build_region_labels(
        atlas_xlsx: str, region_map: RegionMap, first_hemisphere: str,
    ) -> List[str]:
        df = pd.read_excel(atlas_xlsx).sort_values("Position in .mat file")
        names = [region_map.get(int(rid), "name") for rid in df["ID_ALLEN"]]

        second_hemisphere = "Left" if first_hemisphere == "Right" else "Right"
        return (
            [f"{first_hemisphere} {n}" for n in names]
            + [f"{second_hemisphere} {n}" for n in names]
        )


def compute_fc(ts: np.ndarray) -> np.ndarray:
    """Static Pearson FC matrix (regions × regions)."""
    fc = np.corrcoef(ts.T)
    return np.nan_to_num(fc, nan=0.0, posinf=0.0, neginf=0.0)


def compute_fcd(
        ts: np.ndarray,
        win_len: int,
        win_sp: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sliding-window FCD.

    Returns
    -------
    FCD      : (n_win, n_win)  correlation matrix of windowed FC vectors
    fcs      : (n_win, n_edges) upper-triangle FC per window
    speed_ts : (n_win - 1,)    frame-to-frame dissimilarity
    """
    n_t, n_r   = ts.shape
    triu_idx   = np.triu_indices(n_r, k=1)
    fc_stack   = []
    speed_list = []
    fc_prev    = None

    for t0 in range(0, n_t - win_len, win_sp):
        fc     = np.corrcoef(ts[t0 : t0 + win_len, :].T)
        fc     = np.nan_to_num(fc, nan=0.0, posinf=0.0, neginf=0.0)
        fc_vec = fc[triu_idx]
        fc_stack.append(fc_vec)

        if fc_prev is not None:
            r = np.corrcoef(np.vstack([fc_vec, fc_prev]))[0, 1]
            speed_list.append(1.0 - (r if np.isfinite(r) else 0.0))
        fc_prev = fc_vec

    fcs      = np.array(fc_stack)           # (n_win, n_edges)
    speed_ts = np.array(speed_list)         # (n_win - 1,)
    FCD      = np.corrcoef(fcs)
    FCD      = np.nan_to_num(FCD, nan=0.0, posinf=0.0, neginf=0.0)
    return FCD, fcs, speed_ts


def compute_sfc(sc: np.ndarray, fc: np.ndarray) -> np.ndarray:
    """
    Per-region structural-functional coupling (SFC_i).

    For each region i, the SC and FC matrices are first reduced to their
    i-th row (excluding the self-connection on the diagonal), giving a
    structural and a functional connectivity vector for that region.
    SFC_i is the Pearson correlation coefficient between these two
    vectors. `sc` and `fc` must be square, same shape, and share the same
    region ordering.

    Returns
    -------
    sfc : (n_regions,) array, SFC_i for each region i.
    """
    if sc.shape != fc.shape:
        raise ValueError(f"sc shape {sc.shape} != fc shape {fc.shape}")
    n = sc.shape[0]
    off_diag = ~np.eye(n, dtype=bool)
    sfc = np.empty(n)
    for i in range(n):
        r = np.corrcoef(sc[i, off_diag[i]], fc[i, off_diag[i]])[0, 1]
        sfc[i] = r if np.isfinite(r) else 0.0
    return sfc


def compute_sfc_network(sfc: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
    """
    Network-level structural-functional coupling (SFC_N): the average of
    the per-region SFC_i values (see `compute_sfc`) for the regions
    belonging to network N. `sfc` should already be restricted to that
    network's regions. `weights[i]` is wi, the weight of region i; if
    None (default), regions are averaged unweighted.
    """
    if weights is None:
        return float(np.mean(sfc))
    return float(np.average(sfc, weights=weights))


def get_fmri_feature_cfg(json_path: Optional[str] = None, tags_to_keep: Optional[list] = None) -> dict:
    """
    Build a vbi feature-extraction config restricted to the features in
    features.json tagged "all" or "fmri" (i.e. the metrics that are
    meaningful for fMRI/BOLD data).
    """
    json_path = json_path or VBI_FEATURES_JSON
    dict_features = _vbi_load_json(json_path)

    # default behavior
    if tags_to_keep is None:
        tags_to_keep = {"all", "fmri"}
    else:
        tags_to_keep = set(tags_to_keep)

    cfg = {}

    for domain, feats in dict_features.items():
        kept = {}

        for name, spec in feats.items():
            if spec.get("use") != "yes":
                continue
            tag = spec.get("tag", "all")
            tags = tag if isinstance(tag, list) else [tag]
            tags = set(tags)
            if tags.intersection(tags_to_keep):
                kept[name] = spec
        if kept:
            cfg[domain] = kept

    return cfg


def extract_fmri_features(
    bold: BoldTimeSeries,
    tags_to_keep: Optional[list] = None,
    cfg: Optional[dict] = None,
    tr_s: Optional[float] = None,
) -> Tuple[List[float], List[str], List[str]]:
    """
    Extract vbi's fMRI-relevant features for a single BoldTimeSeries.

    Each feature is computed independently (one vbi.calc_features call
    per feature): a few features in features.json (e.g. calc_mi, calc_te,
    calc_entropy, hmm_stat) require optional dependencies (JPype/Java,
    ssm) that may not be installed. If a feature raises, it is filled
    with NaN instead of aborting the extraction for the whole mouse.

    Parameters
    ----------
    bold : BoldTimeSeries
    tags_to_keep : list or None
        Forwarded to `get_fmri_feature_cfg` when `cfg` is None.
    cfg : dict or None
        Feature-extraction config; defaults to `get_fmri_feature_cfg()`.
    tr_s : float or None
        Repetition time in seconds, used as 1/fs and for fcd_stat's
        window length. Defaults to `bold.tr_ms / 1000.0`.

    Returns
    -------
    values : list of float
    labels : list of str
    failed : list of str
        Names of the features that raised an exception (filled with NaN).
    """

    cfg = cfg if cfg is not None else get_fmri_feature_cfg(tags_to_keep=tags_to_keep)
    tr_s = tr_s if tr_s is not None else bold.tr_ms / 1000.0
    fs = 1.0 / tr_s
    ts = bold.ts.T  # vbi expects [n_regions x n_samples]

    values: List[float] = []
    labels: List[str] = []
    failed: List[str] = []

    for domain, feats in cfg.items():
        for name, spec in feats.items():
            spec = deepcopy(spec)
            params = spec.get("parameters") or {}
            if "TR" in params:
                params["TR"] = tr_s
            single_cfg = {domain: {name: spec}}
            try:
                vals, labs, _ = _vbi_calc_features(ts, fs, single_cfg)
            except Exception as exc:
                warnings.warn(f"Feature '{name}' failed ({exc}); filling with NaN.")
                vals, labs = [np.nan], [name]
                failed.append(name)
            values.extend(vals)
            labels.extend(labs)

    return values, labels, failed


def extract_fmri_features_df(
    bold_list: Sequence[BoldTimeSeries],
    mouse_ids: Optional[Sequence[str]] = None,
    tags_to_keep: Optional[list] = None,
    cfg: Optional[dict] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Extract vbi's fMRI-relevant features for a list of BoldTimeSeries and
    collect them into a (n_mice x n_features) DataFrame indexed by mouse id.

    Parameters
    ----------
    bold_list : sequence of BoldTimeSeries
    mouse_ids : sequence of str or None
    tags_to_keep : list or None
        Forwarded to `get_fmri_feature_cfg` when `cfg` is None.
    cfg : dict or None
        Feature-extraction config; defaults to `get_fmri_feature_cfg(tags_to_keep=tags_to_keep)`.
    verbose : bool
    """
    cfg = cfg if cfg is not None else get_fmri_feature_cfg(tags_to_keep=tags_to_keep)
    mouse_ids = list(mouse_ids) if mouse_ids is not None else [b.name for b in bold_list]

    rows = []
    labels = None
    for bold in tqdm.tqdm(bold_list, disable=not verbose, desc="Extracting fMRI features"):
        values, labels, _failed = extract_fmri_features(bold, cfg=cfg)
        rows.append(values)

    df = pd.DataFrame(rows, columns=labels, index=mouse_ids)
    df.index.name = "mouse_id"
    return df


class PlotBold:
    """
    Plotting helpers for BoldTimeSeries data: raw time-series traces,
    static FC matrices and FCD matrices.
    """

    SPINE_C    = "#333333"
    LABEL_SIZE = 14
    TITLE_SIZE = 16
    TICK_SIZE  = 12

    CMAP_FC  = "RdBu_r"
    CMAP_FCD = "viridis"

    def style_ax(self, ax):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(self.SPINE_C)
        ax.spines["bottom"].set_color(self.SPINE_C)
        ax.tick_params(axis="both", labelsize=self.TICK_SIZE, color=self.SPINE_C)

    def plot_timeseries(
        self,
        bold: BoldTimeSeries,
        region_labels: Union[str, Sequence[str], None] = None,
        out_path: Optional[str] = None,
        title: Optional[str] = None,
        save: bool = False,
    ):
        """
        Plot raw BOLD time series.

        Parameters
        ----------
        bold : BoldTimeSeries
        region_labels : str, sequence of str, or None
            A single region label, a list of region labels, or None
            (default) to plot every region in `bold.region_labels`.
        out_path : str or None
            Where to save the figure. Required if `save=True`.
        save : bool
            If True, save the figure to `out_path` and close it.
            If False (default), leave the figure open and return it.
        """
        if region_labels is None:
            labels = bold.region_labels
        elif isinstance(region_labels, str):
            labels = [region_labels]
        else:
            labels = list(region_labels)

        idx = []
        for lab in labels:
            if lab not in bold.region_labels:
                raise ValueError(f"Unknown region label: {lab}")
            idx.append(bold.region_labels.index(lab))

        t = np.arange(bold.n_timepoints) * bold.tr_ms / 1000.0

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        colors = plt.cm.viridis(np.linspace(0, 1, len(idx)))
        for color, i, lab in zip(colors, idx, labels):
            ax.plot(t, bold.ts[:, i], color=color, linewidth=1.1, alpha=0.85, label=lab)

        ax.set_xlabel("Time (s)", fontsize=self.LABEL_SIZE, labelpad=8)
        ax.set_ylabel("BOLD signal", fontsize=self.LABEL_SIZE, labelpad=8)
        ax.set_title(
            title or f"{bold.name} — BOLD time series",
            fontsize=self.TITLE_SIZE, fontweight="bold", pad=12,
        )
        self.style_ax(ax)
        if len(idx) <= 15:
            ax.legend(fontsize=9, frameon=True, framealpha=0.9, edgecolor="#CCCCCC", loc="upper right")
        fig.tight_layout()

        if save:
            if out_path is None:
                raise ValueError("out_path is required when save=True")
            fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            print(f"Saved {out_path}")
        return fig, ax

    def plot_matrix(self, data, title, cmap, cb_label, out_path=None, vmin=None, vmax=None, save=False):
        """Plot a single matrix (FC or FCD); save it if `save=True`."""
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        im = ax.imshow(data, aspect="auto", cmap=cmap, interpolation="nearest", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=self.TITLE_SIZE, fontweight="bold", pad=10)
        cb = plt.colorbar(im, ax=ax, shrink=0.85)
        cb.set_label(cb_label, fontsize=11)
        fig.tight_layout()
        if save:
            if out_path is None:
                raise ValueError("out_path is required when save=True")
            fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            print(f"Saved {out_path}")
        return fig, ax

    def plot_fc(self, fc, out_path=None, title=None, vmin=-1, vmax=1, save=False):
        """Plot a static FC matrix; vmin/vmax default to [-1, 1] (Pearson r)."""
        return self.plot_matrix(
            fc, title or "Static FC", self.CMAP_FC, "Pearson r",
            out_path=out_path, vmin=vmin, vmax=vmax, save=save,
        )

    def plot_fcd(self, fcd, out_path=None, title=None, vmin=-1, vmax=1, save=False):
        """Plot an FCD matrix; vmin/vmax default to [-1, 1] (FC-vector correlation)."""
        return self.plot_matrix(
            fcd, title or "FCD", self.CMAP_FCD, "FC-vector correlation",
            out_path=out_path, vmin=vmin, vmax=vmax, save=save,
        )
