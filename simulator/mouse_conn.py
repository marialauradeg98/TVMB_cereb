import h5py
import numpy as np
from tvb.datatypes import connectivity


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
    """

    WEIGHT_MODES = ("raw", "log_scale")

    def __init__(
        self,
        path: str,
        speed: float = 4.0,
        weight_mode: str = "raw",
        voxel_size_mm: float = None,
    ):
        if weight_mode not in self.WEIGHT_MODES:
            raise ValueError(f"weight_mode must be one of {self.WEIGHT_MODES}")
        self.path = path
        self.speed = speed
        self.weight_mode = weight_mode
        self.voxel_size_mm = voxel_size_mm
        self._load()

    # ── raw loading ───────────────────────────────────────────────────────────

    def _load(self):
        with h5py.File(self.path, "r") as f:
            self.region_labels = np.array(f["region_labels"][()]).astype("<U128")
            self.centres       = np.array(f["centres"][()])
            self.weights       = np.array(f["weights"][()],       dtype=np.float64)
            self.tract_lengths = np.array(f["tract_lengths"][()], dtype=np.float64)
        np.fill_diagonal(self.weights, 0.0)

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

    def preprocess_tract_lengths(self, tl: np.ndarray) -> np.ndarray:
        """Convert tract lengths from voxel units to mm if voxel_size_mm is set."""
        if self.voxel_size_mm is not None:
            return tl * self.voxel_size_mm
        return tl

    def preprocess_centres(self, centres: np.ndarray) -> np.ndarray:
        """Convert centres from voxel units to mm if voxel_size_mm is set."""
        if self.voxel_size_mm is not None:
            return centres * self.voxel_size_mm
        return centres

    # ── TVB connectivity ──────────────────────────────────────────────────────

    def get_tvb_connectivity(self) -> connectivity.Connectivity:
        """
        Build and return a configured TVB Connectivity object applying
        the weight_mode and tract_length preprocessing selected at init.
        """
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
        return conn

    # ── cerebellar indices ────────────────────────────────────────────────────

    def get_cereb_indices(self, cereb_labels: np.ndarray) -> np.ndarray:
        """Return indices of cerebellar regions within region_labels."""
        cereb_set = set(cereb_labels)
        return np.array([
            i for i, r in enumerate(self.region_labels)
            if r in cereb_set
        ])
