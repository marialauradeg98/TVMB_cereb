import os

BASE     = os.path.dirname(os.path.abspath(__file__))

from voxcell import RegionMap
from simulator.mouse_conn import MouseConnectivity, PlotConn

CONNECTIVITY_FILE = os.path.join(BASE, "../conn_data/Conn_Allen/ConnAllen.h5")
RVM_FILE          = os.path.join(BASE, "../conn_data/Conn_Allen//RegionVolumeMapping.h5")
JSON_FILE         = os.path.join(BASE, "../conn_data/1.json")
VOXEL_MM          = 0.025
FIGS_PATH = os.path.join(BASE, "../figs")
os.makedirs(FIGS_PATH, exist_ok=True)

EXCLUDE_LABELS = {
    "Right Hippocampo-amygdalar transition area",
    "Left Hippocampo-amygdalar transition area",
}


if __name__ == "__main__":

    plot_conn = PlotConn()

    print("Loading connectivity…")
    region_map = RegionMap.load_json(JSON_FILE)
    mc = MouseConnectivity(
        CONNECTIVITY_FILE,
        voxel_size_mm=VOXEL_MM,
        exclude_labels=EXCLUDE_LABELS,
        region_map=region_map,
    )
    labels        = mc.region_labels
    macros        = mc.macros
    weights_raw   = mc.weights
    tract_lengths = mc.preprocess_tract_lengths(mc.tract_lengths)
    centres       = mc.preprocess_centres(mc.centres)
    n             = len(labels)
    print(f"  {n} regions after filtering")

    weights_log      = MouseConnectivity.log_scale(weights_raw)
    weights_raw_norm = weights_raw / weights_raw.max()
    THRESHOLD_PCT    = 75

    plot_conn.plot_matrix(
        weights_raw_norm, "Weights matrix (raw, norm.)",
        plot_conn.CMAP_W, "weight [0, 1]", vmin=0, vmax=1,
        out_path=os.path.join(FIGS_PATH, "fig_weights_raw.png"),
    )
    plot_conn.plot_histogram(
        weights_raw_norm, "Weights distribution (raw, non-zero)", "Raw weight",
        color=plot_conn.C_W, fmt=".4f",
        out_path=os.path.join(FIGS_PATH, "fig_weights_raw_hist.png"),
    )

    plot_conn.plot_matrix(
        weights_log, "Weights matrix (log-scaled)",
        plot_conn.CMAP_W, "weight [0, 1]", vmin=0, vmax=1,
        out_path=os.path.join(FIGS_PATH, "fig_weights_log.png"),
    )
    plot_conn.plot_histogram(
        weights_log, "Weights distribution (log-scaled, non-zero)", "Log-scaled weight",
        color=plot_conn.C_W, fmt=".2f", loc="upper left",
        out_path=os.path.join(FIGS_PATH, "fig_weights_log_hist.png"),
    )

    plot_conn.plot_matrix(
        tract_lengths, "Tract lengths matrix (mm)",
        plot_conn.CMAP_TL, "mm",
        out_path=os.path.join(FIGS_PATH, "fig_tract_lengths.png"),
    )
    plot_conn.plot_histogram(
        tract_lengths, "Tract lengths distribution (non-zero, mm)", "Tract length (mm)",
        color=plot_conn.C_TL, fmt=".2f",
        out_path=os.path.join(FIGS_PATH, "fig_tract_lengths_hist.png"),
    )

    plot_conn.plot_network_3d(
        weights_raw, centres, macros,
        surf_pts=PlotConn.brain_surface_points(RVM_FILE, voxel_mm=VOXEL_MM),
        out_path=os.path.join(FIGS_PATH, "fig_network_3d.png"),
    )
    plot_conn.plot_macroarea_distribution(
        labels, macros,
        out_path=os.path.join(FIGS_PATH, "fig_regions_per_macroarea.png"),
    )

    mc.set_weights(weights_log)
    mc.threshold_weights(THRESHOLD_PCT)
    n_edges = int((mc.weights > 0).sum())
    plot_conn.plot_matrix(
        mc.weights, f"Log-scaled weights — Threshold percetile  {THRESHOLD_PCT}% ({n_edges} edges)",
        plot_conn.CMAP_W, "weight [0, 1]", vmin=0, vmax=1,
        out_path=os.path.join(FIGS_PATH, f"fig_w_threshold_{THRESHOLD_PCT}.png"),
    )