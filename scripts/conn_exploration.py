import os

BASE     = os.path.dirname(os.path.abspath(__file__))

from voxcell import RegionMap
from simulator.mouse_conn import MouseConnectivity, PlotConn
import numpy as np

CONNECTIVITY_FILE = os.path.join(BASE, "../conn_data/Conn_Allen/ConnAllen.h5")
RVM_FILE          = os.path.join(BASE, "../conn_data/Conn_Allen//RegionVolumeMapping.h5")
JSON_FILE         = os.path.join(BASE, "../conn_data/1.json")
VOXEL_MM          = 0.025
FIGS_PATH = os.path.join(BASE, "../figs_conn")
os.makedirs(FIGS_PATH, exist_ok=True)

if __name__ == "__main__":

    plot_conn = PlotConn()

    print("Loading connectivity…")
    region_map = RegionMap.load_json(JSON_FILE)
    mc = MouseConnectivity(
        CONNECTIVITY_FILE,
        voxel_size_mm=VOXEL_MM,
        region_map=region_map,
    )
    labels        = mc.region_labels
    macros        = mc.macros
    weights_raw   = mc.weights
    mc.tract_lengths = mc.preprocess_tract_lengths(mc.tract_lengths)
    mc.centres       = mc.preprocess_centres(mc.centres)
    n             = len(labels)
    print(f"  {n} regions after filtering")

    weights_log      = MouseConnectivity.log_scale(weights_raw)
    mc.set_weights(weights_log)
    mc.speed = 3.
    mc.set_delays()
    weights_raw_norm = weights_raw / weights_raw.max()
    THRESHOLD_PCT    = 75

    plot_conn.plot_matrix(
        weights_raw_norm, "Weights matrix (raw)",
        plot_conn.CMAP_W, "weight [0, 1]", vmin=0, vmax=1,
        out_path=os.path.join(FIGS_PATH, "fig_weights_raw.png"),
        save=True,
    )
    plot_conn.plot_histogram(
        weights_raw_norm, "Weights distribution (raw, non-zero)", "Raw weight",
        color=plot_conn.C_W, fmt=".4f",
        out_path=os.path.join(FIGS_PATH, "fig_weights_raw_hist.png"),
        save=True,
    )

    plot_conn.plot_matrix(
        weights_log, "Weights matrix (log-scaled)",
        plot_conn.CMAP_W, "weight [0, 1]", vmin=0, vmax=1,
        out_path=os.path.join(FIGS_PATH, "fig_weights_log.png"),
        save=True,
    )
    plot_conn.plot_histogram(
        weights_log, "Weights distribution (log-scaled, non-zero)", "Log-scaled weight",
        color=plot_conn.C_W, fmt=".2f", loc="upper left",
        out_path=os.path.join(FIGS_PATH, "fig_weights_log_hist.png"),
        save=True,
    )

    plot_conn.plot_matrix(
        mc.tract_lengths, "Tract lengths matrix (mm)",
        plot_conn.CMAP_TL, "mm",
        out_path=os.path.join(FIGS_PATH, "fig_tract_lengths.png"),
        save=True,
    )
    plot_conn.plot_histogram(
        mc.tract_lengths, "Tract lengths distribution (non-zero, mm)", "Tract length (mm)",
        color=plot_conn.C_TL, fmt=".2f",
        out_path=os.path.join(FIGS_PATH, "fig_tract_lengths_hist.png"),
        save=True,
    )

    plot_conn.plot_network_3d(
        weights_raw, mc.centres, macros,
        surf_pts=PlotConn.brain_surface_points(RVM_FILE, voxel_mm=VOXEL_MM),
        out_path=os.path.join(FIGS_PATH, "fig_network_3d.png"),
        save=True,
    )
    plot_conn.plot_macroarea_distribution(
        labels, macros,
        out_path=os.path.join(FIGS_PATH, "fig_regions_per_macroarea.png"),
        save=True,
    )

    mc.set_weights(weights_log)
    mc.threshold_weights(THRESHOLD_PCT)
    n_edges = int((mc.weights > 0).sum())
    plot_conn.plot_matrix(
        mc.weights, f"Log-scaled weights — Threshold percetile  {THRESHOLD_PCT}% ({n_edges} edges)",
        plot_conn.CMAP_W, "weight [0, 1]", vmin=0, vmax=1,
        out_path=os.path.join(FIGS_PATH, f"fig_w_threshold_{THRESHOLD_PCT}.png"),
        save=True,
    )

    plot_conn.plot_matrix(
        mc.delays, "Delays matrix",
        'Reds', "[ms]",
        out_path=None,
        save=None,
    )

    plot_conn.plot_histogram(
        mc.delays, "Delays Distribution", "[ms]",
        color='pink', fmt=".2f",
        out_path=os.path.join(FIGS_PATH, "fig_delays_hist.png"),
        save=True,
    )

"""
# --- Trova il tratto più lungo ---
tract_upper = np.triu(tract_lengths, k=1)  # solo triangolo superiore: niente diagonale né duplicati simmetrici
i, j = np.unravel_index(np.argmax(tract_upper), tract_upper.shape)
max_length = tract_upper[i, j]

region_i, region_j = labels[i], labels[j]
print(f"Tratto più lungo: {max_length:.2f} mm")
print(f"Tra le regioni: '{region_i}' (idx {i}) <-> '{region_j}' (idx {j})")

# --- Top 10 ---
N_TOP = 10
flat_sorted = np.argsort(tract_upper, axis=None)[::-1][:N_TOP]
top_pairs = [np.unravel_index(idx, tract_upper.shape) for idx in flat_sorted]

print(f"\nTop {N_TOP} tratti più lunghi:")
for rank, (a, b) in enumerate(top_pairs, 1):
    print(f"{rank}. {labels[a]} <-> {labels[b]}: {tract_upper[a, b]:.2f} mm")
"""