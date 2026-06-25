import os
import time

import numpy as np
from voxcell import RegionMap

from simulator.bold_analysis import BoldTimeSeries, PlotBold
from simulator.models import GastModel
from simulator.mouse_conn import MouseConnectivity
from simulator.simulator import Simulator

BASE              = os.path.dirname(os.path.abspath(__file__))
REGION_MAP_JSON   = os.path.join(BASE, "../conn_data/1.json")
CONNECTIVITY_FILE = os.path.join(BASE, "../conn_data/Conn_Allen/ConnAllen.h5")
OUTPUT_PATH       = os.path.join(BASE, "../sim_data")
FIGS_PATH         = os.path.join(BASE, "../figs_bold")
VOXEL_MM          = 0.025

WE       = 1.0     # global coupling
J        = 15.0    # self-excitation
SIGMA    = 0.1      # noise level (sigma_u = 0.1 * SIGMA)

BOLD_TR_MS = 1000.0  # Simulator 1 timepoint/s (== emp TR)


if __name__ == "__main__":

    # set structural connectivity
    region_map = RegionMap.load_json(REGION_MAP_JSON)
    mc = MouseConnectivity(
        CONNECTIVITY_FILE,
        voxel_size_mm=VOXEL_MM,
        region_map=region_map,
    )

    mc.tract_lengths = mc.preprocess_tract_lengths(mc.tract_lengths)  # voxel -> mm
    mc.set_weights(MouseConnectivity.log_scale(mc.weights)) # log scale weights
    print(f"  SC: {mc.weights.shape[0]} regions")

    sim = Simulator(mc, GastModel)

    theta = GastModel.default_theta._replace(
        we=WE, J=J, sigma_V=SIGMA, sigma_u=0.1 * SIGMA,
    )

    print(f"Running simulation (we={WE}, J={J}, sigma={SIGMA})…")
    t0 = time.perf_counter()
    bold = sim.run(theta, seed=42)
    elapsed_s = time.perf_counter() - t0
    print(f"  BOLD shape: {bold.shape}")
    print(f"  Simulation lasts {elapsed_s:.1f} s")

    out_file = os.path.join(OUTPUT_PATH, "1sim_gast_bold.npz")
    Simulator.save_bold(bold, out_file)
    print(f"Saved {out_file}")

    print("Plotting simulated BOLD time series…")
    sim_bold = BoldTimeSeries.from_array(
        ts=np.asarray(bold[:, :, 0]),
        region_labels=mc.region_labels,
        tr_ms=BOLD_TR_MS,
        name="one_sim",
        macros=mc.macros,
        acronyms=mc.acronyms,
    )

    os.makedirs(FIGS_PATH, exist_ok=True)
    PlotBold().plot_timeseries(
        sim_bold,
        out_path=os.path.join(FIGS_PATH, "one_sim_bold_ts.png"),
        title=f"Simulated BOLD time series (we={WE}, J={J}, sigma={SIGMA})",
        save=True,
    )