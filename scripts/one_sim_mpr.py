import os
import time

import numpy as np
from voxcell import RegionMap

from scripts.simulation_scratch import sigma_v
from simulator.bold_analysis import BoldTimeSeries, PlotBold
from simulator.models import MontbrioModel
from simulator.mouse_conn import MouseConnectivity
from simulator.simulator import Simulator

DT         = 0.1        # ms — integration step
NUM_SKIP   = 10         # ms — TemporalAverage/BOLD sampling period
NUM_TIME   = 60000     # ms — 5 min
BOLD_TR_MS = 1000.0     # 1 timepoint/s (== emp TR)


if __name__ == "__main__":

    BASE              = os.path.dirname(os.path.abspath(__file__))
    REGION_MAP_JSON   = os.path.join(BASE, "../conn_data/1.json")
    CONNECTIVITY_FILE = os.path.join(BASE, "../conn_data/Conn_Allen/ConnAllen.h5")
    OUTPUT_PATH       = os.path.join(BASE, "../sim_data")
    FIGS_PATH         = os.path.join(BASE, "../figs_bold")
    VOXEL_MM          = 0.025
    BOLD_TR_MS        = 1000.0   # BOLD repetition time (1 timepoint / s)

    # load connectome
    region_map = RegionMap.load_json(REGION_MAP_JSON)
    mc = MouseConnectivity(
        CONNECTIVITY_FILE,
        voxel_size_mm=VOXEL_MM,
        region_map=region_map,
    )
    mc.tract_lengths = mc.preprocess_tract_lengths(mc.tract_lengths)
    mc.centres       = mc.preprocess_centres(mc.centres)
    mc.set_delays()
    np.fill_diagonal(mc.weights, 0.0)
    weights_log      = MouseConnectivity.log_scale(mc.weights)
    mc.set_weights(weights_log)
    mc.speed = 3.
    mc.set_delays()
    sigma_r = 0.001
    sigma_v = sigma_r * 2

    sim = Simulator(mc, MontbrioModel, dt=DT, num_skip=NUM_SKIP, num_time=NUM_TIME)

    theta = MontbrioModel.default_theta
    print('Default parameter set: ', theta)
    theta = MontbrioModel.default_theta._replace(
        cr = 0.004, c_v = 0.,
        sigma_r = sigma_r, sigma_v=sigma_v,
        eta=-4.6, tau=1., Delta = 0.7, J=14.5, I=0
    )
    print('Actual parameter set: ', theta)

    print(f"Running simulation (J={theta.J}, eta={theta.eta}, G={theta.cr}, dt={DT})…")
    t0 = time.perf_counter()
    tavg, bold = sim.run(theta, seed=42, return_tavg=True)
    elapsed_s = time.perf_counter() - t0
    print(f"  BOLD shape: {bold.shape}  (tavg shape: {tavg.shape})")
    print(f"  Simulation lasts {elapsed_s:.1f} s")

    out_file = os.path.join(OUTPUT_PATH, "1sim_mpr_bold.npz")
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    np.savez(out_file, bold=bold, tavg=tavg)
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
        out_path=os.path.join(FIGS_PATH, "one_sim_bold_ts_mpr.png"),
        title=f"Simulated BOLD time series (J={theta.J}, eta={theta.eta}, G={theta.cr})",
        save=True,
    )
