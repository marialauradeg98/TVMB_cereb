from tvb.simulator.lab import *
from tvb.simulator.backend.nb_mpr import NbMPRBackend
import time
from voxcell import RegionMap
from simulator.mouse_conn import MouseConnectivity
from simulator.bold_analysis import BoldTimeSeries, PlotBold
import matplotlib.pyplot as plt
import numpy as np


def tavg_to_bold(tavg_t, tavg_r, conn, tavg_period=10.0, decimate=100):
    """
    Convert TemporalAverage r to BOLD via Balloon-Windkessel.
    tavg_r : [T, N]   firing rate (SVAR=0 already applied upstream)
    Returns bold_t [T_bold ms], bold_d [T_bold, 1, N, 1]
    """
    if tavg_r.ndim == 2:
        tavg_r = tavg_r[:, np.newaxis, :, np.newaxis]
    tsr = TimeSeriesRegion(
        connectivity=conn, data=tavg_r,
        time=tavg_t, sample_period=tavg_period)
    tsr.configure()
    bm = BalloonModel(time_series=tsr, dt=tavg_period/1000.)
    bold_out = bm.evaluate()
    bold_t = bold_out.time[::decimate] * 1000.   # s → ms
    bold_d = bold_out.data[::decimate, :]
    return bold_t, bold_d


# simulation set
dt = 0.1
nsigma       = 0.004
seed         = 42
sim_len      = 60000     # 5 minutes simulation
G = 0.004
TR = 1e3                # for BOLD signals
DECIMATE  = 100         # TR = 10ms × 100 = 1000ms = 1s
R0, V0 = 0.1, -2.0      # initial conditions (r, V), matching simulation_scratch.py

t0 = time.time()

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

# Build TVB connectivity from file
conn = connectivity.Connectivity()
conn.weights = mc.weights
conn.centres = mc.centres
conn.region_labels = mc.region_labels
conn.tract_lengths = mc.tract_lengths
conn.speed = np.r_[mc.speed] # m/s Rabuffo 2021
conn.configure()

mpr = models.MontbrioPazoRoxin(
    eta   = np.r_[-4.6],
    J     = np.r_[14.5],
    Delta = np.r_[0.7],
    tau   = np.r_[1],
)

# random per-region initial conditions (r, V), constant over the whole delay
n_hist = int(np.ceil(conn.tract_lengths.max() / conn.speed[0] / dt)) + 10
rng = np.random.default_rng(0)
r0 = rng.uniform(0.0, 2.0, size=(conn.number_of_regions, 1))
v0 = rng.uniform(-2.0, 1.0, size=(conn.number_of_regions, 1))
ic = np.zeros((n_hist, mpr.nvar, conn.number_of_regions, 1))
ic[:, 0] = r0   # r ~ U[0, 2] per region
ic[:, 1] = v0   # V ~ U[-2, 1] per region

sim = simulator.Simulator(
    model = mpr,
    connectivity = conn,
    coupling = coupling.Linear(a=np.r_[G]),
    conduction_speed = np.inf,
    integrator = integrators.HeunStochastic(dt = dt, noise = noise.Additive(nsig=np.r_[nsigma, nsigma*2], noise_seed=seed)),
    monitors = [monitors.TemporalAverage(period=10.)],
    initial_conditions = ic,
)

sim.configure()
runner = NbMPRBackend()
(TemporalAverage_time, TemporalAverage_data), = runner.run_sim(sim, simulation_length=sim_len)
TemporalAverage_time *= 10

duration = time.time() - t0
print(f'Job duration : {duration}')


os.makedirs('figs', exist_ok=True)

r_st      = TemporalAverage_data[:,0,:,0]
V_st      = TemporalAverage_data[:,1,:,0]

os.makedirs(FIGS_PATH, exist_ok=True)

# phase portrait r vs V (one trajectory per node)
plt.figure()
plt.plot(r_st, V_st)
plt.xlabel('r'); plt.ylabel('V'); plt.title('backend — phase r vs V')
plt.savefig(os.path.join(FIGS_PATH, "backend_phase_rV.png"), dpi=120)

# r time series (all nodes)
plt.figure()
plt.plot(TemporalAverage_time, r_st)
plt.xlabel('Time [ms]'); plt.ylabel('r'); plt.title('backend — r time series')
plt.savefig(os.path.join(FIGS_PATH, "backend_r_ts.png"), dpi=120)

# V time series (all nodes)
plt.figure()
plt.plot(TemporalAverage_time, V_st)
plt.xlabel('Time [ms]'); plt.ylabel('V'); plt.title('backend — V time series')
plt.savefig(os.path.join(FIGS_PATH, "backend_V_ts.png"), dpi=120)

exit()
bold_t, bold_d = tavg_to_bold(TemporalAverage_time, r_st, conn, tavg_period=10., decimate=DECIMATE)

print("Plotting simulated BOLD time series…")

sim_bold = BoldTimeSeries.from_array(
    ts= bold_d[:, 0, :, 0],   # [T, svar, node, mode] -> [T, node]
    region_labels= mc.region_labels,
    tr_ms=1000,
    name="one_sim",
    macros=mc.macros,
    acronyms=mc.acronyms,
)
os.makedirs(FIGS_PATH, exist_ok=True)
PlotBold().plot_timeseries(
    sim_bold,
    out_path=os.path.join(FIGS_PATH, "one_sim_bold_ts_mpr_backend.png"),
    title=f"Simulated BOLD time series (J={mpr.J}, eta={mpr.eta}, G={mpr.cr})",
    save=True,
)