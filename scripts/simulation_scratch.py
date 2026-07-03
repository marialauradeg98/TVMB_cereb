import os
import numpy as np
import jax
import jax.numpy as jp
import vbjax as vb
import tqdm
from voxcell import RegionMap
from simulator.mouse_conn import MouseConnectivity
from simulator.bold_analysis import BoldTimeSeries, PlotBold

def log_scale(w_in):
    w = jp.nan_to_num(w_in, nan=0.0)
    w_positive = w > 0

    w_min = jp.min(jp.where(w_positive, w, jp.inf))
    w_scaled = w / w_min * jp.exp(1)

    w_log = jp.where(w_positive, jp.log(w_scaled), 0.0)
    w_out = w_log / jp.max(w_log)
    return w_out

BASE              = os.path.dirname(os.path.abspath(__file__))
REGION_MAP_JSON   = os.path.join(BASE, "../conn_data/1.json")
CONNECTIVITY_FILE = os.path.join(BASE, "../conn_data/Conn_Allen/ConnAllen.h5")
OUTPUT_PATH       = os.path.join(BASE, "../sim_data")
FIGS_PATH         = os.path.join(BASE, "../figs_bold")
VOXEL_MM          = 0.025
BOLD_TR_MS        = 1000.0   # BOLD repetition time (1 timepoint / s)

BASE              = os.path.dirname(os.path.abspath(__file__))
REGION_MAP_JSON   = os.path.join(BASE, "../conn_data/1.json")
CONNECTIVITY_FILE = os.path.join(BASE, "../conn_data/Conn_Allen/ConnAllen.h5")
OUTPUT_PATH       = os.path.join(BASE, "../sim_data")
FIGS_PATH         = os.path.join(BASE, "../figs_bold")
VOXEL_MM          = 0.025

# load connectome
region_map = RegionMap.load_json(REGION_MAP_JSON)
mc = MouseConnectivity(
    CONNECTIVITY_FILE,
    voxel_size_mm=VOXEL_MM,
    region_map=region_map,
)
mc.tract_lengths = mc.preprocess_tract_lengths(mc.tract_lengths)  # voxel -> mm

W, L = mc.weights, mc.tract_lengths
W = jp.fill_diagonal(W, 0, inplace=False)
W = log_scale(W)


# setup delays
dt = 0.1
dh = vb.make_delay_helper(weights=W, lengths=L, dt=dt, v_c = 4.)

# define parameters for dfun
params = {
    'dh': dh,
    'theta': vb.mpr_default_theta._replace(eta=-4.6),
    'k': 0.006
}

# define our model
def dfun(buf, rv, t: int, p):
    crv = vb.delay_apply(p['dh'], t, buf)                  # compute delay coupling
    return vb.mpr_dfun(rv, p['k']*crv, p['theta'])         # compute dynamics

# initial conditions (r, V), applied to the whole delay history
R0, V0 = 0.1, -2.0
INIT_STATE = jp.r_[R0, V0]

# buf should cover all delays + noise for time steps to take
chunk_len = int(10 / dt) # 10 ms
buf = jp.zeros((dh.max_lag + chunk_len + 1 , 2, dh.n_from))
buf = buf.at[:dh.max_lag+1].add( INIT_STATE.reshape(2,1) )

# compile model and enable continuations
_, run_chunk = vb.make_sdde(dt, dh.max_lag, dfun, gfun=1e-3, unroll=10, adhoc=vb.mpr_r_positive)
cont_chunk = vb.make_continuation(run_chunk, chunk_len, dh.max_lag, dh.n_from, n_svar=2, stochastic=True)

# setup time avg and bold monitors
ta_buf, ta_step, ta_samp = vb.make_timeavg((2, dh.n_from))
ta_samp = vb.make_offline(ta_step, ta_samp)
bold_buf, bold_step, bold_samp = vb.make_bold((dh.n_from,), dt / 1000, vb.bold_default_theta)  # dt in s; BOLD on r only
bold_samp = vb.make_offline(bold_step, bold_samp)

# run chunk w/ monitors
def chunk_ta_bold(sim, key):
    sim['buf'], rv = cont_chunk(sim['buf'], sim['params'], key)
    sim['ta_buf'], ta = ta_samp(sim['ta_buf'], rv)
    sim['bold_buf'], bold = bold_samp(sim['bold_buf'], rv[:, 0])   # firing rate r only
    return sim, (ta, bold)

@jax.jit
def run_one_second(sim, key):
    keys = jax.random.split(key, 100) # 100 * 10 ms
    return jax.lax.scan(chunk_ta_bold, sim, keys)


# pack buffers and run it one minute
sim = {
    'params': params,
    'buf': buf,
    'ta_buf': ta_buf,
    'bold_buf': bold_buf
}
ta, bold = [], []
keys = jax.random.split(jax.random.PRNGKey(42), 60)
for i, key in enumerate(tqdm.tqdm(keys)):
    sim, (ta_i, bold_i) = run_one_second(sim, key)
    ta.append(ta_i)
    bold.append(bold_i)
ta = jp.array(ta).reshape((-1, 2, dh.n_from))     # (T, 2, n_from) — r, V at 10 ms
bold = jp.array(bold).reshape((-1, dh.n_from))    # (T, n_from)    — r-BOLD at 10 ms

# decimate BOLD from the 10 ms monitor period to TR = BOLD_TR_MS (1 s)
bold_ms = chunk_len * dt                                  # BOLD sample period = 10 ms
bold = bold[:: int(round(BOLD_TR_MS / bold_ms))]         # -> TR = 1 s

print(bold.shape)

# save
os.makedirs(OUTPUT_PATH, exist_ok=True)
np.savez(os.path.join(OUTPUT_PATH, "1sim_mpr_scratch_bold.npz"),
         tavg=np.asarray(ta), bold=np.asarray(bold))

# --- plots, same style as the other one_sim_* scripts ---
os.makedirs(FIGS_PATH, exist_ok=True)
eta, k = params['theta'].eta, params['k']

# firing rate r (temporal average, 10 ms sampling period)
tavg_ts = BoldTimeSeries.from_array(
    ts=np.asarray(ta[:, 0]),
    region_labels=mc.region_labels,
    tr_ms=10.0,
    name="one_sim_tavg",
    macros=mc.macros,
    acronyms=mc.acronyms,
)
PlotBold().plot_timeseries(
    tavg_ts,
    out_path=os.path.join(FIGS_PATH, "one_sim_tavg_ts_scratch.png"),
    title=f"Simulated firing rate r (tavg) — scratch (eta={eta}, k={k})",
    save=True,
)

# BOLD (r-driven) at TR = BOLD_TR_MS
sim_bold = BoldTimeSeries.from_array(
    ts=np.asarray(bold),
    region_labels=mc.region_labels,
    tr_ms=BOLD_TR_MS,
    name="one_sim",
    macros=mc.macros,
    acronyms=mc.acronyms,
)
PlotBold().plot_timeseries(
    sim_bold,
    out_path=os.path.join(FIGS_PATH, "one_sim_bold_ts_scratch.png"),
    title=f"Simulated BOLD time series — scratch (eta={eta}, k={k})",
    save=True,
)
print("saved figures to", FIGS_PATH)
