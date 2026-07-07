import os
import numpy as np
import jax
import jax.numpy as jp
import vbjax as vb
import tqdm
from voxcell import RegionMap
from simulator.mouse_conn import MouseConnectivity
from simulator.bold_analysis import BoldTimeSeries, PlotBold
import matplotlib.pyplot as plt

# define our model
def dfun(buf, rv, t: int, p):
    crv = vb.delay_apply(p['dh'], t, buf)                  # compute delay coupling
    return vb.mpr_dfun(rv, p['k']*crv, p['theta'])         # compute single_node_dynamics

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

# setup delays
dt = 0.1
dh = vb.make_delay_helper(weights=mc.weights, lengths=mc.tract_lengths, dt=dt, v_c = mc.speed)

# define parameters for dfun
params = {
    'dh': dh,
    'theta': vb.mpr_default_theta._replace(eta=-4.6, tau=1., Delta = 0.7, J=14.5, I=0),
    'k': 0.004
}

# initial conditions (r, V), applied to the whole delay history
n_nodes = dh.n_from
init_key = jax.random.PRNGKey(0)
r_key, v_key = jax.random.split(init_key)
r0 = jax.random.uniform(r_key, (n_nodes,), minval=0.0, maxval=2.0)
v0 = jax.random.uniform(v_key, (n_nodes,), minval=-2.0, maxval=1.0)
INIT_STATE = jp.stack([r0, v0], axis=0)

# buf should cover all delays + noise for time steps to take
chunk_len = int(10 / dt) # 10 ms
buf = jp.zeros((dh.max_lag + chunk_len + 1 , 2, dh.n_from))
# NB: cont_chunk does np.roll(buf, -chunk_len) BEFORE integrating, so the
# initial history must live in [chunk_len : chunk_len+max_lag+1] to survive
# the first roll and land in buf[0:max_lag+1]. Putting it in [:max_lag+1]
# would be rolled out and the sim would start from (0, 0) instead of the IC.
buf = buf.at[chunk_len : chunk_len + dh.max_lag + 1].add( INIT_STATE.reshape(1, 2, n_nodes) )

# noise amplitude: different sigma on r and V (shape (2, 1) -> broadcast over nodes)
sigma_r = 0.004
sigma_v = sigma_r * 2
gfun = jp.array([[sigma_r], [sigma_v]])

# compile model and enable continuations
_, run_chunk = vb.make_sdde(dt, dh.max_lag, dfun, gfun=gfun, unroll=10, adhoc=vb.mpr_r_positive)
cont_chunk = vb.make_continuation(run_chunk, chunk_len, dh.max_lag, dh.n_from, n_svar=2, stochastic=True)

# setup time avg and bold monitors
ta_buf, ta_step, ta_samp = vb.make_timeavg((2, dh.n_from))
ta_samp = vb.make_offline(ta_step, ta_samp)
bold_buf, bold_step, bold_samp = vb.make_bold((dh.n_from,), dt / 1000, vb.bold_default_theta)  # dt in s; BOLD on r only
bold_samp = vb.make_offline(bold_step, bold_samp)

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


# phase portrait r vs V (one trajectory per node)
plt.figure()
plt.plot(ta[:, 0], ta[:, 1])
plt.xlabel('r'); plt.ylabel('V'); plt.title('scratch — phase r vs V')
plt.savefig(os.path.join(FIGS_PATH, "scratch_phase_rV.png"), dpi=120)

# r time series (all nodes)
plt.figure()
plt.plot(ta[:, 0])
plt.xlabel('sample (10 ms)'); plt.ylabel('r'); plt.title('scratch — r time series')
plt.savefig(os.path.join(FIGS_PATH, "scratch_r_ts.png"), dpi=120)

# V time series (all nodes)
plt.figure()
plt.plot(ta[:, 1])
plt.xlabel('sample (10 ms)'); plt.ylabel('V'); plt.title('scratch — V time series')
plt.savefig(os.path.join(FIGS_PATH, "scratch_V_ts.png"), dpi=120)



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
