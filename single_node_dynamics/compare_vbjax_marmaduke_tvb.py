import numpy as np
import jax
import jax.numpy as jp
import vbjax as vb
import tqdm
from mpr_analysis import *
from tvb.simulator.lab import *
from tvb.simulator.backend.nb_mpr import NbMPRBackend

def dfun(buf, rv, t: int, p):
    crv = vb.delay_apply(p['dh'], t, buf)                  # compute delay coupling
    return vb.mpr_dfun(rv, p['k']*crv, p['theta'])         # compute single_node_dynamics

# run chunk w/ monitors
def chunk_ta_bold(sim, key):
    sim['buf'], rv = cont_chunk(sim['buf'], sim['params'], key)
    sim['ta_buf'], ta = ta_samp(sim['ta_buf'], rv)
    return sim, ta

@jax.jit
def run_sim(sim, key):
    keys = jax.random.split(key, n_chunks)
    return jax.lax.scan(chunk_ta_bold, sim, keys)

params_ = {
    "tau": 1.,
    "Delta": 0.7,
    "eta": -4.6,
    "J": 14.5,
    "I": 0.,
}

v_c = 4.
dt = 0.1
# per-variable noise amplitudes: gfun returns shape (2, 1) and broadcasts
# over the (2, n_from) state, so r and V get independent sigmas.
sigma_r = 0.
sigma_V = 2 * sigma_r
gfun = lambda x, p: jp.array([sigma_r, sigma_V]).reshape(2, 1)

# --- monitor timing ---
# The time-average monitor emits ONE sample per chunk (make_offline averages
# the whole chunk), so the monitor sampling period == chunk_len * dt.
sim_len = 10000.                        # total simulated time [ms]
sampling_period = 0.1                   # monitor period [ms]
chunk_len = int(sampling_period / dt)  # integration steps per averaging window
n_chunks = int(sim_len / sampling_period)  # number of monitor samples

assert np.isclose(n_chunks * dt * chunk_len, sim_len)

# (3x3 toy matrix)
C= jp.ones((3,3))
L = jp.ones((3,3))
idelays = (L[C != 0.0] / v_c / dt).astype(np.uint32)

# setup delays
dh = vb.make_delay_helper(weights=C, lengths=L, dt=dt, v_c = v_c)

# define parameters for dfun
params = {
    'dh': dh,
    'theta': vb.mpr_default_theta._replace(
        cr =0., cv = 0.,
        eta = params_['eta'], Delta = params_['Delta'],
        tau = params_['tau'], I = params_['I'], J = params_['J']
    ),
    'k': 0.
}

# initial conditions (r, V), applied to the whole delay history
x0 = [1.5, 0.] # [r0, V0]
INIT_STATE = jp.r_[jp.array(x0)]

# buf should cover all delays + noise for time steps to take
buf = jp.zeros((dh.max_lag + chunk_len + 1 , 2, dh.n_from))

# NB: cont_chunk does np.roll(buf, -chunk_len) BEFORE integrating, so the
# initial history must live in [chunk_len : chunk_len+max_lag+1] to survive
# the first roll and land in buf[0:max_lag+1]. Putting it in [:max_lag+1]
# would be rolled out and the sim would start from (0, 0) instead of x0.
buf = buf.at[chunk_len : chunk_len + dh.max_lag + 1].add( INIT_STATE.reshape(2,1) )

# compile model and enable continuations
_, run_chunk = vb.make_sdde(dt, dh.max_lag, dfun, gfun=gfun, unroll=10, adhoc=vb.mpr_r_positive)
cont_chunk = vb.make_continuation(run_chunk, chunk_len, dh.max_lag, dh.n_from, n_svar=2, stochastic=True)

# setup time avg
ta_buf, ta_step, ta_samp = vb.make_timeavg((2, dh.n_from))
ta_samp = vb.make_offline(ta_step, ta_samp)
print('ta_step:',ta_step)

# pack buffers and run
sim = {
    'params': params,
    'buf': buf,
    'ta_buf': ta_buf,
}
ta = []
keys = jax.random.split(jax.random.PRNGKey(42), 1)
for i, key in enumerate(tqdm.tqdm(keys)):
    sim, ta_i = run_sim(sim, key)
    ta.append(ta_i)


raw = jp.array(ta).reshape((-1, 2, dh.n_from))     # (T, 2, n_from) — r, V at sampling_period
print('ta:', raw.shape)
t = np.arange(1, n_chunks + 1) * sampling_period   # right edge of each averaging window [ms]

# TVB mpr backend
conn = connectivity.Connectivity()
conn.weights = np.array(C)
conn.centres = np.ones((3,3))
conn.region_labels = np.array(['Region 1', 'Region 2', 'Region 3'])
conn.tract_lengths = np.array(L)
conn.speed = np.r_[v_c]
conn.configure()

mpr = models.MontbrioPazoRoxin(
    eta   = np.r_[params_['eta']],
    J     = np.r_[params_['J']],
    Delta = np.r_[params_['Delta']],
    tau   = np.r_[params_['tau']],
    I = np.r_[params_['I']],
)

n_hist = int(np.ceil(conn.tract_lengths.max() / conn.speed[0] / dt)) + 10
ic = np.zeros((n_hist, mpr.nvar, conn.number_of_regions, 1))
ic[:, 0] = x0[0]   # r0
ic[:, 1] = x0[1]   # V0

G = 0.
sim = simulator.Simulator(
    model = mpr,
    connectivity = conn,
    coupling = coupling.Linear(a=np.r_[G]),
    conduction_speed = 4.,
    integrator = integrators.HeunStochastic(dt = dt, noise = noise.Additive(nsig=np.r_[sigma_r,sigma_V], noise_seed=42)),
    monitors = [monitors.TemporalAverage(period=sampling_period)],
    initial_conditions = ic,
)


sim.configure()
runner = NbMPRBackend()
(time, data), = runner.run_sim(sim, simulation_length=sim_len)

fig, ax = plt.subplots(1,2)
ax[0].set_title('V timeseries')
ax[0].plot(t, raw[:, 1, 0] , color='red', label='V vbjax')
ax[0].plot(time, data[:, 1, 0, 0], color = 'gray', label = 'V mpr', linestyle = '--', zorder = 10, alpha=0.5)
ax[0].legend()
ax[1].set_title('r timeseries')
ax[1].plot(t, raw[:, 0, 0], color='black', label = 'r vbjax')
ax[1].plot(time, data[:, 0, 0, 0], color = 'gray', label = 'r mpr', linestyle = '--', zorder = 10, alpha=0.5)
ax[1].legend()
plt.show()

fig, ax = plt.subplots(1,2)
ax[0].set_title('V timeseries')
ax[0].plot(t, raw[:, 1, 0] , color='red', label='V vbjax')
ax[0].plot(time, data[:, 1, 0, 0], color = 'gray', label = 'V mpr', linestyle = '--', zorder = 10, alpha=0.5)
ax[0].set_xlim(0,100)
ax[0].legend()
ax[1].set_title('r timeseries')
ax[1].plot(t, raw[:, 0, 0], color='black', label = 'r vbjax')
ax[1].plot(time, data[:, 0, 0, 0], color = 'gray', label = 'r mpr', linestyle = '--', zorder = 10, alpha=0.5)
ax[1].set_xlim(0,100)
ax[1].legend()
plt.show()
