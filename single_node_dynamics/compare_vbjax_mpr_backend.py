from mpr_analysis import *
import scipy
import gmv_codes.montbrio_model as mpr
import jax.numpy as jp
from gmv_codes.simulation_utils_mpr import run_bold_sweep
from tvb.simulator.lab import *
from tvb.simulator.backend.nb_mpr import NbMPRBackend

dt = 0.1
sigma       = 0. # 0 if deterministic
seed         = 42
sim_len      = 10000
v_c = 4.

params = {
    "tau": 1.,
    "Delta": 0.7,
    "eta": -4.6,
    "J": 14.5,
    "I": 0.,
}

# disconnected regions (3x3 toy matrix)
C= np.ones((3,3))
L = np.ones((3,3))
idelays = (L[C != 0.0] / v_c / dt).astype(np.uint32)

x0 = [1.5, 0.] # [r0, V0]
setup = {
    "Seids": scipy.sparse.csr_matrix(C),
    "idelays": idelays,
    "params": mpr.default_theta,
    "v_c": v_c,
    "horizon": 650,
    "num_item": 1,
    "dt": dt,
    "num_skip": 10,
    "num_time": sim_len,
    "init_state": jp.array(x0).reshape(2,1),
    "num_svar": 2,
    "noise": sigma,
}


theta = mpr.default_theta._replace(
    cr = 0,
    cv = 0.,
    eta = params['eta'] ,
    Delta = params['Delta'] ,
    J = params['J'] ,
    I = params['I'] ,
    tau = params['tau'] ,
    sigma_r= setup["noise"],
    sigma_V= 2 * setup["noise"],
)
setup["params"] = theta

_ , raw = run_bold_sweep((theta, setup))
raw = np.array(raw)
print('Sim vbjax executed!')
print('Raw shape', raw.shape)

# TVB mpr backend
conn = connectivity.Connectivity()
conn.weights = C
conn.centres = np.ones((3,3))
conn.region_labels = np.array(['Region 1', 'Region 2', 'Region 3'])
conn.tract_lengths = L
conn.speed = np.r_[v_c]
conn.configure()

mpr = models.MontbrioPazoRoxin(
    eta   = np.r_[params['eta']],
    J     = np.r_[params['J']],
    Delta = np.r_[params['Delta']],
    tau   = np.r_[params['tau']],
    I = np.r_[params['I']],
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
    integrator = integrators.HeunStochastic(dt = dt, noise = noise.Additive(nsig=np.r_[sigma,2*sigma], noise_seed=seed)),
    monitors = [monitors.TemporalAverage(period=setup['num_skip'])],
    initial_conditions = ic,
)



# t = jp.arange(0, sim_len, setup['num_skip']) # num_skip = sampling period (10 = 10 ms)
# print(t.shape)

sim.configure()
runner = NbMPRBackend()
(t, data), = runner.run_sim(sim, simulation_length=sim_len)
print('Sim TVB MPR backend executed!')
print('Raw dim: ', data.shape)



fig, ax = plt.subplots(1,2)
ax[0].set_title('V timeseries')
ax[0].plot(t, raw[:, 1, 0, 0] , color='red', label='V vbjax')
ax[0].plot(t, data[:, 1, 0, 0], color = 'gray', label = 'V TVB', linestyle = '--', zorder = 10, alpha=0.5)
ax[0].legend()
ax[1].set_title('r timeseries')
ax[1].plot(t, raw[:, 0, 0, 0], color='black', label = 'r vbjax')
ax[1].plot(t, data[:, 0, 0, 0], color = 'gray', label = 'r mpr', linestyle = '--', zorder = 10, alpha=0.5)
ax[1].legend()
plt.show()

fig, ax = plt.subplots(1,2)
ax[0].set_title('V timeseries')
#ax[0].plot(t, raw[:, 1, 0, 0] , color='red', label='V vbjax')
ax[0].plot(t, data[:, 1, 0, 0], color = 'gray', label = 'V TVB', linestyle = '--', zorder = 10, alpha=0.5)
ax[0].set_xlim(0,1000)
ax[0].legend()
ax[1].set_title('r timeseries')
#ax[1].plot(t, raw[:, 0, 0, 0], color='black', label = 'r vbjax')
ax[1].plot(t, data[:, 0, 0, 0], color = 'gray', label = 'r TVB', linestyle = '--', zorder = 10, alpha=0.5)
ax[1].set_xlim(0,1000)
ax[1].legend()
plt.show()
