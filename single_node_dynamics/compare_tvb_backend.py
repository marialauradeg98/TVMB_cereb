from tvb.simulator.lab import *
from tvb.simulator.backend.nb_mpr import NbMPRBackend
from mpr_analysis import *

dt = 0.1
nsigma       = 0. # 0 if deterministic
seed         = 42
sim_len      = 10000

# disconnected regions (3x3 toy matrix)
conn = connectivity.Connectivity()
conn.weights = np.zeros((3,3))
conn.centres = np.ones((3,3))
conn.region_labels = np.array(['Region 1', 'Region 2', 'Region 3'])
conn.tract_lengths = np.ones((3,3))
conn.speed = np.r_[4.] # m/s Rabuffo 2021
conn.configure()

params = {
    "tau": 1.,
    "Delta": 0.7,
    "eta": -4.6,
    "J": 14.5,
    "I": 0.,
}


mpr = models.MontbrioPazoRoxin(
    eta   = np.r_[params['eta']],
    J     = np.r_[params['J']],
    Delta = np.r_[params['Delta']],
    tau   = np.r_[params['tau']],
    I = np.r_[params['I']],
)

x0 = [1.5, 0.] # [r0, V0]

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
    integrator = integrators.HeunStochastic(dt = dt, noise = noise.Additive(nsig=np.r_[nsigma,2*nsigma], noise_seed=seed)),
    monitors = [monitors.TemporalAverage(period=dt)],
    initial_conditions = ic,
)

sim.configure()
runner = NbMPRBackend()
(t, data), = runner.run_sim(sim, simulation_length=sim_len)

print(data.shape)

time, r, V = compute_timeseries(params, x0, dt = dt, T=sim_len)


fig, ax = plt.subplots(1,2)
ax[0].set_title('V timeseries')
ax[0].plot(t, data[:, 1, 0, 0] , color='red', label='V TVB')
ax[0].plot(time, V, color = 'gray', label = 'V mpr', linestyle = '--', zorder = 10, alpha=0.5)
ax[0].set_xlim(0,100)
ax[0].legend()
ax[1].set_title('r timeseries')
ax[1].plot(t, data[:, 0, 0, 0], color='black', label = 'r TVB')
ax[1].plot(time, r, color = 'gray', label = 'r mpr', linestyle = '--', zorder = 10, alpha=0.5)
ax[1].set_xlim(0,100)
ax[1].legend()
plt.show()

