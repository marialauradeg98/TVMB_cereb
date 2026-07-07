from mpr_analysis import *
import scipy
import gmv_codes.montbrio_model as mpr
import jax.numpy as jp
from gmv_codes.simulation_utils_mpr import run_bold_sweep

dt = 0.1
sigma       = 0. # 0 if deterministic
seed         = 42
sim_len      = 1000
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
    "num_skip": 1,
    "num_time": 10000,
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
t = jp.arange(0, sim_len, setup['num_skip']*dt) # num_skip = sampling period (10 = 10 ms)
print(t.shape)


raw = np.array(raw)
print('Raw shape', raw.shape)

time, r, V = compute_timeseries(params, x0, dt = dt, T=sim_len)

fig, ax = plt.subplots(1,2)
ax[0].set_title('V timeseries')
ax[0].plot(t, raw[:, 1, 0, 0] , color='red', label='V vbjax')
ax[0].plot(time, V, color = 'gray', label = 'V mpr', linestyle = '--', zorder = 10, alpha=0.5)
ax[0].legend()
ax[1].set_title('r timeseries')
ax[1].plot(t, raw[:, 0, 0, 0], color='black', label = 'r vbjax')
ax[1].plot(time, r, color = 'gray', label = 'r mpr', linestyle = '--', zorder = 10, alpha=0.5)
ax[1].legend()
plt.show()

fig, ax = plt.subplots(1,2)
ax[0].set_title('V timeseries')
ax[0].plot(t, raw[:, 1, 0, 0] , color='red', label='V TVB')
ax[0].plot(time, V, color = 'gray', label = 'V mpr', linestyle = '--', zorder = 10, alpha=0.5)
ax[0].set_xlim(0,100)
ax[0].legend()
ax[1].set_title('r timeseries')
ax[1].plot(t, raw[:, 0, 0, 0], color='black', label = 'r TVB')
ax[1].plot(time, r, color = 'gray', label = 'r mpr', linestyle = '--', zorder = 10, alpha=0.5)
ax[1].set_xlim(0,100)
ax[1].legend()
plt.show()
