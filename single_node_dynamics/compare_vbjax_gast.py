import scipy
import gmv_codes.gast_model as gm
import jax.numpy as jp
from gmv_codes.simulation_utils import run_bold_sweep
import numpy as np
import matplotlib.pyplot as plt
from gast_analysis import *

params = {
    'C': 100.0,
    'k': 0.7,
    'Delta': 0.5,
    'v_r': -60.0,
    'v_theta': -40.0,
    'g': 1.0,
    'E': 0.0,
    'I': 46.5,
    'tau_u': 33.33,
    'b': -2.0,
    'kappa': 100.0,
    'tau_s': 6.0,
    'Js': 1.0,
    'J': 15.0,
    'we': 0.0,
    'sigma_V': 0.,
    'sigma_u': 0.,
    'c_exc': 0.
}

v_c = 4.0
dt = 0.1
sigma = 0 # deterministic
sim_len = 1000 # ms (?)

x0 = np.array([.01, -55.0, 0.0, 0.0])

# (3x3 toy matrix)
C= jp.ones((3,3))
L = jp.ones((3,3))
idelays = (L[C != 0.0] / v_c / dt).astype(np.uint32)


# setup parameters
setup = {
    "Seids": scipy.sparse.csr_matrix(C),
    "idelays": idelays,
    "params": gm.default_theta,
    "v_c": v_c,
    "horizon": 650,
    "num_item": 1,
    "dt": dt,
    "num_skip": 10,
    "num_time": sim_len,
    "init_state": jp.array(x0).reshape(4,1),
    "num_svar": 4,
    "noise": sigma,
}

theta = gm.default_theta._replace(
    we=0., # decoupled
    sigma_V=setup["noise"],
    sigma_u=0.1 * setup["noise"],
)

# update parameters
setup["params"] = theta

bold, raw, heun_cont = run_bold_sweep((theta, setup))

print(raw.shape)
print('Integration count: ', heun_cont)
# setup['num_skip']*dt
t = jp.arange(0, sim_len, setup['num_skip']) # (if num_skip = sampling period temporal average)

# integrate with odeint
time, r, v, u, s = compute_timeseries(params, x0, dt = dt, T=sim_len)


fig, ax = plt.subplots(1,4)
ax[0].set_title('r timeseries')
ax[0].plot(t, raw[:, 0, 0, 0] , color='red', label='r vbjax')
ax[0].plot(time, r, color = 'gray', label = 'r gast', linestyle = '--', zorder = 10, alpha=0.5)
ax[0].legend(loc='upper right')
ax[1].set_title('V timeseries')
ax[1].plot(t, raw[:, 1, 0, 0], color='black', label = 'V vbjax')
ax[1].plot(time, v,  color = 'gray', label = 'v gast', linestyle = '--', zorder = 10, alpha=0.5)
ax[1].legend(loc='upper right')
ax[2].set_title('u timeseries')
ax[2].plot(t, raw[:, 2, 0, 0], color='blue', label = 'u vbjax')
ax[2].plot(time, u, color = 'gray', label = 'u gast', linestyle = '--', zorder = 10, alpha=0.5)
ax[2].legend(loc='upper right')
ax[3].set_title('s timeseries')
ax[3].plot(t, raw[:, 3, 0, 0], color='green', label = 's vbjax')
ax[3].plot(time, s, color = 'gray', label = 's gast', linestyle = '--', zorder = 10, alpha=0.5)
ax[3].legend(loc='upper right')
plt.show()

