import os
import numpy as np
import jax.numpy as jp
import scipy.sparse
import montbrio_model as mpr
from simulation_utils_mpr import run_bold_sweep
import matplotlib.pyplot as plt


sigma   =  0. # deterministic
v_c = 4.
dt = 0.1

# ------------------------
# Load subject-specific data
# ------------------------
C0 = np.ones((3,3))* 1e-4
L0 = np.ones((3,3))

C = np.array([[0.5488, 0.7152, 0.6028],
              [0.5449, 0.4237, 0.6459],
              [0.4376, 0.8918, 0.9637]])

L = np.array([[5.4881, 7.1519, 6.0276],
              [5.4488, 4.2365, 6.4589],
              [4.3759, 8.9177, 9.6366]])

plt.figure()
plt.title('Tracts')
plt.imshow(L)
plt.colorbar()
plt.show()

plt.figure()
plt.title('Weights')
plt.imshow(C)
plt.colorbar()
plt.show()

idelays = (L[C != 0.0] / v_c / dt).astype(np.uint32)


# ------------------------
# Model setup
# ------------------------

setup = {
    "Seids": scipy.sparse.csr_matrix(C),
    "idelays": idelays,
    "params": mpr.default_theta,
    "v_c": v_c,
    "horizon": 650,
    "num_item": 1,
    "dt": dt,
    "num_skip": 1,
    "num_time": 500,
    "init_state": jp.array([0.25, 0.5]).reshape(2,1),
    "num_svar": 2,
    "noise": sigma,
}

# ------------------------
# Parameter update
# ------------------------

theta = mpr.default_theta._replace(
    cr = 0,
    cv = 0.,
    eta = -4.6,
    sigma_r= setup["noise"],
    sigma_V= 2 * setup["noise"],
)

setup["params"] = theta

# ------------------------
# Run simulation
# ------------------------

bold, raw = run_bold_sweep((theta, setup))

raw = np.array(raw)
print('Raw shape', raw.shape)

bold = np.asarray(bold)
print('Bold shape ', bold.shape)

# plot results
plt.figure()
plt.title('Raw r')
plt.plot(raw[: , 0 , :, 0], label = 'r')
#plt.xlim(20, 100)
plt.legend()
plt.show()

plt.figure()
plt.title('Raw v')
plt.plot(raw[: , 1 , :, 0], label = 'v')
plt.legend()
plt.show()

plt.figure()
plt.title('Phase plane')
plt.plot(raw[: , 0 , :, 0], raw[: , 1 , :, 0])
plt.show()



plt.figure()
plt.title('Bold')
plt.plot(bold[:, :, 0])
plt.show()