import os
import sys
import numpy as np
import jax.numpy as jp
import scipy.sparse
import gast_model as gm

from paths import Paths
from simulation_utils import run_bold_sweep
from simulator.mouse_conn import MouseConnectivity
from voxcell import RegionMap

REGION_MAP_JSON   = "../../../conn_data/1.json"
CONNECTIVITY_FILE = "../../../conn_data/Conn_Allen/ConnAllen.h5"
OUTPUT_PATH       = "../../../sim_data"
FIGS_PATH         =  "../../../figs_bold"
VOXEL_MM          = 0.025

region_map = RegionMap.load_json(REGION_MAP_JSON)
mc = MouseConnectivity(
    CONNECTIVITY_FILE,
    voxel_size_mm=VOXEL_MM,
    region_map=region_map,
)

mc.tract_lengths = mc.preprocess_tract_lengths(mc.tract_lengths)  # voxel -> mm
mc.set_weights(MouseConnectivity.log_scale(mc.weights)) # log scale weights

# ------------------------
# Parse args
# ------------------------

sigma   =  0.001
v_c = 3.9
dt = 0.1

# ------------------------
# Load subject-specific data
# ------------------------

L = mc.tract_lengths
C = mc.weights
C = C /np.max(C)
idelays = (L[C != 0.0] / v_c / dt).astype(np.uint32)


# ------------------------
# Model setup
# ------------------------

setup = {
    "Seids": scipy.sparse.csr_matrix(C),
    "idelays": idelays,
    "params": gm.default_theta,
    "v_c": v_c,
    "horizon": 650,
    "num_item": 1,
    "dt": dt,
    "num_skip": 10,
    "num_time": 300000,
    "init_state": jp.array([.01, -55.0, 0.0, 0.0]).reshape(4,1),
    "num_svar": 4,
    "noise": sigma,
}

# ------------------------
# Parameter update
# ------------------------

theta = gm.default_theta._replace(
    sigma_V=setup["noise"],
    sigma_u=0.1 * setup["noise"],
)

setup["params"] = theta

# ------------------------
# Run simulation
# ------------------------

bold = run_bold_sweep((theta, setup))

bold = np.asarray(bold)
print(bold.shape)

output_file = 'prova'
os.makedirs(output_file, exist_ok=True)
np.savez(output_file, bold=bold)