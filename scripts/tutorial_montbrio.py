import numpy as np
import pandas as pd
import time as tm
import matplotlib.pyplot as plt
from tvb.simulator.integrators import HeunStochastic
from voxcell import RegionMap
from simulator.mouse_conn import MouseConnectivity
from tvb.simulator.lab import *

def plot_raw(t, r,V):
    fig = plt.figure(figsize=(12,4))
    plt.subplot(121)
    plt.plot(t[50:1000], r[50:1000, :30])
    plt.title('Firing rate Activity', fontsize=14)
    plt.xlabel('Time [ms]', fontsize=14)

    plt.subplot(122)
    plt.plot(r[50:1000, :30], V[50:1000, :30])
    plt.title('Phase space', fontsize=14)
    plt.xlabel('r', fontsize=14);plt.ylabel('V', fontsize=14)

    plt.tight_layout()

    plt.show()


def save_steady_state(region_labels, r, V, filename):
    """Salva lo stato stazionario (ultimo istante) di ogni regione in un DataFrame pandas.

    Parameters
    ----------
    region_labels : array-like
        Nomi delle regioni.
    r, V : ndarray
        Array temporali (time, n_regioni) delle variabili di stato.
    filename : str
        Percorso del file .csv in cui salvare il DataFrame.
    """
    df = pd.DataFrame({
        "region": np.asarray(region_labels),
        "r": r[-1, :],
        "V": V[-1, :],
    })
    df.to_csv(filename, index=False)
    print(f"Stato stazionario salvato in {filename}")
    return df


CONNECTIVITY_FILE = "../conn/conn_data/Conn_Allen/ConnAllen.h5"
RVM_FILE          = "../conn/conn_data/Conn_Allen/RegionVolumeMapping.h5"
JSON_FILE         = "../conn/conn_data/1.json"
VOXEL_MM          = 0.025

region_map = RegionMap.load_json(JSON_FILE)
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

conn = connectivity.Connectivity()
conn.weights = mc.weights
conn.tract_lengths = mc.tract_lengths
conn.region_labels = mc.region_labels
conn.centres = mc.centres
conn.speed = np.array(mc.speed)

conn.configure()

params = {
    "eta": -4.6,
    "J": 14.5,
    "Delta": 0.7,
    "tau": 1.
}

mpr = models.MontbrioPazoRoxin(
    J     = np.r_[params['J']],
    eta     = np.r_[params['eta']],
    Delta = np.r_[params['Delta']],
    tau   = np.r_[params['tau']],
)

G = 0.001
con_coupling = coupling.Linear(a= np.array(G))

dt = 0.1
heun_det = integrators.HeunDeterministic(dt=dt)

mon_tavg = monitors.TemporalAverage(period=0.1)

sim = simulator.Simulator(
    model=mpr,
    connectivity=conn,
    coupling=con_coupling,
    conduction_speed=mc.speed,
    integrator=heun_det,
    monitors=[mon_tavg]
)

sim.configure()

tic = tm.time()
sim_len= 1000.
(Tavg_time, Tavg_data), = sim.run(simulation_length=sim_len)

r      = Tavg_data[:,0,:,0]
V      = Tavg_data[:,1,:,0]

steady_state_df = save_steady_state(
    conn.region_labels, r, V, "steady_state_montbrio_det.csv"
)

plot_raw(Tavg_time, r, V)


seed = 42
sigma = 0.01
noise = noise.Additive(nsig = np.array([sigma, sigma*2]), noise_seed = seed)
heun_stoc = HeunStochastic(dt=dt, noise = noise)

sim = simulator.Simulator(
    model=mpr,
    connectivity=conn,
    coupling=con_coupling,
    conduction_speed=mc.speed,
    integrator=heun_stoc,
    monitors=[mon_tavg]
)

sim.configure()

(Tavg_time, Tavg_data), = sim.run(simulation_length=sim_len)
r      = Tavg_data[:,0,:,0]
V      = Tavg_data[:,1,:,0]

plot_raw(Tavg_time, r, V)


