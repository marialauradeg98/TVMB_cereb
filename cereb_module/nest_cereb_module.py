import os
import json

import numpy as np
import yaml
import nest
from bsb import from_storage
from mpi4py import MPI

# --- MPI -------------------------------------------------------------------
COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()


def log(*args, **kwargs):
    """Print only from the MPI root rank to avoid duplicated output."""
    if RANK == 0:
        print(*args, **kwargs)


BASE         = os.path.dirname(os.path.abspath(__file__))
CEREB        = os.path.join(BASE, "cerebellar-models")
HDF5         = os.path.join(CEREB, "mouse_cerebellum.hdf5")
CIRCUIT_YAML = os.path.join(CEREB, "circuit.yaml")
SIM_NAME     = "nest_basal_activity"
OUTPUT_DIR   = os.path.join(BASE, "nest_results")

# OpenMP threads per MPI process (override with env NEST_THREADS)
LOCAL_NUM_THREADS = int(os.environ.get("NEST_THREADS", "1"))

# extra (non weight/delay/receptor) synapse parameters we forward to NEST,
# e.g. for tsodyks_synapse
TSODYKS_KEYS = ("U", "u", "x", "y", "tau_rec", "tau_fac", "tau_psc")


def load_sim_config(circuit_yaml=CIRCUIT_YAML, sim_name=SIM_NAME):
    """Read the embedded NEST simulation block from circuit.yaml."""
    with open(circuit_yaml) as f:
        cfg = yaml.safe_load(f)
    return cfg["simulations"][sim_name]


def build_neuron_params(constants):
    """Turn a cell_model 'constants' dict into NEST params.

    Distribution specs (e.g. ``V_m: {distribution: uniform, min, max}``) are
    converted into per-neuron ``nest.random`` parameters.
    """
    params = {}
    for key, value in (constants or {}).items():
        if isinstance(value, dict) and "distribution" in value:
            dist = value["distribution"]
            if dist == "uniform":
                params[key] = nest.random.uniform(min=value["min"], max=value["max"])
            elif dist == "normal":
                params[key] = nest.random.normal(
                    mean=value.get("mean", 0.0), std=value.get("std", 1.0)
                )
            else:
                raise ValueError(f"Unsupported distribution '{dist}' for '{key}'")
        else:
            params[key] = value
    return params


def create_populations(scaffold, sim):
    """Create every cell population in NEST; return node collections and gid offsets."""
    populations = [ct.name for ct in scaffold.get_cell_types()]
    n_neurons = {ps.cell_type.name: len(ps) for ps in scaffold.get_placement_sets()}

    nodes, first_gid = {}, {}
    for ct in populations:
        model = sim["cell_models"][ct]["model"]
        params = build_neuron_params(sim["cell_models"][ct].get("constants"))
        nc = nest.Create(model, n_neurons[ct], params=params or None)
        nodes[ct] = nc
        first_gid[ct] = nc[0].global_id
        log(f"  created {n_neurons[ct]:>6d} x {model:<15s} for {ct}")
    return populations, n_neurons, nodes, first_gid


def connect_populations(scaffold, sim, first_gid):
    """Wire every connectivity set following its connection_model synapses."""
    resolution = float(sim["resolution"])
    for cs in scaffold.get_connectivity_sets():
        tag = cs.tag
        conn_cfg = sim["connection_models"].get(tag)
        if conn_cfg is None:
            log(f"  [skip] no connection_model for '{tag}'")
            continue

        pre_name, post_name = cs.pre_type.name, cs.post_type.name
        pre_locs, post_locs = cs.load_connections().as_globals().all()
        pre_locs, post_locs = np.asarray(pre_locs), np.asarray(post_locs)
        if pre_locs.shape[0] == 0:
            log(f"  [empty] {tag}")
            continue

        # column 0 = cell id; map BSB local id -> contiguous NEST gid
        source = (pre_locs[:, 0] + first_gid[pre_name]).astype(int).tolist()
        target = (post_locs[:, 0] + first_gid[post_name]).astype(int).tolist()
        n = len(source)

        for syn in conn_cfg["synapses"]:
            model = syn.get("model", "static_synapse")

            if model == "gap_junction":
                # gap junctions require a symmetric connection and neuron-model
                # support for secondary events; guard so it can't crash the run.
                try:
                    nest.Connect(
                        source, target,
                        {"rule": "one_to_one", "make_symmetric": True},
                        {"synapse_model": "gap_junction", "weight": np.full(n, syn["weight"])},
                    )
                    log(f"  {tag:<28s} gap_junction        ({n})")
                except Exception as exc:  # noqa: BLE001
                    log(f"  [warn] gap junction '{tag}' skipped: {exc}")
                continue

            syn_spec = {
                "synapse_model": model,
                "weight": np.full(n, syn["weight"]),
                "delay": np.full(n, syn.get("delay", resolution)),
            }
            if syn.get("receptor_type") is not None:
                syn_spec["receptor_type"] = int(syn["receptor_type"])
            for k in TSODYKS_KEYS:
                if k in syn:
                    syn_spec[k] = syn[k]

            nest.Connect(source, target, {"rule": "one_to_one"}, syn_spec)
            rec = syn.get("receptor_type", "-")
            log(f"  {tag:<28s} {model:<15s} rec={rec} ({n})")


def attach_devices(sim, nodes):
    """Create generator devices from config + one spike recorder per population."""
    resolution = float(sim["resolution"])

    # stimulation / background devices (spike_recorders in the config are replaced
    # by our per-population recorders below)
    for dev_name, dev in sim.get("devices", {}).items():
        device = dev["device"]
        if device == "spike_recorder":
            continue
        gen_params = {"rate": dev["rate"]} if "rate" in dev else {}
        gen = nest.Create(device, params=gen_params)
        for tgt in dev.get("targetting", {}).get("cell_models", []):
            nest.Connect(
                gen, nodes[tgt],
                syn_spec={"weight": dev.get("weight", 1.0), "delay": dev.get("delay", resolution)},
            )
        log(f"  device '{dev_name}': {device} -> {dev.get('targetting', {}).get('cell_models')}")

    recorders = {}
    for ct, nc in nodes.items():
        rec = nest.Create("spike_recorder")
        nest.Connect(nc, rec, syn_spec={"delay": resolution})
        recorders[ct] = rec
    return recorders


def save_results(results, duration, resolution, populations):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for ct, r in results.items():
        np.savez(
            os.path.join(OUTPUT_DIR, f"{ct}.npz"),
            times=r["times"], senders=r["senders"],
            n_neurons=r["n_neurons"], first_gid=r["first_gid"],
        )
    meta = {
        "duration": duration,
        "resolution": resolution,
        "populations": populations,
        "n_neurons": {ct: int(results[ct]["n_neurons"]) for ct in populations},
    }
    with open(os.path.join(OUTPUT_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    log(f"Saved NEST results to {OUTPUT_DIR}")


def run_simulation(save=True):
    sim = load_sim_config()
    duration = float(sim["duration"])
    resolution = float(sim["resolution"])
    seed = int(sim.get("seed", 1234))
    modules = sim.get("modules", ["cerebmodule"])

    log("Loading circuit from", HDF5)
    # The config stored in the HDF5 has a code file-dependency resolved relative
    # to the CWD; it lives under the cerebellar-models package folder, so run from
    # there. All our paths (HDF5, yaml, output) are absolute, so chdir is safe.
    os.chdir(CEREB)
    scaffold = from_storage(HDF5)

    nest.ResetKernel()
    for m in modules:
        nest.Install(m)
    nest.SetKernelStatus(
        {"resolution": resolution, "rng_seed": seed, "local_num_threads": LOCAL_NUM_THREADS}
    )

    log("Creating populations...")
    populations, n_neurons, nodes, first_gid = create_populations(scaffold, sim)

    log("Connecting populations...")
    connect_populations(scaffold, sim, first_gid)

    log("Attaching devices...")
    recorders = attach_devices(sim, nodes)

    log(f"Simulating {duration} ms ...")
    nest.Simulate(duration)

    # Each MPI rank only records its local neurons, so gather spikes across ranks
    # and let rank 0 concatenate and save (otherwise every rank clobbers the files
    # with partial data).
    results = {}
    for ct in populations:
        ev = recorders[ct].get("events")
        local_times = np.asarray(ev["times"], dtype=float)
        local_senders = np.asarray(ev["senders"], dtype=np.int64)
        all_times = COMM.gather(local_times, root=0)
        all_senders = COMM.gather(local_senders, root=0)
        if RANK == 0:
            times = np.concatenate(all_times) if all_times else np.array([], dtype=float)
            senders = np.concatenate(all_senders) if all_senders else np.array([], dtype=np.int64)
            results[ct] = {
                "times": times,
                "senders": senders,
                "n_neurons": n_neurons[ct],
                "first_gid": first_gid[ct],
            }
            log(f"  {ct:<15s}: {len(times):>8d} spikes")

    if save and RANK == 0:
        save_results(results, duration, resolution, populations)
    return results if RANK == 0 else None


if __name__ == "__main__":
    run_simulation(save=True)
