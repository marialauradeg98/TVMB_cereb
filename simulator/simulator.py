"""
Simulation of tvb network using vbjax
From: Giacomo Preti
"""

import os
from typing import Optional, Type

import jax
import jax.numpy as jp
import numpy as np
import scipy.sparse
import vbjax as vb

from simulator.models import BaseModel
from simulator.mouse_conn import MouseConnectivity


class Simulator:
    """
    Runs network simulations with a BOLD monitor on a mouse structural
    connectome, for any model implementing the BaseModel contract
    (simulator/models.py: Theta/default_theta, NUM_SVAR,
    COUPLING_VAR_IDX, DEFAULT_INIT_STATE, dfun, noise).

    Combines the setup from GMV/snakeproject2/run_simulation.py (sparse
    weights + delays) with the integration machinery from
    GMV/snakeproject2/simulation_utils.py (delayed coupling, Heun
    integrator, BOLD monitor), adapted to (a) take a single
    MouseConnectivity SC (shared across mice) instead of per-subject
    human DK-atlas files, and (b) be agnostic to the model's number of
    state variables, coupling variable and noise structure, instead of
    hardcoding GastModel. The delayed network input passed to model.dfun
    is raw/unscaled — each model applies its own coupling gain
    internally (see simulator/models.py).

    Any SC preprocessing — weights (log-scaling,
    percentile thresholding) and tract_lengths (voxel→mm conversion) —
    is expected to already be applied to `mouse_conn` by the caller
    (e.g. via MouseConnectivity.set_weights/threshold_weights and
    mouse_conn.preprocess_tract_lengths) before constructing the
    Simulator; see scripts/one_sim_gast.py for an example.

    Example
    -------
    mc = MouseConnectivity(CONNECTIVITY_FILE, voxel_size_mm=0.025, region_map=region_map)
    mc.tract_lengths = mc.preprocess_tract_lengths(mc.tract_lengths)
    sim = Simulator(mc, GastModel)
    theta = GastModel.default_theta._replace(we=1.0, J=15.0, sigma_V=0.1, sigma_u=0.01)
    bold = sim.run(theta)
    """

    def __init__(
        self,
        mouse_conn: MouseConnectivity,
        model: Type[BaseModel],
        v_c: float = 4.,
        dt: float = 0.1,
        horizon: int = 650,
        num_item: int = 1,
        num_skip: int = 10,
        num_time: int = 300000,
        init_state: Optional[jp.ndarray] = None,
    ):
        self.mouse_conn = mouse_conn
        self.model      = model
        self.v_c        = v_c
        self.dt         = dt
        self.horizon    = horizon
        self.num_item   = num_item
        self.num_skip   = num_skip
        self.num_time   = num_time
        self.init_state = (
            init_state if init_state is not None else model.DEFAULT_INIT_STATE
        )

        self.csr_weights, self.idelays = self._setup_connectivity()
        if self.idelays.max() >= self.horizon - 2:
            raise ValueError(
                f"max idelay {self.idelays.max()} too large for horizon "
                f"{self.horizon}; increase horizon or v_c"
            )



    def _setup_connectivity(self):
        """
        Build the sparse weights matrix and integer delays (idelays) from
        mouse_conn.weights/tract_lengths, the way run_simulation.py did
        for the per-subject DK-atlas weights/lengths. tract_lengths is
        used as-is: convert it to mm beforehand (e.g. via
        mouse_conn.preprocess_tract_lengths) if needed.
        """
        C = self.mouse_conn.weights
        C = C / np.max(C)
        L = self.mouse_conn.tract_lengths
        idelays = (L[C != 0.0] / self.v_c / self.dt).astype(np.uint32)
        csr_weights = scipy.sparse.csr_matrix(C)
        return csr_weights, idelays

    # ── integration, adapted from simulation_utils.make_jp_runsim_for_bold ──

    def _make_run_sim(self, params: BaseModel.Theta):
        """
        Build the jitted simulation function: delayed coupling, a Heun
        integrator for model.dfun (noise from model.noise), and a BOLD
        monitor fed by the model's coupling variable — same machinery as
        simulation_utils.make_jp_runsim_for_bold, closed over this
        Simulator's model/connectivity/integration settings.
        """
        model         = self.model
        num_svar      = model.NUM_SVAR
        coupling_idx  = model.COUPLING_VAR_IDX
        csr_weights   = self.csr_weights
        idelays       = self.idelays
        horizon       = self.horizon
        num_item      = self.num_item
        num_time      = self.num_time
        dt            = self.dt
        num_skip      = self.num_skip

        num_out_node, num_node = csr_weights.shape
        horizonm1 = horizon - 1
        j_indices = jp.array(csr_weights.indices)
        j_weights = jp.array(csr_weights.data)
        idelays2  = jp.array(horizon + np.c_[idelays, idelays - 1].T)

        _csr_rows = np.concatenate([
            i * np.ones(n, 'i') for i, n in enumerate(np.diff(csr_weights.indptr))
        ])
        j_csr_rows = jp.array(_csr_rows)

        def cfun(buffer, t):
            wxij = j_weights.reshape(-1, 1) * buffer[j_indices, (t - idelays2) & horizonm1]
            cx = jp.zeros((2, num_out_node, num_item))
            cx = cx.at[:, j_csr_rows].add(wxij)
            return cx

        def dfun(x, cx):
            # raw, unscaled coupling: each model's dfun applies its own
            # coupling gain internally (e.g. GastModel.Theta.we,
            # MontbrioModel.Theta.cr)
            Ce_aff = cx.reshape(num_node, num_item)
            return model.dfun(x, Ce_aff, params)

        def heun(x, cx, key):
            z = model.noise(key, num_node, num_item, params)
            dx1 = dfun(x, cx[0])
            dx2 = dfun(x + dt * dx1 + z, cx[1])
            return x + dt / 2 * (dx1 + dx2) + z

        dt_bold = num_skip / 1000
        bold_buf0, bold_step, bold_samp = vb.make_bold(
            shape=(num_node, num_item),
            dt=dt_bold,
            p=vb.bold_default_theta,
        )

        def op(sim, T):
            buffer   = sim['buf']
            bold_buf = sim['bold']
            keys     = jax.random.split(sim['rng_key'], num_skip + 1)
            x        = sim['x']
            assert x.shape == (num_svar, num_node, num_item)

            for i in range(num_skip):
                t = i + T * num_skip
                cx = cfun(buffer, t)
                x = heun(x, cx, keys[i])
                buffer = buffer.at[:, t % horizon].set(x[coupling_idx])

            bold_buf = bold_step(bold_buf, x[coupling_idx])
            _, bold_t = bold_samp(bold_buf)

            sim['x']       = x
            sim['buf']     = buffer
            sim['bold']    = bold_buf
            sim['rng_key'] = keys[-1]
            return sim, bold_t

        def run_sim_jp(rng_key, init_state):
            buffer = jp.zeros((num_node, horizon, num_item)) + init_state[coupling_idx]
            init = {
                'buf':     buffer,
                'bold':    bold_buf0,
                'x':       jp.zeros((num_svar, num_node, num_item)) + init_state.reshape(-1, 1, 1),
                'rng_key': rng_key,
            }
            ts = jp.r_[:num_time // num_skip]

            _, bold = jax.lax.scan(op, init, ts)
            cut = 2000
            return bold[cut::100]  # downsample to 1 timepoint per second

        return jax.jit(run_sim_jp)

    # ── run, adapted from simulation_utils.run_bold_sweep ─────────────

    def run(self, params: BaseModel.Theta, seed: int = 42) -> np.ndarray:
        """
        Run one BOLD simulation for the given model parameters (e.g.
        model.default_theta._replace(...) for a parameter sweep point).
        """
        run_sim_jp = self._make_run_sim(params)
        rng_key = jax.random.PRNGKey(seed)
        bold = run_sim_jp(rng_key, self.init_state)
        bold.block_until_ready()
        return np.asarray(bold)

    @staticmethod
    def save_bold(bold: np.ndarray, output_file: str):
        """Save a BOLD array to output_file (.npz, key 'bold')."""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        np.savez(output_file, bold=bold)