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

    def __init__(
        self,
        mouse_conn: MouseConnectivity,
        model: Type[BaseModel],
        v_c: float = 4.,
        dt: float = 0.1,
        horizon: Optional[int] = None,
        num_item: int = 1,
        num_skip: int = 10,        # ms — monitor sampling period
        num_time: int = 300000,    # ms — total simulated time
        init_state: Optional[jp.ndarray] = None,
    ):
        # num_time (total time) and num_skip (sampling period) are in
        # MILLISECONDS. They are converted to integration steps via dt inside
        # _make_run_sim (n_inner = num_skip/dt steps per sample, n_samples =
        # num_time/num_skip samples), so a run keeps the same duration and
        # sampling period at any dt — change dt alone. dt_bold = num_skip/1000 s
        # is then correct (num_skip is already ms).
        self.mouse_conn = mouse_conn
        self.model      = model
        self.v_c        = v_c
        self.dt         = dt
        self.num_item   = num_item
        self.num_skip   = num_skip
        self.num_time   = num_time
        self.init_state = (
            init_state if init_state is not None else model.DEFAULT_INIT_STATE
        )

        self.csr_weights, self.idelays = self._setup_connectivity()
        max_idelay = int(self.idelays.max())
        # The delay ring buffer reads with a bitwise mask ((t-…) & (horizon-1))
        # but writes with % horizon; these agree ONLY when horizon is a power
        # of two, so horizon must be a power of two > max_idelay + 1. Auto-size
        # it when not given (idelays grow like 1/dt, so a fixed value overflows
        # at small dt, and a non-power-of-two mis-reads the delayed values).
        if horizon is None:
            self.horizon = self._next_pow2(max_idelay + 2)
        else:
            if horizon & (horizon - 1) != 0:
                raise ValueError(
                    f"horizon must be a power of two (got {horizon}); the "
                    f"delay buffer is indexed with a bitwise-mask modulo."
                )
            if max_idelay >= horizon - 2:
                raise ValueError(
                    f"max idelay {max_idelay} too large for horizon "
                    f"{horizon}; increase horizon or v_c"
                )
            self.horizon = horizon

    @staticmethod
    def _next_pow2(n: int) -> int:
        """Smallest power of two >= n."""
        p = 1
        while p < n:
            p <<= 1
        return p

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

        # num_skip / num_time are in ms; convert to integration steps via dt.
        n_inner   = int(round(num_skip / dt))        # steps per monitor sample
        n_samples = int(round(num_time / num_skip))  # number of samples

        # dt_bold in seconds: num_skip is the sampling period in ms (/1000 -> s).
        dt_bold = num_skip / 1000
        bold_buf0, bold_step, bold_samp = vb.make_bold(
            shape=(num_node, num_item),
            dt=dt_bold,
            p=vb.bold_default_theta,
        )

        def op(sim, T):
            buffer   = sim['buf']
            keys     = jax.random.split(sim['rng_key'], n_inner + 1)
            x        = sim['x']

            # inner loop over the n_inner integration steps of one sampling
            # window, Python-unrolled so XLA can fuse the coupling and ring
            # buffer updates (a lax.scan here threads the whole buffer through
            # the carry and is ~12x slower). n_inner = num_skip/dt: fine for
            # the usual dt (e.g. 100 at dt=0.1); a very small dt makes it large
            # (2000 at dt=0.005) and compilation slow — that's the trade-off.
            ta_sum = jp.zeros((num_node, num_item))
            for i in range(n_inner):
                t = i + T * n_inner
                cx = cfun(buffer, t)
                x = heun(x, cx, keys[i])
                buffer = buffer.at[:, t % horizon].set(x[coupling_idx])
                ta_sum = ta_sum + x[coupling_idx]

            # temporal average of the coupling variable over the window
            # (sampling period = num_skip ms), fed to the BOLD monitor.
            tavg = ta_sum / n_inner
            bold_buf = bold_step(sim['bold'], tavg)
            _, bold_t = bold_samp(bold_buf)

            sim = {
                'x':       x,
                'buf':     buffer,
                'bold':    bold_buf,
                'rng_key': keys[-1],
            }
            return sim, (tavg, bold_t)

        def run_sim_jp(rng_key, init_state):
            buffer = jp.zeros((num_node, horizon, num_item)) + init_state[coupling_idx]
            init = {
                'buf':     buffer,
                'bold':    bold_buf0,
                'x':       jp.zeros((num_svar, num_node, num_item)) + init_state.reshape(-1, 1, 1),
                'rng_key': rng_key,
            }
            ts = jp.r_[:n_samples]

            _, (tavg, bold) = jax.lax.scan(op, init, ts)
            cut = 2000
            # tavg at full sampling resolution (num_skip ms); bold downsampled
            # to ~1 timepoint per second.
            return tavg[cut:], bold[cut::100]

        return jax.jit(run_sim_jp)

    # ── run, adapted from simulation_utils.run_bold_sweep ─────────────

    def run(self, params: BaseModel.Theta, seed: int = 42,
            return_tavg: bool = False):
        """
        Run one BOLD simulation for the given model parameters (e.g.
        model.default_theta._replace(...) for a parameter sweep point).

        Returns the BOLD array (downsampled to ~1 timepoint/s). With
        return_tavg=True, returns (tavg, bold) where tavg is the temporal
        average of the coupling variable at the sampling period (num_skip ms),
        transient removed.
        """
        run_sim_jp = self._make_run_sim(params)
        rng_key = jax.random.PRNGKey(seed)
        tavg, bold = run_sim_jp(rng_key, self.init_state)
        bold.block_until_ready()
        if return_tavg:
            return np.asarray(tavg), np.asarray(bold)
        return np.asarray(bold)

    @staticmethod
    def save_bold(bold: np.ndarray, output_file: str):
        """Save a BOLD array to output_file (.npz, key 'bold')."""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        np.savez(output_file, bold=bold)