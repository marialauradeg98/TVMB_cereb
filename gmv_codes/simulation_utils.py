import numpy as np
import scipy
import jax
import os
import vbjax as vb
import gmv_codes.gast_model as gm

PROJECT_DIR = os.path.abspath(os.path.join(os.getcwd(), '../..'))
RESOURCES_DIR = os.path.join(PROJECT_DIR, 'resources')

def setup_delays(L, Ceids, dt, v_c=3.9):
    """
    Docstring per setup_delays
    :param L: Patient tract lenghts
    :param Ceids: Stacked connectomes (output from stack_connectomes)
    :param dt: integration time step
    :param v_c: conductance velocity
    """
    Leids = np.vstack([L, L, L, L])
    return (Leids[Ceids != 0.0] / v_c / dt).astype(np.uint32)

def setup_ja(zscoresdf, weights, pid, mean, std):
    """
    Setup the self-excitation parameter based on regional zscores from centile,
    in the same order as the columns of the weights matrix.

    :param zscoresdf: DataFrame of zscores (rows=patients, columns=regions)
    :param weights: DataFrame or array of connectome (columns define order)
    :param pid: patient ID
    :param mean: scalar or array to sweep
    :param std: scalar or array to sweep
    :return: np.ndarray of shape (num_combinations, num_regions)
    """
    def rescale_from_zscores(z_scores, mean, std):
        return mean + std * z_scores

    # Get patient zscores
    pid_scores = zscoresdf.loc[int(pid)].drop(index=['age','sex'])

    # Map pid_scores to weights columns
    param_vector = pid_scores

    condition = weights.columns.isin(param_vector.index)

    # If mean/std are arrays (e.g., parameter sweep)
    mean = np.atleast_1d(mean)
    std = np.atleast_1d(std)
    n_comb = len(mean)
    n_regions = len(weights.columns)

    filled_matrix = np.zeros((n_comb, n_regions))

    for i, (m, s) in enumerate(zip(mean, std)):
        rescaled = rescale_from_zscores(param_vector, mean=m, std=s)
        # Fill with rescaled where condition is True, 13 otherwise
        filled_matrix[i, :] = np.where(condition,
                                       weights.columns.map(rescaled),
                                       13)
    filled_matrix[filled_matrix < 0] = 0

    return filled_matrix.reshape(-1,1)

# =============
# INTEGRATION
# =============

def make_jp_runsim_for_bold(
        csr_weights: scipy.sparse.csr_matrix,
        idelays: np.ndarray,
        params: np.ndarray,
        horizon: int,
        rng_seed=43,
        num_item=8,
        num_svar=10,
        num_time=1000,
        dt=0.1,
        num_skip=5,
):
    """
    Integration function with BOLD monitor for the mdoel with time delays.

    :param csr_weights: Stacked connectivity matrices
    :type csr_weights: scipy.sparse.csr_matrix
    :param idelays: Delays computed from the Length matrix
    :type idelays: np.ndarray
    :param params: Nodes parameters
    :type params: np.ndarray
    :param horizon: Maximum delay in number of steps
    :type horizon: int
    :param rng_seed: Random seed for JAX PRNG, defaults to 43
    :param num_item: Number of simuations to run
    :param num_svar: Number of variables
    :param num_time: Total time for simulation in ms
    :param dt: Integration step in ms
    :param num_skip: Number of steps not to be recorded
    """
    import jax
    import jax.numpy as jp



    num_out_node, num_node = csr_weights.shape
    horizonm1 = horizon - 1
    j_indices = jp.array(csr_weights.indices)
    j_weights = jp.array(csr_weights.data)
    j_indptr = jp.array(csr_weights.indptr)
    assert idelays.max() < horizon-2
    idelays2 = jp.array(horizon + np.c_[idelays, idelays-1].T)

    _csr_rows = np.concatenate([i*np.ones(n, 'i')
                                for i, n in enumerate(np.diff(csr_weights.indptr))])
    j_csr_rows = jp.array(_csr_rows)

    def cfun(buffer, t):
        wxij = j_weights.reshape(-1,1) * buffer[j_indices, (t - idelays2) & horizonm1]
        cx = jp.zeros((2, num_out_node, num_item))
        cx = cx.at[:, j_csr_rows].add(wxij)
        return cx

    def dfun(x, cx):
        # cx.shape = (3*num_node, num_node, ...)
        Ce_aff = cx.reshape(num_node, num_item)
        return gm.dfun(x,
                       params.we*Ce_aff,
                       params)

    def heun(x, cx, key):
        z = jp.zeros((num_svar, num_node, num_item))
        z = z.at[1:3].set(jax.random.normal(key, (2,num_node,num_item)))
        z = z.at[1].multiply(params.sigma_V)
        z = z.at[2].multiply(params.sigma_u)
        dx1 = dfun(x, cx[0])
        dx2 = dfun(x + dt*dx1 + z, cx[1])
        return x + dt/2*(dx1 + dx2) + z

    # --- BOLD monitor ---
    dt_bold = num_skip / 1000
    bold_buf0, bold_step, bold_samp = vb.make_bold(
        shape=(num_node, num_item),
        dt=dt_bold,
        p=vb.bold_default_theta,
    )

    def op(sim, T):
        buffer = sim['buf']
        bold_buf = sim['bold']
        keys = jax.random.split(sim['rng_key'], num_skip+1)
        x = sim['x']
        heun_count = sim['heun_count']
        assert x.shape == (num_svar, num_node, num_item)

        for i in range(num_skip):
            t = i + T*num_skip
            cx = cfun(buffer, t)
            x = heun(x, cx, keys[i])
            heun_count = heun_count + 1
            buffer = buffer.at[:, t % horizon].set(x[0])

        bold_buf = bold_step(bold_buf, x[0])
        _, bold_t = bold_samp(bold_buf)

        sim['x'] = x
        sim['buf'] = buffer
        sim['bold'] = bold_buf
        sim['rng_key'] = keys[-1]
        sim['heun_count'] = heun_count
        return sim, (bold_t, x)

    def run_sim_jp(rng_key, init_state):
        buffer = jp.zeros((num_node, horizon, num_item)) + init_state[0]
        init = {
            'buf': buffer,
            'bold': bold_buf0,
            'x': jp.zeros((num_svar, num_node, num_item)) + init_state.reshape(-1,1,1),
            'rng_key': rng_key,
            'heun_count': jp.array(0, dtype=jp.int32),
        }
        ts = jp.r_[:num_time//num_skip]

        final_state , ys = jax.lax.scan(op, init, ts)
        bold, x = ys
        cut=0
        return bold[cut::100], x[cut:], final_state['heun_count'] # downsample to 1 timepoint per second

    return jax.jit(run_sim_jp)

def run_bold_sweep(p, seed=42):

    theta, setup = p
    run_sim_jp = make_jp_runsim_for_bold(
        csr_weights=setup['Seids'],
        idelays=setup['idelays'],
        params=theta,
        horizon=setup['horizon'],
        num_item=setup['num_item'],
        dt=setup['dt'],
        num_skip=setup['num_skip'],
        num_time=setup['num_time'],
        num_svar=setup['num_svar'],
    )

    rng_key = jax.random.PRNGKey(seed)
    xs = run_sim_jp(rng_key, setup['init_state'])
    jax.tree_util.tree_map(lambda a: a.block_until_ready(), xs)
    return xs