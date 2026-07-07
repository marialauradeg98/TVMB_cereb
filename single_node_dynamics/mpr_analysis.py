import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import os



def mpr_model(x, t, params):
    r, V = x
    dr = (1 / params['tau']) * (params['Delta'] / (np.pi * params['tau']) + 2 * r * V)
    dV = (1 / params['tau']) * (V ** 2 + params['eta'] + params['J'] * params['tau'] * r
                                + params['I'] - (np.pi ** 2) * (r ** 2) * (params['tau'] ** 2))
    return [dr, dV]


def solve_mpr(params, x0, t, dt):
    r0, V0 = x0
    sol = odeint(mpr_model, [r0, V0], t, args=(params,), hmax=dt)
    r, V = sol[:, 0], sol[:, 1]
    return r, V


def find_fixed_points(eta, J, Delta=1.0):
    coeffs = [-np.pi**2, J, eta, 0.0, Delta**2 / (4 * np.pi**2)]
    roots = np.roots(coeffs)
    fixed_points = []
    for r in roots:
        if np.isreal(r) and r.real > 1e-9:
            r = r.real
            V = -Delta / (2 * np.pi * r)
            fixed_points.append((r, V))
    return fixed_points


def classify_fixed_point(r, V, J):
    trace = 4 * V
    det = 4 * V**2 - 2 * r * J + 4 * np.pi**2 * r**2
    D = trace**2 - 4 * det
    lambda1 = (trace + np.sqrt(complex(D))) / 2
    lambda2 = (trace - np.sqrt(complex(D))) / 2
    stable = (trace < 0) and (det > 0)
    kind = 'node' if D >= 0 else 'focus'
    return stable, kind, lambda1, lambda2


def classify_point(eta, J, Delta=1.0):
    fps = find_fixed_points(eta, J, Delta)
    stable_kinds = []
    for r, V in fps:
        stable, kind, _, _ = classify_fixed_point(r, V, J)
        if stable:
            stable_kinds.append(kind)

    if len(stable_kinds) >= 2:
        return 2
    elif len(stable_kinds) == 1:
        return 0 if stable_kinds[0] == 'node' else 1
    else:
        return 3


def build_phase_map(n_grid=300, eta_range=(-15, 0), J_range=(0, 20), Delta=1.0):
    eta_tilde = np.linspace(*eta_range, n_grid)
    J_tilde = np.linspace(*J_range, n_grid)
    phase_map = np.zeros((n_grid, n_grid), dtype=int)

    for i, J_t in enumerate(J_tilde):
        for j, eta_t in enumerate(eta_tilde):
            eta = eta_t * Delta
            J = J_t * np.sqrt(Delta)
            phase_map[i, j] = classify_point(eta, J, Delta)

    return phase_map, eta_tilde, J_tilde


def bifurcation_diagram(J, Delta=1.0, eta_range=(-10, 5), n_points=1000):
    etas = np.linspace(*eta_range, n_points)

    r_stable_node, eta_stable_node = [], []
    r_stable_focus, eta_stable_focus = [], []
    r_unstable, eta_unstable = [], []
    v_stable_node, v_stable_focus, v_unstable = [], [], []

    for eta in etas:
        fps = find_fixed_points(eta, J, Delta)
        for r, V in fps:
            stable, kind, _, _ = classify_fixed_point(r, V, J)
            if stable and kind == 'node':
                r_stable_node.append(r); v_stable_node.append(V); eta_stable_node.append(eta)
            elif stable and kind == 'focus':
                r_stable_focus.append(r); v_stable_focus.append(V); eta_stable_focus.append(eta)
            else:
                r_unstable.append(r); v_unstable.append(V); eta_unstable.append(eta)

    return {
        'stable_node': (np.array(eta_stable_node), np.array(r_stable_node), np.array(v_stable_node)),
        'stable_focus': (np.array(eta_stable_focus), np.array(r_stable_focus), np.array(v_stable_focus)),
        'unstable': (np.array(eta_unstable), np.array(r_unstable), np.array(v_unstable)),
    }


def compute_vector_field(params, r_range, V_range, n_grid=20):
    r_vals = np.linspace(*r_range, n_grid)
    V_vals = np.linspace(*V_range, n_grid)
    R, Vg = np.meshgrid(r_vals, V_vals)
    dR, dV = mpr_model([R, Vg], 0, params)
    return R, Vg, dR, dV


def plot_bifurcation_diagram(bif_data, J, Delta, params, savepath=None):
    fig, (ax_r, ax_v) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    eta_sn, r_sn, v_sn = bif_data['stable_node']
    eta_sf, r_sf, v_sf = bif_data['stable_focus']
    eta_u, r_u, v_u = bif_data['unstable']

    sqrt_Delta = np.sqrt(Delta)

    ax_r.plot(eta_sn, r_sn / sqrt_Delta, '.', color='red', markersize=2, label='Stable node')
    ax_r.plot(eta_sf, r_sf / sqrt_Delta, '.', color='tab:blue', markersize=2, label='Stable focus')
    ax_r.plot(eta_u, r_u / sqrt_Delta, '--', color='tab:gray', markersize=2, label='Unstable')
    ax_r.set_ylabel(r'$r/\Delta^{1/2}$')
    ax_r.set_title(f'Bifurcation diagram (J = {J:.2f}, Delta = {Delta:.2f})')
    ax_r.axvline(params['eta'], color='orange', linestyle='-', label = 'Current eta', alpha=0.5)
    ax_r.set_xlim(-10, 0)
    ax_r.set_ylim(0, 2)
    ax_r.legend(loc='upper left')

    ax_v.plot(eta_sn, v_sn, '.', color='red', markersize=2)
    ax_v.plot(eta_sf, v_sf, '.', color='tab:blue', markersize=2)
    ax_v.plot(eta_u, v_u, '--', color='tab:gray', markersize=2)
    ax_v.axvline(params['eta'], color='orange', linestyle='-', label = 'Current eta', alpha =0.5)
    ax_v.set_ylabel('V')
    ax_v.set_xlabel(r'$\bar{\eta}$')
    ax_v.set_xlim(-10, 0)
    ax_v.set_ylim(-3, 0)

    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath, dpi=150)
    plt.show()


def plot_phase_diagram(phase_map, eta_tilde, J_tilde, params=None, savepath=None):
    cmap = ListedColormap(['white', 'gray', 'cyan', 'red'])

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.pcolormesh(eta_tilde, J_tilde, phase_map, cmap=cmap, vmin=0, vmax=3, shading='auto')
    ax.set_xlabel(r'$\bar{\eta}/\Delta$')
    ax.set_ylabel(r'$J/\Delta^{1/2}$')
    ax.set_title('Phase diagram')

    legend_elements = [
        Patch(facecolor='white', edgecolor='k', label='Stable node'),
        Patch(facecolor='gray', label='Stable focus'),
        Patch(facecolor='cyan', label='Bistability'),
        Patch(facecolor='red', label='No stable FP'),
    ]

    if params is not None:
        eta_t_point = params['eta'] / params['Delta']
        J_t_point = params['J'] / np.sqrt(params['Delta'])
        regime = classify_point(params['eta'], params['J'], params['Delta'])

        ax.plot(eta_t_point, J_t_point, marker='*', color='red',
                markersize=18, markeredgewidth=1.5, label='Current parameter set')

        print(f'eta/Delta = {eta_t_point:.3f}, J/sqrt(Delta) = {J_t_point:.3f}')
        print(f'Regime: {REGIME_LABELS[regime]}')

    ax.legend(handles=legend_elements, loc='upper left',
              bbox_to_anchor=(1.02, 1.0), framealpha=0.9)
    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, dpi=150)

    plt.show()


def plot_phase_plane(params, x0_list, r_range=(0, 2), V_range=(-3, 1),
                                       T=100., dt=1e-3, n_grid=20, savepath=None):
    R, Vg, dR, dV = compute_vector_field(params, r_range, V_range, n_grid)
    magnitude = np.sqrt(dR**2 + dV**2)
    magnitude[magnitude == 0] = 1.
    dR_norm, dV_norm = dR / magnitude, dV / magnitude

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.quiver(R, Vg, dR_norm, dV_norm, color='gray', alpha=0.6, angles='xy')

    t = np.arange(int(T / dt)) * dt
    colors = plt.cm.tab10(np.linspace(0, 1, len(x0_list)))

    for x0, c in zip(x0_list, colors):
        r_traj, V_traj = solve_mpr(params, x0, t, dt)
        ax.plot(r_traj, V_traj, color=c, lw=1.5,
                label=f'IC: r0={x0[0]:.2f}, V0={x0[1]:.2f}')
        ax.plot(x0[0], x0[1], 'o', color=c, markersize=7, markeredgecolor='black')
        ax.plot(r_traj[-1], V_traj[-1], 'x', color=c, markersize=9, markeredgewidth=2)

    fps = find_fixed_points(params['eta'], params['J'], params['Delta'])
    for r, V in fps:
        stable, kind, _, _ = classify_fixed_point(r, V, params['J'])
        marker = 'o' if stable else '^'
        fc = 'black' if stable else 'white'
        ax.plot(r, V, marker=marker, color=fc, markeredgecolor='black',
                markersize=11, zorder=5)

    ax.set_xlim(r_range)
    ax.set_ylim(V_range)
    ax.set_xlabel('r')
    ax.set_ylabel('V')
    ax.set_title(f"Phase plane (eta={params['eta']}, J={params['J']}, Delta={params['Delta']})")
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), framealpha=0.9)
    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, dpi=150)
    plt.show()

def compute_timeseries(params, x0, T=100., dt=1e-3):
    t = np.arange(int(T / dt)) * dt
    r, V = solve_mpr(params, x0, t, dt)
    return t, r, V

def run_mpr_analysis(params, dt = 1e-3, T=1000.):

    # 1) Map of eigenvalues

    phase_map, eta_tilde, J_tilde = build_phase_map(
        n_grid=300, eta_range=(-15, 0), J_range=(0, 25), Delta=params['Delta']
    )

    plot_phase_diagram(phase_map, eta_tilde, J_tilde, params=params)

    # 2) Fix J and Delta and do bifurcutaion diagrams r-eta and v-eta
    J_fixed = params['J']
    Delta_fixed = params['Delta']
    bif_data = bifurcation_diagram(J_fixed, Delta_fixed, eta_range=(-10, 0), n_points=2000)
    plot_bifurcation_diagram(bif_data, J_fixed, Delta_fixed, params)

    # 3) Phase portrait for a set of parameters

    # random initial conditions
    n_ic = 6    # number of trajectories
    r_range = (0, 2)
    V_range = (-3, 1)
    seed = 42

    rng = np.random.default_rng(seed)
    r0_vals = rng.uniform(*r_range, n_ic)
    V0_vals = rng.uniform(*V_range, n_ic)
    x0_list = list(zip(r0_vals, V0_vals))

    plot_phase_plane(params, x0_list, r_range=(0, 2), V_range=(-3, 1), T=T, dt=dt, n_grid=20)


REGIME_LABELS = {0: 'stable node', 1: 'stable focus', 2: 'bistability', 3: 'no stable FP'}

if __name__ == "__main__":

    OUTPUT_PATH = 'figs'
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # TVB defaults
    params = {
        "tau": 1.,
        "Delta": 0.5,
        "eta": -5.,
        "J": 15.,
        "I": 0.,
    }

    run_mpr_analysis(params)

    x0 = [1.5, 0.]
    t, r, V = compute_timeseries(params, x0, dt = 0.1, T=100.)

    fig, ax = plt.subplots(1,2)
    ax[0].set_title('V timeseries')
    ax[0].plot(t, V, color='red')
    ax[1].set_title('r timeseries')
    ax[1].plot(t, r, color='black')
    plt.show()
