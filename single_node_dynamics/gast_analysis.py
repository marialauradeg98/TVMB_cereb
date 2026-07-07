import numpy as np
from scipy.integrate import odeint

def gast_model(x, t, params):
    r, v, u, s = x
    dr = ((params['Delta'] * params['k']**2 * (v - params['v_r'])) / (np.pi * params['C']) + r * (params['k'] * (2 * v - params['v_r'] - params['v_theta']) - params['g']*s)) / params['C']
    dv = (params['k'] * v * (v - params['v_r'] - params['v_theta']) - np.pi * params['C'] * r * (params['Delta'] + np.pi * params['C'] * r/ params['k']) + params['k'] * params['v_r'] * params['v_theta'] - u + params['I'] + params['g'] * s * (params['E'] - v)) / params['C']
    du = (params['b'] * (v - params['v_r']) -u) / params['tau_u'] + params['kappa'] * r
    ds = - s / params['tau_s'] + params['Js'] * params['c_exc'] + params['J'] * r
    return [dr, dv, du, ds]

def solve_gast(params, x0, t, dt):
    r0, v0, u0, s0 = x0
    sol = odeint(gast_model, [r0, v0, u0, s0], t, args=(params,), hmax=dt)
    r, v, u, s = sol[:, 0], sol[:, 1], sol[:, 2], sol[:, 3]
    return r, v, u, s

def compute_timeseries(params, x0, T=100., dt=1e-3):
    t = np.arange(int(T / dt)) * dt
    r, v, u, s = solve_gast(params, x0, t, dt)
    return t, r, v, u, s