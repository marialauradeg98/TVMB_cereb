import jax.numpy as jp
import collections  

# Model implementation

Theta = collections.namedtuple(
    typename='Theta',
    field_names='C, k, Delta, v_r, v_theta, g, E, I, tau_u, b, kappa, tau_s, J, Js, we, sigma_V, sigma_u')

default_theta = Theta(
    C=100.0, k=0.7, Delta=0.5, v_r=-60.0, v_theta=-40.0, g=1.0, E=0.0, I=46.5, 
    tau_u=33.33, b=-2.0, kappa=100.0, tau_s=6.0, Js=1.0, J=15.0, we=0.0, sigma_V=0., sigma_u=0.,
)

def dfun(y, cy, p: Theta):
    "Adaptive QIF model with dopamine modulation."

    r, v, u, s = y
    c_exc = cy
    C, k, Delta, v_r, v_theta, g, E, I, tau_u, b, kappa, tau_s, J, Js, *_ = p

    dr = ((Delta * k**2 * (v - v_r)) / (jp.pi * C) + r * (k * (2 * v - v_r - v_theta) - g*s)) / C
    dv = (k * v * (v - v_r - v_theta) - jp.pi * C * r * (Delta + jp.pi * C * r/ k) + k * v_r * v_theta - u + I + g * s * (E - v)) / C
    du = (b * (v - v_r) -u) / tau_u + kappa * r
    ds = - s / tau_s + Js * c_exc + J * r

    return jp.array([dr, dv, du, ds])

def net(y, p):
    "Canonical form for network of dopa nodes."
    Ce, node_params = p
    r = y[0]
    c_exc = node_params.we * Ce @ r
  
    return dfun(y, (c_exc), node_params)

def stay_positive(y, _):
    # at, set are JAX function used for immutable updates to an array
    # if where<0 is true, set the value to 0, conversely it leaves the original value 
    # in this way r, is never negative
    y = y.at[0].set( jp.where(y[0]<0, 0, y[0]) ) #r

    return y