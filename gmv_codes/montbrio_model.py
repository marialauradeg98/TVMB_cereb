import jax.numpy as jp
import collections

# Model implementation
Theta = collections.namedtuple(
    typename='Theta',
    field_names= 'Delta, eta, J, I, cr, cv, tau, sigma_r, sigma_V'
)

default_theta = Theta(
    Delta = 1.0, eta = -5.0, J = 15.0, I=0, cr=1., cv=0., tau = 1., sigma_r = 0., sigma_V = 0.
)


def dfun(y, cy, p: Theta):
    r,V = y

    Delta, eta, J, I, cr, cv, tau, *_= p

    I_c = cr * cy[0] + cv * cy[1]

    dr = (1 / tau) * (Delta / (jp.pi * tau) + 2 * r * V)
    dV = (1 / tau) * (V ** 2 + eta + J * tau *
                        r + I + I_c - (jp.pi ** 2) * (r ** 2) * (tau ** 2))

    return jp.array([dr,dV])

def net(y, p):
    Ce, node_params = p
    r, V = y[0], y[1]
    c_r = Ce @ r
    c_v = Ce @ V
    cy = (c_r, c_v)
    return dfun(y, cy, node_params)

def stay_positive(y, _):
    # at, set are JAX function used for immutable updates to an array
    # if where<0 is true, set the value to 0, conversely it leaves the original value
    # in this way r, is never negative
    y = y.at[0].set( jp.where(y[0]<0, 0, y[0]) ) #r

    return y
