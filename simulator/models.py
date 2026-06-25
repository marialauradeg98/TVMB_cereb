"""
Models for TVB simulations.
From: Giacomo Preti (AMU)
"""

import collections

import jax
import jax.numpy as jp
import vbjax as vb


class BaseModel:
    """
    Contract a model must satisfy to be used with Simulator
    (simulator/simulator.py), which is agnostic to the number of state
    variables, the coupling variable, and the noise structure of any
    given model.
    Class attributes (set by subclasses)
    -------------------------------------
    Theta : collections.namedtuple
        Parameter namedtuple for the model.
    default_theta : Theta
        Default parameter values.
    NUM_SVAR : int
        Number of state variables (first axis of the state array y).
    COUPLING_VAR_IDX : int
        Index, within the NUM_SVAR state variables, of the variable
        that (a) is exchanged between nodes through the connectome and
        (b) drives the BOLD monitor.
    DEFAULT_INIT_STATE : jax.numpy.ndarray, shape (NUM_SVAR, 1)
        Default initial condition.

    Methods (implemented by subclasses)
    ------------------------------------
    dfun(y, cy, p) -> dy
        Right-hand side of the ODE; dy has shape (NUM_SVAR, ...). cy is
        the *raw*, unscaled delayed network input (Simulator does not
        apply any coupling gain) — each model is responsible for
        scaling it with its own field (e.g. GastModel.Theta.we,
        MontbrioModel.Theta.cr), exactly as in the model's "canonical"
        un-delayed form (see e.g. GastModel.net).
    noise(key, num_node, num_item, p) -> z
        Stochastic increment for one Heun step, shape
        (NUM_SVAR, num_node, num_item). Return 0. for a deterministic
        model (the default).
    """

    Theta = None
    default_theta = None
    NUM_SVAR = None
    COUPLING_VAR_IDX = 0
    DEFAULT_INIT_STATE = None

    @staticmethod
    def dfun(y, cy, p):
        raise NotImplementedError

    @staticmethod
    def noise(key, num_node, num_item, p):
        return 0.0


class GastModel(BaseModel):
    Theta = collections.namedtuple(
        typename='Theta',
        field_names='C, k, Delta, v_r, v_theta, g, E, I, tau_u, b, kappa, tau_s, J, Js, we, sigma_V, sigma_u')

    default_theta = Theta(
        C=100.0, k=0.7, Delta=0.5, v_r=-60.0, v_theta=-40.0, g=1.0, E=0.0, I=46.5,
        tau_u=33.33, b=-2.0, kappa=100.0, tau_s=6.0, Js=1.0, J=15.0, we=0.0, sigma_V=0., sigma_u=0.,
    )

    NUM_SVAR = 4             # r, v, u, s
    COUPLING_VAR_IDX = 0     # r
    DEFAULT_INIT_STATE = jp.array([.01, -55.0, 0.0, 0.0]).reshape(4, 1)

    @staticmethod
    def dfun(y, cy, p: Theta):
        r, v, u, s = y
        C, k, Delta, v_r, v_theta, g, E, I, tau_u, b, kappa, tau_s, J, Js, we, *_ = p
        c_exc = we * cy

        dr = ((Delta * k**2 * (v - v_r)) / (jp.pi * C) + r * (k * (2 * v - v_r - v_theta) - g*s)) / C
        dv = (k * v * (v - v_r - v_theta) - jp.pi * C * r * (Delta + jp.pi * C * r/ k) + k * v_r * v_theta - u + I + g * s * (E - v)) / C
        du = (b * (v - v_r) -u) / tau_u + kappa * r
        ds = - s / tau_s + Js * c_exc + J * r

        return jp.array([dr, dv, du, ds])

    @staticmethod
    def noise(key, num_node, num_item, p: Theta):
        z = jp.zeros((GastModel.NUM_SVAR, num_node, num_item))
        z = z.at[1:3].set(jax.random.normal(key, (2, num_node, num_item)))
        z = z.at[1].multiply(p.sigma_V)
        z = z.at[2].multiply(p.sigma_u)
        return z

    @staticmethod
    def net(y, p):
        Ce, node_params = p
        r = y[0]
        c_exc = Ce @ r  # raw coupling; dfun applies node_params.we internally

        return GastModel.dfun(y, c_exc, node_params)

    @staticmethod
    def stay_positive(y, _):
        # at, set are JAX function used for immutable updates to an array
        # if where<0 is true, set the value to 0, conversely it leaves the original value
        # in this way r, is never negative
        y = y.at[0].set( jp.where(y[0]<0, 0, y[0]) ) #r

        return y


class MontbrioModel(BaseModel):
    """
    Montbrio-Pazo-Roxin mean-field model (state: r, V), reusing the
    dynamics already implemented in vbjax
    (vbjax.neural_mass.mpr_dfun/mpr_default_theta/mpr_r_positive).

    vbjax's mpr_dfun expects coupling as a 2-tuple (c_r, c_v), for
    coupling through both the rate r and the potential V, scaled
    internally by Theta.cr/cv (no separate global-coupling field is
    needed: cr already plays that role for the r channel — see
    examples/mpr_autotune.py's network_model, which scales the raw
    coupling by a plain `coupling_strength` before calling mpr_dfun;
    here that role is played by cr, already part of MPRTheta). Simulator
    only delays/couples a single state variable (COUPLING_VAR_IDX), so
    dfun forwards the raw coupling as the r-channel and leaves the
    V-channel at 0 — matching the default cv=0.0, i.e. no V-coupling.

    vbjax's own examples add noise via a single shared `gfun` scalar
    (e.g. mpr_autotune.py, delays-hcp.py: vb.make_sde(..., gfun=1e-5)),
    applied identically to both r and V. Here it's differentiated per
    variable instead (sigma_r, sigma_V), the same way GastModel.noise
    has separate sigma_V/sigma_u — vbjax's MPRTheta has no such fields,
    so Theta extends it with sigma_r/sigma_V (default 0.0, deterministic).
    """

    Theta = collections.namedtuple(
        typename="MontbrioTheta",
        field_names=list(vb.MPRTheta._fields) + ["sigma_r", "sigma_V"],
    )
    default_theta = Theta(*vb.mpr_default_theta, sigma_r=0.0, sigma_V=0.0)

    NUM_SVAR = 2             # r, V
    COUPLING_VAR_IDX = 0     # r
    DEFAULT_INIT_STATE = jp.array([0.0, -2.0]).reshape(2, 1)  # vb.mpr_default_state

    @staticmethod
    def dfun(y, cy, p: Theta):
        return vb.mpr_dfun(y, (cy, 0.0), p)

    @staticmethod
    def noise(key, num_node, num_item, p: Theta):
        z = jax.random.normal(key, (MontbrioModel.NUM_SVAR, num_node, num_item))
        z = z.at[0].multiply(p.sigma_r)
        z = z.at[1].multiply(p.sigma_V)
        return z

    @staticmethod
    def stay_positive(y, _):
        return vb.mpr_r_positive(y, _)
