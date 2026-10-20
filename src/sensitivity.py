"""Sensitivity analysis: finite differences, analytical checks, implicit and forward sensitivities.

Math doc: Section 7.1 (definitions, Eq 41), Section 7.2 (analytical check,
Eq 42-43), Section 7.3 (finite differences, Eq 44, step size), Section 7.4
(implicit differentiation through the shooting residual, Eq 45) and
Section 7.5 (forward sensitivity equations, Eq 46). The smooth max of
Section 7.6 lives in metrics.py.

Outputs y of interest: duty Q, max stream-to-stream difference, peak wall rate
max_t R_w. Parameters p: flow ratio, fin height, ramp rate, Colburn multiplier
zeta. Everything a study perturbs lives in the flat ``params`` dict, so every
sensitivity is "copy params, change one key, re-run".

Imports: numpy, scipy.sparse (augmented Jacobian pattern),
scipy.integrate.solve_ivp, src.correlations, src.steady and src.transient.

The ``model`` callable
----------------------
Several functions take ``model(params) -> y``, a function that runs whatever
solve is needed and returns one float. For example:

    def duty(p):
        props = properties.build_surrogates(p)          # rebuild if p changes properties
        return steady.energy_balance(steady.solve_steady(p, props), p, props)["Q_h"]

If the perturbed key cannot change the properties (fin geometry, flows,
zeta_j, ramp size inside the fitted range), the model may reuse one ``props``
built outside, which is much faster.
"""

import numpy as np
from scipy import sparse
from scipy.integrate import solve_ivp

from src import correlations, steady, transient


def with_parameter(params, key, value):
    """Copy of params with one entry changed.

    Parameters
    ----------
    params : dict
        Flat parameter dict.
    key : str
        Key to change (must already exist, to catch typos).
    value : object
        New value.

    Returns
    -------
    new_params : dict
        Shallow copy with new_params[key] = value; params is unchanged.
    """
    # 1. If key not in params: raise KeyError (a typo would otherwise add a
    #    new key that nothing reads, and the "sensitivity" would be zero).
    # 2. new_params = dict(params); new_params[key] = value; return it.
    # Pitfall: never write params[key] = ... on config.BASE_CASE or on a
    #    dict shared with other runs. If value is a numpy array (e.g.
    #    colburn_beta), store a copy so later in-place edits cannot leak back.
    # Check: after with_parameter(params, "b_h", 1.0), params["b_h"] is
    #    unchanged.
    raise NotImplementedError


def optimal_rel_step(rtol):
    """Relative finite-difference step that balances truncation and round-off.

    Section 7.3: the central difference has truncation error O(dp^2) and
    round-off error O(eta/dp), where eta is the solver's relative tolerance;
    balancing them gives dp/p ~ eta^(1/3).

    Parameters
    ----------
    rtol : float
        Relative tolerance eta of the solver used inside the model.

    Returns
    -------
    rel_step : float
        Suggested relative step dp/p [-].
    """
    # 1. Return rtol ** (1/3).
    # Pitfall: with the default rtol ~1e-6 this is 1e-2 and the derivative
    #    is only good to ~1e-4. Tighten the solver (params["sens_rtol"] ~1e-8
    #    for bvp_tol and ivp_rtol) whenever outputs are differenced.
    # Check: optimal_rel_step(1e-9) == 1e-3.
    raise NotImplementedError


def central_difference(model, params, key, rel_step):
    """Local sensitivity dy/dp by a central difference.

    Eq 44: S_y,p ~ [y(p + dp) - y(p - dp)] / (2 dp), with dp = rel_step |p|.

    Parameters
    ----------
    model : callable
        model(params) -> float (see module docstring).
    params : dict
        Base parameters (with tightened solver tolerances, see
        optimal_rel_step).
    key : str
        Parameter to perturb.
    rel_step : float
        Relative step dp/p [-].

    Returns
    -------
    dy_dp : float
        Sensitivity [units of y per unit of p].
    """
    # 1. p0 = params[key]; dp = rel_step * abs(p0) (if p0 == 0, use an
    #    absolute step and say so).
    # 2. y_plus = model(with_parameter(params, key, p0 + dp));
    #    y_minus = model(with_parameter(params, key, p0 - dp)).
    # 3. Return (y_plus - y_minus) / (2 dp).
    # Pitfall: keep N, the mesh-related settings and the tolerances identical
    #    in the two runs; otherwise you difference two different numerical
    #    errors.
    # Pitfall: "flow ratio" mdot_c/mdot_h at fixed mdot_h is the key "mdot_c":
    #    the normalized sensitivity to the ratio equals the normalized
    #    sensitivity to mdot_c.
    # Check: for the constant-property effectiveness, the result matches
    #    steady.depsilon_dNTU times dNTU/dp (Section 7.2), see
    #    epsilon_chain_rule.
    raise NotImplementedError


def normalized_sensitivity(dy_dp, p0, y0):
    """Dimensionless sensitivity.

    Eq 41: S~_y,p = (p/y) dy/dp. A value of 0.5 means a 1 % change in p moves
    y by 0.5 %.

    Parameters
    ----------
    dy_dp : float
        Local sensitivity.
    p0 : float
        Parameter value at the base point.
    y0 : float
        Output value at the base point.

    Returns
    -------
    S_tilde : float
        Normalized sensitivity [-].
    """
    # 1. Return p0 / y0 * dy_dp. Raise ValueError if y0 == 0 (normalization
    #    undefined; report dy_dp instead).
    # Check: y = c p^n gives S~ = n for any c.
    raise NotImplementedError


def sensitivity_table(model, params, keys, rel_step):
    """Rank parameters by normalized sensitivity of one output.

    Section 7.1: parameters are ranked by |S~|.

    Parameters
    ----------
    model : callable
        model(params) -> float.
    params : dict
        Base parameters.
    keys : list of str
        Parameters to study, e.g. ["mdot_c", "b_h", "b_c", "ramp_t", "zeta_j"].
        Fin height on both sides is two keys; their S~ add (to first order) for
        a change of both.
    rel_step : float
        Relative step, usually optimal_rel_step(params["sens_rtol"]).

    Returns
    -------
    table : list of dict
        One row per key, sorted by |S_tilde| descending, each
        {"key", "p0", "y0", "dy_dp", "S_tilde"}.
    """
    # 1. y0 = model(params) once.
    # 2. For each key: dy_dp = central_difference(model, params, key, rel_step);
    #    S_tilde = normalized_sensitivity(dy_dp, params[key], y0).
    # 3. Sort rows with sorted(..., key=lambda r: abs(r["S_tilde"]), reverse=True).
    # Check: a parameter the output cannot depend on (e.g. ramp_t for a
    #    steady output) gets S_tilde ~ 0 (at the round-off level).
    raise NotImplementedError


def epsilon_chain_rule(params, props, key, rel_step):
    """Effectiveness sensitivity from the analytical chain rule (constant properties).

    Eq 42-43: d eps/dp = (d eps/d NTU) (L/C_min) dU'/dp, with
    dU'/dp = U'^2 [ (1/a_h^2) da_h/dp + (1/a_c^2) da_c/dp ] and d eps/d NTU
    from steady.depsilon_dNTU. Only the cheap conductance calculation is
    differenced, never a BVP. Used to test the full finite-difference
    sensitivity code (Section 7.2).

    Parameters
    ----------
    params : dict
        Base parameters.
    props : dict
        CONSTANT-property surrogates (properties.constant_surrogate), so that
        U' is uniform along the core.
    key : str
        A parameter that changes U' but not C_min (fin geometry such as b_h,
        or zeta_j).
    rel_step : float
        Relative step for da/dp.

    Returns
    -------
    deps_dp : float
        Effectiveness sensitivity [1/units of p].
    """
    # 1. Base point: geom = correlations.exchanger_geometry(params);
    #    a_h, a_c = correlations.conductances(...) (constants);
    #    U' = overall_conductance(a_h, a_c); C_min, C_r; NTU = U' L / C_min.
    # 2. da_h/dp and da_c/dp by central differences of the conductances,
    #    rebuilding geom for each perturbed copy of params (geometry keys change
    #    D_h, Re, j, eta_f and A'; that is the whole point).
    # 3. dU'/dp from Eq 43, then deps/dp = depsilon_dNTU(NTU, C_r) * L/C_min * dU'/dp.
    # Check: agrees with central_difference(model=effectiveness from a full
    #    constant-property solve_steady, key) to ~1e-5 relative (with
    #    bvp_tol = 1e-10).
    raise NotImplementedError


def shooting_sensitivity(params, props, key, rel_step):
    """Cold-outlet sensitivity by implicit differentiation of the shooting residual.

    Eq 45: at the converged root r(sigma, p) = 0,
      d sigma/dp = -(dr/dp) / (dr/d sigma),
    so the sensitivity of the cold outlet T_c(0) = sigma needs only residual
    evaluations (single IVPs), not new root solves.

    Parameters
    ----------
    params : dict
        Base parameters.
    props : dict
        Surrogates (must stay valid for the perturbed parameter).
    key : str
        Parameter to perturb.
    rel_step : float
        Relative step for dr/dp.

    Returns
    -------
    dsigma_dp : float
        Sensitivity of the cold outlet temperature [K per unit of p].
    """
    # 1. sigma* = steady.solve_steady_shooting(params, props)["T_c"][0].
    # 2. dr/dsigma: central difference of steady.shooting_residual in sigma
    #    (step ~1e-4 K) at fixed params.
    # 3. dr/dp: central difference of steady.shooting_residual(sigma*, p+-, ...)
    #    at fixed sigma*, with geom rebuilt from each perturbed copy p+-.
    # 4. Return -(dr/dp)/(dr/dsigma).
    # Pitfall: dr/dsigma becomes huge at large NTU (1 - C_r), the same
    #    ill-conditioning that makes shooting delicate (Section 5.2). Use
    #    tight solve_ivp tolerances inside shooting_residual.
    # Check: agrees with central_difference on a model returning
    #    solve_steady(p, props)["T_c"][0].
    raise NotImplementedError


def forward_sensitivity_rhs(t, z, rhs, rhs_plus, rhs_minus, dp, eps):
    """Right-hand side of the augmented state-plus-sensitivity system.

    Eq 46: for dy/dt = F(y, p), the sensitivity s = dy/dp obeys
      ds/dt = (dF/dy) s + dF/dp.
    The Jacobian-vector product is formed by a directional difference, so the
    full Jacobian is never built:
      (dF/dy) s ~ [F(y + eps s) - F(y - eps s)] / (2 eps)
      dF/dp     ~ [F(y; p + dp) - F(y; p - dp)] / (2 dp)

    Parameters
    ----------
    t : float
        Time [s].
    z : ndarray, shape (6N,)
        z = [y, s]: the state (3N) followed by its sensitivity (3N).
    rhs, rhs_plus, rhs_minus : callable
        rhs(t, y) -> dy/dt at the base parameters and at p +- dp. Build them
        once in simulate_forward_sensitivity as closures over
        transient.transient_rhs with their own params and geom.
    dp : float
        Parameter step used for dF/dp.
    eps : float
        Scalar step for the directional difference.

    Returns
    -------
    dzdt : ndarray, shape (6N,)
        [dy/dt, ds/dt].
    """
    # 1. Split: y = z[:3N], s = z[3N:].
    # 2. F0 = rhs(t, y).
    # 3. Jv = (rhs(t, y + h s) - rhs(t, y - h s)) / (2 h), with
    #    h = eps * max(1, |y|) / max(|s|, tiny) so the probe moves y by about
    #    eps relative (vector norms). If s == 0, Jv = 0.
    # 4. Fp = (rhs_plus(t, y) - rhs_minus(t, y)) / (2 dp).
    # 5. Return np.concatenate((F0, Jv + Fp)).
    # Pitfall: y +- h s must stay inside the surrogate range; keep eps small
    #    (~1e-6).
    # Check: integrating this and comparing s(t) with the central difference
    #    of two full simulate() runs agrees to ~1e-4 relative.
    raise NotImplementedError


def simulate_forward_sensitivity(params, props, scenario, key, t_end, t_eval=None, rel_step=None):
    """Integrate the state and its sensitivity to one parameter together.

    Section 7.5: integrating [y; s] together avoids differencing two stiff
    solutions. s(0) = dy0/dp comes from the steady sensitivities.

    Parameters
    ----------
    params : dict
        Base parameters.
    props : dict
        Surrogates.
    scenario : str
        Upset scenario name.
    key : str
        Parameter p.
    t_end : float
        Final time [s].
    t_eval : ndarray, shape (n_t,), optional
        Saved times [s].
    rel_step : float, optional
        Relative step for p; default optimal_rel_step(params["sens_rtol"]).

    Returns
    -------
    result : dict
        Same fields as transient.simulate, plus "s_T_h", "s_T_c", "s_T_w",
        each shape (n_t, N): sensitivities of the temperatures to p
        [K per unit of p].
    """
    # 1. geom, geom_plus, geom_minus from correlations.exchanger_geometry on
    #    params and its two perturbed copies (dp = rel_step |p0|).
    # 2. Closures rhs, rhs_plus, rhs_minus over transient.transient_rhs.
    # 3. y0 = transient.discrete_steady_state(params, props, scenario);
    #    s0 = [discrete_steady_state(p+) - discrete_steady_state(p-)] / (2 dp).
    # 4. Sparsity: with P = transient.jacobian_sparsity(N, axial_conduction),
    #    the augmented pattern is sparse.bmat([[P, None], [P, P]]). Row i of
    #    ds/dt depends on y and s only through the cells that F_i depends on,
    #    so the lower-left block has the same pattern as P.
    # 5. solve_ivp(lambda t, z: forward_sensitivity_rhs(t, z, rhs, rhs_plus,
    #    rhs_minus, dp, eps), ..., method="BDF", jac_sparsity=pattern), with
    #    the same scenario step control as transient.simulate.
    # 6. Unpack y and s histories into (n_t, N) arrays.
    # Note: if the right-hand side were written in jax.numpy, the products
    #    in forward_sensitivity_rhs could come from automatic differentiation
    #    (math doc Section 7.5). That needs team agreement (CLAUDE.md).
    # Check: s at the final time of a "none" run equals s0 (a steady state
    #    stays steady, and so does its sensitivity).
    raise NotImplementedError
