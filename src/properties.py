"""Thermophysical properties: CoolProp tables and fast polynomial surrogates.

Math doc: Section 4.4 (thermophysical properties), Section 8.4 (property
surrogates, Eq 51) and Section 5.1 (property continuation, Eq 23).

Why surrogates: CoolProp is slow and is never called inside an ODE or BVP
right-hand side (CLAUDE.md). We call CoolProp once, at build time, fit each
property at fixed pressure as a polynomial in scaled temperature, and evaluate
the polynomial everywhere else. A surrogate must raise outside its fitted range.

Imports: numpy (fitting and evaluation), scipy.interpolate.CubicSpline (spline
fallback) and CoolProp.CoolProp.PropsSI / PhaseSI (build time only). It does not
import any other src module; correlations.py, steady.py and transient.py import
this one.

The surrogate dict (one per stream; ``props = {"h": surr_h, "c": surr_c}``)
-------------------------------------------------------------------------
    "fluid"       str     CoolProp fluid name
    "p"           float   pressure the fit was made at [Pa]
    "kind"        str     "poly" (default) or "spline" (fallback near the cp peak)
    "T_lo","T_hi" float   valid range [K]; check_range raises outside it
    "T_bar","dT"  float   scaling T~ = (T - T_bar)/dT, so T~ is in [-1, 1] [K]
    "coef"        dict    "cp","mu","k","rho" -> coefficient array, shape (K+1,),
                          increasing powers of T~ (or CubicSpline objects)
    "hhat_coef"   array   shape (K+2,): integral of the cp polynomial in T~
    "hhat_ref"    float   specific enthalpy at T_bar from CoolProp [J/kg]
    "mean"        dict    "cp","mu","k","rho" -> mean value over the range
                          (phi_bar in Eq 23 and in the steady initial guess)
    "lam"         float   continuation parameter lambda (1.0 = full properties)
    "max_rel_err" dict    max relative error of each fit on a validation grid

Property names: "cp" [J/(kg K)], "mu" [Pa s], "k" [W/(m K)], "rho" [kg/m^3],
and specific enthalpy "hhat" [J/kg] (written h-hat in the math doc; not "h",
which means the heat transfer coefficient and the hot side elsewhere).
"""

import numpy as np
from CoolProp.CoolProp import PhaseSI, PropsSI
from scipy.interpolate import CubicSpline


def coolprop_table(fluid, p, T):
    """Evaluate CoolProp properties along an isobar (build time only).

    Section 4.4. This is the only function that calls CoolProp for property
    values, and it must never be called from an ODE/BVP right-hand side.

    Parameters
    ----------
    fluid : str
        CoolProp fluid name, e.g. "Methane".
    p : float
        Pressure [Pa].
    T : ndarray, shape (n,)
        Temperatures [K].

    Returns
    -------
    table : dict of ndarray, each shape (n,)
        "cp" [J/(kg K)], "mu" [Pa s], "k" [W/(m K)], "rho" [kg/m^3],
        "hhat" [J/kg].
    """
    # 1. For each property, call PropsSI(output, "T", T, "P", p, fluid) with the
    #    CoolProp output keys CPMASS, VISCOSITY, CONDUCTIVITY, DMASS, HMASS.
    #    PropsSI accepts a numpy array for T; if it complains, loop over T.
    # 2. Return them in a dict with the keys above, each as a float array of
    #    shape (n,).
    # Pitfall: CoolProp's enthalpy has an arbitrary reference state. Only
    #    enthalpy differences are meaningful, so never compare hhat from two
    #    different fluids or reference-state settings.
    # Check: methane at 300 K, 101325 Pa gives cp = 2235.9 J/(kg K) (NIST);
    #    tests/test_setup.py already checks this value through PropsSI.
    raise NotImplementedError


def check_single_phase(fluid, p, T_lo, T_hi):
    """Confirm a temperature range is single phase and report its margins.

    Sections 4.4 and 8.4: fit ranges must stay inside the single-phase
    operating window. At subcritical pressure the danger is condensation
    (T near T_sat); at supercritical pressure it is the sharp cp peak near the
    pseudo-critical temperature T_pc.

    Parameters
    ----------
    fluid : str
        CoolProp fluid name.
    p : float
        Pressure [Pa].
    T_lo, T_hi : float
        Range to check [K], T_lo < T_hi.

    Returns
    -------
    report : dict
        "supercritical" bool; "T_sat" [K] or None; "T_pc" [K] or None;
        "margin" [K] = T_lo - (T_sat or T_pc), the distance from the bottom
        of the range to the danger point; "cp_ratio" = max cp / min cp over
        the range (a measure of how hard the fit will be).

    Raises
    ------
    ValueError
        If any part of [T_lo, T_hi] is two-phase or below T_sat.
    """
    # 1. Get the critical pressure p_crit = PropsSI("pcrit", fluid).
    # 2. If p < p_crit (subcritical): T_sat = PropsSI("T", "P", p, "Q", 1, fluid).
    #    Raise ValueError if T_lo <= T_sat. margin = T_lo - T_sat.
    # 3. If p >= p_crit (supercritical): there is no phase change, but cp peaks
    #    at the pseudo-critical temperature. Sample cp on a fine grid (0.1-0.5 K)
    #    from T_crit to T_crit + ~60 K with coolprop_table and take
    #    T_pc = T at max cp (np.argmax). margin = T_lo - T_pc.
    #    Raise ValueError if T_lo <= T_pc (the range straddles the peak).
    # 4. Sample PhaseSI("T", T, "P", p, fluid) at a few points across the range
    #    as a second check; every phase string should be one of "gas",
    #    "supercritical_gas" or "supercritical".
    # 5. cp_ratio = max/min of cp over np.linspace(T_lo, T_hi, ~200).
    # Pitfall: being above T_pc is not enough. cp still rises steeply on the
    #    shoulder of the peak (methane at 6 MPa: cp(210 K)/cp(300 K) = 2.4), so
    #    report cp_ratio and let the fit error decide.
    # Check: methane at 1.5 MPa gives T_sat = 158.5 K; at 6 MPa gives
    #    T_pc = 199.6 K. A range starting at 150 K at 1.5 MPa must raise.
    raise NotImplementedError


def fit_surrogate(fluid, p, T_lo, T_hi, degree, n_points):
    """Fit polynomial surrogates for one stream at fixed pressure.

    Section 8.4, Eq 51: each property is a polynomial in the scaled
    temperature T~ = (T - T_bar)/dT, fitted by linear least squares on a
    Vandermonde matrix. Enthalpy is the exact integral of the cp polynomial,
    so d(hhat)/dT = cp holds exactly (needed for energy closure, Eq 16, and
    for the enthalpy-form transient, Eq 32-33).

    Parameters
    ----------
    fluid : str
        CoolProp fluid name.
    p : float
        Pressure [Pa].
    T_lo, T_hi : float
        Fit range [K]. Must already include the safety margin.
    degree : int
        Polynomial degree K (Eq 51).
    n_points : int
        Number of CoolProp sample points used for the fit.

    Returns
    -------
    surr : dict
        Surrogate dict as described in the module docstring, with kind "poly",
        lam = 1.0 and max_rel_err filled in for cp, mu, k, rho and hhat.
    """
    # 1. Scaling: T_bar = (T_lo + T_hi)/2, dT = (T_hi - T_lo)/2.
    # 2. Fit points: T_fit = np.linspace(T_lo, T_hi, n_points), shape (n_points,).
    #    Table = coolprop_table(fluid, p, T_fit).
    # 3. Vandermonde matrix V = np.vander(T~_fit, degree + 1, increasing=True),
    #    shape (n_points, degree + 1). Scaling T~ into [-1, 1] keeps V well
    #    conditioned; fitting in raw kelvin would not.
    # 4. For each name in ("cp", "mu", "k", "rho"): solve
    #    min ||V c - phi||_2 with np.linalg.lstsq(V, table[name], rcond=None);
    #    c has shape (degree + 1,).
    # 5. Enthalpy from cp: hhat(T) = hhat(T_bar) + integral of cp from T_bar to T
    #                              = hhat_ref + dT * P(T~),
    #    where P = np.polynomial.polynomial.polyint(c_cp) (P(0) = 0) and
    #    hhat_ref = CoolProp hhat at T_bar. The factor dT converts the integral
    #    over T~ back to an integral over T.
    # 6. Mean values: mean[name] = average of the sampled table values
    #    (phi_bar in Eq 23).
    # 7. Validation: evaluate every fit on a different grid,
    #    T_val = np.linspace(T_lo, T_hi, ~3 * n_points + 1), so you are not
    #    testing only at the fit points. For each property
    #    max_rel_err[name] = max |fit/CoolProp - 1|. For hhat, compare
    #    differences hhat(T) - hhat(T_lo), not absolute values, because
    #    CoolProp's enthalpy zero is arbitrary.
    # 8. Return the dict with kind "poly", lam = 1.0 and the fields above.
    # Spline fallback (Section 8.4): if a polynomial cannot meet the target
    #    near the cp peak, store CubicSpline(T_fit, table[name]) objects in
    #    "coef" instead, set kind "spline", and get hhat from
    #    CubicSpline(...).antiderivative(). CubicSpline extrapolates by default,
    #    so check_range must be called before evaluating it.
    # Check: methane at 1.5 MPa on [225, 310] K with degree 6 gives
    #    max_rel_err below 1e-5 for every property; at 6 MPa, cp error is
    #    ~8e-4. Also check that the numerical derivative of enthalpy matches
    #    evaluate(surr, "cp", T) to round-off.
    raise NotImplementedError


def build_surrogates(params, scenario="none"):
    """Build hot- and cold-side surrogates covering every temperature a run can reach.

    By the maximum principle, every stream and wall temperature stays between
    the coldest and the hottest inlet temperature over the whole run, so the
    fit range is [min inlet, max inlet] widened by ``params["fit_T_margin"]``.

    Parameters
    ----------
    params : dict
        Flat parameter dict (config.BASE_CASE copy). Uses fluid, p_h, p_c,
        T_h_in, T_c_in, ramp_dT (for "feed_ramp"), fit_degree, fit_n_points,
        fit_T_margin, fit_max_rel_err.
    scenario : str
        "none", "feed_ramp" or "cold_trip"; decides whether the hot inlet
        temperature moves during the run.

    Returns
    -------
    props : dict
        {"h": hot surrogate at p_h, "c": cold surrogate at p_c}.

    Raises
    ------
    ValueError
        If a range is not single phase or a fit misses fit_max_rel_err.
    """
    # 1. Coldest inlet over the run: T_min = T_c_in. Hottest: T_max = T_h_in,
    #    plus ramp_dT if scenario == "feed_ramp" and ramp_dT > 0. (If ramp_dT
    #    is negative, the ramp lowers the hot inlet; T_min must then be
    #    min(T_c_in, T_h_in + ramp_dT).)
    # 2. T_lo = T_min - fit_T_margin, T_hi = T_max + fit_T_margin. Use the same
    #    range for both sides: in a cold trip the cold stream can warm almost
    #    to T_h_in, and the hot stream can cool almost to T_c_in.
    # 3. For each side (p_h for "h", p_c for "c"): check_single_phase, then
    #    fit_surrogate with fit_degree and fit_n_points.
    # 4. If any max_rel_err exceeds fit_max_rel_err, raise ValueError naming the
    #    side and property (or switch that side to the spline fallback). Do not
    #    silently accept a poor fit.
    # 5. Print or return each surrogate's max_rel_err: CLAUDE.md asks us to
    #    report it.
    # Pitfall: rebuild props whenever a pressure, an inlet temperature or the
    #    ramp size changes (e.g. inside sensitivity studies). A stale surrogate
    #    either raises out of range or silently uses the wrong pressure.
    # Check: with the base case (6 MPa / 1.5 MPa, 300 K / 235 K) and degree 6,
    #    both sides build and every max_rel_err is below 1e-3.
    raise NotImplementedError


def constant_surrogate(cp, mu, k, rho, T_lo, T_hi, hhat_ref=0.0):
    """Build a constant-property surrogate (no CoolProp).

    Used for the constant-property verification against the analytical
    epsilon-NTU solution (Section 3.5) and for unit tests of the solvers.

    Parameters
    ----------
    cp : float
        Heat capacity [J/(kg K)].
    mu : float
        Viscosity [Pa s].
    k : float
        Thermal conductivity [W/(m K)].
    rho : float
        Density [kg/m^3].
    T_lo, T_hi : float
        Valid range [K] (still enforced, to catch solver bugs).
    hhat_ref : float, optional
        Enthalpy at the range midpoint [J/kg].

    Returns
    -------
    surr : dict
        Surrogate dict of kind "poly" and degree 0, so evaluate() and
        enthalpy() need no special case. max_rel_err is 0 for every property.
    """
    # 1. T_bar and dT as in fit_surrogate.
    # 2. coef[name] = np.array([value]) for each property (degree-0 polynomial).
    # 3. hhat_coef = polyint(coef["cp"]) = [0, cp]. Then
    #    enthalpy = hhat_ref + dT * cp * T~ = hhat_ref + cp (T - T_bar), the
    #    exact constant-cp enthalpy.
    # 4. mean[name] = value; lam = 1.0; max_rel_err all 0.0; fluid "constant";
    #    p = None.
    # Check: evaluate(surr, "cp", T) returns cp for any T in range, and
    #    enthalpy(surr, T2) - enthalpy(surr, T1) == cp (T2 - T1).
    raise NotImplementedError


def continuation_surrogate(surr, lam):
    """Return a copy of a surrogate with continuation parameter lambda set.

    Section 5.1, Eq 23: phi(T; lambda) = (1 - lambda) phi_bar + lambda phi(T)
    for phi in {cp, mu, k}. lambda = 0 is the constant-property problem;
    lambda = 1 is the full variable-property problem.

    Parameters
    ----------
    surr : dict
        Surrogate dict.
    lam : float
        Continuation parameter in [0, 1].

    Returns
    -------
    surr_lam : dict
        New dict, identical except surr_lam["lam"] = lam. The blend itself is
        applied inside evaluate(), so this works for polynomial and spline
        surrogates alike.
    """
    # 1. Make a shallow copy: dict(surr). Do not modify the caller's surrogate.
    # 2. Set the copy's "lam" to lam and return it.
    # Note: Eq 23 blends cp, mu and k only. rho and hhat are not blended; the
    #    steady solve does not use rho, and hhat is only used for the duty after
    #    the lambda = 1 solve.
    # Check: evaluate(continuation_surrogate(s, 0.0), "cp", T) equals
    #    s["mean"]["cp"] for every T; with lam = 1 it equals evaluate(s, "cp", T);
    #    the original s["lam"] is still 1.0.
    raise NotImplementedError


def check_range(surr, T):
    """Raise if any temperature is outside a surrogate's fitted range.

    CLAUDE.md: surrogates must raise an error outside their fitted range.

    Parameters
    ----------
    surr : dict
        Surrogate dict.
    T : float or ndarray, any shape
        Temperatures [K].

    Raises
    ------
    ValueError
        If any T < T_lo or T > T_hi, or any T is NaN. The message gives the
        offending min/max and the valid range, which makes debugging solver
        overshoots much faster.
    """
    # 1. T = np.asarray(T). Use np.any (not Python `if T < ...`) so arrays of
    #    any shape work, including solve_bvp's (m,) and solve_ivp's (N,).
    # 2. Raise ValueError if np.any(np.isnan(T)) or np.any(T < T_lo) or
    #    np.any(T > T_hi). Include np.min(T), np.max(T), T_lo, T_hi and
    #    surr["p"] in the message.
    # Pitfall: if a solver keeps tripping this, the fix is a wider
    #    fit_T_margin or a better initial guess, not removing the check.
    # Check: T exactly at T_lo and T_hi passes; T_lo - 1e-9 raises; a (2, m)
    #    array with one bad entry raises.
    raise NotImplementedError


def evaluate(surr, name, T):
    """Evaluate one property surrogate.

    Eq 51 (polynomial in scaled temperature), with the continuation blend of
    Eq 23 applied for cp, mu and k when surr["lam"] < 1.

    Parameters
    ----------
    surr : dict
        Surrogate dict for ONE stream. Pass the hot surrogate with hot
        temperatures and the cold one with cold temperatures.
    name : str
        "cp", "mu", "k" or "rho".
    T : float or ndarray, any shape
        Temperatures [K].

    Returns
    -------
    phi : ndarray, same shape as T
        Property value in SI units (see module docstring).
    """
    # 1. check_range(surr, T).
    # 2. T~ = (T - T_bar)/dT.
    # 3. kind "poly": phi = np.polynomial.polynomial.polyval(T~, coef[name])
    #    (Horner's rule; works elementwise for any array shape).
    #    kind "spline": phi = coef[name](T).
    # 4. If name is in ("cp", "mu", "k") and lam != 1:
    #    phi = (1 - lam) * mean[name] + lam * phi   (Eq 23).
    # 5. Return phi with the same shape as T.
    # Pitfall: never call CoolProp here. This function runs inside every
    #    right-hand-side evaluation.
    # Check: at the fit points the value matches coolprop_table to within
    #    max_rel_err; shape (2, 7) in gives shape (2, 7) out.
    raise NotImplementedError


def enthalpy(surr, T):
    """Evaluate specific enthalpy from the integrated cp surrogate.

    hhat(T) = hhat_ref + dT * P(T~), with P the integral of the cp polynomial
    (see fit_surrogate step 5). Used for the duty (Eq 16) and the
    enthalpy-form advective fluxes of the transient model (Eq 32-33).

    Parameters
    ----------
    surr : dict
        Surrogate dict for one stream.
    T : float or ndarray, any shape
        Temperatures [K].

    Returns
    -------
    hhat : ndarray, same shape as T
        Specific enthalpy [J/kg] (only differences are physical).
    """
    # 1. check_range(surr, T).
    # 2. kind "poly": hhat = hhat_ref + dT * polyval((T - T_bar)/dT, hhat_coef).
    #    kind "spline": hhat = hhat_ref + (antiderivative spline)(T) minus its
    #    value at T_bar.
    # 3. Ignore lam here (see continuation_surrogate).
    # Check: (enthalpy(s, T + 1e-3) - enthalpy(s, T - 1e-3)) / 2e-3 equals
    #    evaluate(s, "cp", T) to ~1e-8 relative; differences
    #    enthalpy(s, T) - enthalpy(s, T_lo) match CoolProp HMASS differences
    #    to within max_rel_err["hhat"].
    raise NotImplementedError
