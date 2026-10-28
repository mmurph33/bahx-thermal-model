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
    "degree"      int     polynomial degree K (Eq 51)
    "n_points"    int     number of CoolProp samples used for the fit
    "coef"        dict    "cp","mu","k","rho" -> for kind "poly", a coefficient
                          array, shape (K+1,), in increasing powers of T~; for
                          kind "spline", a CubicSpline in kelvin.
                          "hhat" -> for "poly", the tuple (hhat_ref, P, T_bar, dT)
                          with P, shape (K+2,), the integral of the cp
                          polynomial in T~; for "spline", the tuple
                          (antiderivative of the cp spline, offset) [J/kg]
    "hhat_ref"    float   specific enthalpy at T_bar from CoolProp [J/kg]
    "mean"        dict    "cp","mu","k","rho","hhat" -> mean value over the range
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
from scipy.optimize import curve_fit

# Two fitters are kept so the team can compare them (Section 8.4):
# - fit_surrogate_curvefit: polynomial least squares only. build_surrogates uses it.
# - fit_surrogate_cubicspline: polynomial first, then a cubic spline through the
#   CoolProp points if the polynomial misses a tight tolerance. CoolProp data are
#   not noisy, so an interpolating spline is a reasonable fallback near the cp peak.


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
    #    CoolProp output keys CPMASS, VISCOSITY, CONDUCTIVITY, DMASS, HMASS,
    #    looping over T.
    # 2. Return them in a dict with the keys above, each as a float array of
    #    shape (n,).
    # Pitfall: CoolProp's enthalpy has an arbitrary reference state. Only
    #    enthalpy differences are meaningful, so never compare hhat from two
    #    different fluids or reference-state settings.
    # Check: methane at 300 K, 101325 Pa gives cp = 2235.9 J/(kg K) (NIST);
    #    tests/test_setup.py already checks this value through PropsSI.

    # Ensure T is a numpy array
    T = np.asarray(T, dtype = float)

    if T.ndim != 1:
        raise ValueError('T must be a 1D array')

    cp = np.empty(T.shape)
    mu = np.empty(T.shape)
    k = np.empty(T.shape)
    rho = np.empty(T.shape)
    hhat = np.empty(T.shape)

    for i, temp in enumerate(T):
        cp[i] = PropsSI("CPMASS", "T", temp, "P", p, fluid)
        mu[i] = PropsSI("VISCOSITY", "T", temp, "P", p, fluid )
        k[i] = PropsSI("CONDUCTIVITY", "T", temp, "P", p, fluid)
        rho[i] = PropsSI("DMASS", "T", temp, "P", p, fluid)
        hhat[i] = PropsSI("HMASS", "T", temp, "P", p, fluid)

    return {
        "cp": cp,
        "mu": mu,
        "k": k,
        "rho": rho,
        "hhat": hhat
    }

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
        If any part of [T_lo, T_hi] is two-phase or below T_sat, or (at
        supercritical pressure) at or below the cp peak T_pc.
    """
    # 1. Get the critical pressure p_crit = PropsSI("pcrit", fluid).
    # 2. If p < p_crit (subcritical): T_sat = PropsSI("T", "P", p, "Q", 1, fluid).
    #    Raise ValueError if T_lo <= T_sat. margin = T_lo - T_sat.
    # 3. If p >= p_crit (supercritical): there is no phase change, but cp peaks
    #    at the pseudo-critical temperature. Sample cp on a 0.2 K grid from
    #    T_crit to T_crit + 60 K with coolprop_table and take
    #    T_pc = T at max cp (np.argmax). margin = T_lo - T_pc.
    #    Raise ValueError if T_lo <= T_pc (the range straddles the peak).
    # 4. Sample PhaseSI("T", T, "P", p, fluid) at a few points across the range
    #    as a second check; every phase string should be one of "gas",
    #    "supercritical_gas" or "supercritical".
    # 5. cp_ratio = max/min of cp over np.linspace(T_lo, T_hi, 200).
    # Pitfall: being above T_pc is not enough. cp still rises steeply on the
    #    shoulder of the peak (methane at 6 MPa: cp(210 K)/cp(300 K) = 2.4), so
    #    report cp_ratio and let the fit error decide.
    # Check: methane at 1.5 MPa gives T_sat = 158.5 K; at 6 MPa gives
    #    T_pc = 199.6 K. A range starting at 150 K at 1.5 MPa must raise, and
    #    so must a range starting at 195 K at 6 MPa.

    # Verify T_lo < T_hi
    if T_lo >= T_hi:
        raise ValueError("T_lo must be lower than T_hi")

    # Verify P > 0 
    if p <= 0:
        raise ValueError("Pressure must be positive")
    
    # Find critical pressure
    p_crit = PropsSI("pcrit", fluid)

    if p < p_crit: # For sub-critical pressure
        T_sat = PropsSI("T", "P", p, "Q", 1, fluid)
        # Q = 1 to ask CoolProp for saturated-vapor temperature

        if T_lo <= T_sat:
            raise ValueError("Multiphase")

        T_pc = None
        margin = T_lo - T_sat
    else:
        # Here the fluid is supercritical & there is no phase change
        # However, cp still peaks at the pseudo-critical temperature
        T_crit = PropsSI("Tcrit", fluid)

        # Sample cp around T_crit to find the pseudo-critical temperature
        T_scan = np.arange(
            T_crit + 0.1,
            T_crit + 60.0,
            0.2
        )

        cp_scan = coolprop_table(fluid, p, T_scan)["cp"]

        # T_pc = T at max cp
        T_pc = T_scan[np.argmax(cp_scan)]

        # A fit range that reaches the cp peak cannot be fit by a smooth
        # polynomial, and the stream would be liquid-like below it.
        if T_lo <= T_pc:
            raise ValueError(
                f"T_lo = {T_lo} K is at or below the pseudo-critical "
                f"temperature T_pc = {T_pc:.1f} K at p = {p} Pa"
            )

        T_sat = None
        margin = T_lo - T_pc

    # Check phase through a few points
    T_phase = np.linspace(T_lo, T_hi, 5)
    allowed_phases = {
        'gas',
        'supercritical_gas',
        'supercritical'
    }

    for temp in T_phase:
        phase = PhaseSI("T", temp, "P", p, fluid)
        if phase not in allowed_phases:
            # Raise error if the phase is not something we would expect
            raise ValueError(f"Unexpected phase {phase} at T = {temp}K")

    # Calculate the ratio of cp
    T_fit = np.linspace(T_lo, T_hi, 200)
    cp_fit = coolprop_table(fluid, p, T_fit)["cp"]
    cp_ratio = np.max(cp_fit) / np.min(cp_fit)

    # Return
    return {
        "supercritical": p > p_crit,
        "T_sat": T_sat,
        "T_pc": T_pc,
        "margin": margin,
        "cp_ratio": cp_ratio
    }


def fit_surrogate_cubicspline(fluid, p, T_lo, T_hi, degree, n_points):
    """Fit polynomial surrogates, falling back to cubic splines if needed.

    Section 8.4, Eq 51: each property is a polynomial in the scaled
    temperature T~ = (T - T_bar)/dT, fitted by linear least squares on a
    Vandermonde matrix. Enthalpy is the exact integral of the cp polynomial,
    so d(hhat)/dT = cp holds exactly (needed for energy closure, Eq 16, and
    for the enthalpy-form transient, Eq 32-33). If any polynomial misses a
    tolerance of 1e-5, every property is refit as a cubic spline through the
    CoolProp points instead (the spline fallback of Section 8.4).

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
        Surrogate dict as described in the module docstring, with kind "poly"
        or "spline", lam = 1.0 and max_rel_err filled in for cp, mu, k, rho
        and hhat.
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
    # 7. Validation on a different, denser grid T_val (3 n_points + 1 points),
    #    so the fit is not tested only at its own points:
    #    max_rel_err[name] = max |fit - CoolProp| / |CoolProp|. For hhat,
    #    compare differences hhat(T) - hhat(T_lo), not absolute values,
    #    because CoolProp's enthalpy zero is arbitrary.
    # 8. If every error is below 1e-5, return the polynomial surrogate.
    # 9. Otherwise (spline fallback): CubicSpline(T_fit, table[name]) in
    #    kelvin for each property; hhat from the antiderivative of the cp
    #    spline, offset so hhat(T_bar) = hhat_ref. Validate on T_val the same
    #    way and return with kind "spline". CubicSpline extrapolates, so
    #    evaluate/enthalpy must call check_range first (they do).
    # Check: methane at 1.5 MPa on [225, 310] K with degree 6 gives
    #    max_rel_err below 1e-5 for every property (poly); at 6 MPa the cp
    #    error is ~8e-4, so this function returns a spline there, whose cp
    #    matches CoolProp at 260 K to ~1e-6.

    # Validate Input
    if T_lo >= T_hi:
        raise ValueError("T_lo cannot be higher than T_hi")

    if degree < 0:
        raise ValueError("Degree cannot be negative")

    if n_points < degree + 1:
        raise ValueError("Overfitting - n_points be at least degree + 1")

    # Scaling with T_bar
    T_bar = (T_lo + T_hi) / 2
    dT = (T_hi - T_lo) / 2

    T_fit = np.linspace(T_lo, T_hi, n_points)
    T_scale_fit = (T_fit - T_bar) / dT

    # Reference properties at fitting point
    table = coolprop_table(fluid, p, T_fit)
    property_names = ("cp", "mu", "k", "rho")

    # Define Vandermonde matrix
    V = np.vander(T_scale_fit, N = degree + 1, increasing = True)
    coefficients = {}
    for name in property_names:
        coefficients[name] = np.linalg.lstsq(V, np.asarray(table[name]), rcond = None)[0]

    # Find enthalpy with cp polynomial
    hhat_ref = float(
        coolprop_table(fluid, p, np.array([T_bar]))["hhat"][0]
    )

    # cp(T~) = c[0] + c1 * T~ + ...
    cp_integral = np.polynomial.polynomial.polyint(coefficients["cp"])
    cp_integral[0] = 0.0

    coefficients["hhat"] = (
        hhat_ref,
        cp_integral,
        T_bar,
        dT
    )


    # Mean values for properties
    mean = {
        name: float(np.mean(table[name]))
        for name in (*property_names, "hhat")
    }

    # Validate fit on different grid
    T_val = np.linspace(T_lo, T_hi, 3*n_points + 1)
    val_table = coolprop_table(fluid, p, T_val)
    T_scale_val = (T_val - T_bar) / dT

    max_rel_err = {}

    for name in property_names:
        fit_values = np.polynomial.polynomial.polyval(
            T_scale_val, coefficients[name] 
        )
        reference = np.asarray(val_table[name])

        # Avoid divison by 0
        denom = np.maximum(np.abs(reference), 1e-14)
        max_rel_err[name] = float(
            np.max(np.abs(fit_values - reference) / denom)
        )

    # Compare different in hhat (since CoolProp's enthalpy 0 is arbitrary)
    h_fit = hhat_ref + dT * np.polynomial.polynomial.polyval(
        T_scale_val, cp_integral
    )

    h_ref = np.asarray(val_table["hhat"])
    h_fit_delta = h_fit - h_fit[0]
    h_ref_delta = h_ref - h_ref[0]

    # Normalize by the overall reference enthalpy change to avoid
    # dividing by values near zero at the lower endpoint.
    h_scale = max(
        float(np.max(np.abs(h_ref_delta))),
        1e-14,
    )
    max_rel_err["hhat"] = float(
        np.max(np.abs(h_fit_delta - h_ref_delta)) / h_scale
    )

    # Select polynomial or CubicSpline based on error
    tolerance = 1e-5

    # If errors are already within tolerance, return as a polynomial
    if all(err <= tolerance for err in max_rel_err.values()):
        return {
            "kind": "poly",
            "fluid": fluid,
            "p": p,
            "T_lo": T_lo,
            "T_hi": T_hi,
            "T_bar": T_bar,
            "dT": dT,
            "degree": degree,
            "n_points": n_points,
            "coef": coefficients,
            "mean": mean,
            "lam": 1.0,
            "max_rel_err": max_rel_err,
            "hhat_ref": hhat_ref,
        }

    # If polynomial did not meet tolerance requirement
    spline_coef = {
        name: CubicSpline(
            T_fit, np.asarray(table[name]), extrapolate=True
        )
        for name in property_names
    }

    cp_antiderivative = spline_coef["cp"].antiderivative()
    h_ref_spline = float(cp_antiderivative(T_bar))

    # Define hhat(T_bar) using CoolProp as the enthalpy reference.
    hhat_offset = hhat_ref - h_ref_spline

    spline_coef["hhat"] = (
        cp_antiderivative,
        hhat_offset,
    )

    # Validate spline fits on the same independent grid.
    spline_max_rel_err = {}

    for name in property_names:
        fit_values = spline_coef[name](T_val)
        reference = np.asarray(val_table[name])
        denom = np.maximum(np.abs(reference), 1e-14)

        spline_max_rel_err[name] = float(
            np.max(np.abs(fit_values - reference) / denom)
        )

    h_spline = cp_antiderivative(T_val) + hhat_offset
    h_spline_delta = h_spline - h_spline[0]

    spline_max_rel_err["hhat"] = float(
        np.max(np.abs(h_spline_delta - h_ref_delta)) / h_scale
    )

    return {
        "kind": "spline",
        "fluid": fluid,
        "p": p,
        "T_lo": T_lo,
        "T_hi": T_hi,
        "T_bar": T_bar,
        "dT": dT,
        "degree": degree,
        "n_points": n_points,
        "coef": spline_coef,
        "mean": mean,
        "lam": 1.0,
        "max_rel_err": spline_max_rel_err,
        "hhat_ref": hhat_ref,
    }



def fit_surrogate_curvefit(fluid, p, T_lo, T_hi, degree, n_points):
    """Fit polynomial property surrogates using scipy.optimize.curve_fit.

    Fits cp, mu, k, and rho as polynomials in scaled temperature.
    Enthalpy is obtained by integrating the fitted cp polynomial.

    Parameters
    ----------
    fluid : str
        CoolProp fluid name.
    p : float
        Pressure [Pa].
    T_lo, T_hi : float
        Fit range [K].
    degree : int
        Polynomial degree.
    n_points : int
        Number of CoolProp samples used for fitting.

    Returns
    -------
    surr : dict
        Polynomial surrogate compatible with the same evaluate()
        implementation and validation convention as the polynomial
        branch of fit_surrogate_cubicspline().
    """
    # 1. Scale temperature to [-1, 1] as in Eq 51.
    # 2. For each of cp, mu, k, rho: start from the linear least-squares
    #    (lstsq) solution and refine it with curve_fit. The model is linear in
    #    its coefficients, so curve_fit converges straight back to the lstsq
    #    optimum; it is kept so the two fitting tools can be compared.
    # 3. Enthalpy = hhat_ref + dT * (integral of the cp polynomial), exactly
    #    as in fit_surrogate_cubicspline step 5, so d(hhat)/dT = cp.
    # 4. Sampled means (phi_bar of Eq 23).
    # 5. Max relative errors on a separate, denser validation grid (hhat
    #    compared as differences from T_lo).
    # Check: base case (6 MPa and 1.5 MPa on [225, 310] K, degree 6) gives
    #    hot cp error ~8e-4 and cold errors below 1e-5.

    # Validate Input
    if T_lo >= T_hi:
        raise ValueError("T_lo must be less than T_hi.")
    if degree < 0:
        raise ValueError("degree must be nonnegative.")
    if n_points < degree + 1:
        raise ValueError("n_points must be at least degree + 1.")

    # 1. Scale temperature to [-1, 1].
    T_bar = 0.5 * (T_lo + T_hi)
    dT = 0.5 * (T_hi - T_lo)

    T_fit = np.linspace(T_lo, T_hi, n_points)
    T_tilde_fit = (T_fit - T_bar) / dT

    table = coolprop_table(fluid, p, T_fit)
    property_names = ("cp", "mu", "k", "rho")

    # Polynomial with coefficients in ascending order:
    # c[0] + c[1]*x + ... + c[degree]*x**degree
    def polynomial(x, *coefficients):
        return np.polynomial.polynomial.polyval(x, coefficients)

    # 2. Fit each property using nonlinear least squares.
    coefficients = {}

    initial_guess = np.zeros(degree + 1)

    for name in property_names:
        y = np.asarray(table[name], dtype=float)

        # Use a linear least-squares fit as a stable initial guess.
        # curve_fit then optimizes the same polynomial model.
        V = np.vander(
            T_tilde_fit, N=degree + 1, increasing=True
        )
        initial_guess = np.linalg.lstsq(V, y, rcond=None)[0]

        popt, _ = curve_fit(
            polynomial,
            T_tilde_fit,
            y,
            p0=initial_guess,
            maxfev=100000,
        )

        coefficients[name] = np.asarray(popt)

    # 3. Construct enthalpy by integrating the cp polynomial.
    hhat_ref = float(
        coolprop_table(
            fluid, p, np.array([T_bar])
        )["hhat"][0]
    )

    cp_integral = np.polynomial.polynomial.polyint(
        coefficients["cp"]
    )
    cp_integral[0] = 0.0

    coefficients["hhat"] = (
        hhat_ref,
        cp_integral,
        T_bar,
        dT,
    )

    # 4. Compute sampled means.
    mean = {
        name: float(np.mean(table[name]))
        for name in (*property_names, "hhat")
    }

    # 5. Validate on a separate, denser temperature grid.
    T_val = np.linspace(T_lo, T_hi, 3 * n_points + 1)
    T_tilde_val = (T_val - T_bar) / dT
    val_table = coolprop_table(fluid, p, T_val)

    max_rel_err = {}

    for name in property_names:
        fitted = polynomial(
            T_tilde_val, *coefficients[name]
        )
        reference = np.asarray(val_table[name], dtype=float)

        denominator = np.maximum(np.abs(reference), 1e-14)
        max_rel_err[name] = float(
            np.max(np.abs(fitted - reference) / denominator)
        )

    # Compare enthalpy increments because its absolute reference
    # value is arbitrary.
    h_fit = hhat_ref + dT * polynomial(
        T_tilde_val, *cp_integral
    )
    h_ref = np.asarray(val_table["hhat"], dtype=float)

    h_fit_delta = h_fit - h_fit[0]
    h_ref_delta = h_ref - h_ref[0]

    h_scale = max(float(np.max(np.abs(h_ref_delta))), 1e-14)

    max_rel_err["hhat"] = float(
        np.max(np.abs(h_fit_delta - h_ref_delta)) / h_scale
    )

    return {
        "kind": "poly",
        "fluid": fluid,
        "p": p,
        "T_lo": T_lo,
        "T_hi": T_hi,
        "T_bar": T_bar,
        "dT": dT,
        "degree": degree,
        "n_points": n_points,
        "coef": coefficients,
        "mean": mean,
        "lam": 1.0,
        "max_rel_err": max_rel_err,
        "hhat_ref": hhat_ref,
    }


def build_surrogates(params, scenario="none"):
    """Build hot- and cold-side surrogates covering every temperature a run can reach.

    By the maximum principle, every stream and wall temperature stays between
    the coldest and the hottest inlet temperature over the whole run, so the
    fit range is [min inlet, max inlet] widened by ``params["fit_T_margin"]``.

    Note that any changes to pressure, inlet temperature or ramp rate will
    require a new surrogate model.

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
    # 1. Coldest inlet over the run: T_min = T_c_in. Hottest: T_max = T_h_in.
    #    For "feed_ramp" the hot inlet moves from T_h_in to T_h_in + ramp_dT,
    #    so widen the range to include both ends of the ramp (a negative
    #    ramp_dT can lower T_min; T_max never drops below the initial T_h_in).
    # 2. T_lo = T_min - fit_T_margin, T_hi = T_max + fit_T_margin. Use the same
    #    range for both sides: in a cold trip the cold stream can warm almost
    #    to T_h_in, and the hot stream can cool almost to T_c_in.
    # 3. For each side (p_h for "h", p_c for "c"): check_single_phase, then
    #    fit_surrogate_curvefit with fit_degree and fit_n_points.
    # 4. If any max_rel_err exceeds fit_max_rel_err, raise ValueError naming the
    #    side and property. Do not silently accept a poor fit.
    # 5. Print each surrogate's max_rel_err: CLAUDE.md asks us to report it.
    # Pitfall: rebuild props whenever a pressure, an inlet temperature or the
    #    ramp size changes (e.g. inside a sweep of ramp rates). A stale surrogate
    #    either raises out of range or silently uses the wrong pressure.
    # Check: with the base case (6 MPa / 1.5 MPa, 300 K / 235 K) and degree 6,
    #    both sides build and every max_rel_err is below 1e-3. With
    #    ramp_dT = -10 the range is still [225, 310] K.

    # Set max and min temperature
    T_min = params["T_c_in"]
    T_max = params["T_h_in"]

    if scenario == "feed_ramp":
        # Both ends of the ramp must be inside the range, whatever its sign.
        T_h_ramped = params["T_h_in"] + params["ramp_dT"]
        T_min = min(T_min, T_h_ramped)
        T_max = max(T_max, T_h_ramped)

    # Add margins to ensure surrogate covers full range
    T_lo = T_min - params["fit_T_margin"]
    T_hi = T_max + params["fit_T_margin"]

    # Build surrogate at each stream's operating pressure
    props = {}
    for side, pressure_key in (("h", "p_h"), ("c", "p_c")):
        pressure = params[pressure_key]

        # check entire range in single phase
        check_single_phase(
            params["fluid"],
            pressure, 
            T_lo, 
            T_hi
        )

        # Fit property surrogate
        surrogate = fit_surrogate_curvefit(
            params["fluid"],
            pressure,
            T_lo,
            T_hi,
            params["fit_degree"],
            params["fit_n_points"],
        )


        # Validate the fitting error.
        max_rel_err = surrogate["max_rel_err"]

        for prop, err in max_rel_err.items():
            if err > params["fit_max_rel_err"]:
                raise ValueError(
                    f"{side}-side {prop} fit error too high: "
                    f"{err:.6e} > {params['fit_max_rel_err']:.6e}"
                )

        props[side] = surrogate

        print(
            f"{side}-side surrogate: "
            f"P={pressure:.6g} Pa, "
            f"T=[{T_lo:.2f}, {T_hi:.2f}] K, "
            f"max_rel_err={max_rel_err}"
        )

    return props



def constant_surrogate(cp, mu, k, rho, T_lo, T_hi, hhat_ref=0.0):
    """Build a constant-property surrogate using the standard surrogate format.

    The surrogate is compatible with evaluate(), enthalpy(), and the
    continuation framework. No CoolProp calls are required.

    Parameters
    ----------
    cp : float
        Heat capacity [J/(kg K)].
    mu : float
        Dynamic viscosity [Pa s].
    k : float
        Thermal conductivity [W/(m K)].
    rho : float
        Density [kg/m^3].
    T_lo, T_hi : float
        Valid temperature range [K].
    hhat_ref : float, optional
        Specific enthalpy at T_bar [J/kg].

    Returns
    -------
    surr : dict
        Standardized polynomial surrogate dictionary.
    """
    # Used for the constant-property verification against the analytical
    # epsilon-NTU solution (Section 3.5) and for unit tests of the solvers.
    # Check: evaluate(surr, "cp", T) returns cp for any T in range, and
    #    enthalpy(surr, T2) - enthalpy(surr, T1) == cp (T2 - T1).
    # 1. Validate inputs.
    if not np.isfinite(T_lo) or not np.isfinite(T_hi):
        raise ValueError("T_lo and T_hi must be finite.")

    if T_lo >= T_hi:
        raise ValueError("T_lo must be less than T_hi.")

    properties = {"cp": cp, "mu": mu, "k": k, "rho": rho}

    for name, value in properties.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive.")

    if not np.isfinite(hhat_ref):
        raise ValueError("hhat_ref must be finite.")

    # 2. Temperature scaling, consistent with the fitted surrogates.
    T_bar = (T_lo + T_hi) / 2
    dT = (T_hi - T_lo) / 2

    # 3. Constant property polynomials.
    # Coefficients are in ascending powers of scaled temperature.
    coef = {
        name: np.array([value], dtype=float)
        for name, value in properties.items()
    }

    # 4. Integrated cp polynomial.
    # cp(T_scaled) = cp, so its integral is cp * T_scaled.
    # enthalpy() multiplies this integral by dT.
    cp_integral = np.polynomial.polynomial.polyint(coef["cp"])
    cp_integral[0] = 0.0

    # Match the hhat representation used by the polynomial fit functions.
    coef["hhat"] = (hhat_ref, cp_integral, T_bar, dT)

    # 5. Mean properties, including the enthalpy reference.
    mean = {
        "cp": cp,
        "mu": mu,
        "k": k,
        "rho": rho,
        "hhat": hhat_ref,
    }

    # 6. Constant properties have zero fitting error.
    max_rel_err = {
        "cp": 0.0,
        "mu": 0.0,
        "k": 0.0,
        "rho": 0.0,
        "hhat": 0.0,
    }

    # 7. Return the standardized surrogate dictionary.
    return {
        "kind": "poly",
        "fluid": "constant",
        "p": None,
        "T_lo": T_lo,
        "T_hi": T_hi,
        "T_bar": T_bar,
        "dT": dT,
        "degree": 0,
        "n_points": 1,
        "coef": coef,
        "mean": mean,
        "lam": 1.0,
        "max_rel_err": max_rel_err,
        "hhat_ref": hhat_ref,
    }


# Bridge between constant and temperature dependent surrogate to ensure
# steady state and transient can be solved numerically
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

    if not 0.0 <= lam <= 1.0:
        raise ValueError("lamdha must be between 0 and 1")

    surr_lam = dict(surr)
    surr_lam["lam"] = lam

    return surr_lam


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
    # 2. Raise ValueError if any T is NaN, below T_lo or above T_hi. Include
    #    the min and max of T, T_lo, T_hi and surr["p"] in the message.
    # Pitfall: if a solver keeps tripping this, the fix is a wider
    #    fit_T_margin or a better initial guess, not removing the check.
    # Check: T exactly at T_lo and T_hi passes; T_lo - 1e-9 raises; a (2, m)
    #    array with one bad entry raises.

    T = np.asarray(T, dtype = float)
    T_lo = surr["T_lo"]
    T_hi = surr["T_hi"]

    # Detect any NaN or out of range temperature
    invalid = (
        np.any(np.isnan(T))
        or np.any(T < T_lo)
        or np.any(T > T_hi)
    )

    if invalid:
        raise ValueError(
            f"Temperature outside surrogate range or contains NaN: "
            f"T_min={np.nanmin(T):.6g} K, "
            f"T_max={np.nanmax(T):.6g} K, "
            f"valid range=[{T_lo:.6g}, {T_hi:.6g}] K, "
            f"p={surr['p']} Pa."
        )


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
    #    kind "spline": phi = coef[name](T). The splines are built in kelvin
    #    (fit_surrogate_cubicspline), so they take T, not T~.
    # 4. If name is in ("cp", "mu", "k") and lam != 1:
    #    phi = (1 - lam) * mean[name] + lam * phi   (Eq 23).
    # 5. Return phi with the same shape as T.
    # Pitfall: never call CoolProp here. This function runs inside every
    #    right-hand-side evaluation.
    # Check: at the fit points the value matches coolprop_table to within
    #    max_rel_err; shape (2, 7) in gives shape (2, 7) out.
    if name not in ("cp", "mu", "k", "rho"):
        raise ValueError(f"Unknown property {name!r} detected")

    # Check temperature bound
    check_range(surr, T)

    # Scale temperature
    T = np.asarray(T, dtype = float)
    T_scaled = (T - surr["T_bar"]) / surr["dT"]

    # Evaluate underlying surrogate model type (poly, cubicspline)
    kind = surr["kind"]
    coef = surr["coef"][name]


    if kind == "poly":
        phi = np.polynomial.polynomial.polyval(T_scaled, coef)
    elif kind == "spline":
        # Splines were fit against T in kelvin, not the scaled T~.
        phi = coef(T)
    else:
        raise ValueError(f"Unknown surrogate kind: {kind!r}")

    # Apply continuation blend to cp, mu, and k
    lam = surr.get("lam", 1.0)
    
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"Continuation parameter lam={lam} is outside [0, 1].")

    if name in ("cp", "mu", "k") and lam != 1.0:
        phi = (1.0 - lam) * surr["mean"][name] + lam * phi

    # 5. Preserve the input shape, including scalar inputs.
    return np.asarray(phi).reshape(T.shape)


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
    # 2. kind "poly": unpack (hhat_ref, P, T_bar, dT) from coef["hhat"] and
    #    return hhat_ref + dT * polyval((T - T_bar)/dT, P).
    #    kind "spline": unpack (antiderivative, offset) from coef["hhat"] and
    #    return antiderivative(T) + offset (in kelvin, like the cp spline).
    # 3. Ignore lam here (see continuation_surrogate): d(hhat)/dT equals the
    #    full cp surrogate, not the blended one.
    # Check: (enthalpy(s, T + 1e-3) - enthalpy(s, T - 1e-3)) / 2e-3 equals
    #    evaluate(s, "cp", T) to ~1e-8 relative; differences
    #    enthalpy(s, T) - enthalpy(s, T_lo) match CoolProp HMASS differences
    #    to within max_rel_err["hhat"].

    # Validate temperature range
    check_range(surr, T)
    T = np.asarray(T, dtype = float)

    # Determine hhat depending on the kind of surrogate model
    kind = surr["kind"]
    if kind == "poly":
        hhat_ref, cp_integral, T_bar, dT = surr["coef"]["hhat"]
        T_scaled = (T - T_bar) / dT
        hhat = hhat_ref + dT * np.polynomial.polynomial.polyval(T_scaled, cp_integral)

    elif kind == "spline":
        cp_integral, hhat_offset = surr["coef"]["hhat"]
        hhat = cp_integral(T) + hhat_offset
    else:
        raise ValueError(f"Unsupported type {kind}")

    return np.asarray(hhat).reshape(T.shape)
