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
from scipy.optimize import curve_fit

# CubicSpline is used instead of curve_fit as we know that data from CoolProp is accurate
# Thus, we want our best fit line to pass through these points as we are not interpolating
# noisy data


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
        If any part of [T_lo, T_hi] is two-phase or below T_sat.
    """
    # Verify T_lo < T_hi
    if T_lo >= T_hi:
        raise ValueError("T_lo must be lower than T_hi")

    # Verify P > 0 
    if p < 0:
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

        T_sat = None
        margin = T_lo = T_pc

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
        "supercritical": p <= p_crit,
        "T_sat": T_sat,
        "T_pc": T_pc,
        "margin": margin,
        "cp_ratio": cp_ratio
    }


def fit_surrogate_cubicspline(fluid, p, T_lo, T_hi, degree, n_points):
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


    # Set max and min temperature
    T_min = params["T_c_in"]
    T_max = params["T_h_in"]

    if scenario == "feed_ramp":
        ramp_dT = params["ramp_dT"]
        if params["ramp_dT"] < 0:
            T_min = min(T_min, T_max + ramp_dT)
            T_max = max(T_min, T_max + ramp_dT)

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

    # Validate input
    if T_lo > T_hi:
        raise ValueError('T_lo cannot be higher than T_hi')

    if cp <= 0 or mu <= 0 or k <= 0 or rho <= 0:
        raise ValueError('cp, mu, k, rho cannot be negative')

    T_bar = (T_lo + T_hi) / 2
    dT = (T_hi - T_lo) / 2

    # Constant properties
    mean = {
        "cp": cp,
        "mu": mu,
        "k": k,
        "rho": rho
    }

    # Constant values
    coeff = {}
    for name,value in mean.items():
        coeff[name] = np.array([value], dtype = float)

    # Enthlapy: integral of cp with respect to dT
    hhat_coeff = np.array([0.0, cp * dT], dtype = float)

    # max_rel_err = 0 since they are constant
    max_rel_err = {
        "cp": 0.0,
        "mu": 0.0,
        "k": 0.0,
        "rho": 0.0
    }

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
        "coef": coeff,
        "mean": mean,
        "lam": 1.0,
        "max_rel_err": max_rel_err,
        "hhat_ref": hhat_ref,
        "hhat_coeff": hhat_coeff
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
    if name not in ("cp", "mu", "k", "rho"):
        raise ValueError(f"Unknown property {name!r} detected")

    # Check temperature bound
    check_range(surr, T)

    # Scale temperature
    T = np.asarray(T, dtype = float)
    T_scaled = (T - surr["T_bar"]) / surr["T_bar"]

    # Evaluate underlying surrogate model type (poly, cubicspline)
    kind = surr["kind"]
    coeff = surr["coeff"][name]


    if kind == "poly":
        phi = np.polynomial.polynomial.polyval(T_scaled, coef)
    elif kind == "spline":
        phi = coef(T_scaled)
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

    # Validate temperature range
    check_range(surr, T)
    T = np.asarray(T, dtype = float)

    # Determine hhat depending on the kind of surrogate model
    kind = surr["kind"]
    hhat_ref = surr["hhat_ref"]
    hhat_coeff = surr["hhat_coeff"]
    T_bar = surr["T_bar"]
    dT = surr["dT"]

    if kind == "poly":
        T_scaled = (T - T_bar) / dT
        hhat = hhat_ref + dT * np.polynomial.polynomial.polyval(T_scaled, hhat_coeff)
    elif kind == "spline":
        cp_spline = surr["coeff"]["cp"]
        cp_integral = cp_spline.antiderivative()
        hhat = hhat_ref + cp_integral(T) - cp_integral(T_bar)
    else:
        raise ValueError(f"Unsupported type {kind}")

    return np.asarray(hhat).reshape(T.shape)
