"""Thermal stress check against the ALPEMA limits (stretch goal).

Math doc: Section 9.2 (fully restrained thermal stress, Eq 53), Section 6.6
(ALPEMA temperature-difference metrics, Eq 40) and Eq 5 (at steady state the
two stream-to-metal differences split T_h - T_c in the ratio a_c : a_h).
Proposal, Section 4: sigma ~ E alpha Delta T / (1 - nu).

ALPEMA sets no stress limit. Its operating-window table limits temperature
differences (stream-to-stream and stream-to-metal < ~30 degC) and rates, and
the pressure design follows a vessel code such as ASME VIII Div. 1. So the
check here has two parts:
  1. pass/fail of the largest stream-to-metal and stream-to-stream
     differences against config.ALPEMA_LIMITS;
  2. the local fully restrained stress of Eq 53, an upper bound used to
     compare locations and operating cases, with two reference stresses: the
     ALPEMA 30 degC limit converted by the same Eq 53, and the yield strength
     of the annealed core alloy (config.MATERIAL_LIMITS).
Rates of change are not checked here (they are zero at steady state; see
metrics.wall_rate and metrics.nozzle_rate for the transient).

Overlap with metrics.py (flagged): metrics.py has stubs for stream_to_stream,
stream_to_metal and thermal_stress with the same equations. This module is
self-contained so the stress check does not depend on metrics.py; once
metrics.py is filled in, keep one copy.

Like metrics.py, this module imports no other src module and never imports
config: callers pass params, config.ALPEMA_LIMITS and the yield strength in.

Array conventions: temperature fields reduce over the LAST axis (cells), so
every function works on a steady profile, shape (n,), and on a transient
history, shape (n_t, N), alike.

Units: K for temperatures and differences (a difference in K equals the same
difference in degC; never add 273.15 to a difference), Pa for stresses, m for x.

Imports: numpy.
"""

import numpy as np

# Names of the three local temperature differences, in plotting order.
DIFFERENCES = ("hot-to-metal", "metal-to-cold", "stream-to-stream")


def temperature_differences(T_h, T_c, T_w):
    """Local hot-to-metal, metal-to-cold and stream-to-stream differences.

    Eq 40 (the terms inside the max). At steady state the first two split the
    third in the ratio a_c : a_h (Eq 5), so they add up to T_h - T_c.

    Parameters
    ----------
    T_h, T_c, T_w : ndarray, shape (n,) or (n_t, N)
        Hot stream, cold stream and wall temperatures [K].

    Returns
    -------
    dT : dict of ndarray, each the shape of T_h
        "hot-to-metal" = T_h - T_w, "metal-to-cold" = T_w - T_c,
        "stream-to-stream" = T_h - T_c [K]. All positive in normal operation.
    """
    return {
        "hot-to-metal": T_h - T_w,
        "metal-to-cold": T_w - T_c,
        "stream-to-stream": T_h - T_c,
    }


def thermal_stress(dT, E, alpha_T, nu):
    """Thermal stress in a fully restrained plate.

    Eq 53: sigma_th ~ E alpha_T Delta T / (1 - nu). The plate would expand by
    alpha_T Delta T in both in-plane directions; full restraint turns that
    strain into a biaxial stress. An upper bound, because a real core is only
    partly restrained (fins and parting sheets can bend).

    Parameters
    ----------
    dT : float or ndarray
        Local temperature difference [K].
    E : float
        Young's modulus [Pa] (params["E_m"]).
    alpha_T : float
        Thermal expansion coefficient [1/K] (params["alpha_T"]).
    nu : float
        Poisson's ratio [-] (params["nu_m"]).

    Returns
    -------
    sigma : float or ndarray, same shape as dT
        Thermal stress [Pa].
    """
    # Check: E = 70 GPa, alpha_T = 23e-6 1/K, nu = 0.33 gives ~2.4 MPa per kelvin
    #    (math doc Section 9.2).
    return E * alpha_T * dT / (1.0 - nu)


def alpema_difference_check(T_h, T_c, T_w, limits):
    """Compare the largest temperature differences with the ALPEMA limits.

    Eq 40: Delta T_sm = max over cells of max(|T_h - T_w|, |T_w - T_c|) and
    Delta T_ss = max over cells of |T_h - T_c|. For a transient history the
    max is also taken over all saved times, so one pass/fail covers the run.

    Parameters
    ----------
    T_h, T_c, T_w : ndarray, shape (n,) or (n_t, N)
        Stream and wall temperatures [K].
    limits : dict
        config.ALPEMA_LIMITS; uses "dT_stream_metal" and "dT_stream_stream"
        [degC = K for a difference].

    Returns
    -------
    checks : dict
        "stream-to-metal" and "stream-to-stream", each a dict
        {"value": largest difference [K], "limit": [K],
         "margin": limit - value [K], "ok": value < limit}.
    """
    dT = temperature_differences(T_h, T_c, T_w)
    values = {
        "stream-to-metal": float(np.max(np.maximum(np.abs(dT["hot-to-metal"]),
                                                   np.abs(dT["metal-to-cold"])))),
        "stream-to-stream": float(np.max(np.abs(dT["stream-to-stream"]))),
    }
    limit = {"stream-to-metal": limits["dT_stream_metal"],
             "stream-to-stream": limits["dT_stream_stream"]}
    return {name: {"value": v, "limit": limit[name], "margin": limit[name] - v,
                   "ok": bool(v < limit[name])}
            for name, v in values.items()}


def stress_check(x, T_h, T_c, T_w, params, limits, sigma_yield):
    """Full stress check for one temperature field (steady or transient).

    Combines temperature_differences, thermal_stress and
    alpema_difference_check, and converts the two reference levels between
    stress and temperature difference with Eq 53.

    Parameters
    ----------
    x : ndarray, shape (n,) or (N,)
        Positions of the temperature values along the core [m].
    T_h, T_c, T_w : ndarray, shape (n,) or (n_t, N)
        Stream and wall temperatures [K] (steady result or transient history).
    params : dict
        Uses "E_m" [Pa], "alpha_T" [1/K], "nu_m" [-].
    limits : dict
        config.ALPEMA_LIMITS.
    sigma_yield : float
        Yield strength of the core metal [Pa]
        (config.MATERIAL_LIMITS["sigma_y_3003_O"]).

    Returns
    -------
    result : dict
        "dT"           dict of local differences [K] (temperature_differences)
        "sigma"        dict of local stresses [Pa], same keys and shapes
        "per_K"        stress per kelvin of difference [Pa/K]
        "sigma_alpema" stress at the ALPEMA stream-to-metal limit [Pa]
        "sigma_yield"  yield strength passed in [Pa]
        "dT_yield"     difference at which Eq 53 reaches the yield strength [K]
        "peaks"        dict per difference: {"sigma" [Pa], "dT" [K], "x" [m],
                       "index" (tuple into the arrays)} at the largest |sigma|
        "check"        ALPEMA pass/fail (alpema_difference_check)
    """
    E, alpha_T, nu = params["E_m"], params["alpha_T"], params["nu_m"]
    per_K = thermal_stress(1.0, E, alpha_T, nu)

    dT = temperature_differences(T_h, T_c, T_w)
    sigma = {name: thermal_stress(d, E, alpha_T, nu) for name, d in dT.items()}

    peaks = {}
    for name, s in sigma.items():
        # Index of the largest |sigma|; for a history this is (time index, cell index).
        idx = np.unravel_index(np.argmax(np.abs(s)), np.shape(s))
        peaks[name] = {"sigma": float(s[idx]), "dT": float(dT[name][idx]),
                       "x": float(x[idx[-1]]), "index": idx}

    return {
        "dT": dT,
        "sigma": sigma,
        "per_K": per_K,
        "sigma_alpema": per_K * limits["dT_stream_metal"],
        "sigma_yield": sigma_yield,
        "dT_yield": sigma_yield / per_K,
        "peaks": peaks,
        "check": alpema_difference_check(T_h, T_c, T_w, limits),
    }


def summary_text(result):
    """Readable summary of a stress_check result, in K and MPa.

    Parameters
    ----------
    result : dict
        From stress_check.

    Returns
    -------
    text : str
        The ALPEMA pass/fail lines, the peak stress of each difference with
        its location, and the two reference stresses.
    """
    MPa = 1e6
    lines = ["ALPEMA check on the temperature differences:"]
    for name, c in result["check"].items():
        verdict = "OK" if c["ok"] else "EXCEEDED"
        lines.append(f"  {name:17s} max {c['value']:5.1f} K   limit {c['limit']:.0f} K   "
                     f"margin {c['margin']:5.1f} K   {verdict}")
    lines.append(f"Fully restrained thermal stress (Eq 53): {result['per_K'] / MPa:.2f} MPa "
                 "per K of difference")
    for name in DIFFERENCES:
        p = result["peaks"][name]
        lines.append(f"  {name:17s} peak {p['sigma'] / MPa:5.1f} MPa at x = {p['x']:.2f} m "
                     f"(Delta T = {p['dT']:.1f} K)")
    lines.append(f"  ALPEMA limit as a stress: {result['sigma_alpema'] / MPa:.1f} MPa")
    lines.append(f"  Yield strength {result['sigma_yield'] / MPa:.0f} MPa is reached at "
                 f"Delta T = {result['dT_yield']:.1f} K")
    return "\n".join(lines)
