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

Eq 53 is an index, not a design stress: at the ALPEMA limit itself it gives
~72 MPa, twice the 35 MPa yield strength of annealed 3003. A real core is
only partly restrained (fins and parting sheets bend), which is why the math
doc calls this the hand-off point to a structural finite element model.

One copy of each equation: the maxima of Eq 40 come from metrics.py
(stream_to_stream, stream_to_metal) and Eq 53 is written only here.

Steady or transient: use stress_check for a steady profile (the BVP grid
already includes x = 0 and x = L) and stress_check_transient for a
transient.simulate result. The transient version adds the two end faces of
the core (metrics.with_end_faces), where the largest differences sit during a
trip or a startup; the cell averages alone under-report them and the ALPEMA
verdict then changes with the number of cells.

This module never imports config: callers pass params, config.ALPEMA_LIMITS
and the yield strength in.

Array conventions: temperature fields reduce over the LAST axis (positions
along the core), so every function works on a steady profile, shape (n,), and
on a transient history, shape (n_t, n), alike.

Units: K for temperatures and differences (a difference in K equals the same
difference in degC; never add 273.15 to a difference), Pa for stresses, m for x.

Imports: numpy and src.metrics.
"""

import numpy as np

# Import through the src package (run code from the repo root: pytest, or
# python -m scripts.<name>), like the other src modules.
from src import metrics

# Names of the three local temperature differences, in plotting order.
DIFFERENCES = ("hot-to-metal", "metal-to-cold", "stream-to-stream")


def temperature_differences(T_h, T_c, T_w):
    """Local hot-to-metal, metal-to-cold and stream-to-stream differences.

    Eq 40 (the terms inside the max). At steady state the first two split the
    third in the ratio a_c : a_h (Eq 5), so they add up to T_h - T_c.

    Parameters
    ----------
    T_h, T_c, T_w : ndarray, shape (n,) or (n_t, n)
        Hot stream, cold stream and wall temperatures [K].

    Returns
    -------
    dT : dict of ndarray, each the shape of T_h
        "hot-to-metal" = T_h - T_w, "metal-to-cold" = T_w - T_c,
        "stream-to-stream" = T_h - T_c [K]. All positive in normal operation.
    """
    # 1. Three subtractions, kept signed (a negative value means the heat
    #    flows the other way, e.g. metal warmer than the hot gas).
    # Check: hand-made arrays give the differences by inspection, and
    #    hot-to-metal + metal-to-cold = stream-to-stream.
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
        Local temperature difference (stream-to-metal or stream-to-stream) [K].
    E : float
        Young's modulus [Pa] (params["E_m"]).
    alpha_T : float
        Thermal expansion coefficient [1/K] (params["alpha_T"]).
    nu : float
        Poisson's ratio [-] (params["nu_m"]).

    Returns
    -------
    sigma : float or ndarray, same shape as dT
        Thermal stress [Pa]. Keeps the sign of dT.
    """
    # 1. sigma = E alpha_T dT / (1 - nu).
    # Check: E = 70 GPa, alpha_T = 23e-6 1/K, nu = 0.33 gives ~2.4 MPa per kelvin
    #    (math doc Section 9.2).
    return E * alpha_T * dT / (1.0 - nu)


def alpema_difference_check(T_h, T_c, T_w, limits):
    """Compare the largest temperature differences with the ALPEMA limits.

    Eq 40: Delta T_sm = max over positions of max(|T_h - T_w|, |T_w - T_c|)
    and Delta T_ss = max over positions of |T_h - T_c|. For a transient
    history the max is also taken over all saved times, so one pass/fail
    covers the run.

    Parameters
    ----------
    T_h, T_c, T_w : ndarray, shape (n,) or (n_t, n)
        Stream and wall temperatures [K]. For a transient pass the fields
        with the end faces (metrics.with_end_faces), not the cells alone.
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
    # 1. Largest stream-to-metal and stream-to-stream differences along the
    #    core from metrics.stream_to_metal and metrics.stream_to_stream
    #    (Eq 40), then np.max over the saved times for a history.
    # 2. Compare each with its limit: ok = value < limit, margin = limit - value.
    # Pitfall: on the cells of a transient alone the maximum misses the inlet
    #    faces. For the base-case cold trip with N = 50 the cells give a
    #    stream-to-metal maximum of 29.9 K, which "passes" the 30 K limit; the
    #    converged value is ~52.6 K. Use stress_check_transient.
    # Check: with a_h = a_c the stream-to-metal value is half the
    #    stream-to-stream value (Eq 5); a difference just under / just over
    #    the limit flips "ok".
    values = {
        "stream-to-metal": float(np.max(metrics.stream_to_metal(T_h, T_c, T_w))),
        "stream-to-stream": float(np.max(metrics.stream_to_stream(T_h, T_c))),
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
    x : ndarray, shape (n,)
        Positions of the temperature values along the core [m].
    T_h, T_c, T_w : ndarray, shape (n,) or (n_t, n)
        Stream and wall temperatures [K]: a steady result, or a transient
        history with the end faces added (stress_check_transient does this).
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
        "dT_max"       dict per difference: largest |difference| along the
                       core [K]; a float for a steady profile, shape (n_t,)
                       (one value per saved time) for a history
        "sigma_max"    the same for the stress [Pa]
        "per_K"        stress per kelvin of difference [Pa/K]
        "sigma_alpema" stress at the ALPEMA stream-to-metal limit [Pa]
        "sigma_yield"  yield strength passed in [Pa]
        "dT_yield"     difference at which Eq 53 reaches the yield strength [K]
        "peaks"        dict per difference: {"sigma" [Pa], "dT" [K], "x" [m],
                       "index" (tuple into the arrays)} at the largest |sigma|
        "check"        ALPEMA pass/fail (alpema_difference_check)
    """
    # 1. per_K = thermal_stress(1 K): Eq 53 is linear, so one number converts
    #    any difference to a stress and any stress back to a difference.
    # 2. Local differences (temperature_differences) and their stresses.
    # 3. Largest |difference| and |stress| along the core (last axis).
    # 4. Peak of each difference: np.argmax of |sigma| over the whole array;
    #    np.unravel_index turns it into (position index,) for a profile or
    #    (time index, position index) for a history.
    # 5. Reference levels: sigma_alpema = per_K * limit, dT_yield = sigma_yield / per_K.
    # Check: on a hand-made profile whose gap grows to 17 K at x = L the
    #    stream-to-stream peak is 17 K * per_K at x = L.
    E, alpha_T, nu = params["E_m"], params["alpha_T"], params["nu_m"]
    per_K = thermal_stress(1.0, E, alpha_T, nu)

    dT = temperature_differences(T_h, T_c, T_w)
    sigma = {name: thermal_stress(d, E, alpha_T, nu) for name, d in dT.items()}
    dT_max = {name: np.max(np.abs(d), axis=-1) for name, d in dT.items()}
    sigma_max = {name: per_K * d for name, d in dT_max.items()}

    peaks = {}
    for name, s in sigma.items():
        # Index of the largest |sigma|; for a history this is (time index, position index).
        idx = np.unravel_index(np.argmax(np.abs(s)), np.shape(s))
        peaks[name] = {"sigma": float(s[idx]), "dT": float(dT[name][idx]),
                       "x": float(x[idx[-1]]), "index": idx}

    return {
        "dT": dT,
        "sigma": sigma,
        "dT_max": dT_max,
        "sigma_max": sigma_max,
        "per_K": per_K,
        "sigma_alpema": per_K * limits["dT_stream_metal"],
        "sigma_yield": sigma_yield,
        "dT_yield": sigma_yield / per_K,
        "peaks": peaks,
        "check": alpema_difference_check(T_h, T_c, T_w, limits),
    }


def stress_check_transient(result, params, limits, sigma_yield):
    """Full stress check for a transient run, including the two end faces of the core.

    stress_check on the fields from metrics.with_end_faces: the N cells plus
    the face at x = 0 (hot inlet, cold outlet) and the face at x = L (cold
    inlet, hot outlet). The faces carry the largest differences during a trip
    or a startup, and they make the result nearly independent of N.

    Parameters
    ----------
    result : dict
        Result dict from transient.simulate.
    params : dict
        Uses "L" [m] and the metal properties "E_m" [Pa], "alpha_T" [1/K],
        "nu_m" [-].
    limits : dict
        config.ALPEMA_LIMITS.
    sigma_yield : float
        Yield strength of the core metal [Pa].

    Returns
    -------
    check : dict
        Everything stress_check returns, on arrays of shape (n_t, N + 2), plus
        "t"  saved times [s], shape (n_t,)
        "x"  positions of the N + 2 columns [m]: 0, the cell centres, L
        and each entry of "peaks" also has "t", the time of that peak [s].
    """
    # 1. T_h, T_c, T_w with the end faces: metrics.with_end_faces(result).
    # 2. Positions: 0 for the first face, the cell centres, L for the last face.
    # 3. stress_check on those arrays.
    # 4. Time of each peak from its time index (the first entry of "index").
    # Check: for the base-case cold trip the stream-to-stream peak equals the
    #    largest nozzle difference T_h,N - T_c_in exactly, on any grid, and the
    #    stream-to-metal verdict is the same for N = 50 and N = 100.
    T_h_f, T_c_f, T_w_f = metrics.with_end_faces(result)
    x_f = np.concatenate(([0.0], result["x"], [params["L"]]))

    check = stress_check(x_f, T_h_f, T_c_f, T_w_f, params, limits, sigma_yield)
    check["t"] = result["t"]
    check["x"] = x_f
    for peak in check["peaks"].values():
        peak["t"] = float(result["t"][peak["index"][0]])
    return check


def summary_text(result):
    """Readable summary of a stress_check result, in K and MPa.

    Parameters
    ----------
    result : dict
        From stress_check or stress_check_transient.

    Returns
    -------
    text : str
        The ALPEMA pass/fail lines, the peak stress of each difference with
        its location (and time, for a transient), and the two reference
        stresses.
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
        when = f", t = {p['t']:.0f} s" if "t" in p else ""
        lines.append(f"  {name:17s} peak {p['sigma'] / MPa:5.1f} MPa at x = {p['x']:.2f} m{when} "
                     f"(Delta T = {p['dT']:.1f} K)")
    lines.append(f"  ALPEMA limit as a stress: {result['sigma_alpema'] / MPa:.1f} MPa")
    lines.append(f"  Yield strength {result['sigma_yield'] / MPa:.0f} MPa is reached at "
                 f"Delta T = {result['dT_yield']:.1f} K")
    return "\n".join(lines)
