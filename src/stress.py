"""Thermal stress check against the ALPEMA limits (stretch goal).

WHAT THIS FILE DOES (the short version you can say out loud)
-------------------------------------------------------------
The thermal model gives three temperatures at every position along the core:
the hot gas T_h, the cold gas T_c and the aluminium wall T_w. This file
  1. subtracts them to get the local temperature differences,
  2. checks the largest differences against the ALPEMA limits (pass/fail),
  3. turns each difference into a stress with sigma = E alpha_T dT / (1 - nu),
  4. finds where (and, for a transient, when) the stress is largest,
  5. prints a readable summary.
It is a calculations file only, like properties.py: it never solves the model
and never plots. scripts/plot_stress.py and the report notebook call it. Run
it directly (python -m src.stress, from the repo root) to see the steady base
case and the startup from rest checked with real model output.

WHERE THE EQUATIONS COME FROM
-----------------------------
Math doc: Section 9.2 (fully restrained thermal stress, Eq 53), Section 6.6
(ALPEMA temperature-difference metrics, Eq 40) and Eq 5 (at steady state the
two stream-to-metal differences split T_h - T_c in the ratio a_c : a_h).
Proposal, Section 4: sigma ~ E alpha Delta T / (1 - nu).

WHY THE CHECK HAS TWO PARTS
---------------------------
ALPEMA sets no stress limit. Its operating-window table limits temperature
differences (stream-to-stream and stream-to-metal < ~30 degC) and rates, and
the pressure design follows a vessel code such as ASME VIII Div. 1. So:
  1. pass/fail of the largest stream-to-metal and stream-to-stream
     differences against config.ALPEMA_LIMITS (this is the real check);
  2. the local fully restrained stress of Eq 53, an upper bound used to
     compare locations and operating cases, with two reference stresses: the
     ALPEMA 30 degC limit converted by the same Eq 53, and the yield strength
     of the annealed core alloy (config.MATERIAL_LIMITS).
Rates of change are not checked here (they are zero at steady state; see
metrics.wall_rate and metrics.nozzle_rate for the transient).

WHY THE STRESS IS ONLY AN INDEX
-------------------------------
Eq 53 is an index, not a design stress: at the ALPEMA limit itself it gives
~72 MPa, twice the 35 MPa yield strength of annealed 3003. If it were a real
stress, every exchanger run at the ALPEMA limit would yield. A real core is
only partly restrained (fins and parting sheets bend and take up the strain),
which is why the math doc calls this the hand-off point to a structural
finite element model. Use it to say WHERE and IN WHICH CASE the thermal
loading is worst, not to predict failure.

ONE COPY OF EACH EQUATION
-------------------------
The maxima of Eq 40 come from metrics.py (stream_to_stream, stream_to_metal)
and Eq 53 is written only here, so the two files can never disagree.

STEADY OR TRANSIENT
-------------------
Use stress_check for a steady profile (the BVP grid already includes x = 0
and x = L) and stress_check_transient for a transient.simulate result. The
transient version adds the two end faces of the core (metrics.with_end_faces),
where the largest differences sit during a trip or a startup; the cell
averages alone under-report them and the ALPEMA verdict then changes with the
number of cells.

The functions never import config: callers pass params, config.ALPEMA_LIMITS
and the yield strength in, so a test can run them on any made-up numbers.
Only the demo at the bottom (which runs when you execute this file) reads
config.

ARRAY CONVENTIONS
-----------------
Temperature fields reduce over the LAST axis (positions along the core), so
every function works on a steady profile, shape (n,), and on a transient
history, shape (n_t, n), alike. Row = one saved time, column = one position.

UNITS
-----
K for temperatures and differences (a difference in K equals the same
difference in degC; never add 273.15 to a difference), Pa for stresses, m for
x, s for t. Conversion to MPa happens only when printing.

Imports: numpy and src.metrics.
"""

import numpy as np  # array maths: subtraction, max, argmax over whole arrays at once

# Import through the src package (run code from the repo root: pytest, or
# python -m scripts.<name>), like the other src modules.
from src import metrics  # Eq 40 maxima (stream_to_stream, stream_to_metal) and with_end_faces

# DIFFERENCES: the names of the three local temperature differences, in the
# order the plots and the printed summary list them. Used as dictionary keys
# everywhere below, so a typo in a key shows up as a KeyError instead of a
# silently wrong number.
#   "hot-to-metal"     = T_h - T_w   (hot gas minus wall)
#   "metal-to-cold"    = T_w - T_c   (wall minus cold gas)
#   "stream-to-stream" = T_h - T_c   (hot gas minus cold gas; the sum of the other two)
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
    # Physics: heat flows hot gas -> wall -> cold gas, so in normal operation
    # T_h > T_w > T_c and all three differences are positive. Each one is a
    # temperature "step" the metal has to sit across. The bigger the step,
    # the more one part of the core wants to expand relative to its
    # neighbour, and the bigger the thermal stress.
    #
    # The differences are kept SIGNED (no abs here): a negative value means
    # heat is flowing the other way, e.g. during a startup the idle metal can
    # be warmer than the cold gas just entering, or than the hot gas.
    #
    # The subtraction is elementwise, so it works the same on one profile
    # (shape (n,)) and on a whole history (shape (n_t, n)).

    # dT_hw [K]: hot gas minus wall at every point. Drives heat INTO the wall
    # from the hot side (Eq 5: a_h * dT_hw is the heat flux per unit length).
    dT_hw = T_h - T_w

    # dT_wc [K]: wall minus cold gas at every point. Drives heat OUT of the
    # wall into the cold side (a_c * dT_wc).
    dT_wc = T_w - T_c

    # dT_hc [K]: hot gas minus cold gas at every point; the total driving
    # force of the exchanger (U' * dT_hc). Always equals dT_hw + dT_wc,
    # because T_w cancels out.
    dT_hc = T_h - T_c

    # Check: hand-made arrays give the differences by inspection, and
    #    hot-to-metal + metal-to-cold = stream-to-stream.
    return {
        "hot-to-metal": dT_hw,
        "metal-to-cold": dT_wc,
        "stream-to-stream": dT_hc,
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
    # Derivation (how to explain Eq 53 in three steps):
    #   1. Free expansion: a piece of metal that is dT warmer than its
    #      surroundings wants to grow by the thermal strain
    #          strain_th = alpha_T * dT                     [-]
    #      e.g. 23e-6 1/K * 10 K = 2.3e-4, i.e. 0.023 % longer.
    #   2. Full restraint: if the neighbouring metal holds it at its original
    #      size in BOTH in-plane directions (x and y), the elastic strain must
    #      cancel the thermal strain exactly. Hooke's law for a plate with equal
    #      stress sigma in x and y gives
    #          strain_x = (sigma - nu * sigma) / E - alpha_T * dT = 0
    #   3. Solve for sigma:
    #          sigma = E * alpha_T * dT / (1 - nu)
    #      The 1/(1 - nu) factor (~1.5 for aluminium) is there because the
    #      restraint acts in two directions at once (biaxial); a bar held at
    #      its two ends only (uniaxial) would give E * alpha_T * dT.
    #
    # Sign: sigma has the sign of dT here; it says which side is hotter, not
    # whether the metal is in tension or compression. The check below only
    # uses the size |sigma|.
    #
    # Inputs (all SI):
    #   dT      [K]   temperature difference the metal sits across
    #   E       [Pa]  Young's modulus, stiffness of aluminium, 70e9 Pa = 70 GPa
    #   alpha_T [1/K] linear thermal expansion coefficient, 23e-6 1/K
    #   nu      [-]   Poisson's ratio, 0.33 (how much a material shrinks
    #                 sideways when stretched)
    #
    # Check: E = 70 GPa, alpha_T = 23e-6 1/K, nu = 0.33 gives ~2.4 MPa per kelvin
    #    (math doc Section 9.2): 70e9 * 23e-6 / 0.67 = 2.40e6 Pa/K.
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
    # This is the actual ALPEMA check: one number per difference for the whole
    # core (and, for a history, for the whole run), compared with 30 K.
    #
    # Pitfall: on the cells of a transient alone the maximum misses the inlet
    #    faces. For the base-case cold trip with N = 50 the cells give a
    #    stream-to-metal maximum of 29.9 K, which "passes" the 30 K limit; the
    #    converged value is ~52.6 K. Use stress_check_transient.
    # Check: with a_h = a_c the stream-to-metal value is half the
    #    stream-to-stream value (Eq 5); a difference just under / just over
    #    the limit flips "ok".

    # dT_sm_along [K]: largest stream-to-metal difference along the core.
    # metrics.stream_to_metal takes the worse of |T_h - T_w| and |T_w - T_c|
    # at each point, then the max over positions (last axis). Result: a single
    # number for a steady profile, or one number per saved time, shape (n_t,),
    # for a history.
    dT_sm_along = metrics.stream_to_metal(T_h, T_c, T_w)

    # dT_ss_along [K]: largest |T_h - T_c| along the core, same shapes as above.
    dT_ss_along = metrics.stream_to_stream(T_h, T_c)

    # values [K]: the single worst value of each difference, over positions
    # AND (for a history) over all saved times. np.max of a single number is
    # that number, so the same line handles steady and transient. float()
    # turns the numpy scalar into a plain Python number for printing.
    values = {
        "stream-to-metal": float(np.max(dT_sm_along)),
        "stream-to-stream": float(np.max(dT_ss_along)),
    }

    # limit [K]: the ALPEMA limit for each difference, read from the limits
    # dict (config.ALPEMA_LIMITS, both 30 degC). A limit in degC is the same
    # number in K because it is a difference, not an absolute temperature.
    limit = {
        "stream-to-metal": limits["dT_stream_metal"],
        "stream-to-stream": limits["dT_stream_stream"],
    }

    # checks: the result, one entry per difference.
    checks = {}
    for name in values:                        # name: "stream-to-metal", then "stream-to-stream"
        value = values[name]                   # value [K]: worst difference found by the model
        lim = limit[name]                      # lim [K]: what ALPEMA allows
        margin = lim - value                   # margin [K]: room left; negative = limit exceeded by that much
        ok = bool(value < lim)                 # ok: True = passes ALPEMA (strictly below the limit)
        checks[name] = {"value": value, "limit": lim, "margin": margin, "ok": ok}
    return checks


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
    # Check: on a hand-made profile whose gap grows to 17 K at x = L the
    #    stream-to-stream peak is 17 K * per_K at x = L.

    # --- Step 1: material properties of the aluminium core (from config.BASE_CASE) ---
    E = params["E_m"]            # E [Pa]: Young's modulus, 70e9 Pa
    alpha_T = params["alpha_T"]  # alpha_T [1/K]: thermal expansion coefficient, 23e-6 1/K
    nu = params["nu_m"]          # nu [-]: Poisson's ratio, 0.33

    # --- Step 2: the conversion factor between kelvin and pascal ---
    # per_K [Pa/K]: stress produced by a 1 K difference (~2.40e6 Pa/K =
    # 2.40 MPa per K). Eq 53 is linear in dT, so this one number converts any
    # difference to a stress (multiply) and any stress back to a difference
    # (divide). This is the single most useful number to quote.
    per_K = thermal_stress(1.0, E, alpha_T, nu)

    # --- Step 3: local differences and local stresses at every point (and time) ---
    # dT: dict of three arrays [K], each the same shape as T_h.
    dT = temperature_differences(T_h, T_c, T_w)

    # sigma: dict of three arrays [Pa], the same keys and shapes as dT; Eq 53
    # applied point by point.
    sigma = {}
    for name in dT:                                    # name: one of DIFFERENCES
        sigma[name] = thermal_stress(dT[name], E, alpha_T, nu)

    # --- Step 4: the worst value along the core ---
    # dT_max: dict [K]. For each difference, the largest SIZE (|dT|) over the
    # positions (axis=-1, the last axis). Steady profile -> one number;
    # history -> one number per saved time, shape (n_t,), so it can be
    # plotted against time.
    dT_max = {}
    for name in dT:
        dT_max[name] = np.max(np.abs(dT[name]), axis=-1)

    # sigma_max: dict [Pa]. The same thing as a stress. Because Eq 53 is
    # linear, the largest stress is just per_K times the largest difference.
    sigma_max = {}
    for name in dT_max:
        sigma_max[name] = per_K * dT_max[name]

    # --- Step 5: where (and when) each stress peaks ---
    # peaks: dict per difference with the stress, difference, position, and
    # array index of the single largest |sigma| in the whole field.
    peaks = {}
    for name in sigma:
        s = sigma[name]                        # s [Pa]: local stress array for this difference
        # flat_idx: position of the largest |s| if the array were laid out as
        # one long list (np.argmax works on the flattened array).
        flat_idx = np.argmax(np.abs(s))
        # idx: the same position as a tuple of indices into the original
        # array: (position index,) for a steady profile, or
        # (time index, position index) for a history.
        idx = np.unravel_index(flat_idx, np.shape(s))
        sigma_peak = float(s[idx])             # sigma_peak [Pa]: stress at the peak (signed)
        dT_peak = float(dT[name][idx])         # dT_peak [K]: temperature difference at the peak (signed)
        x_peak = float(x[idx[-1]])             # x_peak [m]: position of the peak; the last index is always the position
        peaks[name] = {"sigma": sigma_peak, "dT": dT_peak, "x": x_peak, "index": idx}

    # --- Step 6: the two reference levels, each as a stress AND as a difference ---
    # sigma_alpema [Pa]: the ALPEMA 30 K stream-to-metal limit converted to a
    # stress with Eq 53 (~72 MPa). Lets the stress plot show where ALPEMA sits.
    sigma_alpema = per_K * limits["dT_stream_metal"]

    # dT_yield [K]: the temperature difference at which Eq 53 reaches the
    # yield strength (35e6 Pa / 2.40e6 Pa/K = 14.6 K). It is BELOW the 30 K
    # ALPEMA limit, which is how we know Eq 53 overestimates the real stress
    # (see the module docstring).
    dT_yield = sigma_yield / per_K

    # --- Step 7: the ALPEMA pass/fail (the real check) ---
    # check: dict from alpema_difference_check, worst stream-to-metal and
    # stream-to-stream values with limits, margins and ok flags.
    check = alpema_difference_check(T_h, T_c, T_w, limits)

    return {
        "dT": dT,
        "sigma": sigma,
        "dT_max": dT_max,
        "sigma_max": sigma_max,
        "per_K": per_K,
        "sigma_alpema": sigma_alpema,
        "sigma_yield": sigma_yield,   # sigma_yield [Pa]: passed in, kept so summary_text can print it
        "dT_yield": dT_yield,
        "peaks": peaks,
        "check": check,
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
    # Why the end faces: the transient model stores one AVERAGE temperature
    # per cell. When a stream changes temperature within a few millimetres of
    # its inlet (a startup, a trip), the inlet cell average hides the sharp
    # step right at the inlet. The face values put that step back:
    #   face at x = 0: T_h = T_h_in(t) (hot gas entering), T_c = last cold cell
    #   face at x = L: T_c = T_c_in(t) (cold gas entering), T_h = last hot cell
    #   metal on each face = the metal of the neighbouring cell.
    # Check: for the base-case cold trip the stream-to-stream peak equals the
    #    largest nozzle difference T_h,N - T_c_in exactly, on any grid, and the
    #    stream-to-metal verdict is the same for N = 50 and N = 100.

    # T_h_f, T_c_f, T_w_f [K]: hot, cold and metal histories, each shape
    # (n_t, N + 2): column 0 = face at x = 0, columns 1..N = the cells,
    # column N + 1 = face at x = L ("_f" = "with faces").
    T_h_f, T_c_f, T_w_f = metrics.with_end_faces(result)

    # x_f [m]: positions matching those N + 2 columns: 0, the N cell centres
    # (result["x"]), then L (params["L"], the core length).
    x_f = np.concatenate(([0.0], result["x"], [params["L"]]))

    # check: the full stress check (stress_check above) run on the face-added
    # history. Every array in it has shape (n_t, N + 2).
    check = stress_check(x_f, T_h_f, T_c_f, T_w_f, params, limits, sigma_yield)

    check["t"] = result["t"]   # t [s]: saved times of the run, shape (n_t,)
    check["x"] = x_f           # x [m]: positions of the columns, shape (N + 2,)

    # Add the time of each peak. peak["index"] is (time index, position
    # index), so its first entry picks the saved time.
    for name in check["peaks"]:
        peak = check["peaks"][name]          # peak: the dict for one difference
        i_t = peak["index"][0]               # i_t: row of the history where the peak sits
        peak["t"] = float(result["t"][i_t])  # t [s]: time of the peak
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
    MPa = 1e6  # MPa [Pa/MPa]: divide a stress in Pa by this to print it in MPa

    # lines: the summary, one string per printed line, joined at the end.
    lines = ["ALPEMA check on the temperature differences:"]

    # Part 1: the pass/fail lines (the real ALPEMA check).
    for name in result["check"]:                       # name: "stream-to-metal", "stream-to-stream"
        c = result["check"][name]                      # c: {"value", "limit", "margin", "ok"}
        verdict = "OK" if c["ok"] else "EXCEEDED"      # verdict: the word printed at the end of the line
        lines.append(f"  {name:17s} max {c['value']:5.1f} K   limit {c['limit']:.0f} K   "
                     f"margin {c['margin']:5.1f} K   {verdict}")

    # Part 2: the conversion factor, then the peak stress of each difference.
    lines.append(f"Fully restrained thermal stress (Eq 53): {result['per_K'] / MPa:.2f} MPa "
                 "per K of difference")
    for name in DIFFERENCES:
        p = result["peaks"][name]                      # p: {"sigma", "dT", "x", "index"} (+ "t" for a transient)
        # when: ", t = ... s" for a transient (its peaks carry a time), empty for steady.
        when = f", t = {p['t']:.0f} s" if "t" in p else ""
        lines.append(f"  {name:17s} peak {p['sigma'] / MPa:5.1f} MPa at x = {p['x']:.2f} m{when} "
                     f"(Delta T = {p['dT']:.1f} K)")

    # Part 3: the two reference levels.
    lines.append(f"  ALPEMA limit as a stress: {result['sigma_alpema'] / MPa:.1f} MPa")
    lines.append(f"  Yield strength {result['sigma_yield'] / MPa:.0f} MPa is reached at "
                 f"Delta T = {result['dT_yield']:.1f} K")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Demo: run this file to see the check on real model output.
#     python -m src.stress          (from the repo root, with the venv active)
# Nothing below runs when another file does `from src import stress`.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import contextlib  # to silence the property-fit printout
    import io          # an in-memory text buffer to send that printout to

    from src import config, properties, steady, transient

    # params: a COPY of the base case (geometry, flows, inlet temperatures,
    # metal properties, grid size N, solver tolerances), so the demo can never
    # change config.BASE_CASE itself.
    params = dict(config.BASE_CASE)

    # limits: the ALPEMA limits (30 K differences; rates, not used here).
    limits = config.ALPEMA_LIMITS

    # sigma_yield [Pa]: minimum yield strength of annealed 3003 aluminium, 35e6 Pa.
    sigma_yield = config.MATERIAL_LIMITS["sigma_y_3003_O"]

    # props_steady, props_startup: polynomial fits of methane's properties (cp,
    # mu, k, rho) for each stream. The fit range depends on the scenario: the
    # startup needs a wider temperature range because the core starts at its
    # idle temperature. build_surrogates prints its fit errors; we hide them.
    with contextlib.redirect_stdout(io.StringIO()):
        props_steady = properties.build_surrogates(params, "none")
        props_startup = properties.build_surrogates(params, "startup")

    # --- Case 1: steady base case ---
    # s: steady solution dict; s["x"] [m] positions, s["T_h"], s["T_c"],
    # s["T_w"] [K] the three temperature profiles, each shape (n,).
    s = steady.solve_steady(params, props_steady)
    steady_result = stress_check(s["x"], s["T_h"], s["T_c"], s["T_w"], params, limits, sigma_yield)
    print("=== Steady base case ===")
    print(summary_text(steady_result))

    # --- Case 2: startup from rest ---
    # t_response [s]: time for the whole exchanger to respond (math doc
    # Section 6.2); used to run long enough for the startup to settle.
    t_response = transient.time_scales(params, props_steady)["t_response"]
    # t_end [s]: end of the run = when the flows start opening
    # (startup_t_start) + how long they take to open fully (startup_t_flow) +
    # 15 response times, the same run length as scripts/plot_stress.py.
    t_end = params["startup_t_start"] + params["startup_t_flow"] + 15.0 * t_response
    # t_eval [s]: the 1201 saved times the solver reports, evenly spaced.
    t_eval = np.linspace(0.0, t_end, 1201)
    # r: transient result dict; r["t"] [s] saved times, r["x"] [m] cell
    # centres, r["T_h"], r["T_c"], r["T_w"] [K] histories, shape (n_t, N).
    r = transient.simulate(params, props_startup, "startup", t_end, t_eval=t_eval)
    startup_result = stress_check_transient(r, params, limits, sigma_yield)
    print(f"\n=== Startup from rest (N = {params['N']} cells, {t_end:.0f} s) ===")
    print(summary_text(startup_result))
    # Reading the startup peak at t = 0 s, x = L: that is the face where the
    # cold gas enters. Before any flow, the idle core sits at its idle
    # temperature while the cold inlet gas is much colder; the face value is
    # the difference the first cold gas to enter will see.
