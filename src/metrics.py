"""Thermal-fatigue and ALPEMA metrics, grid convergence, smooth max and thermal stress.

Math doc: Section 6.6 (ALPEMA metrics, Eq 39-40, hourly window), Section 6.3
(observed order and Richardson extrapolation, Eq 35), Section 7.6 (smooth max,
Eq 47, kept for the startup-ramp optimization stretch goal), Section 9.2
(thermal stress, Eq 53) and Eq 5 (stream-to-metal differences at steady state).

Proposal (Section 4, "Thermal Fatigue Metrics"): |T_h - T_w|, |T_w - T_c|,
|T_h - T_c|, max dT_w/dt and max dT_w/dx, then sigma ~ E alpha Delta T/(1 - nu).
All are covered here; max dT_w/dx (wall_gradient) is in the proposal but not yet
in the math doc (flagged).

Pure post-processing on numpy arrays: this module imports no other src module,
so it can be written and tested at any time. Callers pass the ALPEMA limits in
(config.ALPEMA_LIMITS); nothing is hard-coded here.

Units: rates are reported in K/min (= degC/min) and differences in K (= degC).
A temperature *difference* never gets a 273.15 offset.

Array conventions: functions that take temperature fields reduce over the
LAST axis (cells), so they work on a steady profile, shape (n,), and on a
transient history, shape (n_t, N), alike.

Imports: numpy, warnings and scipy.special.logsumexp.
"""

import warnings

import numpy as np
from scipy.special import logsumexp


def wall_rate(dTw_dt):
    """Largest metal temperature rate over all cells, in K/min.

    Eq 39: R_w(t) = max_i |dT_w,i/dt| * 60 s/min. A metal-temperature proxy
    for thermal-fatigue loading; ALPEMA itself measures fluid temperatures at
    the nozzles (flagged), which nozzle_rate covers.

    Parameters
    ----------
    dTw_dt : ndarray, shape (n_t, N)
        Wall temperature rates from the RHS at saved times [K/s].

    Returns
    -------
    R_w : ndarray, shape (n_t,)
        Peak wall rate at each time [K/min].
    """
    # 1. R_w = 60 * np.max(np.abs(dTw_dt), axis=-1).
    # Pitfall: the rates must come from the right-hand side at the saved times
    #    (transient.simulate does this), not from np.gradient of T_w (CLAUDE.md).
    # Check: a hand-made array with one entry of -0.1 K/s gives 6 K/min.
    # Peak |dT_w/dt| over the cells (last axis), converted from K/s to K/min.
    return 60.0 * np.max(np.abs(dTw_dt), axis=-1)


def nozzle_rate(dTh_dt, dTc_dt):
    """Largest outlet-stream temperature rate, in K/min.

    Eq 39: R_end(t) = max(|dT_h,N/dt|, |dT_c,1/dt|) * 60 s/min. The hot outlet
    is at x = L (last cell) and the cold outlet at x = 0 (first cell).

    Parameters
    ----------
    dTh_dt, dTc_dt : ndarray, shape (n_t, N)
        Hot and cold rates from the RHS at saved times [K/s].

    Returns
    -------
    R_end : ndarray, shape (n_t,)
        Peak outlet rate at each time [K/min].
    """
    # 1. hot outlet = dTh_dt[:, -1]; cold outlet = dTc_dt[:, 0].
    #    (Swapping these would report the INLET cells, which barely move.)
    # 2. R_end = 60 * np.maximum(np.abs(hot outlet), np.abs(cold outlet)).
    # Note: ALPEMA also monitors the inlet nozzles. The inlet rates are the
    #    imposed boundary conditions, so they are checked in check_alpema from
    #    the scenario's ramp rate rather than here.
    # Check: with dTh_dt[:, -1] = 0.05 K/s and everything else 0, R_end = 3 K/min.
    # The hot stream leaves at x = L (last cell); the cold stream leaves at
    # x = 0 (first cell).
    dT_hot_outlet = dTh_dt[..., -1]
    dT_cold_outlet = dTc_dt[..., 0]
    return 60.0 * np.maximum(np.abs(dT_hot_outlet), np.abs(dT_cold_outlet))


def stream_to_stream(T_h, T_c):
    """Largest hot-to-cold temperature difference over the cells.

    Eq 40: Delta T_ss = max_i |T_h,i - T_c,i|.

    Parameters
    ----------
    T_h, T_c : ndarray, shape (n,) or (n_t, N)
        Stream temperatures [K].

    Returns
    -------
    dT_ss : float or ndarray, shape () or (n_t,)
        Maximum stream-to-stream difference [K].
    """
    # 1. np.max(np.abs(T_h - T_c), axis=-1).
    # Check: on the constant-property analytical profile this equals
    #    Delta_0 (T_h_in - T_c_in) from Eq 14 (at the warm end if hot is C_min,
    #    at the cold end if cold is C_min).
    return np.max(np.abs(T_h - T_c), axis=-1)


def stream_to_metal(T_h, T_c, T_w):
    """Largest stream-to-metal temperature difference over the cells.

    Eq 40: Delta T_sm = max_i max(|T_h,i - T_w,i|, |T_w,i - T_c,i|). At steady
    state these follow Eq 5.

    Parameters
    ----------
    T_h, T_c, T_w : ndarray, shape (n,) or (n_t, N)
        Stream and wall temperatures [K].

    Returns
    -------
    dT_sm : float or ndarray, shape () or (n_t,)
        Maximum stream-to-metal difference [K].
    """
    # 1. diff = np.maximum(np.abs(T_h - T_w), np.abs(T_w - T_c)) (elementwise).
    #    (The proposal lists Delta T_hw and Delta T_cw separately; return them
    #    separately too if the plots need both.)
    # 2. Return np.max(diff, axis=-1).
    # Check: at steady state with a_h = a_c, each difference is half the
    #    stream-to-stream difference (Eq 5).
    # Worse of the hot-to-metal and metal-to-cold differences in each cell.
    diff = np.maximum(np.abs(T_h - T_w), np.abs(T_w - T_c))
    return np.max(diff, axis=-1)


def wall_gradient(T_w, x):
    """Largest axial wall temperature gradient, in K/m.

    Proposal, Section 4: max |dT_w/dx|, a measure of the in-plane thermal
    strain along the core. The math doc does not define it yet (flagged): add
    it to Section 6.6 with an equation number before citing it in the report.

    Parameters
    ----------
    T_w : ndarray, shape (n,) or (n_t, N)
        Wall temperatures (steady profile or transient history) [K].
    x : ndarray, shape (n,) or (N,)
        Positions of the T_w values: the steady grid or the cell centres [m].

    Returns
    -------
    dTw_dx_max : float or ndarray, shape () or (n_t,)
        Largest |dT_w/dx| over the core at each time [K/m].
    """
    # 1. dTw_dx = np.gradient(T_w, x, axis=-1): second-order central
    #    differences inside the core, one-sided at the two ends; x may be
    #    non-uniform.
    # 2. Return np.max(np.abs(dTw_dx), axis=-1).
    # Note: np.gradient is fine here because this is a SPATIAL derivative of
    #    the wall profile at one instant. The CLAUDE.md rule against np.gradient
    #    is about TIME derivatives (rates come from the right-hand side).
    # Pitfall: a coarse grid smooths steep wall fronts and under-predicts this
    #    peak, like R_w; include it in the N, 2N, 4N grid study.
    # Check: a linear profile T_w = 250 + 10 x returns 10 K/m exactly; for the
    #    constant-property balanced case (C_r = 1, linear profiles) it equals
    #    the analytical slope.
    dTw_dx = np.gradient(T_w, x, axis=-1)
    return np.max(np.abs(dTw_dx), axis=-1)


def hourly_change(t, T, window=3600.0):
    """Largest temperature change over the preceding window at each saved time.

    Section 6.6: the 60 degC/h limit is checked with
    max_i |T_i(t) - T_i(t - 3600 s)|.

    Parameters
    ----------
    t : ndarray, shape (n_t,)
        Saved times, increasing [s].
    T : ndarray, shape (n_t, N)
        Temperature history to check (T_w as in the math doc, or outlet
        temperatures, which is what ALPEMA measures) [K].
    window : float, optional
        Window length [s] (config.ALPEMA_LIMITS["hour_window"]).

    Returns
    -------
    dT_hour : ndarray, shape (n_t,)
        Largest change over the window ending at each t [K].
    """
    # 1. Lagged times t_lag = t - window. Before the run started the exchanger
    #    sat at its initial steady state, so wherever t_lag < t[0] use the
    #    initial temperatures T[0].
    # 2. Interpolate T at t_lag column by column: np.interp(t_lag, t, T[:, i])
    #    for each cell (np.interp clamps t_lag < t[0] to T[0], which is exactly
    #    the rule in step 1), or scipy.interpolate.interp1d(t, T, axis=0).
    # 3. dT_hour = np.max(np.abs(T - T_lag), axis=-1).
    # Pitfall: interpolation between saved times is only as good as the
    #    t_eval spacing; use a fine t_eval (or sol.sol dense output). Runs
    #    shorter than one window can only under-estimate the hourly change.
    # Check: a linear ramp of 2 K/min held for 2 h gives dT_hour = 120 K once
    #    t >= 1 h.
    t = np.asarray(t, dtype=float)
    T = np.asarray(T, dtype=float)
    t_lag = t - window

    # Temperature one window earlier, cell by cell. np.interp holds T[0] for
    # t_lag < t[0]: before the run the exchanger sat at its initial state.
    T_lag = np.empty_like(T)
    for i in range(T.shape[1]):
        T_lag[:, i] = np.interp(t_lag, t, T[:, i])

    return np.max(np.abs(T - T_lag), axis=-1)


def alpema_metrics(result, window=3600.0):
    """All ALPEMA comparison metrics for one transient run.

    Section 6.6, Eq 39-40 plus the hourly window.

    Parameters
    ----------
    result : dict
        Result dict from transient.simulate.
    window : float, optional
        Hourly window [s].

    Returns
    -------
    m : dict
        Time series, each shape (n_t,): "t", "R_w" [K/min], "R_end" [K/min]
        (None for the quasi-steady model), "dT_ss" [K], "dT_sm" [K],
        "dT_hour" [K] (from the outlet temperatures T_h,N and T_c,1, the
        nozzle quantities ALPEMA measures), "dT_hour_wall" [K] (the T_w
        version the math doc writes), "dTw_dx" [K/m] (proposal metric;
        ALPEMA sets no limit on it).
        Peaks: "max_R_w", "max_R_end" (None for the quasi-steady model),
        "max_dT_ss", "max_dT_sm", "max_dT_hour", "max_dT_hour_wall",
        "max_dTw_dx", and "t_max_R_w" (time of the peak wall rate) [s].
    """
    # 1. R_w = wall_rate(result["dTw_dt"]).
    # 2. R_end = nozzle_rate(result["dTh_dt"], result["dTc_dt"]) unless those
    #    are None (quasi-steady model).
    # 3. dT_ss = stream_to_stream(T_h, T_c); dT_sm = stream_to_metal(T_h, T_c, T_w);
    #    dTw_dx = wall_gradient(T_w, result["x"]).
    # 4. dT_hour from hourly_change on the outlet histories
    #    np.column_stack((T_h[:, -1], T_c[:, 0])), and a second version on T_w
    #    as the math doc writes it.
    # 5. Peaks with np.max; time of the peak R_w with np.argmax.
    # Check: on a scenario "none" run every rate is ~0 and dT_ss, dT_sm are
    #    constant in time.
    t = result["t"]
    T_h = result["T_h"]
    T_c = result["T_c"]
    T_w = result["T_w"]

    R_w = wall_rate(result["dTw_dt"])
    if result["dTh_dt"] is None or result["dTc_dt"] is None:
        R_end = None  # quasi-steady model: gas is algebraic, no gas rates
    else:
        R_end = nozzle_rate(result["dTh_dt"], result["dTc_dt"])

    dT_ss = stream_to_stream(T_h, T_c)
    dT_sm = stream_to_metal(T_h, T_c, T_w)
    dTw_dx = wall_gradient(T_w, result["x"])

    # Hourly change at the two outlet nozzles (hot leaves at x = L, cold at
    # x = 0), and the wall version of the math doc.
    T_outlets = np.column_stack((T_h[:, -1], T_c[:, 0]))
    dT_hour = hourly_change(t, T_outlets, window)
    dT_hour_wall = hourly_change(t, T_w, window)

    return {
        "t": t,
        "R_w": R_w,
        "R_end": R_end,
        "dT_ss": dT_ss,
        "dT_sm": dT_sm,
        "dT_hour": dT_hour,
        "dT_hour_wall": dT_hour_wall,
        "dTw_dx": dTw_dx,
        "max_R_w": float(np.max(R_w)),
        "max_R_end": None if R_end is None else float(np.max(R_end)),
        "max_dT_ss": float(np.max(dT_ss)),
        "max_dT_sm": float(np.max(dT_sm)),
        "max_dT_hour": float(np.max(dT_hour)),
        "max_dT_hour_wall": float(np.max(dT_hour_wall)),
        "max_dTw_dx": float(np.max(dTw_dx)),
        "t_max_R_w": float(t[np.argmax(R_w)]),
    }


def check_alpema(m, limits, event, ramp_rate=None):
    """Compare metrics with the ALPEMA Integrity Operating Window limits.

    Section 6.6 and the ALPEMA IOW table: rates < 1 degC/min for frequent
    events; < 5 degC/min and < 60 degC per hour for infrequent events
    (startup, shutdown, trips); stream-to-stream and stream-to-metal
    differences < ~30 degC.

    Parameters
    ----------
    m : dict
        From alpema_metrics.
    limits : dict
        config.ALPEMA_LIMITS (passed in; never hard-code the numbers).
    event : str
        "frequent" or "infrequent".
    ramp_rate : float, optional
        Imposed inlet ramp rate |ramp_dT/ramp_t| * 60 [K/min]. ALPEMA measures
        at the inlet nozzles too, so the imposed rate must itself meet the rate
        limit (the math doc's R_end covers outlets only; flagged).

    Returns
    -------
    checks : dict
        One entry per criterion, e.g. "rate_wall", "rate_outlet",
        "rate_inlet", "hourly", "dT_stream_stream", "dT_stream_metal", each a
        dict {"value": ..., "limit": ..., "ok": bool, "margin": limit - value}.
        Criteria that do not apply (hourly for "frequent", R_end for the
        quasi-steady model, rate_inlet without ramp_rate) are omitted.
    """
    # 1. Rate limit: limits["rate_frequent"] if event == "frequent", else
    #    limits["rate_infrequent"]; raise ValueError for any other event name.
    # 2. Compare max_R_w, max_R_end and ramp_rate (if given) with the rate limit.
    # 3. For "infrequent", compare max_dT_hour with
    #    limits["change_per_hour_infrequent"].
    # 4. Compare max_dT_ss with limits["dT_stream_stream"] and max_dT_sm with
    #    limits["dT_stream_metal"].
    # 5. ok = value < limit. Report the margin so near-misses are visible.
    # Pitfall: model values are in K and K/min; ALPEMA limits are in degC and
    #    degC/min. For differences and rates these are the same numbers;
    #    adding 273.15 anywhere here is a bug.
    # Check: hand-made metrics just under and just over each limit flip "ok".
    if event == "frequent":
        rate_limit = limits["rate_frequent"]
    elif event == "infrequent":
        rate_limit = limits["rate_infrequent"]
    else:
        raise ValueError(f"event must be 'frequent' or 'infrequent', not {event!r}")

    def criterion(value, limit):
        """One comparison: value and limit in K (or K/min), ok if below."""
        return {"value": value, "limit": limit, "ok": value < limit, "margin": limit - value}

    checks = {"rate_wall": criterion(m["max_R_w"], rate_limit)}
    if m["max_R_end"] is not None:
        checks["rate_outlet"] = criterion(m["max_R_end"], rate_limit)
    if ramp_rate is not None:
        checks["rate_inlet"] = criterion(abs(ramp_rate), rate_limit)
    if event == "infrequent":
        checks["hourly"] = criterion(m["max_dT_hour"], limits["change_per_hour_infrequent"])
    checks["dT_stream_stream"] = criterion(m["max_dT_ss"], limits["dT_stream_stream"])
    checks["dT_stream_metal"] = criterion(m["max_dT_sm"], limits["dT_stream_metal"])
    return checks


def observed_order(y_N, y_2N, y_4N):
    """Observed order of grid convergence and Richardson extrapolation.

    Eq 35:
      p_obs = ln(|y_N - y_2N| / |y_2N - y_4N|) / ln 2
      y*    = y_4N + (y_4N - y_2N) / (2^p_obs - 1)
    First-order upwinding should give p_obs ~ 1. Report y*, not y_N: a coarse
    grid smears fronts and under-predicts peak rates (the unsafe side of an
    ALPEMA check).

    Parameters
    ----------
    y_N, y_2N, y_4N : float
        The same output computed with N, 2N and 4N cells.

    Returns
    -------
    p_obs : float
        Observed order of accuracy [-].
    y_star : float
        Richardson-extrapolated value (same units as y).
    """
    # 1. d1 = y_N - y_2N; d2 = y_2N - y_4N.
    # 2. If d1 and d2 have opposite signs, convergence is not monotone and
    #    Eq 35 is meaningless: warn (warnings.warn) and return p_obs = nan,
    #    y_star = y_4N. Same if d2 == 0.
    # 3. p_obs = np.log(abs(d1)/abs(d2)) / np.log(2).
    # 4. y_star = y_4N + (y_4N - y_2N) / (2**p_obs - 1).
    # Pitfall: the time-integration error must be much smaller than the
    #    spatial error, or p_obs measures the solver tolerance instead. Use a
    #    tight ivp_rtol (~1e-8) for grid studies.
    # Check: y_N = 1 + 1/N style data (exactly first order) gives p_obs = 1 and
    #    y_star = 1; second-order data gives p_obs = 2.
    d1 = y_N - y_2N
    d2 = y_2N - y_4N

    # Eq 35 assumes monotone convergence: successive differences of one sign.
    if d2 == 0.0 or d1 * d2 <= 0.0:
        warnings.warn(
            f"Grid convergence is not monotone (y_N - y_2N = {d1:.3e}, "
            f"y_2N - y_4N = {d2:.3e}); p_obs is undefined, returning y_4N."
        )
        return np.nan, y_4N

    p_obs = np.log(abs(d1) / abs(d2)) / np.log(2.0)
    y_star = y_4N + (y_4N - y_2N) / (2.0**p_obs - 1.0)
    return float(p_obs), float(y_star)


def smax(z, omega, axis=None):
    """Smooth maximum (log-sum-exp).

    Eq 47: smax_omega(z) = (1/omega) ln sum_k exp(omega z_k), which satisfies
    max z <= smax <= max z + ln(K)/omega. Used in the startup-ramp optimization
    stretch goal (the constraints of Eq 52, where the location of the max jumps
    between candidate ramps and a plain max would make the optimizer stall),
    and optionally to smooth the laminar-floor kink in correlations.py. It is
    defined in math doc Section 7.6; the rest of that sensitivity section is
    out of scope per the proposal.

    Parameters
    ----------
    z : ndarray
        Values (any shape).
    omega : float
        Sharpness [1/units of z]; larger is closer to the true max.
    axis : int or None, optional
        Axis to reduce over (None = all entries).

    Returns
    -------
    s : float or ndarray
        Smooth maximum, same units as z.
    """
    # 1. s = scipy.special.logsumexp(omega * z, axis=axis) / omega.
    #    logsumexp subtracts the max internally, so exp never overflows;
    #    np.log(np.sum(np.exp(...))) would overflow for omega z > ~700.
    # Check: max(z) <= smax(z, omega) <= max(z) + ln(K)/omega for random z;
    #    large omega approaches max(z).
    return logsumexp(omega * np.asarray(z), axis=axis) / omega


def thermal_stress(dT, E, alpha_T, nu):
    """Thermal stress in a fully restrained plate (stretch goal).

    Eq 53: sigma_th ~ E alpha_T Delta T / (1 - nu), an upper bound because real
    cores are only partly restrained.

    Parameters
    ----------
    dT : float or ndarray
        Local temperature difference (stream-to-metal or stream-to-stream) [K].
    E : float
        Young's modulus [Pa] (params["E_m"]).
    alpha_T : float
        Thermal expansion coefficient [1/K].
    nu : float
        Poisson's ratio [-] (params["nu_m"]).

    Returns
    -------
    sigma : float or ndarray, same shape as dT
        Thermal stress [Pa].
    """
    # 1. sigma = E alpha_T dT / (1 - nu).
    # Check: E = 70 GPa, alpha_T = 23e-6 1/K, nu = 0.33 gives ~2.4 MPa per kelvin
    #    (math doc Section 9.2).
    return E * alpha_T * dT / (1.0 - nu)
