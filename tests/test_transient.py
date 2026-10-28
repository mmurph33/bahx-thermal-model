"""Verification of the transient model (CLAUDE.md "Verification"; math doc Sections 5.4, 6.3).

The first three tests are the verification checks from CLAUDE.md and the math
doc. The rest turn the "# Check:" comments in src/transient.py into unit
tests. Never loosen a tolerance to make a test pass; report the failure and
its likely cause instead (CLAUDE.md).
"""

import math

import numpy as np
import pytest

from src import config, correlations, metrics, properties, steady, transient


def base_case_params(**changes):
    """config.BASE_CASE copy with changes, failing clearly if a TODO is unfilled.

    Parameters
    ----------
    **changes
        Values to override in the copy.

    Returns
    -------
    params : dict
        Base-case params with the changes applied.
    """
    params = dict(config.BASE_CASE)
    missing = [key for key, value in params.items() if value is None]
    if missing:
        pytest.fail(f"config.BASE_CASE values still None: {missing}")
    params.update(changes)
    return params


@pytest.fixture(scope="module")
def props_none():
    """Base-case surrogates for scenario "none", built once for this module."""
    return properties.build_surrogates(base_case_params(), "none")


def test_transient_stays_at_steady_state(props_none):
    """A transient started from steady state stays there.

    CLAUDE.md check 3 (first half). Start from the steady state of the
    DISCRETE model (transient.discrete_steady_state), not from the BVP
    solution. With first-order upwinding, the BVP profile sampled at the cell
    centres leaves the two inlet cells with dT/dt ~ u T'/2, which is O(1) and
    does not shrink with N (flagged against math doc Section 6.1), so a
    BVP start would drift no matter how fine the grid.
    """
    # 1. params = dict(config.BASE_CASE) (fail clearly if TODOs are unfilled)
    #    with a small grid for speed (N = 50) and tight integration
    #    (ivp_rtol = ivp_atol = 1e-8).
    #    props = properties.build_surrogates(params, "none").
    params = base_case_params(N=50, ivp_rtol=1e-8, ivp_atol=1e-8)
    props = props_none

    # 2. y0 = transient.discrete_steady_state(params, props, "none").
    y0 = transient.discrete_steady_state(params, props, "none")

    # 3. Residual: geom = correlations.exchanger_geometry(params);
    #    assert max |transient.transient_rhs(0.0, y0, params, props, geom, "none")|
    #    < 1e-8 K/s.
    geom = correlations.exchanger_geometry(params)
    assert np.max(np.abs(transient.transient_rhs(0.0, y0, params, props, geom, "none"))) < 1e-8

    # 4. Hold: t_end = 5 * transient.time_scales(params, props)["t_response"];
    #    r = transient.simulate(params, props, "none", t_end, y0=y0).
    t_end = 5.0 * transient.time_scales(params, props)["t_response"]
    r = transient.simulate(params, props, "none", t_end, y0=y0)

    # 5. assert the largest drift max |T(t) - T(0)| over all times, cells and
    #    the three fields is < 1e-6 K, and max metrics.wall_rate(r["dTw_dt"])
    #    < 1e-4 K/min.
    drift = max(np.max(np.abs(r[field] - r[field][0])) for field in ("T_h", "T_c", "T_w"))
    assert drift < 1e-6
    assert np.max(metrics.wall_rate(r["dTw_dt"])) < 1e-4

    # 6. (Diagnostic, no assert) The rates at transient.steady_to_cells(...)
    #    are largest in hot cell 1 and cold cell N. Print them to show why
    #    the BVP is not a valid initial state.
    y_bvp = transient.steady_to_cells(steady.solve_steady(params, props), params["N"])
    dTh_dt, dTc_dt, _ = transient.unpack_state(transient.transient_rhs(0.0, y_bvp, params, props, geom, "none"))
    print(
        f"\nBVP start: |dT_h/dt| hot cell 1 = {abs(dTh_dt[0]):.2f} K/s "
        f"(interior max {np.max(np.abs(dTh_dt[1:])):.3f}); "
        f"|dT_c/dt| cold cell N = {abs(dTc_dt[-1]):.2f} K/s "
        f"(interior max {np.max(np.abs(dTc_dt[:-1])):.3f})"
    )


def test_transient_relaxes_to_new_steady_state():
    """After a disturbance the transient relaxes to the new steady solution.

    CLAUDE.md check 3 (second half) / math doc Section 5.4 item 4. Apply a
    quick, small hot-inlet ramp, wait until the solution stops changing, and
    compare with the steady state at the new inlet temperature: tightly
    against the discrete steady state, and against the continuous BVP through
    first-order convergence in N.
    """
    # 1. params = dict(config.BASE_CASE) with a short ramp: ramp_dT = 5 K,
    #    ramp_t = 10 s, ramp_t_start = 5 s, ramp_sharpness = 20; N = 50;
    #    ivp_rtol = ivp_atol = 1e-8. props = properties.build_surrogates(params,
    #    "feed_ramp") so the ramped inlet is inside the fitted range.
    params = base_case_params(
        ramp_dT=5.0, ramp_t=10.0, ramp_t_start=5.0, ramp_sharpness=20.0,
        N=50, ivp_rtol=1e-8, ivp_atol=1e-8,
    )
    props = properties.build_surrogates(params, "feed_ramp")
    p_new = dict(params, T_h_in=params["T_h_in"] + params["ramp_dT"])
    s_new = steady.solve_steady(dict(p_new, bvp_tol=1e-8), props)

    def relax(N):
        """Run the ramp on N cells until the state stops changing; return it."""
        p = dict(params, N=N)
        geom = correlations.exchanger_geometry(p)
        # 2. Run r = transient.simulate(params, props, "feed_ramp", t_end) and
        #    stop on a convergence criterion, not a fixed time: keep extending
        #    t_end until the max |dy/dt| at the final time is small. The stub
        #    said 1e-6 K/s, but a residual rate leaves the state about
        #    (rate x slowest time constant ~ 20 s) from steady, which at
        #    1e-6 K/s is ~2e-5 K, the size of the step-3 assertion itself. So
        #    the criterion is tightened to 1e-8 K/s (stricter, not looser).
        t_end = 10.0 * transient.time_scales(p, props)["t_response"]
        for _ in range(6):
            r = transient.simulate(p, props, "feed_ramp", t_end, t_eval=np.linspace(0.0, t_end, 201))
            y_final = transient.pack_state(r["T_h"][-1], r["T_c"][-1], r["T_w"][-1])
            rate = np.max(np.abs(transient.transient_rhs(t_end, y_final, p, props, geom, "feed_ramp")))
            if rate < 1e-8:
                return y_final
            t_end *= 2.0
        pytest.fail(f"N = {N}: still changing at {rate:.1e} K/s after t = {t_end / 2:.0f} s")

    # 3. New steady target: p_new = dict(params,
    #    T_h_in=params["T_h_in"] + params["ramp_dT"]);
    #    y_new = transient.discrete_steady_state(p_new, props, "none").
    #    assert max |y_final - y_new| < 1e-5 K.
    y_final = relax(params["N"])
    y_new = transient.discrete_steady_state(p_new, props, "none")
    assert np.max(np.abs(y_final - y_new)) < 1e-5

    # 4. Against the BVP: for N in (25, 50, 100) compute
    #    err_N = max |y_final(N) - transient.steady_to_cells(
    #                 steady.solve_steady(p_new, props), N)|
    #    over the cells. First-order upwinding means err_N / err_2N should be
    #    ~2: assert 1.6 < err_N/err_2N < 2.5 for both pairs. (A fixed
    #    tolerance in kelvin would just test the chosen N.)
    err = {}
    for N in (25, 50, 100):
        y_N = y_final if N == params["N"] else relax(N)
        err[N] = np.max(np.abs(y_N - transient.steady_to_cells(s_new, N)))
    assert 1.6 < err[25] / err[50] < 2.5
    assert 1.6 < err[50] / err[100] < 2.5


def test_grid_convergence_of_transient_metric(record_property):
    """Transient outputs converge at the expected first order in N.

    CLAUDE.md check 4 / math doc Section 6.3, Eq 35: report N, 2N, 4N
    results, the observed order p_obs and the Richardson-extrapolated y*
    for transient metrics. Upwinding smears fronts, so a coarse grid
    under-predicts peak rates; grid convergence is required, not optional.
    """
    # 1. params = dict(config.BASE_CASE) with the base feed ramp,
    #    ivp_rtol = ivp_atol = 1e-8 (time error must be much smaller than
    #    space error, or p_obs measures the solver tolerance), and one fixed,
    #    dense t_eval shared by all runs.
    #    props = properties.build_surrogates(params, "feed_ramp").
    params = base_case_params(ivp_rtol=1e-8, ivp_atol=1e-8)
    props = properties.build_surrogates(params, "feed_ramp")
    t_end = params["ramp_t_start"] + params["ramp_t"] + 480.0
    t_eval = np.linspace(0.0, t_end, 1801)
    t_mid = params["ramp_t_start"] + params["ramp_t"] / 2.0

    # 2. For N in (N0, 2 N0, 4 N0) with N0 = 25: r_N = transient.simulate(
    #    dict(params, N=N), props, "feed_ramp", t_end, t_eval=t_eval).
    outlet = []
    peak_R_w = []
    for N in (25, 50, 100):
        r_N = transient.simulate(dict(params, N=N), props, "feed_ramp", t_end, t_eval=t_eval)
        outlet.append(float(np.interp(t_mid, r_N["t"], r_N["T_h"][:, -1])))
        peak_R_w.append(float(np.max(metrics.wall_rate(r_N["dTw_dt"]))))

    # 3. Smooth metric: hot outlet temperature T_h[:, -1] at a fixed time in
    #    the middle of the ramp (np.interp on t). p_obs, y_star =
    #    metrics.observed_order(y_N, y_2N, y_4N);
    #    assert 0.7 < p_obs < 1.3 and the convergence is monotone.
    assert (outlet[0] - outlet[1]) * (outlet[1] - outlet[2]) > 0.0
    p_obs, y_star = metrics.observed_order(*outlet)
    assert 0.7 < p_obs < 1.3

    # 4. Peak metric: max of metrics.wall_rate(r_N["dTw_dt"]) for each N.
    #    Report p_obs and y_star (print, or record_property) but do NOT assert
    #    on them: the location of the max can jump between grids, so this
    #    metric may not be in the asymptotic range yet.
    p_obs_R, y_star_R = metrics.observed_order(*peak_R_w)

    # 5. Report y_star, not y_4N, in the final results.
    record_property("T_h_out_mid_ramp_N_2N_4N", outlet)
    record_property("T_h_out_mid_ramp_p_obs_y_star", (p_obs, y_star))
    record_property("peak_R_w_N_2N_4N", peak_R_w)
    record_property("peak_R_w_p_obs_y_star", (p_obs_R, y_star_R))
    print(
        f"\nT_h,out at mid-ramp: {outlet} K, p_obs = {p_obs:.3f}, y* = {y_star:.4f} K"
        f"\npeak R_w: {peak_R_w} K/min, p_obs = {p_obs_R:.3f}, y* = {y_star_R:.4f} K/min"
    )


# ---------------------------------------------------------------------------
# Unit checks from the "# Check:" comments in src/transient.py
# ---------------------------------------------------------------------------

def test_cell_centers_and_state_packing():
    """Cell centres sit at (i - 1/2) dx; pack/unpack round-trips exactly."""
    x, dx = transient.cell_centers(0.7, 10)
    assert len(x) == 10 and dx == pytest.approx(0.07)
    assert x[0] == pytest.approx(dx / 2) and x[-1] == pytest.approx(0.7 - dx / 2)

    rng = np.random.default_rng(1)
    a, b, c = rng.random(4), rng.random(4), rng.random(4)
    y = transient.pack_state(a, b, c)
    assert np.array_equal(y[:3], [a[0], b[0], c[0]])
    for original, unpacked in zip((a, b, c), transient.unpack_state(y)):
        assert np.array_equal(original, unpacked)


def test_smooth_ramp_properties():
    """psi goes 0 -> 1, is 1/2 at the midpoint, monotone, and never steeper than 1."""
    omega = 20.0
    assert transient.smooth_ramp(-10.0, omega) < 1e-12
    assert transient.smooth_ramp(10.0, omega) == pytest.approx(1.0, abs=1e-12)
    assert transient.smooth_ramp(0.5, omega) == pytest.approx(0.5, abs=1e-12)
    s = np.linspace(-1.0, 2.0, 30001)
    slope = np.diff(transient.smooth_ramp(s, omega)) / np.diff(s)
    assert np.all(slope >= 0.0)
    assert np.max(slope) == pytest.approx(np.tanh(omega / 4.0), rel=1e-6)


def test_boundary_conditions_scenarios():
    """Base values at t = 0; ramp and trip reach their end values; bad names raise."""
    params = base_case_params()
    for scenario in ("none", "cold_trip"):
        T_h_in, T_c_in, mdot_h, mdot_c = transient.boundary_conditions(0.0, params, scenario)
        assert (T_h_in, T_c_in, mdot_h, mdot_c) == (
            params["T_h_in"], params["T_c_in"], params["mdot_h"], params["mdot_c"]
        )

    # The smoothed ramp is not exactly zero at t = 0 (smooth_ramp pitfall).
    # Its offset is ramp_dT psi(s0) with s0 = -ramp_t_start/ramp_t, written
    # here independently with math.log1p; it is 1.7e-4 K for the base case.
    # The config rule ramp_t_start >= 5 ramp_t/omega guarantees
    # psi(s0) <= softplus(-5)/omega.
    omega = params["ramp_sharpness"]
    s0 = -params["ramp_t_start"] / params["ramp_t"]
    psi0 = (math.log1p(math.exp(omega * s0)) - math.log1p(math.exp(omega * (s0 - 1.0)))) / omega
    T_h_in, T_c_in, mdot_h, mdot_c = transient.boundary_conditions(0.0, params, "feed_ramp")
    assert T_h_in - params["T_h_in"] == pytest.approx(params["ramp_dT"] * psi0, rel=1e-9)
    assert psi0 <= math.log1p(math.exp(-5.0)) / omega
    assert (T_c_in, mdot_h, mdot_c) == (params["T_c_in"], params["mdot_h"], params["mdot_c"])

    t_late = params["ramp_t_start"] + 10.0 * params["ramp_t"]
    T_h_in, _, _, _ = transient.boundary_conditions(t_late, params, "feed_ramp")
    assert T_h_in == pytest.approx(params["T_h_in"] + params["ramp_dT"], abs=1e-9)

    t_late = params["trip_t_start"] + 50.0 * params["trip_tau_v"]
    _, _, _, mdot_c = transient.boundary_conditions(t_late, params, "cold_trip")
    assert mdot_c == pytest.approx(params["trip_phi"] * params["mdot_c"], rel=1e-9)

    t = np.linspace(0.0, 100.0, 5)
    assert transient.boundary_conditions(t, params, "cold_trip")[3].shape == (5,)
    with pytest.raises(ValueError):
        transient.boundary_conditions(0.0, params, "ramp")


@pytest.mark.parametrize("axial_conduction", [False, True])
def test_transient_rhs_energy_identity_and_sparsity(props_none, axial_conduction):
    """Uniform state is at rest; storage equals boundary enthalpy flux; pattern covers the Jacobian.

    The energy identity catches a wrong cold-stream sign without integrating:
    the exchange terms cancel pairwise and the ghost cells are adiabatic, so
    sum(M dT/dt) = mdot_h [hhat_h(T_h_in) - hhat_h(T_h,N)] + mdot_c [hhat_c(T_c_in) - hhat_c(T_c,1)].
    """
    props = props_none
    N = 7
    params = base_case_params(N=N, axial_conduction=axial_conduction)
    geom = correlations.exchanger_geometry(params)
    dx = params["L"] / N

    # (a) Uniform state at a common inlet temperature: nothing moves.
    p_uniform = dict(params, T_h_in=270.0, T_c_in=270.0)
    assert np.all(transient.transient_rhs(0.0, np.full(3 * N, 270.0), p_uniform, props, geom, "none") == 0.0)

    # (b) Energy identity at a random in-range state.
    y = np.random.default_rng(0).uniform(240.0, 295.0, 3 * N)
    dTh_dt, dTc_dt, dTw_dt = transient.unpack_state(transient.transient_rhs(0.0, y, params, props, geom, "none"))
    T_h, T_c, _ = transient.unpack_state(y)
    M_h = properties.evaluate(props["h"], "rho", T_h) * geom["h"]["A_ff"] * properties.evaluate(props["h"], "cp", T_h) * dx
    M_c = properties.evaluate(props["c"], "rho", T_c) * geom["c"]["A_ff"] * properties.evaluate(props["c"], "cp", T_c) * dx
    storage = np.sum(M_h * dTh_dt + M_c * dTc_dt + geom["C_m"] * dx * dTw_dt)
    boundary_flux = (
        params["mdot_h"] * (properties.enthalpy(props["h"], params["T_h_in"]) - properties.enthalpy(props["h"], T_h[-1]))
        + params["mdot_c"] * (properties.enthalpy(props["c"], params["T_c_in"]) - properties.enthalpy(props["c"], T_c[0]))
    )
    assert storage == pytest.approx(float(boundary_flux), rel=1e-12)

    # (c) Every nonzero of a central-difference Jacobian lies in the pattern.
    J = np.zeros((3 * N, 3 * N))
    for j in range(3 * N):
        e = np.zeros(3 * N)
        e[j] = 1e-4
        J[:, j] = (
            transient.transient_rhs(0.0, y + e, params, props, geom, "none")
            - transient.transient_rhs(0.0, y - e, params, props, geom, "none")
        ) / 2e-4
    pattern = transient.jacobian_sparsity(N, axial_conduction).toarray()
    assert not np.any((np.abs(J) > 0.0) & (pattern == 0))


def test_quasi_steady_gas_limits(props_none):
    """Uniform wall at T_h_in leaves T_h at T_h_in; huge conductance pins gas to the wall."""
    params = base_case_params(N=10)
    geom = correlations.exchanger_geometry(params)
    T_w = np.full(10, params["T_h_in"])
    T_h, _ = transient.quasi_steady_gas(T_w, 0.0, params, props_none, geom, "none")
    assert np.allclose(T_h, params["T_h_in"], atol=1e-12)

    # Very low flow means NTU_cell >> 1, so each gas cell sits at its wall.
    p_slow = dict(params, mdot_h=1e-7, mdot_c=1e-7)
    T_w = np.linspace(295.0, 240.0, 10)
    T_h, T_c = transient.quasi_steady_gas(T_w, 0.0, p_slow, props_none, geom, "none")
    assert np.max(np.abs(T_h - T_w)) < 1e-2 and np.max(np.abs(T_c - T_w)) < 1e-2
