"""Verification of the transient model (CLAUDE.md "Verification"; math doc Sections 5.4, 6.3).

Each test is skipped until the functions it needs exist. To activate one,
delete its pytest.skip line and turn the numbered comments into code. Never
loosen a tolerance to make a test pass; report the failure and its likely
cause instead (CLAUDE.md).
"""

import pytest

from src import (  # noqa: F401  (used once implemented)
    config,
    correlations,
    metrics,
    properties,
    sensitivity,
    steady,
    transient,
)


def test_transient_stays_at_steady_state():
    """A transient started from steady state stays there.

    CLAUDE.md check 3 (first half). Start from the steady state of the
    DISCRETE model (transient.discrete_steady_state), not from the BVP
    solution. With first-order upwinding, the BVP profile sampled at the cell
    centres leaves the two inlet cells with dT/dt ~ u T'/2, which is O(1) and
    does not shrink with N (flagged against math doc Section 6.1), so a
    BVP start would drift no matter how fine the grid.
    """
    pytest.skip("not implemented yet")
    # 1. params = dict(config.BASE_CASE) (fail clearly if TODOs are unfilled)
    #    with a small grid for speed (N = 50) and tight integration
    #    (ivp_rtol = ivp_atol = 1e-8).
    #    props = properties.build_surrogates(params, "none").
    # 2. y0 = transient.discrete_steady_state(params, props, "none").
    # 3. Residual: geom = correlations.exchanger_geometry(params);
    #    assert max |transient.transient_rhs(0.0, y0, params, props, geom, "none")|
    #    < 1e-8 K/s.
    # 4. Hold: t_end = 5 * transient.time_scales(params, props)["t_response"];
    #    r = transient.simulate(params, props, "none", t_end, y0=y0).
    # 5. assert the largest drift max |T(t) - T(0)| over all times, cells and
    #    the three fields is < 1e-6 K, and max metrics.wall_rate(r["dTw_dt"])
    #    < 1e-4 K/min.
    # 6. (Diagnostic, no assert) The rates at transient.steady_to_cells(...)
    #    are largest in hot cell 1 and cold cell N. Print them to show why
    #    the BVP is not a valid initial state.


def test_transient_relaxes_to_new_steady_state():
    """After a disturbance the transient relaxes to the new steady solution.

    CLAUDE.md check 3 (second half) / math doc Section 5.4 item 4. Apply a
    quick, small hot-inlet ramp, wait until the solution stops changing, and
    compare with the steady state at the new inlet temperature: tightly
    against the discrete steady state, and against the continuous BVP through
    first-order convergence in N.
    """
    pytest.skip("not implemented yet")
    # 1. params = dict(config.BASE_CASE) with a short ramp: ramp_dT = 5 K,
    #    ramp_t = 10 s, ramp_t_start = 5 s, ramp_sharpness = 20; N = 50;
    #    ivp_rtol = ivp_atol = 1e-8. props = properties.build_surrogates(params,
    #    "feed_ramp") so the ramped inlet is inside the fitted range.
    # 2. Run r = transient.simulate(params, props, "feed_ramp", t_end) and stop
    #    on a convergence criterion, not a fixed time: keep extending t_end
    #    (or use a terminal solve_ivp event) until the max |dy/dt| at the
    #    final time is < 1e-6 K/s.
    # 3. New steady target: p_new = sensitivity.with_parameter(params, "T_h_in",
    #    params["T_h_in"] + params["ramp_dT"]);
    #    y_new = transient.discrete_steady_state(p_new, props, "none").
    #    assert max |y_final - y_new| < 1e-5 K.
    # 4. Against the BVP: for N in (25, 50, 100) compute
    #    err_N = max |y_final(N) - transient.steady_to_cells(
    #                 steady.solve_steady(p_new, props), N)|
    #    over the interior cells. First-order upwinding means
    #    err_N / err_2N should be ~2: assert 1.6 < err_N/err_2N < 2.5 for both
    #    pairs. (A fixed tolerance in kelvin would just test the chosen N.)


def test_grid_convergence_of_transient_metric():
    """Transient outputs converge at the expected first order in N.

    CLAUDE.md check 4 / math doc Section 6.3, Eq 35: report N, 2N, 4N
    results, the observed order p_obs and the Richardson-extrapolated y*
    for transient metrics. Upwinding smears fronts, so a coarse grid
    under-predicts peak rates; grid convergence is required, not optional.
    """
    pytest.skip("not implemented yet")
    # 1. params = dict(config.BASE_CASE) with the base feed ramp,
    #    ivp_rtol = ivp_atol = 1e-8 (time error must be much smaller than
    #    space error, or p_obs measures the solver tolerance), and one fixed,
    #    dense t_eval shared by all runs.
    #    props = properties.build_surrogates(params, "feed_ramp").
    # 2. For N in (N0, 2 N0, 4 N0) with N0 = 25: r_N = transient.simulate(
    #    sensitivity.with_parameter(params, "N", N), props, "feed_ramp", t_end,
    #    t_eval=t_eval).
    # 3. Smooth metric: hot outlet temperature T_h[:, -1] at a fixed time in
    #    the middle of the ramp (np.interp on t). p_obs, y_star =
    #    metrics.observed_order(y_N, y_2N, y_4N);
    #    assert 0.7 < p_obs < 1.3 and the convergence is monotone.
    # 4. Peak metric: max of metrics.wall_rate(r_N["dTw_dt"]) for each N.
    #    Report p_obs and y_star (print, or record_property) but do NOT assert
    #    on them: the location of the max can jump between grids, so this
    #    metric may not be in the asymptotic range yet.
    # 5. Report y_star, not y_4N, in the final results.
