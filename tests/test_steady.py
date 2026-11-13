"""Verification of the steady model (CLAUDE.md "Verification"; math doc Section 5.4).

The first three tests are the verification checks from CLAUDE.md and the math
doc. The rest turn the "# Check:" comments in src/steady.py into unit tests.
Never loosen a tolerance to make a test pass; report the failure and its
likely cause instead (CLAUDE.md).
"""

import numpy as np
import pytest

from src import config, correlations, properties, steady

# Fixed fluid properties for the constant-property tests [Pa s], [W/(m K)],
# [kg/m^3]. Their values only set the (constant) conductances.
MU = 1.2e-5
K_GAS = 0.035
RHO = 30.0


def synthetic_params():
    """Explicit test inputs, independent of the team's base-case choices.

    Returns
    -------
    p : dict
        A full params dict (BASE_CASE keys) with every value the steady solver
        needs set here. Flows and L are set by each test.
    """
    p = dict(config.BASE_CASE)
    p.update(
        T_h_in=300.0, T_c_in=235.0, W=1.0,
        s_h=1.525e-3, b_h=6.25e-3, t_fin_h=1.02e-4, l_h=3.175e-3,
        s_c=1.525e-3, b_c=6.25e-3, t_fin_c=1.02e-4, l_c=3.175e-3,
        t_sheet=1.0e-3, rho_m=2700.0, c_m=880.0, k_f=180.0,
        n_out=501, bvp_max_nodes=10000,
        continuation_lams=(0.0, 0.25, 0.5, 0.75, 1.0),
        bvp_tol=1e-8,
    )
    return p


def base_case_params():
    """config.BASE_CASE copy, failing clearly if any TODO value is unfilled.

    Returns
    -------
    params : dict
        Base-case params with bvp_tol tightened to 1e-8.
    """
    params = dict(config.BASE_CASE)
    missing = [key for key, value in params.items() if value is None]
    if missing:
        pytest.fail(f"config.BASE_CASE values still None: {missing}")
    params["bvp_tol"] = 1e-8
    return params


@pytest.fixture(scope="module")
def base_props():
    """Base-case CoolProp surrogates, built once for this module."""
    return properties.build_surrogates(base_case_params())


@pytest.mark.parametrize("case", ["hot_is_cmin", "cold_is_cmin", "balanced"])
def test_constant_property_matches_epsilon_ntu(case):
    """Constant-property BVP solution matches the analytical epsilon-NTU solution.

    CLAUDE.md check 1 / math doc Section 5.4 item 1: with constant properties
    the numerical effectiveness and profiles match Eq 13 and Eq 11 (here the
    unified form in steady.analytical_profiles) to solver tolerance,
    including C_r = 1. Run for hot = C_min, cold = C_min and balanced flow,
    since the analytical formulas differ between these cases.
    """
    # 1. Synthetic parameters, independent of the team's base-case choices:
    #    p = dict(config.BASE_CASE) and fill every key the solver needs with
    #    explicit test values written in this test (e.g. T_h_in = 300,
    #    T_c_in = 235, W = 1, the suggested fin geometry, k_f = 180,
    #    n_out = 501, bvp_max_nodes = 10000,
    #    continuation_lams = (0, .25, .5, .75, 1)), and set bvp_tol = 1e-8:
    #    solve_bvp controls a relative residual, not the solution error, so a
    #    tight tolerance is needed to check eps to 1e-6. (Tightening the solver
    #    is not loosening the test.)
    p = synthetic_params()
    T_h_in, T_c_in = p["T_h_in"], p["T_c_in"]

    # 2. Constant surrogates (no CoolProp): props = {"h":
    #    properties.constant_surrogate(cp_h, mu, k, rho, 200, 320), "c": ...}
    #    with one fixed set of mu, k, rho and cp_h. Pick cp_c and the flows to
    #    set C_r:
    #      hot_is_cmin:  C_r = 0.5 with C_h < C_c
    #      cold_is_cmin: C_r = 0.5 with C_c < C_h
    #      balanced:     cp_c = cp_h * mdot_h / mdot_c, so C_h == C_c exactly
    #                    (C_r = 1 to round-off, which exercises the k -> 0
    #                    branch of analytical_profiles).
    cp_h = 2500.0
    if case == "hot_is_cmin":
        p["mdot_h"], p["mdot_c"], cp_c = 0.1, 0.2, 2500.0
    elif case == "cold_is_cmin":
        p["mdot_h"], p["mdot_c"], cp_c = 0.2, 0.1, 2500.0
    else:
        p["mdot_h"], p["mdot_c"] = 0.1, 0.125
        cp_c = cp_h * p["mdot_h"] / p["mdot_c"]
    props = {
        "h": properties.constant_surrogate(cp_h, MU, K_GAS, RHO, 200.0, 320.0),
        "c": properties.constant_surrogate(cp_c, MU, K_GAS, RHO, 200.0, 320.0),
    }
    C_h = p["mdot_h"] * cp_h
    C_c = p["mdot_c"] * cp_c
    C_min, C_max = min(C_h, C_c), max(C_h, C_c)
    C_r = C_min / C_max
    if case == "balanced":
        assert C_h == C_c

    # 3. Constant conductances: geom = correlations.exchanger_geometry(p);
    #    a_h, a_c = correlations.conductances(...) at any in-range T;
    #    U' = correlations.overall_conductance(a_h, a_c). Choose L so the test
    #    is demanding: NTU (1 - C_r) ~ 3-5 (e.g. NTU = 6 for C_r = 0.5) and
    #    NTU ~ 4 for balanced flow; L = NTU * C_min / U'.
    geom = correlations.exchanger_geometry(p)
    a_h, a_c = correlations.conductances(270.0, 270.0, p["mdot_h"], p["mdot_c"], p, props, geom)
    U_prime = float(correlations.overall_conductance(a_h, a_c))
    NTU = 4.0 if case == "balanced" else 6.0
    p["L"] = NTU * C_min / U_prime

    # 4. Numerical solution: s = steady.solve_steady(p, props).
    s = steady.solve_steady(p, props)

    # 5. Analytical solution on the same grid:
    #    T_h_ex, T_c_ex = steady.analytical_profiles(s["x"], L, T_h_in, T_c_in,
    #    C_h, C_c, U').
    T_h_ex, T_c_ex = steady.analytical_profiles(s["x"], p["L"], T_h_in, T_c_in, C_h, C_c, U_prime)

    # 6. Compare in dimensionless form, theta = (T - T_c_in)/(T_h_in - T_c_in):
    #    assert max |theta_h - theta_h_ex| < 1e-5 and the same for theta_c.
    dT_in = T_h_in - T_c_in
    theta_h = (s["T_h"] - T_c_in) / dT_in
    theta_c = (s["T_c"] - T_c_in) / dT_in
    assert np.max(np.abs(theta_h - (T_h_ex - T_c_in) / dT_in)) < 1e-5
    assert np.max(np.abs(theta_c - (T_c_ex - T_c_in) / dT_in)) < 1e-5

    # 7. Effectiveness: eps_num = C_h (T_h_in - s["T_h"][-1]) / (C_min (T_h_in - T_c_in));
    #    assert abs(eps_num - steady.effectiveness(NTU, C_r)) < 1e-6.
    eps_num = C_h * (T_h_in - s["T_h"][-1]) / (C_min * dT_in)
    assert abs(eps_num - steady.effectiveness(NTU, C_r)) < 1e-6

    # 8. Location of the largest difference (flagged Eq 14 wording): assert
    #    np.argmax(T_h - T_c) is at x = 0 for hot_is_cmin and at x = L for
    #    cold_is_cmin, and the value equals Delta_0 (T_h_in - T_c_in) to 1e-5
    #    in theta. (Delta_0 here is Eq 14's (1 - C_r)/(1 - C_r E), which is
    #    1/(1 + NTU) for balanced flow, where the difference is constant.)
    gap = theta_h - theta_c
    if case == "hot_is_cmin":
        assert np.argmax(gap) == 0
    elif case == "cold_is_cmin":
        assert np.argmax(gap) == gap.size - 1
    if case == "balanced":
        Delta_0 = 1.0 / (1.0 + NTU)
    else:
        E = np.exp(-NTU * (1.0 - C_r))
        Delta_0 = (1.0 - C_r) / (1.0 - C_r * E)
    assert abs(np.max(gap) - Delta_0) < 1e-5


def test_energy_closure(base_props):
    """Hot duty, cold duty and the integrated heat flow agree.

    CLAUDE.md check 2 / math doc Section 5.4 item 2, Eq 16:
    Q_h = Q_c = integral of U'(T_h - T_c) dx, for the VARIABLE-property base
    case (with constant properties the check passes trivially).
    """
    # 1. params = dict(config.BASE_CASE). If any base-case TODO is still None,
    #    fail with a message listing the missing keys (don't skip silently).
    #    Set bvp_tol = 1e-8.
    params = base_case_params()

    # 2. props = properties.build_surrogates(params). The enthalpy surrogate
    #    must be the integral of the cp surrogate, otherwise closure is stuck
    #    at the fit error (~1e-3).
    props = base_props

    # 3. s = steady.solve_steady(params, props);
    #    duty = steady.energy_balance(s, params, props).
    s = steady.solve_steady(params, props)
    duty = steady.energy_balance(s, params, props)

    # 4. assert duty["closure_hc"] < 1e-4    (|Q_h - Q_c| / Q_h)
    #    assert duty["closure_int"] < 1e-4   (|Q_h - Q_int| / Q_h)
    assert duty["closure_hc"] < 1e-4
    assert duty["closure_int"] < 1e-4

    # 5. Sanity: 0 < Q_h < C_min (T_h_in - T_c_in) using the largest cp on
    #    each side (no exchanger beats the maximum possible duty).
    T_range = np.linspace(params["T_c_in"], params["T_h_in"], 200)
    C_h_max = params["mdot_h"] * np.max(properties.evaluate(props["h"], "cp", T_range))
    C_c_max = params["mdot_c"] * np.max(properties.evaluate(props["c"], "cp", T_range))
    Q_max = min(C_h_max, C_c_max) * (params["T_h_in"] - params["T_c_in"])
    assert 0.0 < duty["Q_h"] < Q_max


def test_collocation_matches_shooting(base_props):
    """Collocation and shooting give the same variable-property solution.

    From math doc Section 5.4 item 3; this check is NOT in CLAUDE.md's
    verification list yet (flagged; added at the team's request).
    Two independent methods for the same BVP (Eq 6-8): solve_bvp collocation
    (Section 5.1) and solve_ivp + brentq shooting (Section 5.2, Eq 24).
    """
    # 1. params = dict(config.BASE_CASE) with bvp_tol = 1e-8 (fail clearly if
    #    TODOs are unfilled); props = properties.build_surrogates(params).
    params = base_case_params()
    props = base_props

    # 2. s_col = steady.solve_steady(params, props);
    #    s_sh = steady.solve_steady_shooting(params, props).
    #    Both report on the same grid x = np.linspace(0, L, n_out).
    s_col = steady.solve_steady(params, props)
    s_sh = steady.solve_steady_shooting(params, props)
    assert np.array_equal(s_col["x"], s_sh["x"])

    # 3. assert max |s_col["T_h"] - s_sh["T_h"]| < 1e-3 K and the same for T_c.
    assert np.max(np.abs(s_col["T_h"] - s_sh["T_h"])) < 1e-3
    assert np.max(np.abs(s_col["T_c"] - s_sh["T_c"])) < 1e-3

    # 4. Cold outlet (the shooting unknown sigma): assert
    #    abs(s_col["T_c"][0] - s_sh["T_c"][0]) < 1e-4 K.
    assert abs(s_col["T_c"][0] - s_sh["T_c"][0]) < 1e-4

    # 5. Duty: Q from steady.energy_balance for each; assert relative
    #    difference < 1e-5.
    Q_col = steady.energy_balance(s_col, params, props)["Q_h"]
    Q_sh = steady.energy_balance(s_sh, params, props)["Q_h"]
    assert abs(Q_col - Q_sh) / Q_col < 1e-5
    # Note: if this fails at large NTU (1 - C_r), suspect shooting first (its
    #    known weakness, Section 5.2): tighten its solve_ivp tolerances before
    #    blaming collocation.


# ---------------------------------------------------------------------------
# Unit checks from the "# Check:" comments in src/steady.py
# ---------------------------------------------------------------------------

def test_wall_temperature_eq5():
    """Equal conductances give the mean; stream-to-metal differences follow Eq 5."""
    T_h = np.array([300.0, 280.0])
    T_c = np.array([260.0, 250.0])
    assert np.allclose(steady.wall_temperature(T_h, T_c, 2.0, 2.0), (T_h + T_c) / 2)
    a_h, a_c = 3.0, 1.0
    T_w = steady.wall_temperature(T_h, T_c, a_h, a_c)
    assert np.allclose(T_h - T_w, a_c / (a_h + a_c) * (T_h - T_c))
    assert np.allclose(T_w - T_c, a_h / (a_h + a_c) * (T_h - T_c))


def test_effectiveness_limits():
    """Eq 13 limits: C_r = 0, C_r -> 1 continuity, and a hand value."""
    NTU = 2.5
    assert steady.effectiveness(NTU, 0.0) == pytest.approx(1.0 - np.exp(-NTU), rel=1e-14)
    assert abs(steady.effectiveness(NTU, 1.0 - 1e-9) - steady.effectiveness(NTU, 1.0)) < 1e-9
    assert steady.effectiveness(3.0, 0.5) == pytest.approx(0.8744, abs=1e-4)


@pytest.mark.parametrize("C_h, C_c", [(250.0, 500.0), (500.0, 250.0), (250.0, 250.0)])
def test_analytical_profiles_boundaries_and_effectiveness(C_h, C_c):
    """Inlet conditions hold, the BC residual is zero and eps matches Eq 13."""
    L, U_prime, T_h_in, T_c_in = 0.7, 1500.0, 300.0, 235.0
    x = np.linspace(0.0, L, 201)
    T_h, T_c = steady.analytical_profiles(x, L, T_h_in, T_c_in, C_h, C_c, U_prime)
    assert T_h[0] == pytest.approx(T_h_in, abs=1e-12)
    assert T_c[-1] == pytest.approx(T_c_in, abs=1e-12)
    residual = steady.steady_bc(np.array([T_h[0], T_c[0]]), np.array([T_h[-1], T_c[-1]]), T_h_in, T_c_in)
    assert np.max(np.abs(residual)) < 1e-12

    C_min, C_max = min(C_h, C_c), max(C_h, C_c)
    eps = C_h * (T_h_in - T_h[-1]) / (C_min * (T_h_in - T_c_in))
    assert eps == pytest.approx(steady.effectiveness(U_prime * L / C_min, C_min / C_max), abs=1e-12)

    # Balanced flow (C_r = 1): straight lines, and continuous at C_r = 1 +- 1e-9.
    if C_h == C_c:
        assert np.max(np.abs(np.diff(T_h, 2))) < 1e-10
        T_h_near, _ = steady.analytical_profiles(x, L, T_h_in, T_c_in, C_h * (1 + 1e-9), C_c, U_prime)
        assert np.max(np.abs(T_h_near - T_h)) < 1e-6


def test_initial_guess_satisfies_bcs(base_props):
    """The analytical guess meets both inlet conditions and decreases with x."""
    params = base_case_params()
    geom = correlations.exchanger_geometry(params)
    x = np.linspace(0.0, params["L"], 50)
    y = steady.initial_guess(x, params, base_props, geom)
    residual = steady.steady_bc(y[:, 0], y[:, -1], params["T_h_in"], params["T_c_in"])
    assert np.max(np.abs(residual)) < 1e-10
    assert np.all(np.diff(y[0]) < 0) and np.all(np.diff(y[1]) < 0)


def test_steady_rhs_shapes_and_zero_difference(base_props):
    """steady_rhs works for solve_bvp (2, m) and solve_ivp (2,) shapes; T_h == T_c gives 0."""
    params = base_case_params()
    geom = correlations.exchanger_geometry(params)
    y = np.array([[290.0, 270.0, 250.0], [280.0, 260.0, 240.0]])
    dydx = steady.steady_rhs(None, y, params, base_props, geom)
    assert dydx.shape == (2, 3) and np.all(dydx < 0)
    assert steady.steady_rhs(None, y[:, 0], params, base_props, geom).shape == (2,)
    same = steady.steady_rhs(None, np.array([270.0, 270.0]), params, base_props, geom)
    assert np.all(same == 0.0)


def test_shooting_residual_increasing_and_zero_at_solution(base_props):
    """r(sigma) increases with sigma and is ~0 at the collocation cold outlet."""
    params = base_case_params()
    geom = correlations.exchanger_geometry(params)
    s = steady.solve_steady(params, base_props)
    sigma_star = s["T_c"][0]
    sigmas = np.linspace(params["T_c_in"] + 1.0, params["T_h_in"] - 1.0, 9)
    r = [steady.shooting_residual(sig, params, base_props, geom) for sig in sigmas]
    assert np.all(np.diff(r) > 0)
    assert abs(steady.shooting_residual(sigma_star, params, base_props, geom)) < 1e-4
