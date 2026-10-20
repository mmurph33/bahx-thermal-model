"""Verification of the steady model (CLAUDE.md "Verification"; math doc Section 5.4).

Each test is skipped until the functions it needs exist. To activate one,
delete its pytest.skip line and turn the numbered comments into code. Never
loosen a tolerance to make a test pass; report the failure and its likely
cause instead (CLAUDE.md).
"""

import pytest

from src import config, correlations, properties, steady  # noqa: F401  (used once implemented)


@pytest.mark.parametrize("case", ["hot_is_cmin", "cold_is_cmin", "balanced"])
def test_constant_property_matches_epsilon_ntu(case):
    """Constant-property BVP solution matches the analytical epsilon-NTU solution.

    CLAUDE.md check 1 / math doc Section 5.4 item 1: with constant properties
    the numerical effectiveness and profiles match Eq 13 and Eq 11 (here the
    unified form in steady.analytical_profiles) to solver tolerance,
    including C_r = 1. Run for hot = C_min, cold = C_min and balanced flow,
    since the analytical formulas differ between these cases.
    """
    pytest.skip("not implemented yet")
    # 1. Synthetic parameters, independent of the team's base-case choices:
    #    p = dict(config.BASE_CASE) and fill every key the solver needs with
    #    explicit test values written in this test (e.g. T_h_in = 300,
    #    T_c_in = 235, W = 1, the suggested fin geometry, k_f = 180,
    #    zeta_j = 1, colburn_beta = None, n_out = 501, bvp_max_nodes = 10000,
    #    continuation_lams = (0, .25, .5, .75, 1)), and set bvp_tol = 1e-8:
    #    solve_bvp controls a relative residual, not the solution error, so a
    #    tight tolerance is needed to check eps to 1e-6. (Tightening the solver
    #    is not loosening the test.)
    # 2. Constant surrogates (no CoolProp): props = {"h":
    #    properties.constant_surrogate(cp_h, mu, k, rho, 200, 320), "c": ...}
    #    with one fixed set of mu, k, rho and cp_h. Pick cp_c and the flows to
    #    set C_r:
    #      hot_is_cmin:  C_r = 0.5 with C_h < C_c
    #      cold_is_cmin: C_r = 0.5 with C_c < C_h
    #      balanced:     cp_c = cp_h * mdot_h / mdot_c, so C_h == C_c exactly
    #                    (C_r = 1 to round-off, which exercises the k -> 0
    #                    branch of analytical_profiles).
    # 3. Constant conductances: geom = correlations.exchanger_geometry(p);
    #    a_h, a_c = correlations.conductances(...) at any in-range T;
    #    U' = correlations.overall_conductance(a_h, a_c). Choose L so the test
    #    is demanding: NTU (1 - C_r) ~ 3-5 (e.g. NTU = 6 for C_r = 0.5) and
    #    NTU ~ 4 for balanced flow; L = NTU * C_min / U'.
    # 4. Numerical solution: s = steady.solve_steady(p, props).
    # 5. Analytical solution on the same grid:
    #    T_h_ex, T_c_ex = steady.analytical_profiles(s["x"], L, T_h_in, T_c_in,
    #    C_h, C_c, U').
    # 6. Compare in dimensionless form, theta = (T - T_c_in)/(T_h_in - T_c_in):
    #    assert max |theta_h - theta_h_ex| < 1e-5 and the same for theta_c.
    # 7. Effectiveness: eps_num = C_h (T_h_in - s["T_h"][-1]) / (C_min (T_h_in - T_c_in));
    #    assert abs(eps_num - steady.effectiveness(NTU, C_r)) < 1e-6.
    # 8. Location of the largest difference (flagged Eq 14 wording): assert
    #    np.argmax(T_h - T_c) is at x = 0 for hot_is_cmin and at x = L for
    #    cold_is_cmin, and the value equals Delta_0 (T_h_in - T_c_in) to 1e-5
    #    in theta.


def test_energy_closure():
    """Hot duty, cold duty and the integrated heat flow agree.

    CLAUDE.md check 2 / math doc Section 5.4 item 2, Eq 16:
    Q_h = Q_c = integral of U'(T_h - T_c) dx, for the VARIABLE-property base
    case (with constant properties the check passes trivially).
    """
    pytest.skip("not implemented yet")
    # 1. params = dict(config.BASE_CASE). If any base-case TODO is still None,
    #    fail with a message listing the missing keys (don't skip silently).
    #    Set bvp_tol = 1e-8.
    # 2. props = properties.build_surrogates(params). The enthalpy surrogate
    #    must be the integral of the cp surrogate, otherwise closure is stuck
    #    at the fit error (~1e-3).
    # 3. s = steady.solve_steady(params, props);
    #    duty = steady.energy_balance(s, params, props).
    # 4. assert duty["closure_hc"] < 1e-4    (|Q_h - Q_c| / Q_h)
    #    assert duty["closure_int"] < 1e-4   (|Q_h - Q_int| / Q_h)
    # 5. Sanity: 0 < Q_h < C_min (T_h_in - T_c_in) using the largest cp on
    #    each side (no exchanger beats the maximum possible duty).


def test_collocation_matches_shooting():
    """Collocation and shooting give the same variable-property solution.

    From math doc Section 5.4 item 3; this check is NOT in CLAUDE.md's
    verification list yet (flagged; added at the team's request).
    Two independent methods for the same BVP (Eq 6-8): solve_bvp collocation
    (Section 5.1) and solve_ivp + brentq shooting (Section 5.2, Eq 24).
    """
    pytest.skip("not implemented yet")
    # 1. params = dict(config.BASE_CASE) with bvp_tol = 1e-8 (fail clearly if
    #    TODOs are unfilled); props = properties.build_surrogates(params).
    # 2. s_col = steady.solve_steady(params, props);
    #    s_sh = steady.solve_steady_shooting(params, props).
    #    Both report on the same grid x = np.linspace(0, L, n_out).
    # 3. assert max |s_col["T_h"] - s_sh["T_h"]| < 1e-3 K and the same for T_c.
    # 4. Cold outlet (the shooting unknown sigma): assert
    #    abs(s_col["T_c"][0] - s_sh["T_c"][0]) < 1e-4 K.
    # 5. Duty: Q from steady.energy_balance for each; assert relative
    #    difference < 1e-5.
    # Note: if this fails at large NTU (1 - C_r), suspect shooting first (its
    #    known weakness, Section 5.2): tighten its solve_ivp tolerances before
    #    blaming collocation.
