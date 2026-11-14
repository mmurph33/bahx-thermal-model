"""Unit tests for src/stress.py (thermal stress check against the ALPEMA limits).

All tests use hand-made arrays, so no solver or CoolProp is needed and the
results do not depend on the team's base-case choices.
"""

import numpy as np
import pytest

from src import config, stress

# Aluminium values from math doc Section 9.2, written here so the tests do not
# depend on config.BASE_CASE.
ALUMINIUM = {"E_m": 70e9, "alpha_T": 23e-6, "nu_m": 0.33}


def wall_eq4(T_h, T_c, a_h, a_c):
    """Steady wall temperature, Eq 4: T_w = (a_h T_h + a_c T_c) / (a_h + a_c)."""
    return (a_h * T_h + a_c * T_c) / (a_h + a_c)


def test_temperature_differences_hand_made():
    """The three local differences, and hot-to-metal + metal-to-cold = stream-to-stream."""
    T_h = np.array([300.0, 290.0])
    T_c = np.array([280.0, 275.0])
    T_w = np.array([292.0, 281.0])
    dT = stress.temperature_differences(T_h, T_c, T_w)
    assert np.allclose(dT["hot-to-metal"], [8.0, 9.0])
    assert np.allclose(dT["metal-to-cold"], [12.0, 6.0])
    assert np.allclose(dT["stream-to-stream"], [20.0, 15.0])
    assert np.allclose(dT["hot-to-metal"] + dT["metal-to-cold"], dT["stream-to-stream"])


def test_equal_conductances_split_the_difference_in_half():
    """With a_h = a_c each stream-to-metal difference is half the stream-to-stream one (Eq 5)."""
    T_h = np.array([[300.0, 290.0], [295.0, 280.0]])
    T_c = np.array([[280.0, 260.0], [275.0, 270.0]])
    checks = stress.alpema_difference_check(T_h, T_c, wall_eq4(T_h, T_c, 1.0, 1.0),
                                            config.ALPEMA_LIMITS)
    assert checks["stream-to-metal"]["value"] == pytest.approx(checks["stream-to-stream"]["value"] / 2)


def test_stream_to_metal_takes_the_worse_side():
    """With a_c = 3 a_h the wall sits near T_c, so the hot side carries 3/4 of T_h - T_c (Eq 5)."""
    T_h, T_c = np.array([300.0]), np.array([260.0])
    checks = stress.alpema_difference_check(T_h, T_c, wall_eq4(T_h, T_c, 1.0, 3.0),
                                            config.ALPEMA_LIMITS)
    assert checks["stream-to-metal"]["value"] == pytest.approx(30.0)
    assert checks["stream-to-stream"]["value"] == pytest.approx(40.0)


@pytest.mark.parametrize("dT, ok", [(29.9, True), (30.1, False)])
def test_alpema_check_flips_at_the_limit(dT, ok):
    """A stream-to-stream difference just under / just over 30 K passes / fails."""
    T_h, T_c = np.array([260.0 + dT]), np.array([260.0])
    T_w = wall_eq4(T_h, T_c, 1.0, 1.0)
    c = stress.alpema_difference_check(T_h, T_c, T_w, config.ALPEMA_LIMITS)["stream-to-stream"]
    assert c["ok"] is ok
    assert c["margin"] == pytest.approx(30.0 - dT)


def test_thermal_stress_per_kelvin():
    """E = 70 GPa, alpha_T = 23e-6 1/K, nu = 0.33 give ~2.4 MPa per kelvin (math doc Section 9.2)."""
    assert stress.thermal_stress(1.0, 70e9, 23e-6, 0.33) == pytest.approx(2.4e6, rel=0.01)


def test_thermal_stress_is_linear_and_keeps_shape():
    """Eq 53 is linear in Delta T and works elementwise on arrays."""
    dT = np.array([[0.0, 10.0], [-5.0, 30.0]])
    sigma = stress.thermal_stress(dT, 70e9, 23e-6, 0.33)
    assert sigma.shape == dT.shape
    assert np.allclose(sigma, dT * stress.thermal_stress(1.0, 70e9, 23e-6, 0.33))


def test_stress_check_steady_profile():
    """Peaks, their locations and the reference levels on a hand-made steady profile."""
    x = np.linspace(0.0, 0.7, 8)
    T_h = np.linspace(300.0, 250.0, 8)
    T_c = T_h - np.linspace(7.0, 17.0, 8)        # gap grows toward x = L, as in the base case
    T_w = wall_eq4(T_h, T_c, 1.2, 1.0)
    r = stress.stress_check(x, T_h, T_c, T_w, ALUMINIUM, config.ALPEMA_LIMITS, 35e6)
    per_K = stress.thermal_stress(1.0, 70e9, 23e-6, 0.33)
    assert r["per_K"] == pytest.approx(per_K)
    peak = r["peaks"]["stream-to-stream"]
    assert peak["x"] == pytest.approx(0.7)
    assert peak["dT"] == pytest.approx(17.0)
    assert peak["sigma"] == pytest.approx(17.0 * per_K)
    assert r["sigma_alpema"] == pytest.approx(30.0 * per_K)
    assert r["dT_yield"] == pytest.approx(35e6 / per_K)
    assert r["check"]["stream-to-stream"]["ok"]
    assert "EXCEEDED" not in stress.summary_text(r)


def test_stress_check_transient_history():
    """A history, shape (n_t, N), gives the peak over all times and cells, and its cell's x."""
    x = np.array([0.0, 0.35, 0.7])
    T_h = np.array([[300.0, 280.0, 260.0], [300.0, 285.0, 270.0]])
    T_c = T_h - np.array([[5.0, 8.0, 10.0], [5.0, 40.0, 12.0]])   # 40 K at t index 1, cell 1
    T_w = wall_eq4(T_h, T_c, 1.0, 1.0)
    r = stress.stress_check(x, T_h, T_c, T_w, ALUMINIUM, config.ALPEMA_LIMITS, 35e6)
    assert r["peaks"]["stream-to-stream"]["index"] == (1, 1)
    assert r["peaks"]["stream-to-stream"]["x"] == pytest.approx(0.35)
    assert not r["check"]["stream-to-stream"]["ok"]
    assert "EXCEEDED" in stress.summary_text(r)


def test_alpema_limit_exceeds_annealed_yield():
    """At the ALPEMA 30 degC limit the fully restrained estimate (~72 MPa) is above the
    3003-O yield (35 MPa): Eq 53 is an upper bound, not a design stress."""
    sigma_limit = stress.thermal_stress(config.ALPEMA_LIMITS["dT_stream_metal"], 70e9, 23e-6, 0.33)
    assert sigma_limit == pytest.approx(72.1e6, rel=0.01)
    assert sigma_limit > config.MATERIAL_LIMITS["sigma_y_3003_O"]
