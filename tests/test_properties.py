"""Unit tests for src/properties.py, from the "# Check:" comments in that module.

Includes regression tests for three bugs fixed during review: the spline
branch of evaluate() used the scaled temperature, a negative feed ramp dropped
the margin above T_h_in, and a supercritical range reaching the cp peak was
not rejected.
"""

import numpy as np
import pytest
from CoolProp.CoolProp import PropsSI

from src import config, properties


@pytest.fixture(scope="module")
def base_props():
    """Base-case CoolProp surrogates, built once for this module."""
    return properties.build_surrogates(dict(config.BASE_CASE), "none")


@pytest.fixture(scope="module")
def hot_spline():
    """Hot-side (6 MPa) fit that falls back to cubic splines (cp misses 1e-5)."""
    return properties.fit_surrogate_cubicspline("Methane", 6e6, 225.0, 310.0, 6, 200)


def test_check_range_edges():
    """Exactly T_lo and T_hi pass; just outside, NaN or one bad entry in a 2D array raise."""
    surr = properties.constant_surrogate(2500.0, 1.2e-5, 0.035, 30.0, 200.0, 320.0)
    properties.check_range(surr, np.array([200.0, 320.0]))
    for bad in (200.0 - 1e-9, 320.0 + 1e-9, np.nan):
        with pytest.raises(ValueError):
            properties.check_range(surr, bad)
    T = np.full((2, 5), 250.0)
    T[1, 3] = 330.0
    with pytest.raises(ValueError):
        properties.check_range(surr, T)


def test_constant_surrogate_values_and_enthalpy():
    """evaluate returns the constant; enthalpy differences are cp (T2 - T1)."""
    surr = properties.constant_surrogate(2500.0, 1.2e-5, 0.035, 30.0, 200.0, 320.0)
    T = np.linspace(205.0, 315.0, 12).reshape(3, 4)
    assert np.all(properties.evaluate(surr, "cp", T) == 2500.0)
    assert properties.evaluate(surr, "cp", T).shape == (3, 4)
    dh = properties.enthalpy(surr, 300.0) - properties.enthalpy(surr, 250.0)
    assert dh == pytest.approx(2500.0 * 50.0, rel=1e-14)


def test_continuation_surrogate(base_props):
    """lam = 0 gives the mean for every T; lam = 1 the full fit; the original is untouched."""
    surr = base_props["h"]
    T = np.linspace(230.0, 305.0, 9)
    s0 = properties.continuation_surrogate(surr, 0.0)
    s1 = properties.continuation_surrogate(surr, 1.0)
    assert np.allclose(properties.evaluate(s0, "cp", T), surr["mean"]["cp"])
    assert np.array_equal(properties.evaluate(s1, "cp", T), properties.evaluate(surr, "cp", T))
    assert surr["lam"] == 1.0


def test_base_case_fits_meet_target(base_props):
    """Both sides build with every max_rel_err below fit_max_rel_err; d(hhat)/dT = cp."""
    for side in ("h", "c"):
        surr = base_props[side]
        assert max(surr["max_rel_err"].values()) < config.BASE_CASE["fit_max_rel_err"]
        T = 270.0
        dh_dT = (properties.enthalpy(surr, T + 1e-3) - properties.enthalpy(surr, T - 1e-3)) / 2e-3
        assert dh_dT == pytest.approx(properties.evaluate(surr, "cp", T), rel=1e-7)


def test_spline_fallback_matches_coolprop(hot_spline):
    """Regression: the spline branch must evaluate in kelvin (it returned ~400x cp before the fix)."""
    assert hot_spline["kind"] == "spline"
    cp_ref = PropsSI("CPMASS", "T", 260.0, "P", 6e6, "Methane")
    assert properties.evaluate(hot_spline, "cp", 260.0) == pytest.approx(cp_ref, rel=1e-6)
    T = 260.0
    dh_dT = (properties.enthalpy(hot_spline, T + 1e-3) - properties.enthalpy(hot_spline, T - 1e-3)) / 2e-3
    assert dh_dT == pytest.approx(properties.evaluate(hot_spline, "cp", T), rel=1e-7)


def test_negative_ramp_keeps_margin_above_hot_inlet():
    """Regression: a cooldown ramp still covers the initial T_h_in plus the margin.

    Before the fix, ramp_dT = -10 K gave T_hi = 300 K = T_h_in, with no margin.
    (A ramp large enough to pull T_lo below T_c_in would push the hot fit onto
    the cp-peak shoulder near 210 K, which the polynomial cannot fit.)
    """
    params = dict(config.BASE_CASE, ramp_dT=-10.0)
    props = properties.build_surrogates(params, "feed_ramp")
    margin = params["fit_T_margin"]
    assert props["h"]["T_hi"] == params["T_h_in"] + margin
    assert props["h"]["T_lo"] == params["T_c_in"] - margin


def test_check_single_phase_danger_points():
    """T_sat = 158.5 K at 1.5 MPa and T_pc = 199.6 K at 6 MPa; ranges reaching them raise."""
    sub = properties.check_single_phase("Methane", 1.5e6, 225.0, 310.0)
    assert sub["T_sat"] == pytest.approx(158.5, abs=0.1) and not sub["supercritical"]
    sup = properties.check_single_phase("Methane", 6e6, 225.0, 310.0)
    assert sup["T_pc"] == pytest.approx(199.6, abs=0.3) and sup["supercritical"]
    with pytest.raises(ValueError):
        properties.check_single_phase("Methane", 1.5e6, 150.0, 310.0)
    with pytest.raises(ValueError):
        properties.check_single_phase("Methane", 6e6, 195.0, 310.0)
