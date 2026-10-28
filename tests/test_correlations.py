"""Unit tests for src/correlations.py, from the "# Check:" comments in that module."""

import numpy as np
import pytest

from src import config, correlations, properties

# Suggested fin (Kays & London 1/8-15.61 style) [m].
FIN = dict(s=1.525e-3, b=6.25e-3, t=1.02e-4, l=3.175e-3, W=1.0)


@pytest.fixture(scope="module")
def geom():
    """Base-case exchanger geometry."""
    return correlations.exchanger_geometry(dict(config.BASE_CASE))


def test_osf_geometry_hydraulic_diameter():
    """D_h ~ 2.4 mm for the suggested fin; parallel-channel limit for t -> 0, l -> infinity."""
    g = correlations.osf_geometry(**FIN)
    assert g["D_h"] == pytest.approx(2.4e-3, rel=0.02)
    assert g["A_prime"] == pytest.approx(g["A_p"] + g["A_f"])

    s, b = FIN["s"], FIN["b"]
    g_limit = correlations.osf_geometry(s, b, 1e-12, 1e6, 1.0)
    assert g_limit["D_h"] == pytest.approx(2 * s * b / (s + b), rel=1e-6)


def test_exchanger_geometry_metal_area(geom):
    """The two parting sheets dominate A_m: 2e-3 m^2 against ~0.8e-3 m^2 of fins."""
    sheets = 2 * config.BASE_CASE["t_sheet"] * config.BASE_CASE["W"]
    assert sheets == pytest.approx(2e-3)
    assert geom["A_m"] - sheets == pytest.approx(0.8e-3, rel=0.05)
    assert geom["C_m"] == pytest.approx(config.BASE_CASE["rho_m"] * config.BASE_CASE["c_m"] * geom["A_m"])


def test_colburn_j_value_and_smoothness():
    """j ~ 0.009 at Re = 3400 for the suggested fin, and falls smoothly with Re."""
    alpha, delta, gamma = 0.244, 0.032, 0.067
    assert correlations.colburn_j(3400.0, alpha, delta, gamma) == pytest.approx(0.009, rel=0.06)
    Re = np.logspace(2.0, 4.0, 2000)
    j = correlations.colburn_j(Re, alpha, delta, gamma)
    assert np.all(np.diff(j) < 0)
    # No jump: from Eq 19, d ln j / d ln Re = -0.5403 + 0.1 (1.340) B/(1 + B)
    # with B the turbulent bracket term, so the log-log slope must stay
    # strictly between -0.5403 and -0.5403 + 0.134.
    slope = np.diff(np.log(j)) / np.diff(np.log(Re))
    assert np.all(slope > -0.5403) and np.all(slope < -0.5403 + 0.134)


def test_laminar_nusselt_limits():
    """Square duct Nu_H1 = 3.61; parallel-plate limit 8.235; s/b and b/s are the same duct."""
    assert correlations.laminar_nusselt_rect(1.0) == pytest.approx(3.61, abs=0.01)
    assert correlations.laminar_nusselt_rect(1e-9) == pytest.approx(8.235, abs=1e-6)
    assert correlations.laminar_nusselt_rect(0.25) == pytest.approx(correlations.laminar_nusselt_rect(4.0))


def test_heat_transfer_coefficient_zero_flow_floor(geom):
    """With mdot -> 0 the coefficient tends to the laminar floor: finite and positive."""
    cp, mu, k = 2500.0, 1.2e-5, 0.035
    h_lam = correlations.laminar_nusselt_rect(geom["c"]["alpha_mb"]) * k / geom["c"]["D_h"]
    h = correlations.heat_transfer_coefficient(np.array([0.0, 1e-9]), cp, mu, k, geom["c"])
    assert np.all(np.isfinite(h)) and np.allclose(h, h_lam)


def test_fin_and_surface_efficiency(geom):
    """eta_f ~ 0.86 at h ~ 450; eta_f(h = 0) = 1; eta_o limits from Eq 22."""
    eta_f = correlations.fin_efficiency(np.array([450.0, 0.0]), 180.0, geom["h"])
    assert eta_f[0] == pytest.approx(0.86, abs=0.01)
    assert eta_f[1] == 1.0
    g = geom["h"]
    assert correlations.surface_efficiency(1.0, g) == pytest.approx(1.0)
    assert correlations.surface_efficiency(0.0, g) == pytest.approx(g["A_p"] / g["A_prime"])


def test_constant_properties_give_constant_conductance(geom):
    """A constant-property surrogate gives the same a at every temperature."""
    surr = properties.constant_surrogate(2500.0, 1.2e-5, 0.035, 30.0, 200.0, 320.0)
    a = correlations.side_conductance(np.linspace(210.0, 310.0, 7), 0.1, surr, geom["h"], 180.0)
    assert np.ptp(a) == 0.0 and a[0] > 0.0


def test_conductances_swap_symmetry():
    """With identical surrogates and geometry, swapping hot and cold inputs swaps the outputs."""
    params = dict(config.BASE_CASE)
    for side in ("h", "c"):
        params.update({f"s_{side}": FIN["s"], f"b_{side}": FIN["b"], f"t_fin_{side}": FIN["t"], f"l_{side}": FIN["l"]})
    g = correlations.exchanger_geometry(params)
    surr = properties.constant_surrogate(2500.0, 1.2e-5, 0.035, 30.0, 200.0, 320.0)
    props = {"h": surr, "c": surr}
    a_h, a_c = correlations.conductances(280.0, 260.0, 0.1, 0.2, params, props, g)
    b_h, b_c = correlations.conductances(260.0, 280.0, 0.2, 0.1, params, props, g)
    assert a_h == pytest.approx(b_c) and a_c == pytest.approx(b_h)


def test_overall_conductance_limits():
    """Equal sides give a/2; an infinite cold side gives U' -> a_h."""
    assert correlations.overall_conductance(3000.0, 3000.0) == pytest.approx(1500.0)
    assert correlations.overall_conductance(3000.0, 1e15) == pytest.approx(3000.0, rel=1e-9)
