"""Unit tests for src/metrics.py, from the "# Check:" comments in that module.

metrics.py is pure post-processing, so every test here uses hand-made arrays
and needs no solver or CoolProp.
"""

import numpy as np
import pytest

from src import config, metrics, steady


def test_wall_rate_units():
    """One entry of -0.1 K/s gives 6 K/min."""
    dTw_dt = np.zeros((3, 5))
    dTw_dt[1, 2] = -0.1
    assert np.allclose(metrics.wall_rate(dTw_dt), [0.0, 6.0, 0.0])


def test_nozzle_rate_uses_outlet_cells():
    """The hot outlet is the last cell, the cold outlet the first cell."""
    dTh_dt = np.zeros((2, 4))
    dTc_dt = np.zeros((2, 4))
    dTh_dt[:, -1] = 0.05
    assert np.allclose(metrics.nozzle_rate(dTh_dt, dTc_dt), 3.0)
    # The inlet cells (hot cell 1, cold cell N) must be ignored.
    dTh_dt[:, -1] = 0.0
    dTh_dt[:, 0] = 1.0
    dTc_dt[:, -1] = 1.0
    assert np.allclose(metrics.nozzle_rate(dTh_dt, dTc_dt), 0.0)


def test_stream_to_stream_matches_eq14():
    """On the analytical profile with cold = C_min, the max is Delta_0 (T_h_in - T_c_in) at x = L."""
    L, U_prime, T_h_in, T_c_in = 0.7, 1500.0, 300.0, 235.0
    C_h, C_c = 500.0, 250.0
    x = np.linspace(0.0, L, 401)
    T_h, T_c = steady.analytical_profiles(x, L, T_h_in, T_c_in, C_h, C_c, U_prime)
    C_r = C_c / C_h
    NTU = U_prime * L / C_c
    E = np.exp(-NTU * (1.0 - C_r))
    Delta_0 = (1.0 - C_r) / (1.0 - C_r * E)
    assert metrics.stream_to_stream(T_h, T_c) == pytest.approx(Delta_0 * (T_h_in - T_c_in), rel=1e-12)


def test_stream_to_metal_equal_conductances():
    """With a_h = a_c each stream-to-metal difference is half the stream-to-stream one (Eq 5)."""
    T_h = np.array([[300.0, 290.0], [295.0, 280.0]])
    T_c = np.array([[280.0, 260.0], [275.0, 270.0]])
    T_w = steady.wall_temperature(T_h, T_c, 1.0, 1.0)
    assert np.allclose(metrics.stream_to_metal(T_h, T_c, T_w), metrics.stream_to_stream(T_h, T_c) / 2)


def test_wall_gradient_linear_profile():
    """A linear profile T_w = 250 + 10 x has |dT_w/dx| = 10 K/m everywhere, on any grid."""
    x = np.sort(np.random.default_rng(2).uniform(0.0, 0.7, 30))
    T_w = 250.0 + 10.0 * x
    assert metrics.wall_gradient(T_w, x) == pytest.approx(10.0, rel=1e-10)
    assert np.allclose(metrics.wall_gradient(np.vstack((T_w, T_w)), x), 10.0)


def test_hourly_change_linear_ramp():
    """A 2 K/min ramp held for 2 h gives a 120 K change once a full hour has passed."""
    t = np.linspace(0.0, 7200.0, 721)
    T = np.column_stack((250.0 + (2.0 / 60.0) * t, np.full_like(t, 260.0)))
    dT_hour = metrics.hourly_change(t, T, window=3600.0)
    assert np.allclose(dT_hour[t >= 3600.0], 120.0)
    # Before one hour, the change is measured from the initial state.
    assert np.allclose(dT_hour[t < 3600.0], (2.0 / 60.0) * t[t < 3600.0])


def fake_result(n_t=5, N=4):
    """A hand-made steady "simulate" result: constant profiles, zero rates."""
    t = np.linspace(0.0, 100.0, n_t)
    x = (np.arange(N) + 0.5) * 0.7 / N
    T_h = np.tile(np.linspace(300.0, 255.0, N), (n_t, 1))
    T_c = np.tile(np.linspace(292.0, 235.0, N), (n_t, 1))
    T_w = (T_h + T_c) / 2
    zeros = np.zeros((n_t, N))
    # Inlet temperatures equal to the inlet cells, so the end faces add nothing new here.
    return {"t": t, "x": x, "T_h": T_h, "T_c": T_c, "T_w": T_w,
            "dTh_dt": zeros, "dTc_dt": zeros, "dTw_dt": zeros,
            "T_h_in": np.full(n_t, 300.0), "T_c_in": np.full(n_t, 235.0)}


def test_alpema_metrics_steady_run():
    """For a steady run every rate is zero and the differences are constant in time."""
    m = metrics.alpema_metrics(fake_result())
    assert m["max_R_w"] == 0.0 and m["max_R_end"] == 0.0 and m["max_dT_hour"] == 0.0
    assert np.ptp(m["dT_ss"]) == 0.0 and np.ptp(m["dT_sm"]) == 0.0
    assert m["max_dT_ss"] == pytest.approx(20.0)
    assert m["max_dT_ss_cells"] == pytest.approx(20.0)
    # Nozzle differences: hot in - cold out at x = 0, hot out - cold in at x = L.
    assert np.allclose(m["dT_warm_end"], 300.0 - 292.0)
    assert np.allclose(m["dT_cold_end"], 255.0 - 235.0)

    r = fake_result()
    r["dTh_dt"] = None
    r["dTc_dt"] = None
    m_qs = metrics.alpema_metrics(r)
    assert m_qs["R_end"] is None and m_qs["max_R_end"] is None


@pytest.mark.parametrize("event", ["frequent", "infrequent"])
def test_check_alpema_flips_at_the_limits(event):
    """Values just under each limit pass, just over fail; unknown events raise."""
    limits = config.ALPEMA_LIMITS
    rate_limit = limits["rate_frequent"] if event == "frequent" else limits["rate_infrequent"]

    def metrics_at(offset):
        return {
            "max_R_w": rate_limit + offset,
            "max_R_end": rate_limit + offset,
            "max_dT_hour": limits["change_per_hour_infrequent"] + offset,
            "max_dT_ss": limits["dT_stream_stream"] + offset,
            "max_dT_sm": limits["dT_stream_metal"] + offset,
        }

    under = metrics.check_alpema(metrics_at(-1e-6), limits, event, ramp_rate=rate_limit - 1e-6)
    over = metrics.check_alpema(metrics_at(+1e-6), limits, event, ramp_rate=rate_limit + 1e-6)
    assert all(c["ok"] for c in under.values())
    assert not any(c["ok"] for c in over.values())
    assert ("hourly" in under) == (event == "infrequent")
    assert under["rate_wall"]["margin"] == pytest.approx(1e-6)
    with pytest.raises(ValueError):
        metrics.check_alpema(metrics_at(0.0), limits, "startup")


def test_observed_order():
    """Exactly first- and second-order data give p_obs = 1 and 2 and y* = 1; non-monotone data warn."""
    first = [1.0 + 1.0 / N for N in (10, 20, 40)]
    p_obs, y_star = metrics.observed_order(*first)
    assert p_obs == pytest.approx(1.0) and y_star == pytest.approx(1.0)

    second = [1.0 + 1.0 / N**2 for N in (10, 20, 40)]
    p_obs, y_star = metrics.observed_order(*second)
    assert p_obs == pytest.approx(2.0) and y_star == pytest.approx(1.0)

    with pytest.warns(UserWarning):
        p_obs, y_star = metrics.observed_order(1.0, 1.1, 1.05)
    assert np.isnan(p_obs) and y_star == 1.05


def test_smax_bounds():
    """max z <= smax(z) <= max z + ln(K)/omega, and no overflow for large omega z."""
    z = np.random.default_rng(3).uniform(-5.0, 5.0, 50)
    for omega in (1.0, 10.0, 1000.0):
        s = metrics.smax(z, omega)
        assert np.max(z) <= s <= np.max(z) + np.log(z.size) / omega
    assert metrics.smax(np.array([1000.0, 999.0]), 10.0) == pytest.approx(1000.0, abs=1e-4)


def test_with_end_faces_layout():
    """Column 0 is the x = 0 face, columns 1..N the cells, column N + 1 the x = L face."""
    r = fake_result(n_t=3, N=4)
    r["T_h_in"] = np.full(3, 305.0)
    r["T_c_in"] = np.full(3, 230.0)
    T_h_f, T_c_f, T_w_f = metrics.with_end_faces(r)
    assert T_h_f.shape == T_c_f.shape == T_w_f.shape == (3, 6)
    for full, cells in ((T_h_f, r["T_h"]), (T_c_f, r["T_c"]), (T_w_f, r["T_w"])):
        assert np.array_equal(full[:, 1:-1], cells)
    # x = 0: hot enters at T_h_in, cold leaves at its first-cell value, metal of cell 1.
    assert np.allclose(T_h_f[:, 0], 305.0)
    assert np.array_equal(T_c_f[:, 0], r["T_c"][:, 0])
    assert np.array_equal(T_w_f[:, 0], r["T_w"][:, 0])
    # x = L: cold enters at T_c_in, hot leaves at its last-cell value, metal of cell N.
    assert np.allclose(T_c_f[:, -1], 230.0)
    assert np.array_equal(T_h_f[:, -1], r["T_h"][:, -1])
    assert np.array_equal(T_w_f[:, -1], r["T_w"][:, -1])


def test_end_faces_catch_a_difference_the_cells_hide():
    """Cold gas that warms up inside the inlet cell: only the x = L face shows the real difference.

    Hand-made numbers from the "# Check:" of with_end_faces: T_c,N = 280 K, T_c_in = 235 K
    and T_w,N = 290 K give 10 K on the cells but 55 K on the face.
    """
    n_t, N = 2, 3
    r = {
        "t": np.array([0.0, 1.0]), "x": (np.arange(N) + 0.5) * 0.7 / N,
        "T_h": np.full((n_t, N), 300.0),
        "T_w": np.full((n_t, N), 290.0),
        "T_c": np.full((n_t, N), 280.0),
        "T_h_in": np.full(n_t, 300.0), "T_c_in": np.full(n_t, 235.0),
        "dTh_dt": np.zeros((n_t, N)), "dTc_dt": np.zeros((n_t, N)), "dTw_dt": np.zeros((n_t, N)),
    }
    m = metrics.alpema_metrics(r)
    assert m["max_dT_sm_cells"] == pytest.approx(10.0)   # |T_h - T_w| = |T_w - T_c| = 10 K in every cell
    assert m["max_dT_sm"] == pytest.approx(55.0)         # T_w,N - T_c_in on the x = L face
    assert m["max_dT_ss_cells"] == pytest.approx(20.0)
    assert m["max_dT_ss"] == pytest.approx(65.0)         # T_h,N - T_c_in
    assert np.allclose(m["dT_cold_end"], 65.0)
    # The ALPEMA verdict depends on which one is used: cells pass, faces fail.
    limit = config.ALPEMA_LIMITS["dT_stream_metal"]
    assert m["max_dT_sm_cells"] < limit < m["max_dT_sm"]
