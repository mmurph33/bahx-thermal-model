"""Steady-state model: analytical epsilon-NTU solution and the steady BVP.

Math doc: Section 3 (control-volume balances Eq 1-3, wall elimination Eq 4-5,
the two-ODE system Eq 6, boundary conditions Eq 7-8, dimensionless form Eq 9,
analytical solution Eq 10-14, duty and energy closure Eq 16) and Section 5
(collocation with continuation Eq 23, shooting Eq 24). The proposal makes this
steady model the minimum viable product; shooting is its root-finding backup
("guess exit temperatures") and our cross-check on collocation.

Physics conventions (CLAUDE.md): x runs from 0 to L. The hot stream enters at
x = 0 and flows +x; the cold stream enters at x = L and flows -x. Both dT/dx are
negative in normal operation. The steady problem is a BVP because the two known
inlet temperatures sit at opposite ends.

Imports: numpy, scipy.integrate (solve_bvp, solve_ivp, quad, simpson),
scipy.optimize (brentq), src.properties and src.correlations. Used by
transient.py (initial guess).

The steady result dict (returned by solve_steady and solve_steady_shooting)
-------------------------------------------------------------------------
    "x"        (n_out,)  evenly spaced positions from 0 to L [m]
    "T_h"      (n_out,)  hot stream temperature [K]
    "T_c"      (n_out,)  cold stream temperature [K]
    "T_w"      (n_out,)  wall (metal) temperature from Eq 4 [K]
    "a_h","a_c","U_prime" (n_out,)  conductances per unit length [W/(m K)]
    "sol"      scipy result (solve_bvp result, or the dense solve_ivp result for shooting)
    "method"   "collocation" or "shooting"
"""

import numpy as np
from scipy.integrate import quad, simpson, solve_bvp, solve_ivp
from scipy.optimize import brentq

import correlations, properties


def wall_temperature(T_h, T_c, a_h, a_c):
    """Wall (metal) temperature from the steady wall balance.

    Eq 3 solved for T_w gives Eq 4: T_w = (a_h T_h + a_c T_c)/(a_h + a_c),
    a conductance-weighted average of the two stream temperatures.

    Parameters
    ----------
    T_h, T_c : ndarray, same shape
        Hot and cold stream temperatures [K].
    a_h, a_c : ndarray, same shape
        Side conductances per unit length [W/(m K)].

    Returns
    -------
    T_w : ndarray, same shape
        Wall temperature [K].
    """

    return (a_h * T_h + a_c * T_c) / (a_h + a_c)


def effectiveness(NTU, C_r):
    """Counterflow effectiveness.

    Eq 13: eps = (1 - E)/(1 - C_r E) with E = exp(-NTU (1 - C_r)); for
    balanced flow (C_r = 1) eps = NTU/(1 + NTU).

    Parameters
    ----------
    NTU : float
        Number of transfer units U'L/C_min [-].
    C_r : float
        Capacity ratio C_min/C_max, 0 <= C_r <= 1 [-].

    Returns
    -------
    eps : float
        Effectiveness Q/Q_max [-].
    """

    if C_r == 1:
        return NTU / (1 + NTU)

    k = NTU * (1 - C_r)

    one_minus_E = -np.expm1(-k) # np.expm1 = 1 - np.exp(-k)
    E = np.exp(-k)
    eps = ome_minus_E / (one_minus_E + (1 - C_r) * E)
    return eps


def analytical_profiles(x, L, T_h_in, T_c_in, C_h, C_c, U_prime):
    """Constant-property temperature profiles (verification case and initial guess).

    Section 3.5, Eq 9-12, written so the same formula holds whichever stream
    is C_min. With xi = x/L, NTU_h = U'L/C_h, NTU_c = U'L/C_c and
    k = NTU_h - NTU_c, subtracting the two equations of Eq 9 gives
    dDelta/dxi = -k Delta for Delta = theta_h - theta_c, so
      Delta(xi)   = Delta_0 exp(-k xi)
      theta_h(xi) = 1 - NTU_h Delta_0 g(xi),  g(xi) = (1 - exp(-k xi))/k
      theta_c(xi) = theta_h(xi) - Delta(xi)
      Delta_0     = 1 / (exp(-k) + NTU_h g(1))     (from theta_c(1) = 0)
    For hot = C_min this is exactly Eq 10-12. Eq 11 as printed (with the
    factor 1/(1 - C_r)) only holds when the hot stream is C_min (flagged).

    Parameters
    ----------
    x : ndarray, shape (n,)
        Positions, 0 <= x <= L [m].
    L : float
        Core length [m].
    T_h_in, T_c_in : float
        Inlet temperatures [K].
    C_h, C_c : float
        Heat capacity rates mdot*cp of each stream [W/K].
    U_prime : float
        Overall conductance per unit length [W/(m K)].

    Returns
    -------
    T_h, T_c : ndarray, shape (n,)
        Temperature profiles [K].
    """

    x = np.asarray(x)
    xi = x / L
    NTU_h = U_prime * L / C_h
    NTU_c = U_prime * L / C_c
    k = NTU_h - NTU_c

    def g(z):
        if abs(k) < 1e-12:
            return z # prevent division by very small k
        return -np.expm1(-k*z) / k

    Delta_0 = 1 / (np.exp(-k) + NTU_h * g(1))
    theta_h = 1 - NTU_h * Delta_0 * g(xi)
    theta_c = theta_h - Delta_0 * np.exp(-k * xi)

    T_h = T_c_in + theta_h * (T_h_in - T_c_in)
    T_c = T_c_in + theta_c * (T_h_in - T_c_in)
    return T_h, T_c


def steady_rhs(x, y, params, props, geom):
    """Right-hand side of the steady two-ODE system for solve_bvp / solve_ivp.

    Eq 6 (from Eq 1-4):
      dT_h/dx = -U'(T_h, T_c) (T_h - T_c) / (mdot_h cp_h(T_h))
      dT_c/dx = -U'(T_h, T_c) (T_h - T_c) / (mdot_c cp_c(T_c))

    Parameters
    ----------
    x : ndarray, shape (m,)
        Mesh positions [m] (unused: the system is autonomous, but solve_bvp
        passes it).
    y : ndarray, shape (2, m)
        y[0] = T_h, y[1] = T_c [K] at each mesh point. solve_ivp (shooting)
        passes shape (2,); write the function so both shapes work.
    params : dict
        Uses mdot_h, mdot_c (and what correlations.conductances needs).
    props : dict
        Surrogates; possibly continuation copies (lam < 1).
    geom : dict
        From correlations.exchanger_geometry.

    Returns
    -------
    dydx : ndarray, same shape as y
        [dT_h/dx, dT_c/dx] [K/m].
    """

    T_h = y[0]
    T_c = y[1]

    a_h, a_c = correlations.conductances(
        T_,h, T_c, params["mdot_h"], params["mdot_c"], params, props, geom
    )

    U_prime = correlations.overall_conductance(a_h, a_c)
    q_prime = U_prime * (T_h - T_c) # local heat flow per unit length

    cp_h = properties.evaluate(props["h"], "cp", T_h)
    cp_c = properties.evaulate(props["c"], "cp", T_c)

    dT_h_dx = -q_prime / (params["mdot_h"] * cp_h)
    dT_c_dx = -q_prime / (params["mdot_c"] * cp_c)
    return np.vstack((dT_h_dx, dT_c_dx))


def steady_bc(ya, yb, T_h_in, T_c_in):
    """Boundary residuals for solve_bvp.

    Eq 7-8: T_h(0) = T_h_in (hot enters at x = 0), T_c(L) = T_c_in (cold
    enters at x = L).

    Parameters
    ----------
    ya : ndarray, shape (2,)
        [T_h(0), T_c(0)] [K].
    yb : ndarray, shape (2,)
        [T_h(L), T_c(L)] [K].
    T_h_in, T_c_in : float
        Inlet temperatures [K].

    Returns
    -------
    residual : ndarray, shape (2,)
        [T_h(0) - T_h_in, T_c(L) - T_c_in] [K].
    """

    return np.array([ya[0] - T_h_in, yb[1] - T_c_in])


def initial_guess(x, params, props, geom):
    """Initial guess for solve_bvp: analytical profiles at mean properties.

    Section 5.1: the analytical profile Eq 11 evaluated with mean properties.
    Using the surrogates' mean values (phi_bar of Eq 23) makes this guess the
    exact solution of the lambda = 0 continuation problem.

    Parameters
    ----------
    x : ndarray, shape (m,)
        Initial mesh [m].
    params : dict
        Uses L, T_h_in, T_c_in, mdot_h, mdot_c.
    props : dict
        Surrogates.
    geom : dict
        From correlations.exchanger_geometry.

    Returns
    -------
    y_guess : ndarray, shape (2, m)
        [T_h, T_c] at the mesh points [K].
    """

    props0 = {}
    for side in ("h", "c"):
        props0["side"] = properties.continuation_surrogate(props["side"], 0,0)

    T_bar_h = props0["h"]["T_bar"]
    T_bar_c = props0["c"]["T_bar"]

    a_h, a_c = correlations.conductances(
        T_bar,h, T_bar_c, params["mdot_h"], params["mdot_c"], params, props, geom
    )

    U_prime = correlations.overall_conductances(a_h, a_c)
    C_h = params["mdot_h"] * properties.evaulate(props0["h"], "cp", T_bar_h)
    C_c = params["mdot_c"] * properties.evaulate(props0["c"], "cp", T_bar_c)

    T_h, T_c = analytical_profiles(
        x, params["L"], params["T_h_in"], params["T_c_in"], C_h, C_c, U_prime
    )

    return np.vstack((T_h, T_c))


def solve_steady(params, props):
    """Solve the steady variable-property BVP by collocation.

    Section 5.1: solve_bvp on Eq 6 with boundary conditions Eq 8, starting
    from the analytical guess. If it fails, use property continuation (Eq 23):
    solve lambda = 0, 0.25, ..., 1 in turn, each from the previous solution.

    Parameters
    ----------
    params : dict
        Uses L, T_h_in, T_c_in, mdot_h, mdot_c, bvp_tol, bvp_max_nodes,
        continuation_lams, n_out (plus geometry and model options).
    props : dict
        Surrogates from properties.build_surrogates.

    Returns
    -------
    steady : dict
        Steady result dict (see module docstring), method "collocation".

    Raises
    ------
    RuntimeError
        If the solve fails even with continuation. Report sol.message; do not
        retry with a looser tolerance (CLAUDE.md).
    """

    # Get geometry and boundary conditions
    geom = correlations.exhcanger_geometry(params)
    L = params["L"]
    T_h_in = params["T_h_in"]
    T_c_in = params["T_c_in"]

    # Check continuation_lamdha is valid
    if len(params["continuation_lams"]) == 0 or params["continuation_lams"][-1] != 1.0:
        raise ValueError("continuation_lams must be non-empty and end at 1.0")

    # Solve the bvp for given properties and mesh
    def collocate(props_lam, x_mesh, y_mesh):
        # Define rhs and lhs of the equation
        def rhs(x,y):
            return steady_rhs(x,y,params,props_lam, geom)

        def bc(ya,yb):
            return steady_bc(ya, yb, T_h_in, T_c_in)

        # Find solution to bvp
        sol = solve_bvp(
            rhs, 
            bc,
            x_mesh,
            y_mesh,
            tol = params["bvp_tol"], # tolerance
            max_nodes = params["bvp_max_nodes"]
        )

        return sol

    x0 = np.linspace(0.0, L)
    y0 = initial_guess(x0, params, props, geom)

    # First, try full, temperature-dependent properties
    try:
        solution = collocate(props, x0, y0)
        converged = solution.status == 0
    except ValueError:
        converged = False # solution failed to converge 

    # If the solution failed to converged, switch to continuation model
    if not converged:
        x_mesh = x0
        y_mesh = y0

        for lam in params["continuation_lams"]:
            # Build property set for current parameters
            props_lam = {
                side: properties.continuation_surrogate(props[side], lam)
            }

            # Solve using current properties and previous mesh & solution
            try:
                solution = collocate(props_lam, x_mesh, y_mesh)
            except ValueError as err:
                raise RuntimeError(
                    f"Steady State BVP: property range error at "
                    f"lambda = {lam}: {err}"
                ) from err

            if solution.status != 0:
                raise RuntimeError(
                    f"Steady State BVP did not converge at "
                    f"lambda = {lam}: {solution.message}"
                )

            # Update mesh
            x_mesh = solution.x
            y_mesh = solution.y

        x = np.linspace(0.0, L, params["n_out"])
        T_h, T_c = sol.sol(x)
        a_h, a_c = correlations.conductances(
            T_h, T_c, params["mdot_h"], params["mdot_c"], params, props, geom
        )
        U_prime = correlations.overall_conductance(a_h, a_c)
        T_w = wall_temperature(T_h, T_c, a_h, a_c)
        return {
            "x": x,
            "T_h": T_h,
            "T_c": T_c,
            "T_w": T_w,
            "a_h": a_h,
            "a_c": a_c,
            "U_prime": U_prime,
            "sol": sol,
            "method": "collocation"
        }

def _integrate_from_x0(sigma, params, props, geom, dense_output, stop_if_cold_too_low):
    """Integrate Eq 6 from x = 0 with y(0) = [T_h_in, sigma] (shooting helper).

    Shared by shooting_residual (no dense output) and solve_steady_shooting
    (dense output for the final profiles). With stop_if_cold_too_low, a
    terminal event stops the integration as soon as the cold stream falls to
    T_c_in. The cold stream only cools along +x, so if that happens before
    x = L the trial sigma is too low (T_c(L) < T_c_in), and the integration
    must stop there: left running, T_c would fall out of the surrogate range
    and properties.check_range would raise. The event sits at T_c_in, not at
    the bottom of the surrogate range, because the integrator's trial steps
    overshoot an event by up to a few K before the crossing is detected; the
    fit margin below T_c_in absorbs that overshoot.

    The final integration at the root sigma* turns the event off: there
    T_c(L) = T_c_in to within the root tolerance, so the event can fire a
    hair before x = L by round-off alone, and the profile is wanted all the
    way to x = L.

    Parameters
    ----------
    sigma : float
        Trial cold outlet temperature T_c(0) [K].
    params, props, geom : dict
        As for steady_rhs.
    dense_output : bool
        Whether to return a continuous interpolant (sol.sol).
    stop_if_cold_too_low : bool
        Whether to stop at the event described above.

    Returns
    -------
    sol : scipy OdeResult
        sol.status == 1 means the event fired before x = L.
    """

    T_c_in = params["T_c_in"]

    # Terminal event when T_c reaches bottom of cold surrogate model's range
    def cold_reaches_inlet_temperature(x, y):
        return y[1] - T_c_in

    cold_reaches_inlet_temperature.terminal = True
    cold_reaches_inlet_temperature.direction = -1

    return solve_ivp(
        lambda x, y: steady_rhs(x, y, params, props, geom),
        (0.0, params["L"]),
        [params["T_h_in"], sigma],
        method="LSODA",
        rtol=1.0e-10,
        atol=1.0e-8,
        events=cold_reaches_inlet_temperature if stop_if_cold_too_low else None,
        dense_output=dense_output,
    )


def shooting_residual(sigma, params, props, geom):
    """Shooting residual for a trial cold-outlet temperature.

    Section 5.2, Eq 24: integrate Eq 6 from x = 0 with
    y(0) = [T_h_in, sigma] and return r(sigma) = T_c(L; sigma) - T_c_in.

    Parameters
    ----------
    sigma : float
        Trial cold outlet temperature T_c(0) [K], T_c_in < sigma < T_h_in.
    params : dict
        Uses L, T_h_in, T_c_in, mdot_h, mdot_c.
    props : dict
        Surrogates.
    geom : dict
        From correlations.exchanger_geometry.

    Returns
    -------
    r : float
        Residual [K]. Negative means sigma is too low (T_c falls below T_c_in
        before x = L).
    """
    sol = _integrate_from_x0(
        sigma, params, props, geom, dense_output=False, stop_if_cold_too_low=True
    )
    if sol.status == 1:
        # The cold stream reached T_c_in at x_e < L, so sigma is too low. Return
        # how far below T_c_in it would be at x = L if it kept cooling at the
        # slope it has at x_e: negative (the right sign for brentq), and it goes
        # to zero as x_e -> L, so the residual stays continuous at the root.
        x_e = sol.t_events[0][0]
        y_e = sol.y_events[0][0]
        slope_c = steady_rhs(x_e, y_e, params, props, geom)[1]  # dT_c/dx < 0
        return float(slope_c * (params["L"] - x_e))
    return float(sol.y[1, -1] - params["T_c_in"])


def solve_steady_shooting(params, props):
    """Solve the steady problem by shooting (cross-check for collocation).

    Section 5.2: find sigma with r(sigma) = 0 (Eq 24) using brentq on the
    bracket T_c_in < sigma < T_h_in, then integrate once more for profiles.

    Parameters
    ----------
    params : dict
        As for solve_steady.
    props : dict
        Surrogates.

    Returns
    -------
    steady : dict
        Steady result dict (see module docstring), method "shooting",
        "sol" = the final dense solve_ivp result.
    """
    # Define geometry and boundary conditions
    geom = correlations.exchanger_geometry(params)
    L = params["L"]
    T_h_in = params["T_h_in"]
    T_c_in = params["T_c_in"]

    eps = 1.0e-6  # keep the bracket strictly inside (T_c_in, T_h_in) [K]
    sigma_star = brentq(
        shooting_residual,
        T_c_in + eps,
        T_h_in - eps,
        args=(params, props, geom),
        xtol=1.0e-10,
    )

    sol = _integrate_from_x0(
        sigma_star, params, props, geom, dense_output=True, stop_if_cold_too_low=False
    )
    if sol.status != 0:
        raise RuntimeError(f"Shooting re-integration failed at the root: {sol.message}")

    x = np.linspace(0.0, L, params["n_out"])
    T_h, T_c = sol.sol(x)
    a_h, a_c = correlations.conductances(
        T_h, T_c, params["mdot_h"], params["mdot_c"], params, props, geom
    )
    U_prime = correlations.overall_conductance(a_h, a_c)
    T_w = wall_temperature(T_h, T_c, a_h, a_c)
    return {
        "x": x, "T_h": T_h, "T_c": T_c, "T_w": T_w,
        "a_h": a_h, "a_c": a_c, "U_prime": U_prime,
        "sol": sol, "method": "shooting",
    }


def energy_balance(steady, params, props):
    """Heat duty computed three independent ways (energy closure).

    Eq 16:
      Q_h   = mdot_h [hhat_h(T_h(0)) - hhat_h(T_h(L))] = Heat lost by hot stream
      Q_c   = mdot_c [hhat_c(T_c(0)) - hhat_c(T_c(L))] = Heat gained by cold stream
      Q_int = integral from 0 to L of U'(T_h - T_c) dx = Heat transferred between the streams

    By energy conservation, 
      Q_h = Q_c = Q_int ideally

    Parameters
    ----------
    steady : dict
        Result of solve_steady or solve_steady_shooting.
    params : dict
        Uses mdot_h, mdot_c, L.
    props : dict
        Surrogates (properties.enthalpy is used for hhat).

    Returns
    -------
    duty : dict
        "Q_h", "Q_c", "Q_int" [W]; "closure_hc" = |Q_h - Q_c|/Q_h and
        "closure_int" = |Q_h - Q_int|/Q_h [-].
    """

    T_h = steady["T_h"]
    T_c = steady["T_c"]
    mdot_h = params["mdot_h"]
    mdot_c = params["mdot_c"]

    # Hot stream loses enthalpy between its inlet (x = 0) and outlet (x = L);
    # the cold stream gains it between its inlet (x = L) and outlet (x = 0).
    Q_h = mdot_h * (
        properties.enthalpy(props["h"], T_h[0]) - properties.enthalpy(props["h"], T_h[-1])
    )
    Q_c = mdot_c * (
        properties.enthalpy(props["c"], T_c[0]) - properties.enthalpy(props["c"], T_c[-1])
    )

    # Calculate Q_int by integrated local heat transfer
    # If method == collocation, we have a continuous numerical solution
    if steady["method"] == "collocation":
        geom = correlations.exchanger_geometry(params)
        sol = steady["sol"]

        # Local heat transfer per unit length
        def q_prime(x):
            T_h_x, T_c_x = sol.sol(x)
            a_h, a_c = correlations.conductances(
                T_h_x, T_c_x, mdot_h, mdot_c, params, props, geom
            )
            return correlations.overall_conductance(a_h, a_c) * (T_h_x - T_c_x)

        Q_int, _ = quad(q_prime, 0.0, params["L"], epsrel=1.0e-10, limit=200)
    # If method == shooting, we have a discrete mesh instead
    else:
        # Local heat transfer per unit length
        q_line = steady["U_prime"] * (T_h - T_c)

        Q_int = simpson(q_line, x=steady["x"])

    Q_h = float(Q_h)
    Q_c = float(Q_c)
    Q_int = float(Q_int)

    return {
        "Q_h": Q_h,
        "Q_c": Q_c,
        "Q_int": Q_int,
        "closure_hc": abs(Q_h - Q_c) / Q_h,
        "closure_int": abs(Q_h - Q_int) / Q_h,
    }
