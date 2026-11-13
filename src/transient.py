"""Transient model: method of lines on a finite-volume grid.

Math doc: Section 6.1 (governing PDEs, Eq 27-29), Section 6.2 (time scales and
stiffness, Eq 30-31), Section 6.3 (finite-volume discretization, Eq 32-34, state
ordering and sparsity), Section 6.4 (quasi-steady gas fallback, Eq 36) and
Section 6.5 (upset scenarios, Eq 37-38).

The core is split into N cells of width dx = L/N. Each cell has three states
(hot gas, cold gas, metal), giving 3N stiff ODEs integrated with
solve_ivp(method="BDF") and a sparse Jacobian pattern (CLAUDE.md: no explicit
methods). Advection uses donor-cell (upwind) fluxes in enthalpy form, which
conserve energy exactly even with variable cp (Eq 32-33):
  hot cell i  receives from cell i-1, with T_h,0   = T_h_in(t)  (flow +x)
  cold cell i receives from cell i+1, with T_c,N+1 = T_c_in(t)  (flow -x)
Never use central or downwind differences for advection.

Note on CLAUDE.md: it writes the same upwind stencil in temperature form,
mdot*cp*(T[i] - T[i-1])/dx. The two forms agree at constant cp. Following the
team's decision, this module uses the math doc's enthalpy form; CLAUDE.md
should be updated to match.

Note on the proposal: its cold-stream transient equation has the wrong sign
(+mdot_c cp_c dT_c/dx and -a_c (T_w - T_c)), and it applies the stencil
(T_i - T_i-1)/dx to both streams, which is downwind for the cold stream.
Follow Eq 28 and Eq 33 (and CLAUDE.md) instead.

Imports: numpy, scipy.integrate.solve_ivp, scipy.sparse (Jacobian pattern),
scipy.optimize.root, src.properties, src.correlations and src.steady (initial
guess). metrics.py consumes the result dict.

State vector (Section 6.3), shape (3N,), ordered cell by cell:
  y = [T_h,1, T_c,1, T_w,1, T_h,2, T_c,2, T_w,2, ..., T_h,N, T_c,N, T_w,N]
so T_h = y[0::3], T_c = y[1::3], T_w = y[2::3]. Cell 1 (Python index 0) is at
x = 0, where the hot stream enters and the cold stream leaves.

The result dict (returned by simulate)
--------------------------------------
    "t"        (n_t,)    saved times [s]
    "x"        (N,)      cell-centre positions [m]
    "T_h","T_c","T_w"            (n_t, N)  temperatures [K]
    "dTh_dt","dTc_dt","dTw_dt"   (n_t, N)  rates from the RHS at saved times [K/s]
                                           (dTh_dt, dTc_dt are None for the
                                           quasi-steady model)
    "T_h_in","T_c_in","mdot_h","mdot_c"  (n_t,)  boundary histories [K], [kg/s]
    "sol"      the solve_ivp result (or list of results if the run was split)
    "N"        number of cells
    "model"    "full" or "quasi_steady"
"""

import numpy as np
from scipy import sparse
from scipy.integrate import solve_ivp
from scipy.optimize import root

import correlations, properties, steady


def time_scales(params, props):
    """Characteristic time scales and stiffness estimate.

    Eq 30-31:
      tau_res = rho A_ff L / mdot           gas residence time (each side)
      tau_g   = rho A_ff cp / a             gas-to-metal equilibration (each side)
      tau_m   = C'_m / (a_h + a_c)          local metal relaxation
      kappa   = C'_m / [(rho A_ff cp)_h + (rho A_ff cp)_c]   metal/gas capacity ratio
      S      >~ N (1 + kappa)               stiffness ratio

    Parameters
    ----------
    params : dict
        Uses L, mdot_h, mdot_c, N (plus geometry and metal properties).
    props : dict
        Surrogates; evaluated at their mean properties.

    Returns
    -------
    scales : dict
        "tau_res_h", "tau_res_c", "tau_g_h", "tau_g_c", "tau_m" [s],
        "kappa" [-], "stiffness" [-], "t_response" = (1 + kappa) * max tau_res [s]
        (order of the whole-exchanger response time, Section 6.2).
    """

    # 1. Calculate exhcanger geometry
    geom = correlations.exchanger_geometry(params)

    # 2. Find representative properties (surrogate mean values and properties)
    props0 = {
        side: properties.continuation_surrogate(props[side], 0.0)
        for side in ("h", "c")
    }

    # Representative temperatures are the midpoint temperatures
    # associated with each side's property surrogate.
    T_bar_h = props0["h"]["T_bar"]
    T_bar_c = props0["c"]["T_bar"]

    # Evaluate representative gas density and heat capacity.
    # These are used to calculate the thermal energy stored within each gas passage.
    rho_h = properties.evaluate(props0["h"], "rho", T_bar_h)
    rho_c = properties.evaluate(props0["c"], "rho", T_bar_c)

    cp_h = properties.evaluate(props0["h"], "cp", T_bar_h)
    cp_c = properties.evaluate(props0["c"], "cp", T_bar_c)

    # 3. Calculate hot and cold side conductance
    a_h, a_c = correlations.conductances(
        T_bar_h,
        T_bar_c,
        params["mdot_h"],
        params["mdot_c"],
        params,
        props0,
        geom,
    )

    # 4. Calculate gas residence times
    # tau_res = (gas density * free flow area * length) / mass flow
    tau_res_h = (
        rho_h * geom["h"]["A_ff"] * params["L"] / params["mdot_h"]
    )

    tau_res_c = (
        rho_c * geom["h"]["A_ff"] * params["L"] / params["mdot_c"]
    )

    # 5. Calculate gas to metal equilibriation time
    # tau_g = (gas thermal capacity per unit length) / conductance
    tau_g_h = (
        rho_h * geom["h"]["A_ff"] * cp_h / a_h
    )

    tau_g_c = (
        rho_c * geom["c"]["A_ff"] * cp_c / a_c
    )

    # 6. Calculate metal relaxation time
    C_m = geom["C_m"]
    tau_m = C_m / (a_h + a_c)

    # 7. Calculate metal to gas thermal capacity ratio
    # C'_gas = rho * A_ff * cp
    gas_capacity_h = rho_h * geom["h"]["A_ff"] * cp_h
    gas_capacity_c = rho_c * geom["c"]["A_ff"] * cp_c

    kappa = C_m / (gas_capacity_h + gas_capacity_c)

    # 8. Estimate system stiffness
    # S approximatly N * (1 + kappa)
    stiffness = params["N"] * (1.0 + kappa)

    # 9. Estimate whole exchanger response time
    t_response = (1.0 + kappa) * max(tau_res_h, tau_res_c)

    # 10. Return all characteristic time scales
    return {
        "tau_res_h": float(tau_res_h),
        "tau_res_c": float(tau_res_c),
        "tau_g_h": float(tau_g_h),
        "tau_g_c": float(tau_g_c),
        "tau_m": float(tau_m),
        "kappa": float(kappa),
        "stiffness": float(stiffness),
        "t_response": float(t_response),
    }


def cell_centers(L, N):
    """Cell-centre positions of the finite-volume grid.

    Section 6.3: N cells of width dx = L/N.

    Parameters
    ----------
    L : float
        Core length [m].
    N : int
        Number of cells.

    Returns
    -------
    x : ndarray, shape (N,)
        Cell centres x_i = (i - 1/2) dx for i = 1..N [m].
    dx : float
        Cell width [m].
    """

    # Total exchanger length is divided evenly into N cells
    dx = L / N

    # Center position of each cell is in the middle
    x = (np.arrange(N) + 0.5) * dx

    return x, dx


def pack_state(T_h, T_c, T_w):
    """Interleave the three temperature fields into the state vector.

    Section 6.3 ordering: [T_h,1, T_c,1, T_w,1, ..., T_h,N, T_c,N, T_w,N].

    Parameters
    ----------
    T_h, T_c, T_w : ndarray, shape (N,)
        Cell temperatures [K].

    Returns
    -------
    y : ndarray, shape (3N,)
        State vector [K].
    """

    # Number of finite volume cells
    N = len(T_h)

    y = np.empty(3*N)
    y[0::3] = T_h
    y[1::3] = T_c
    y[2::3] = T_w

    return y


def unpack_state(y):
    """Split the state vector into the three temperature fields.

    Parameters
    ----------
    y : ndarray, shape (3N,) or (3N, n_t)
        State vector, or solve_ivp's sol.y with one column per saved time.

    Returns
    -------
    T_h, T_c, T_w : ndarray, each shape (N,) or (N, n_t)
        Hot, cold and wall temperatures [K]. Views into y (do not modify).
    """

    # Opposite of pack_state()
    T_h = y[0::3]
    T_c = y[1::3]
    T_w = y[2::3]

    return T_h, T_c, T_w


def jacobian_sparsity(N, axial_conduction):
    """Sparsity pattern of d(dy/dt)/dy for solve_ivp.

    Section 6.3: with cell-by-cell ordering each equation touches only its own
    cell and one neighbour, so the Jacobian is banded. Passing the pattern via
    jac_sparsity lets BDF build the Jacobian with a few grouped finite
    differences instead of 3N separate ones.

    Parameters
    ----------
    N : int
        Number of cells.
    axial_conduction : bool
        Whether the wall equation couples neighbouring wall cells (Eq 34).

    Returns
    -------
    pattern : scipy.sparse matrix, shape (3N, 3N)
        1 where an entry may be nonzero, 0 elsewhere.
    """

    # Initialize empty sparse matrix (each row = ODE, each column = state variable)
    pattern = sparse.lil_matrix((3 * N, 3 * N), dtype=int)

    # Loop through each cell
    for i in range(N):
        # State vector index
        h = 3 * i      # Hot
        c = 3 * i + 1  # Cold
        w = 3 * i + 2  # Wall

        # Hot gas energy equation (flow in +x direction)
        pattern[h,h] = 1 # Its own cell temp
        pattern[h,w] = 1 # Its own wall temp

        if i > 0 :
            h_upstream = 3 * (i-1)
            pattern[h, h_upstream] = 1 # Previous cell's hot gas temperature

        # Cold gas energy equation (flow in -x direction)
        pattern[c,c] = 1
        pattern[c,w] = 1

        if i < N - 1:
            c_upstream = 3 * (i-1) + 1
            pattern[c,c_upstream] = 1 # next cell's cold gas temperature

        # Metal wall energy equation
        pattern[w,h] = 1
        pattern[w,c] = 1
        pattern[w,w] = 1

        # Optional axial wall conduction
        if axial_conduction:
            if i > 0:
                w_left = 3 * (i - 1) + 2
                pattern[w, w_left] = 1

            if i < N - 1:
                w_right = 3 * (i + 1) + 2
                pattern[w, w_right] = 1

    # LIL is convient for constructing the matrix
    # CSR is more efficient for sparse numerical operations
    return pattern.tocsr()

# Test script for jacobian_sparsity()
# import matplotlib.pyplot as plt
# N_test = 5
# pattern_no_cond = jacobian_sparsity(N_test, False)
# pattern_cond = jacobian_sparsity(N_test, True)
# fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# axes[0].spy(pattern_no_cond, markersize=10)
# axes[0].set_title("Without Axial Conduction")
# axes[0].set_xlabel("State variable")
# axes[0].set_ylabel("ODE equation")
# axes[1].spy(pattern_cond, markersize=10)
# axes[1].set_title("With Axial Conduction")
# axes[1].set_xlabel("State variable")
# axes[1].set_ylabel("ODE equation")
# plt.tight_layout()
# plt.show()

def smooth_ramp(s, sharpness):
    """Smoothed unit ramp psi(s) for the feed temperature ramp.

    Eq 37 uses a smoothed unit ramp psi, which the math doc does not define
    (flagged). Suggested form, the difference of two softplus functions:
      psi(s) = [softplus(omega s) - softplus(omega (s - 1))] / omega,
      softplus(z) = ln(1 + e^z),
    which rises from 0 (s << 0) to 1 (s >> 1) with corners of width ~1/omega.
    Its slope psi'(s) = sigmoid(omega s) - sigmoid(omega (s - 1)) peaks at
    s = 1/2 with value tanh(omega/4) <= 1, so the smoothing never exceeds the
    nominal ramp rate (a cubic smoothstep would overshoot it by 1.5 times).

    Parameters
    ----------
    s : float or ndarray
        Scaled time (t - t_start)/t_r [-].
    sharpness : float
        omega [-], e.g. 20.

    Returns
    -------
    psi : ndarray, same shape as s
        Ramp fraction, between 0 and 1 [-].
    """
    # 1. softplus(z) = np.logaddexp(0, z) (overflow-safe for large z).
    # 2. psi = (softplus(omega s) - softplus(omega (s - 1))) / omega.
    # Pitfall: psi(0) = ln(2)/omega, not 0, and psi is already slightly
    #    positive before s = 0. Start the ramp several corner widths after
    #    t = 0 (ramp_t_start >~ 5 ramp_t/omega) so the initial state is steady.
    # Check: psi(-10) ~ 0, psi(10) ~ 1, psi(0.5) = 0.5, the maximum slope is
    #    tanh(omega/4), and psi is monotone.
    raise NotImplementedError


def boundary_conditions(t, params, scenario):
    """Inlet temperatures and flows at time t for an upset scenario.

    Section 6.5. "feed_ramp" (Eq 37):
      T_h,in(t) = T_h_in + ramp_dT * psi((t - ramp_t_start)/ramp_t)
    "cold_trip" (Eq 38), for t >= trip_t_start:
      mdot_c(t) = mdot_c [phi + (1 - phi) exp(-(t - trip_t_start)/tau_v)]
    "none": everything constant (used for steady holds and tests).

    Parameters
    ----------
    t : float or ndarray, shape (n_t,)
        Time [s].
    params : dict
        Uses T_h_in, T_c_in, mdot_h, mdot_c and the ramp_* / trip_* keys.
    scenario : str
        "none", "feed_ramp" or "cold_trip".

    Returns
    -------
    T_h_in, T_c_in : float or ndarray, same shape as t
        Inlet temperatures [K].
    mdot_h, mdot_c : float or ndarray, same shape as t
        Mass flows per passage [kg/s].

    Raises
    ------
    ValueError
        For an unknown scenario name.
    """
    # 1. Start from the constant base values (broadcast to the shape of t,
    #    e.g. with np.full_like(t, value, dtype=float), if t is an array).
    # 2. "feed_ramp": add ramp_dT * smooth_ramp((t - ramp_t_start)/ramp_t,
    #    ramp_sharpness) to T_h_in.
    # 3. "cold_trip": mdot_c(t) as above for t >= trip_t_start, unchanged
    #    before. Use np.where so array t works.
    # 4. Unknown scenario: raise ValueError (catches typos like "ramp").
    # Pitfall: Eq 38 has a slope jump at trip_t_start (the math doc does not
    #    smooth it, flagged). simulate() splits the integration there so BDF
    #    never steps across the kink.
    # Pitfall: the nominal ramp rate is r = ramp_dT/ramp_t. ALPEMA limits are
    #    measured at the inlet nozzles too, so r itself must satisfy them
    #    (metrics.check_alpema).
    # Check: at t = 0 every scenario returns the base values (to ~1e-4 K for
    #    the ramp); long after the ramp T_h_in -> T_h_in + ramp_dT; long after
    #    the trip mdot_c -> phi * mdot_c.
    raise NotImplementedError


def transient_rhs(t, y, params, props, geom, scenario):
    """Method-of-lines right-hand side dy/dt for the full 3N model.

    Eq 32-34, for i = 1..N:
      M_h,i dT_h,i/dt = mdot_h [hhat_h(T_h,i-1) - hhat_h(T_h,i)] - a_h,i dx (T_h,i - T_w,i)
      M_c,i dT_c,i/dt = mdot_c [hhat_c(T_c,i+1) - hhat_c(T_c,i)] + a_c,i dx (T_w,i - T_c,i)
      M_w   dT_w,i/dt = (k_m A_m/dx)(T_w,i+1 - 2 T_w,i + T_w,i-1)
                        + a_h,i dx (T_h,i - T_w,i) - a_c,i dx (T_w,i - T_c,i)
    with M_h,i = rho_h,i A_ff,h cp_h,i dx, M_c,i = rho_c,i A_ff,c cp_c,i dx,
    M_w = C'_m dx; inlets T_h,0 = T_h_in(t), T_c,N+1 = T_c_in(t); adiabatic
    wall ends via ghost cells T_w,0 = T_w,1, T_w,N+1 = T_w,N.

    Parameters
    ----------
    t : float
        Time [s].
    y : ndarray, shape (3N,)
        State vector [K] (see module docstring for the ordering).
    params : dict
        Uses L, axial_conduction, k_m (plus what boundary_conditions and
        correlations.conductances need).
    props : dict
        Surrogates. Never CoolProp.
    geom : dict
        From correlations.exchanger_geometry (built once by the caller).
    scenario : str
        Upset scenario name.

    Returns
    -------
    dydt : ndarray, shape (3N,)
        Time derivatives [K/s], same ordering as y.
    """
    # 1. N = y.size // 3; dx = L/N; T_h, T_c, T_w = unpack_state(y). Do not
    #    modify y in place (solve_ivp reuses it).
    # 2. T_h_in, T_c_in, mdot_h, mdot_c = boundary_conditions(t, params, scenario).
    # 3. Donor (upstream) temperatures, each shape (N,):
    #      hot flows +x:  T_h_up = [T_h_in, T_h,1, ..., T_h,N-1]
    #                     (np.concatenate(([T_h_in], T_h[:-1])))
    #      cold flows -x: T_c_up = [T_c,2, ..., T_c,N, T_c_in]
    #                     (np.concatenate((T_c[1:], [T_c_in])))
    #    Getting either direction wrong turns the exchanger into parallel flow
    #    or makes it unstable.
    # 4. Advective heat inflow per cell [W], enthalpy form (Eq 32-33):
    #      adv_h = mdot_h (hhat_h(T_h_up) - hhat_h(T_h))
    #      adv_c = mdot_c (hhat_c(T_c_up) - hhat_c(T_c))
    #    using properties.enthalpy with each stream's own surrogate.
    # 5. Exchange with the metal per cell [W]:
    #      a_h, a_c = correlations.conductances(T_h, T_c, mdot_h, mdot_c, params,
    #                                           props, geom)
    #    (a_c uses T_c and the CURRENT mdot_c(t), which falls during a trip).
    #      Q_hw = a_h dx (T_h - T_w)   heat from hot gas into the metal
    #      Q_wc = a_c dx (T_w - T_c)   heat from the metal into cold gas
    # 6. Axial conduction [W] (only if params["axial_conduction"]): pad T_w
    #    with ghost values, T_pad = [T_w,1, T_w, T_w,N] (shape N+2), then
    #    cond = (k_m A_m/dx)(T_pad[2:] - 2 T_pad[1:-1] + T_pad[:-2]).
    #    Otherwise cond = 0.
    # 7. Cell heat capacities [J/K]: M_h = rho_h A_ff,h cp_h dx with rho_h and
    #    cp_h evaluated at T_h; M_c likewise at T_c; M_w = C_m dx.
    # 8. Rates (Eq 32-34):
    #      dT_h/dt = (adv_h - Q_hw) / M_h
    #      dT_c/dt = (adv_c + Q_wc) / M_c     <- plus sign: cold gas gains heat
    #      dT_w/dt = (cond + Q_hw - Q_wc) / M_w
    # 9. Return pack_state(dT_h/dt, dT_c/dt, dT_w/dt).
    # Pitfall: the cold-stream sign has been written wrong before (CLAUDE.md).
    #    In Eq 28 the advection term is -mdot_c cp_c dT_c/dx because the cold
    #    gas flows in -x; in the finite-volume form that becomes inflow from
    #    cell i+1, and the metal term enters with a PLUS sign.
    # Pitfall: no Python loops over cells; everything above is whole-array
    #    numpy, which matters because BDF calls this thousands of times.
    # Check: (a) uniform state T_h = T_c = T_w = T_h_in = T_c_in gives all
    #    zeros. (b) Energy identity at a random in-range state:
    #    sum(M_h dT_h/dt + M_c dT_c/dt + M_w dT_w/dt)
    #      = mdot_h [hhat_h(T_h_in) - hhat_h(T_h,N)] + mdot_c [hhat_c(T_c_in) - hhat_c(T_c,1)]
    #    to round-off (the exchange terms cancel pairwise and the ghost cells
    #    are adiabatic). This catches a wrong cold sign without integrating.
    #    (c) The pattern from jacobian_sparsity covers the FD Jacobian.
    raise NotImplementedError


def steady_to_cells(steady_result, N):
    """Sample a steady BVP solution at the cell centres (a guess only).

    Parameters
    ----------
    steady_result : dict
        From steady.solve_steady (fields x, T_h, T_c, T_w).
    N : int
        Number of cells.

    Returns
    -------
    y : ndarray, shape (3N,)
        State vector [K].
    """
    # 1. L = steady_result["x"][-1]; x_cells, _ = cell_centers(L, N).
    # 2. np.interp each of T_h, T_c, T_w from the fine steady grid onto x_cells.
    # 3. Return pack_state(...).
    # Pitfall: this is NOT a steady state of the discrete model. First-order
    #    upwinding has its own steady state, O(dx) away from the BVP, and at
    #    the two inlet cells the mismatch gives dT/dt ~ u T'/2, which is O(1)
    #    and does not shrink with N (flagged; the math doc says to start from
    #    the BVP). Use the result only as the guess for discrete_steady_state.
    # Check: transient_rhs at this state is O(dx) in interior cells and
    #    largest in hot cell 1 and cold cell N.
    raise NotImplementedError


def discrete_steady_state(params, props, scenario="none", y_guess=None):
    """Steady state of the discrete 3N model (the correct initial condition).

    Solves F(y) = transient_rhs(0, y) = 0 for the boundary values at t = 0 of
    the given scenario, so that a simulation started here does not move until
    the upset begins (verification check in CLAUDE.md).

    Parameters
    ----------
    params : dict
        Uses N, ivp_rtol, ivp_atol (plus everything transient_rhs needs).
    props : dict
        Surrogates.
    scenario : str, optional
        Scenario whose t = 0 boundary values to use.
    y_guess : ndarray, shape (3N,), optional
        Starting guess. Default: steady_to_cells(steady.solve_steady(params,
        props), N).

    Returns
    -------
    y_ss : ndarray, shape (3N,)
        Discrete steady state [K].

    Raises
    ------
    RuntimeError
        If max |F(y_ss)| is not below the tolerance.
    """
    # 1. geom = correlations.exchanger_geometry(params); N = params["N"].
    # 2. Guess: as described above if y_guess is None.
    # 3. F(y) = transient_rhs(0.0, y, params, props, geom, scenario).
    # 4. Pseudo-transient relaxation: solve_ivp with F held at t = 0 (wrap as
    #    lambda t, y: F(y)), method "BDF", jac_sparsity from
    #    jacobian_sparsity, over ~10 x time_scales(...)["t_response"]. This
    #    is robust because the physics itself relaxes to the steady state.
    # 5. Polish: scipy.optimize.root(F, y_relaxed, method="hybr") (dense
    #    finite-difference Jacobian, fine up to 3N ~ 1000), or
    #    scipy.optimize.least_squares(F, y_relaxed, jac_sparsity=pattern)
    #    for larger N.
    # 6. Check max |F(y_ss)| < ~1e-8 K/s; raise RuntimeError otherwise.
    # Check: tests/test_transient.py::test_transient_stays_at_steady_state;
    #    the gap to steady.solve_steady at the cell centres roughly halves when
    #    N doubles (first-order upwinding).
    raise NotImplementedError


def quasi_steady_gas(T_w, t, params, props, geom, scenario):
    """Gas temperatures for a given wall profile, neglecting gas storage.

    Section 6.4, Eq 36: with the gas accumulation dropped (kappa >> 1), each
    gas equation is a linear first-order ODE with a known source, solved
    exactly cell by cell with properties frozen in the cell:
      T_h,i = T_w,i + (T_h,i-1 - T_w,i) exp(-a_h,i dx/(mdot_h cp_h,i)),  i = 1..N
      T_c,i = T_w,i + (T_c,i+1 - T_w,i) exp(-a_c,i dx/(mdot_c cp_c,i)),  i = N..1

    Parameters
    ----------
    T_w : ndarray, shape (N,)
        Wall temperatures [K].
    t : float
        Time [s] (for the boundary values).
    params, props, geom, scenario
        As for transient_rhs.

    Returns
    -------
    T_h, T_c : ndarray, each shape (N,)
        Gas temperatures [K].
    """
    # 1. Boundary values at t from boundary_conditions; N = T_w.size; dx = L/N.
    # 2. Hot sweep, i = 1 -> N (the direction of hot flow), starting from
    #    T_up = T_h_in: freeze a_h and cp_h at the upstream value T_up (T_h,i
    #    is not known yet), compute NTU_cell = a_h dx/(mdot_h cp_h), then
    #    T_h,i = T_w,i + (T_up - T_w,i) exp(-NTU_cell); T_up = T_h,i.
    # 3. Cold sweep, i = N -> 1 (the direction of cold flow), from T_c_in, the
    #    same way. These sweeps are sequential, so a Python loop is fine here.
    # Pitfall (flagged): Eq 36 is not the no-storage limit of Eq 32-34. The
    #    full donor-cell model gives T_i - T_w = (T_i-1 - T_w)/(1 + NTU_cell)
    #    instead of exp(-NTU_cell), and uses enthalpy instead of cp. Both are
    #    first-order accurate, but comparing this model with the full one then
    #    mixes the kappa assumption with discretization error. For a clean
    #    comparison, use the 1/(1 + NTU_cell) update here (or note the
    #    difference in the report).
    # Check: with T_w uniform and equal to T_h_in, T_h stays at T_h_in; with
    #    very large a, T_h -> T_w and T_c -> T_w cell by cell.
    raise NotImplementedError


def quasi_steady_rhs(t, T_w, params, props, geom, scenario):
    """Wall-only right-hand side of the quasi-steady model.

    Section 6.4: only the N wall temperatures are integrated. The heat
    delivered to the metal in cell i is
      Q_i = mdot_h cp_h (T_h,i-1 - T_h,i) + mdot_c cp_c (T_c,i+1 - T_c,i),
    and M_w dT_w,i/dt = Q_i + axial conduction (Eq 34).

    Parameters
    ----------
    t : float
        Time [s].
    T_w : ndarray, shape (N,)
        Wall temperatures [K].
    params, props, geom, scenario
        As for transient_rhs.

    Returns
    -------
    dTw_dt : ndarray, shape (N,)
        Wall temperature rates [K/s].
    """
    # 1. T_h, T_c = quasi_steady_gas(T_w, t, params, props, geom, scenario).
    # 2. Upstream arrays as in transient_rhs step 3; Q_i from the formula
    #    above (or in enthalpy form, mdot [hhat(T_up) - hhat(T)], to match the
    #    full model).
    # 3. Add the conduction term if axial_conduction (same ghost cells).
    # 4. dTw_dt = (Q + cond) / (C_m dx).
    # Pitfall: the sweeps couple every cell to every upstream cell, so the
    #    Jacobian is dense. Do not pass the banded jacobian_sparsity pattern;
    #    with only N unknowns a dense Jacobian is cheap.
    # Check: at the discrete steady state of this model dTw_dt = 0; with
    #    kappa large, the wall histories match the full model's closely.
    raise NotImplementedError


def simulate(params, props, scenario, t_end, t_eval=None, y0=None, model="full"):
    """Run a transient simulation and collect histories and rates.

    Section 6.3 (full 3N model) or Section 6.4 (quasi-steady wall model),
    integrated with solve_ivp(method="BDF").

    Parameters
    ----------
    params : dict
        Uses N, ivp_rtol, ivp_atol, axial_conduction, trip_t_start, ramp_t
        (plus everything transient_rhs needs).
    props : dict
        Surrogates built for this scenario (properties.build_surrogates(params,
        scenario)), so the ramped inlet stays inside the fitted range.
    scenario : str
        "none", "feed_ramp" or "cold_trip".
    t_end : float
        Final time [s].
    t_eval : ndarray, shape (n_t,), optional
        Times at which to save the solution [s]. Default: an even grid fine
        enough to resolve the upset (e.g. 2001 points).
    y0 : ndarray, optional
        Initial state, shape (3N,) (full) or (N,) (quasi_steady). Default:
        discrete_steady_state(params, props, scenario) (wall part only for
        quasi_steady).
    model : str, optional
        "full" or "quasi_steady".

    Returns
    -------
    result : dict
        Result dict described in the module docstring.

    Raises
    ------
    RuntimeError
        If solve_ivp fails (report sol.message).
    """
    # 1. geom = correlations.exchanger_geometry(params); N = params["N"];
    #    x, dx = cell_centers(L, N).
    # 2. y0 as described above.
    # 3. rhs(t, y) = transient_rhs(t, y, params, props, geom, scenario) for
    #    "full", or quasi_steady_rhs(...) for "quasi_steady".
    # 4. solve_ivp(rhs, (0, t_end), y0, method="BDF", t_eval=t_eval,
    #    rtol=ivp_rtol, atol=ivp_atol, dense_output=True,
    #    jac_sparsity=jacobian_sparsity(N, axial_conduction) for "full" only).
    #    Never use RK45 or another explicit method: the system is stiff.
    # 5. Scenario-specific step control:
    #    - "cold_trip": integrate [0, trip_t_start] and then
    #      [trip_t_start, t_end] as two calls, the second starting from the
    #      first's final state, so BDF never steps across the kink in Eq 38.
    #      Concatenate the saved results.
    #    - "feed_ramp": nothing happens before the ramp, so BDF grows its
    #      step and can jump straight over a short ramp. Set
    #      max_step <= ramp_t/20.
    # 6. If a solve fails, raise RuntimeError with sol.message.
    # 7. Histories: T_h, T_c, T_w = unpack_state(sol.y), each transposed to
    #    shape (n_t, N). For "quasi_steady", T_w = sol.y.T and T_h, T_c come
    #    from quasi_steady_gas at each saved time.
    # 8. Rates from the RHS at the saved times, never np.gradient of the
    #    solution (CLAUDE.md): for each k, F_k = rhs(t_k, y_k), unpack,
    #    and store the rows. For "quasi_steady", dTh_dt and dTc_dt are None
    #    (the gas is algebraic, so nozzle rates are not available).
    # 9. Boundary histories: boundary_conditions(t, params, scenario) on the
    #    saved times.
    # 10. Return the result dict.
    # Check: scenario "none" from the discrete steady state stays put
    #    (tests/test_transient.py). A run with N, 2N, 4N gives observed order
    #    ~1 in a smooth output (metrics.observed_order).
    raise NotImplementedError
