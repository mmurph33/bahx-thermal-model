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

# Import through the src package (run code from the repo root: pytest, or
# python -m scripts.<name>). A bare `import correlations` only works when the
# current directory is src/, and breaks every caller that imports src.transient.
from src import correlations, properties, steady


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
    # Pseudocode: steps 1-10 below. Representative properties are the lambda = 0
    #    surrogates (mean cp, mu, k) evaluated at T_bar (rho at T_bar), with the
    #    conductances at those properties, as in steady.initial_guess.
    # Report kappa: the math doc asks for the base-case value in week 1. If
    #    kappa >> 1, the quasi-steady fallback (Section 6.4) is attractive.
    # Check: a hand calculation for the base case gives kappa of order 5-10
    #    and tau_res of order seconds; the cold gas time tau_g_c is the
    #    fastest scale, which is why the system is stiff.

    # 1. Calculate exchanger geometry
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
        rho_c * geom["c"]["A_ff"] * params["L"] / params["mdot_c"]
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
    # 1. dx = L/N; x = (np.arange(N) + 0.5) * dx.
    # Check: x[0] = dx/2, x[-1] = L - dx/2, len(x) = N.

    # Total exchanger length is divided evenly into N cells
    dx = L / N

    # Center position of each cell is in the middle
    x = (np.arange(N) + 0.5) * dx

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
    # 1. y = np.empty(3N); y[0::3] = T_h; y[1::3] = T_c; y[2::3] = T_w
    #    (or np.column_stack((T_h, T_c, T_w)).ravel()).
    # Check: unpack_state(pack_state(a, b, c)) returns (a, b, c) exactly.

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
    # 1. Return y[0::3], y[1::3], y[2::3]. Slicing along axis 0 works for both
    #    input shapes; for histories, transpose to (n_t, N) afterwards.
    # Check: same as pack_state.

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
    # 1. Start from sparse.lil_matrix((3N, 3N)) (cheap to fill entry by entry).
    # 2. For each cell i (Python index 0..N-1), with rows h = 3i, c = 3i+1,
    #    w = 3i+2:
    #      hot row h:   own T_h (3i), own T_w (3i+2), upstream T_h (3(i-1)) if i > 0
    #      cold row c:  own T_c (3i+1), own T_w (3i+2), upstream T_c (3(i+1)+1) if i < N-1
    #      wall row w:  own T_h, T_c, T_w; plus T_w of cells i-1 and i+1 if
    #                   axial_conduction
    #    a_h depends only on the local T_h and a_c only on the local T_c, so no
    #    other entries appear.
    # 3. Return pattern.tocsr().
    # Pitfall: the band is +-3 even without conduction (the upstream gas
    #    neighbours are 3 positions away). jac_sparsity is ignored if you also
    #    pass jac=.
    # Check: compute a dense finite-difference Jacobian of transient_rhs at a
    #    random in-range state for N = 5; every nonzero must sit inside the
    #    pattern.

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

        # Cold gas energy equation (flow in -x direction, so the upstream
        # cell is i+1, not i-1)
        pattern[c,c] = 1
        pattern[c,w] = 1

        if i < N - 1:
            c_upstream = 3 * (i+1) + 1
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

    # Purpose: Smoothing out inlet temperature ramp

    # Convert to numpy array
    s = np.asarray(s, dtype = float)

    # Calculate the soft plus terms
    # softplus(z) = ln(1 + exp(z)) 
    softplus_1 = np.logaddexp(0.0, sharpness * s)
    softplus_2 = np.logaddexp(0.0, sharpness * (s - 1.0))

    # Calculate smoothed unit ramp
    psi = (softplus_1 - softplus_2) / sharpness
    return psi

def boundary_conditions(t, params, scenario):
    """Inlet temperatures and flows at time t for an upset scenario.

    Section 6.5. "feed_ramp" (Eq 37):
      T_h,in(t) = T_h_in + ramp_dT * psi((t - ramp_t_start)/ramp_t)
    "cold_trip" (Eq 38), for t >= trip_t_start:
      mdot_c(t) = mdot_c [phi + (1 - phi) exp(-(t - trip_t_start)/tau_v)]
    "both_ramp": the Eq 37 ramp applied to both inlets at the same time,
      T_h,in(t) = T_h_in + ramp_dT   * psi((t - ramp_t_start)/ramp_t)
      T_c,in(t) = T_c_in + ramp_dT_c * psi((t - ramp_t_start)/ramp_t)
      (not in the math doc yet, flagged: add it to Section 6.5).
    "startup": both flows open from zero along the same smoothed ramp,
      mdot(t) = mdot * psi((t - startup_t_start)/startup_t_flow),
      with the inlet temperatures held at T_h_in and T_c_in. Start it from a
      uniform core (simulate's default y0 for this scenario). Not in the math
      doc yet (flagged).
    "none": everything constant (used for steady holds and tests).

    Parameters
    ----------
    t : float or ndarray, shape (n_t,)
        Time [s].
    params : dict
        Uses T_h_in, T_c_in, mdot_h, mdot_c and the ramp_* / trip_* /
        startup_* keys.
    scenario : str
        "none", "feed_ramp", "cold_trip", "both_ramp" or "startup".

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

    # Initialize all boundary conditions to base values
    # full_like creates arrays with the same shape as t.
    T_h_in = np.full_like(t, params["T_h_in"], dtype=float)
    T_c_in = np.full_like(t, params["T_c_in"], dtype=float)

    mdot_h = np.full_like(t, params["mdot_h"], dtype=float)
    mdot_c = np.full_like(t, params["mdot_c"], dtype=float)

    # Apply selected scenario:
    if scenario == "none":
        pass
    elif scenario == "feed_ramp":
        # Smoothly increase inlet side temperature
        s = (t - params["ramp_t_start"]) / params["ramp_t"]
        psi = smooth_ramp(s, params["ramp_sharpness"])
        T_h_in = (
            T_h_in + params["ramp_dT"] * psi
        )
    elif scenario == "cold_trip":
        # Exponetially reduce cold side flow afte trip

        # Before trip: mdot_c = normal flow
        # After trip: mdot_c approaches phi * normal flow
        elapsed = t - params["trip_t_start"]

        mdot_c = np.where(
            elapsed >= 0.0,
            params["mdot_c"] * (
                params["trip_phi"]
                + (1.0 - params["trip_phi"])
                * np.exp(-elapsed / params["trip_tau_v"])
            ),
            mdot_c
        )
    elif scenario == "both_ramp":
        # Both inlets follow the same smoothed ramp: hot by ramp_dT, cold by ramp_dT_c.
        s = (t - params["ramp_t_start"]) / params["ramp_t"]
        psi = smooth_ramp(s, params["ramp_sharpness"])
        T_h_in = T_h_in + params["ramp_dT"] * psi
        T_c_in = T_c_in + params["ramp_dT_c"] * psi
    elif scenario == "startup":
        # Both flows open together from zero; the inlet temperatures do not change.
        # (At zero flow the correlations fall back to the laminar h, so this is safe.)
        s = (t - params["startup_t_start"]) / params["startup_t_flow"]
        psi = smooth_ramp(s, params["ramp_sharpness"])
        mdot_h = mdot_h * psi
        mdot_c = mdot_c * psi
    else:
        raise ValueError(
            f"Unknown scenario '{scenario}'. "
            "Choose 'none', 'feed_ramp', 'cold_trip', 'both_ramp' or 'startup'."
        )

    return T_h_in, T_c_in, mdot_h, mdot_c


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
    N = y.size // 3
    dx = params["L"] / N
    T_h, T_c, T_w = unpack_state(y)
    T_h_in, T_c_in, mdot_h, mdot_c = boundary_conditions(t, params, scenario)

    # Donor (upstream) temperature of each cell. Hot flows +x, so cell i is
    # fed by cell i-1 (and cell 1 by the hot inlet). Cold flows -x, so cell i
    # is fed by cell i+1 (and cell N by the cold inlet).
    T_h_up = np.concatenate(([T_h_in], T_h[:-1]))
    T_c_up = np.concatenate((T_c[1:], [T_c_in]))

    # Net enthalpy carried into each cell by the flow [W] (Eq 32-33).
    adv_h = mdot_h * (properties.enthalpy(props["h"], T_h_up) - properties.enthalpy(props["h"], T_h))
    adv_c = mdot_c * (properties.enthalpy(props["c"], T_c_up) - properties.enthalpy(props["c"], T_c))

    # Heat exchanged with the metal in each cell [W]. a_c uses the current
    # mdot_c(t), which falls during a trip.
    a_h, a_c = correlations.conductances(T_h, T_c, mdot_h, mdot_c, params, props, geom)
    Q_hw = a_h * dx * (T_h - T_w)  # hot gas -> metal
    Q_wc = a_c * dx * (T_w - T_c)  # metal -> cold gas

    # Axial conduction in the metal [W] (Eq 34), adiabatic ends via ghost
    # cells T_w,0 = T_w,1 and T_w,N+1 = T_w,N.
    if params["axial_conduction"]:
        T_pad = np.concatenate(([T_w[0]], T_w, [T_w[-1]]))
        cond = params["k_m"] * geom["A_m"] / dx * (T_pad[2:] - 2.0 * T_pad[1:-1] + T_pad[:-2])
    else:
        cond = np.zeros(N)

    # Cell heat capacities [J/K], gas properties at each gas's own temperature.
    M_h = properties.evaluate(props["h"], "rho", T_h) * geom["h"]["A_ff"] * properties.evaluate(props["h"], "cp", T_h) * dx
    M_c = properties.evaluate(props["c"], "rho", T_c) * geom["c"]["A_ff"] * properties.evaluate(props["c"], "cp", T_c) * dx
    M_w = geom["C_m"] * dx

    # Energy balances (Eq 32-34). The cold gas GAINS heat from the metal, so
    # Q_wc enters its balance with a plus sign.
    dTh_dt = (adv_h - Q_hw) / M_h
    dTc_dt = (adv_c + Q_wc) / M_c
    dTw_dt = (cond + Q_hw - Q_wc) / M_w

    return pack_state(dTh_dt, dTc_dt, dTw_dt)


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
    L = steady_result["x"][-1]
    x_cells, _ = cell_centers(L, N)
    T_h = np.interp(x_cells, steady_result["x"], steady_result["T_h"])
    T_c = np.interp(x_cells, steady_result["x"], steady_result["T_c"])
    T_w = np.interp(x_cells, steady_result["x"], steady_result["T_w"])
    return pack_state(T_h, T_c, T_w)


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
    #    Set first_step to a fraction of the fastest discrete time scale: the
    #    guess has large rates in the two inlet cells, and solve_ivp's
    #    automatic first step can push them out of the fitted range.
    # 5. Polish: scipy.optimize.root(F, y_relaxed, method="hybr") (dense
    #    finite-difference Jacobian, fine up to 3N ~ 1000), or
    #    scipy.optimize.least_squares(F, y_relaxed, jac_sparsity=pattern)
    #    for larger N.
    # 6. Check max |F(y_ss)| < ~1e-8 K/s; raise RuntimeError otherwise.
    # Check: tests/test_transient.py::test_transient_stays_at_steady_state;
    #    the gap to steady.solve_steady at the cell centres roughly halves when
    #    N doubles (first-order upwinding).
    geom = correlations.exchanger_geometry(params)
    N = params["N"]
    if y_guess is None:
        y_guess = steady_to_cells(steady.solve_steady(params, props), N)

    def F(y):
        """Discrete model right-hand side with the boundary values held at t = 0."""
        return transient_rhs(0.0, y, params, props, geom, scenario)

    # Pseudo-transient relaxation: let the physics itself carry the guess
    # toward the steady state.
    scales = time_scales(params, props)
    t_relax = 10.0 * scales["t_response"]
    # The guess has rates of ~10-100 K/s in the two inlet cells (see
    # steady_to_cells). solve_ivp's automatic first step is an explicit probe
    # sized from the RMS rate, which dilutes those two cells at large N and
    # can push them out of the fitted range. Start instead at a tenth of the
    # fastest time scale of the discrete system, min(tau_g, tau_res/N)
    # (Section 6.2).
    fastest = min(
        scales["tau_g_h"], scales["tau_g_c"],
        scales["tau_res_h"] / N, scales["tau_res_c"] / N,
    )
    relax = solve_ivp(
        lambda t, y: F(y),
        (0.0, t_relax),
        y_guess,
        method="BDF",
        jac_sparsity=jacobian_sparsity(N, params["axial_conduction"]),
        rtol=params["ivp_rtol"],
        atol=params["ivp_atol"],
        first_step=0.1 * fastest,
    )
    if not relax.success:
        raise RuntimeError(f"Steady-state relaxation failed: {relax.message}")

    # Newton-type polish (MINPACK hybrd). xtol is tight because the residual
    # is in K/s: a state error of 1e-11 K times the fastest rate (~1e3 1/s at
    # large N) is ~1e-8 K/s.
    polish = root(F, relax.y[:, -1], method="hybr", options={"xtol": 1.0e-13})
    y_ss = polish.x

    # Judge convergence by the residual itself, not by root's status flag
    # (hybrd can report slow progress once it is already at round-off).
    residual = np.max(np.abs(F(y_ss)))
    if residual > 1.0e-8:
        raise RuntimeError(
            f"Discrete steady state not converged: max |F| = {residual:.2e} K/s "
            f"({polish.message})"
        )
    return y_ss


def quasi_steady_gas(T_w, t, params, props, geom, scenario):
    """Gas temperatures for a given wall profile, neglecting gas storage.

    Section 6.4, Eq 36: with the gas accumulation dropped (kappa >> 1), each
    gas equation is a linear first-order ODE with a known source, solved
    exactly cell by cell with properties frozen in the cell:
      T_h,i = T_w,i + (T_h,i-1 - T_w,i) exp(-a_h,i dx/(mdot_h cp_h,i)),  i = 1..N
      T_c,i = T_w,i + (T_c,i+1 - T_w,i) exp(-a_c,i dx/(mdot_c cp_c,i)),  i = N..1
    Implemented instead with the donor-cell update of the full model,
    T_i = T_w,i + (T_up - T_w,i)/(1 + NTU_cell), with a and cp at the
    upstream temperature (see the flag below; math doc Eq 36 should be
    updated, or the code switched to exp, by team decision).

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
    # Implemented with the 1/(1 + NTU_cell) update recommended above, so that
    #    with constant properties this model is exactly the full model with
    #    the gas storage removed (see the flag in the docstring).
    N = T_w.size
    dx = params["L"] / N
    k_f = params["k_f"]
    T_h_in, T_c_in, mdot_h, mdot_c = boundary_conditions(t, params, scenario)

    # Hot sweep in the direction of hot flow (x = 0 -> L).
    T_h = np.empty(N)
    T_up = T_h_in
    for i in range(N):
        a_h = correlations.side_conductance(T_up, mdot_h, props["h"], geom["h"], k_f)
        cp_h = properties.evaluate(props["h"], "cp", T_up)
        NTU_cell = a_h * dx / (mdot_h * cp_h)
        # Donor-cell balance with no storage: mdot cp (T_up - T_i) = a dx (T_i - T_w)
        T_h[i] = T_w[i] + (T_up - T_w[i]) / (1.0 + NTU_cell)
        T_up = T_h[i]

    # Cold sweep in the direction of cold flow (x = L -> 0).
    T_c = np.empty(N)
    T_up = T_c_in
    for i in range(N - 1, -1, -1):
        a_c = correlations.side_conductance(T_up, mdot_c, props["c"], geom["c"], k_f)
        cp_c = properties.evaluate(props["c"], "cp", T_up)
        NTU_cell = a_c * dx / (mdot_c * cp_c)
        T_c[i] = T_w[i] + (T_up - T_w[i]) / (1.0 + NTU_cell)
        T_up = T_c[i]

    return T_h, T_c


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
    N = T_w.size
    dx = params["L"] / N
    T_h, T_c = quasi_steady_gas(T_w, t, params, props, geom, scenario)
    T_h_in, T_c_in, mdot_h, mdot_c = boundary_conditions(t, params, scenario)

    # Upstream (donor) temperatures, as in transient_rhs.
    T_h_up = np.concatenate(([T_h_in], T_h[:-1]))
    T_c_up = np.concatenate((T_c[1:], [T_c_in]))

    # Heat delivered to the metal = enthalpy the two gases give up crossing
    # the cell (enthalpy form, so the total telescopes to the inlet/outlet
    # enthalpy flows exactly, as in the full model).
    Q = (
        mdot_h * (properties.enthalpy(props["h"], T_h_up) - properties.enthalpy(props["h"], T_h))
        + mdot_c * (properties.enthalpy(props["c"], T_c_up) - properties.enthalpy(props["c"], T_c))
    )

    if params["axial_conduction"]:
        T_pad = np.concatenate(([T_w[0]], T_w, [T_w[-1]]))
        cond = params["k_m"] * geom["A_m"] / dx * (T_pad[2:] - 2.0 * T_pad[1:-1] + T_pad[:-2])
    else:
        cond = np.zeros(N)

    return (Q + cond) / (geom["C_m"] * dx)


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
        "none", "feed_ramp", "cold_trip", "both_ramp" or "startup".
    t_end : float
        Final time [s].
    t_eval : ndarray, shape (n_t,), optional
        Times at which to save the solution [s]. Default: an even grid fine
        enough to resolve the upset (e.g. 2001 points).
    y0 : ndarray, optional
        Initial state, shape (3N,) (full) or (N,) (quasi_steady). Default:
        discrete_steady_state(params, props, scenario), or for "startup" a
        uniform core at startup_T0 (there is no flow yet); for quasi_steady,
        its wall part polished to the quasi-steady model's own steady state.
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
    # 2. y0 as described above. For "quasi_steady", polish the full model's
    #    wall profile with a root solve on quasi_steady_rhs, so the start is
    #    the steady state of the model actually being integrated.
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
    #    - "startup": max_step <= startup_t_flow/20 for the same reason, and
    #      an explicit small first_step (startup_t_flow/1000), because
    #      solve_ivp's automatic first-step trial extrapolates the cold inlet
    #      cell outside the surrogate range on fine grids and long runs.
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
    if model not in ("full", "quasi_steady"):
        raise ValueError(f"model must be 'full' or 'quasi_steady', not {model!r}")

    geom = correlations.exchanger_geometry(params)
    N = params["N"]
    x, _ = cell_centers(params["L"], N)
    if t_eval is None:
        t_eval = np.linspace(0.0, t_end, 2001)
    t_eval = np.asarray(t_eval, dtype=float)

    # Right-hand side and Jacobian pattern of the chosen model.
    if model == "full":
        def rhs(t, y):
            return transient_rhs(t, y, params, props, geom, scenario)
        jac_sparsity = jacobian_sparsity(N, params["axial_conduction"])
    else:
        def rhs(t, T_w):
            return quasi_steady_rhs(t, T_w, params, props, geom, scenario)
        jac_sparsity = None  # the gas sweeps make this Jacobian dense

    # Initial state: the steady state of the discrete model being integrated.
    # A startup begins from rest instead: no flow yet, so the gas and the metal
    # all sit at the uniform idle temperature startup_T0.
    if y0 is None and scenario == "startup":
        T_0 = np.full(N, float(params["startup_T0"]))
        if model == "full":
            y0 = pack_state(T_0, T_0, T_0)
        else:
            y0 = T_0
    if y0 is None:
        y_full = discrete_steady_state(params, props, scenario)
        if model == "full":
            y0 = y_full
        else:
            # The full model's wall profile is O(dx) away from the quasi-steady
            # model's own steady state (properties are lagged one cell in the
            # sweeps), so polish it with a root solve on the N wall unknowns.
            _, _, T_w_full = unpack_state(y_full)
            y0 = root(lambda T_w: rhs(0.0, T_w), T_w_full, method="hybr").x

    # Feed ramp (and both_ramp): nothing happens before it starts, so BDF would
    # grow its step and could jump straight over a short ramp.
    max_step = np.inf
    first_step = None  # None lets solve_ivp choose its own first step
    if scenario in ("feed_ramp", "both_ramp"):
        max_step = params["ramp_t"] / 20.0
    if scenario == "startup":
        max_step = params["startup_t_flow"] / 20.0
        # solve_ivp chooses its first step by evaluating the right-hand side
        # at the trial state y0 + h0*f(y0), with h0 as large as the whole run.
        # From rest only the cold inlet cell has a (small) rate, from the
        # smoothed ramp's leftover flow at t = 0 (psi(0) ~ 1e-6). Multiplied by
        # h0 = t_end, that pushes the cell far below T_c_in and outside the
        # surrogate range (seen for N = 800 with t_end = 261 s and for N = 400
        # with t_end = 600 s; the smaller the cell, the larger its rate).
        # A small explicit first step skips that trial; BDF grows it from there.
        first_step = min(params["startup_t_flow"] / 1000.0, t_end)

    # Cold trip: Eq 38 has a slope jump at trip_t_start, so integrate up to it
    # and restart from there; BDF never steps across the kink.
    t_breaks = [0.0, t_end]
    if scenario == "cold_trip" and 0.0 < params["trip_t_start"] < t_end:
        t_breaks = [0.0, params["trip_t_start"], t_end]

    sols = []
    t_parts = []
    y_parts = []
    y_start = np.asarray(y0, dtype=float)
    for k in range(len(t_breaks) - 1):
        t_a, t_b = t_breaks[k], t_breaks[k + 1]
        # Each saved time belongs to exactly one segment.
        if k == 0:
            in_segment = (t_eval >= t_a) & (t_eval <= t_b)
        else:
            in_segment = (t_eval > t_a) & (t_eval <= t_b)

        # Implicit BDF only: the system is stiff (Section 6.2).
        sol = solve_ivp(
            rhs,
            (t_a, t_b),
            y_start,
            method="BDF",
            t_eval=t_eval[in_segment],
            rtol=params["ivp_rtol"],
            atol=params["ivp_atol"],
            dense_output=True,
            jac_sparsity=jac_sparsity,
            max_step=max_step,
            first_step=first_step,
        )
        if not sol.success:
            raise RuntimeError(f"Transient solve failed on [{t_a}, {t_b}] s: {sol.message}")
        sols.append(sol)
        t_parts.append(sol.t)
        y_parts.append(sol.y)
        y_start = sol.sol(t_b)  # state at the end of this segment

    t = np.concatenate(t_parts)
    Y = np.concatenate(y_parts, axis=1)  # shape (n_state, n_t)

    # Rates straight from the right-hand side at the saved times (CLAUDE.md:
    # never np.gradient of the solution).
    dY = np.empty_like(Y)
    for k in range(t.size):
        dY[:, k] = rhs(t[k], Y[:, k])

    # Histories, each shape (n_t, N).
    if model == "full":
        T_h, T_c, T_w = (field.T for field in unpack_state(Y))
        dTh_dt, dTc_dt, dTw_dt = (field.T for field in unpack_state(dY))
    else:
        T_w = Y.T
        dTw_dt = dY.T
        dTh_dt = None  # the gas is algebraic in this model, so no gas rates
        dTc_dt = None
        T_h = np.empty_like(T_w)
        T_c = np.empty_like(T_w)
        for k in range(t.size):
            T_h[k], T_c[k] = quasi_steady_gas(T_w[k], t[k], params, props, geom, scenario)

    T_h_in, T_c_in, mdot_h, mdot_c = boundary_conditions(t, params, scenario)

    return {
        "t": t,
        "x": x,
        "T_h": T_h, "T_c": T_c, "T_w": T_w,
        "dTh_dt": dTh_dt, "dTc_dt": dTc_dt, "dTw_dt": dTw_dt,
        "T_h_in": T_h_in, "T_c_in": T_c_in, "mdot_h": mdot_h, "mdot_c": mdot_c,
        "sol": sols[0] if len(sols) == 1 else sols,
        "N": N,
        "model": model,
    }
