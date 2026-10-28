"""Closure relations: fin geometry, heat transfer coefficient and side conductances.

Math doc: Section 4.1 (offset strip fin geometry, Eq 17-18), Section 4.2
(Manglik-Bergles Colburn factor, Eq 19 and 21), Section 4.3 (fin and surface
efficiency, Eq 22), Section 6.5 (laminar floor on h at low flow) and
Section 3.2 (overall conductance, Eq 4).

These functions turn local stream temperatures into the side conductances
a = eta_o h A' [W/(m K)] that drive both the steady BVP and the transient model.
Because h depends on the local properties, a_h and a_c are re-evaluated at every
collocation point and every cell; this is where the model becomes nonlinear.

Imports: numpy and src.properties (evaluate). Used by steady.py and transient.py.

Naming (CLAUDE.md): fin height is b (h is the heat transfer coefficient);
Manglik-Bergles ratios are alpha_mb = s/b, delta_mb = t/l, gamma_mb = t/s.
All functions must accept numpy arrays of any shape (solve_bvp passes (m,)
arrays, solve_ivp passes (N,) arrays): use np.maximum / np.where, never the
built-in max or an `if` on array values.

The side_geom dict (one per side; geom = {"h": ..., "c": ..., "A_m": ..., "C_m": ...})
--------------------------------------------------------------------------------
    "s","b","t","l","W"   fin spacing, height, thickness, strip length, core width [m]
    "alpha_mb","delta_mb","gamma_mb"   Manglik-Bergles ratios [-]
    "D_h"     hydraulic diameter [m]
    "n"       fins per unit width, 1/(s + t) [1/m]
    "A_ff"    free-flow area of one passage [m^2]
    "A_p"     primary (parting-sheet) area per unit length A'_p [m]
    "A_f"     fin area per unit length A'_f [m]
    "A_prime" total heat transfer area per unit length A' [m]
"""

import numpy as np

from src import properties


def osf_geometry(s, b, t, l, W):
    """Geometry of one offset-strip-fin passage.

    Eq 17 (Manglik-Bergles ratios and hydraulic diameter) and Eq 18
    (free-flow area and heat transfer areas per unit length), neglecting
    strip-edge and corner corrections.

    Parameters
    ----------
    s : float
        Clear fin spacing [m].
    b : float
        Clear fin height [m] (plate spacing is b + t).
    t : float
        Fin thickness [m].
    l : float
        Strip length [m].
    W : float
        Core width [m].

    Returns
    -------
    side_geom : dict
        Keys listed in the module docstring.
    """
    ratios = {
        "alpha_mb": s/b,
        "delta_mb": t/l,
        "gamma_mb": t/s
    }

    # Hydraulic diameter
    D_h = 4.0 * s*b*l / (2.0 * (s*l + b*l + t*b) + t*s)

    # Fins per unit width
    n = 1.0 / (s + t)

    # Free flow area
    A_ff = W * n * s * b

    # Primary area per unit length
    # Factor of 2 account both parting sheets
    A_p = 2.0 * W * (1.0 - n * t)

    # Fin area per unit length (two face per fin)
    A_f = 2.0 * W * n * b

    return {
        "s": s, "b": b, "t": t, "l": l, "W": W,
        **ratios,
        "D_h": D_h,
        "n": n,
        "A_ff": A_ff,
        "A_p": A_p,
        "A_f": A_f,
        "A_prime": A_p + A_f,
    }


def exchanger_geometry(params):
    """Geometry of the repeating unit: both passages plus the metal.

    Section 4.1 (A_m below Eq 18) and the Nomenclature (C'_m = rho_m c_m A_m).
    Public solvers call this themselves from ``params``, so a perturbed copy of
    params never meets a stale geometry.

    Parameters
    ----------
    params : dict
        Uses W, s_h, b_h, t_fin_h, l_h, s_c, b_c, t_fin_c, l_c, t_sheet,
        rho_m, c_m.

    Returns
    -------
    geom : dict
        "h": side_geom of the hot passage, "c": side_geom of the cold passage,
        "A_m": metal cross-section of one repeating unit [m^2],
        "C_m": metal heat capacity per unit length C'_m [J/(m K)].
    """

    # Geometry for the hot side
    geom_h = osf_geometry(
        params["s_h"], params["b_h"], params["t_fin_h"], params["l_h"], params["W"]
    )

    # Geometry for the cold side
    geom_c = osf_geometry(
        params["s_c"], params["b_c"], params["t_fin_c"], params["l_c"], params["W"]
    )

    # Two parting sheets plus the fins of both passages.
    # Metal  corss-section of the repeating unit
    A_m = params["W"] * (
        2.0 * params["t_sheet"]
        + geom_h["n"] * geom_h["t"] * geom_h["b"]
        + geom_c["n"] * geom_c["t"] * geom_c["b"]
    )

    # Total metal heat capacity
    C_m = params["rho_m"] * params["c_m"] * A_m

    return {"h": geom_h, "c": geom_c, "A_m": A_m, "C_m": C_m}


def reynolds(mdot, mu, side_geom):
    """Reynolds number based on the hydraulic diameter.

    Section 4.2: Re = G D_h / mu with mass flux G = mdot / A_ff.

    Parameters
    ----------
    mdot : float or ndarray
        Mass flow per passage [kg/s]. Can vary in time (cold trip, Eq 38).
    mu : float or ndarray
        Viscosity at the local stream temperature [Pa s].
    side_geom : dict
        Geometry of this passage.

    Returns
    -------
    Re : ndarray, broadcast shape of mdot and mu
        Reynolds number [-].
    """
    # 1. G = mdot / A_ff.
    # 2. Re = G D_h / mu.
    # Check: base case G ~ 17 kg/(m^2 s), D_h ~ 2.4 mm, mu ~ 1.2e-5 Pa s
    #    gives Re ~ 3400 on the hot side.

    G = mdot / side_geom["A_ff"]
    Re = G * side_geom["D_h"] / mu

    return Re


def colburn_j(Re, alpha_mb, delta_mb, gamma_mb):
    """Manglik-Bergles Colburn factor for offset strip fins.

    Eq 19:
      j = 0.6522 Re^-0.5403 alpha^-0.1541 delta^0.1499 gamma^-0.0678
          * [1 + 5.269e-5 Re^1.340 alpha^0.504 delta^0.456 gamma^-1.055]^0.1

    Parameters
    ----------
    Re : ndarray
        Reynolds number [-].
    alpha_mb, delta_mb, gamma_mb : float
        Geometry ratios s/b, t/l, t/s [-].

    Returns
    -------
    j : ndarray, same shape as Re
        Colburn factor [-].
    """

    # Laminar
    laminar_part = (
        0.6522 
        * Re ** -0.5403
        * alpha_mb ** -0.1541
        * delta_mb ** 0.1499
        * gamma_mb ** -0.0678
    )

    # Turbulent
    turblent_part = (
        1.0
        + 5.269e-5
        * Re ** 1.340
        * alpha_mb ** 0.504
        * delta_mb ** 0.456
        * gamma_mb ** -1.055
    ) ** 0.1

    return laminar_part * turbulent_part


def laminar_nusselt_rect(alpha_mb):
    """Fully developed laminar Nusselt number of a rectangular duct.

    Section 6.5: during a cold-stream trip the cold Re falls below the
    correlation's range, so h is bounded below by the fully developed laminar
    value h_lam = Nu_fd(alpha) k / D_h for a duct of aspect ratio alpha = s/b.

    Parameters
    ----------
    alpha_mb : float
        Channel aspect ratio s/b, between 0 and 1 [-].

    Returns
    -------
    Nu : float
        Fully developed laminar Nusselt number [-].
    """

    # Nu_H1 = 8.235 (1 - 2.0421 a + 3.0853 a^2 - 2.4765 a^3 + 1.0578 a^4 - 0.1861 a^5)
    coeffs_H1 = [8.235 * c for c in (1.0, -2.0421, 3.0853, -2.4765, 1.0578, -0.1861)]
    return np.polynomial.polynomial.polyval(alpha_mb, coeffs_H1)


def heat_transfer_coefficient(mdot, cp, mu, k, side_geom):
    """Convective heat transfer coefficient on one side.

    Eq 21: h = j G cp Pr^(-2/3), Pr = cp mu / k, with j from the
    Manglik-Bergles correlation (Eq 19), bounded below by the laminar value
    (Section 6.5). The proposal writes the same thing as h = j rho u cp Pr^(-2/3),
    since rho u = G.

    Parameters
    ----------
    mdot : float or ndarray
        Mass flow per passage [kg/s].
    cp, mu, k : ndarray
        Local properties of THIS stream at its own temperature
        [J/(kg K)], [Pa s], [W/(m K)].
    side_geom : dict
        Geometry of this passage.

    Returns
    -------
    h : ndarray, broadcast shape of the inputs
        Heat transfer coefficient [W/(m^2 K)].
    """
    # 1. G = mdot / A_ff; Re = reynolds(mdot, mu, side_geom).
    # 2. j = colburn_j(Re, alpha_mb, delta_mb, gamma_mb) with this side's ratios.
    # 3. Pr = cp mu / k; h_corr = j G cp Pr^(-2/3)   (Eq 21).
    # 4. Laminar floor: h_lam = laminar_nusselt_rect(alpha_mb) k / D_h;
    #    h = np.maximum(h_corr, h_lam). At design flow the floor is inactive.
    # Pitfall: np.maximum has a kink where the two branches cross, which can
    #    make BDF take tiny steps during a trip. If that happens, replace it
    #    with the smooth max of Eq 47 (metrics.smax) and say so in the report.
    # Pitfall: mdot can be a function of time (cold trip); pass the current
    #    value, never the initial one.
    # Check: base case h ~ 350-500 W/(m^2 K) on both sides. With mdot -> 0
    #    the result tends to h_lam (no division by zero, no negative h).
    
    # Ensure no division by 0
    Re_safe = np.maximum(Re, 1.0e-12)

    # colburn j
    j = colburn_j(Re_safe, side_geom["alpha_mb"], side_geom["delta_mb"], side_geom["gamma_mb"])

    # Prandtl number
    Pr = cp * mu / k

    # heat transfer coefficient
    h_corr = j * G * cp * Pr ** (-2.0 / 3.0)
    h_lam = laminar_nusselt_rect(side_geom["alpha_mb"]) * k / side_geom["D_h"]

    return np.maximum(h_corr, h_lam)


def fin_efficiency(h, k_f, side_geom):
    """Fin efficiency of an offset strip fin.

    Eq 22 (Shah & Sekulic, with the strip-edge correction):
      m = [ (2 h / (k_f t)) (1 + t/l) ]^(1/2),  ell_f = b/2,
      eta_f = tanh(m ell_f) / (m ell_f).
    The fin spans two parting sheets at nearly the same temperature, so it
    acts as two fins of length b/2 with an adiabatic midplane (CLAUDE.md:
    ell_f = b/2).

    Parameters
    ----------
    h : ndarray
        Heat transfer coefficient [W/(m^2 K)].
    k_f : float
        Fin conductivity [W/(m K)].
    side_geom : dict
        Uses t, l, b.

    Returns
    -------
    eta_f : ndarray, same shape as h
        Fin efficiency [-], between 0 and 1.
    """
    # 1. m = sqrt(2 h (1 + t/l) / (k_f t)).
    # 2. ell_f = b/2; x = m ell_f.
    # 3. eta_f = tanh(x)/x. As x -> 0, eta_f -> 1; it is safe for h > 0, but
    #    guard x == 0 with np.where if you ever pass h = 0.
    # Note: the proposal writes m = sqrt(2 h/(k_f t_f)) with a fin length L_f.
    #    Eq 22 adds the strip-edge factor (1 + t/l) and fixes L_f = ell_f = b/2;
    #    follow Eq 22 (the difference is ~1.6 % in m for the suggested fin).
    # Check: base case (h ~ 450, k_f = 180, t = 0.1 mm, b = 6.25 mm) gives
    #    m ell_f ~ 0.7 and eta_f ~ 0.86. Larger h or taller fins -> lower eta_f.

    # Fin thickeness
    t = side_geom["t"]
    m = np.sqrt(2.0 * h * (1.0 + t / side_geom["l"]) / (k_f * t))

    ell_f = side_geom["b"] / 2.0
    x = m * ell_f
    # tanh(x)/x -> 1 as x -> 0; np.where guards the x == 0 (h = 0) case.
    x_safe = np.where(x == 0.0, 1.0, x)
    return np.where(x == 0.0, 1.0, np.tanh(x_safe) / x_safe)


def surface_efficiency(eta_f, side_geom):
    """Overall surface efficiency.

    Eq 22: eta_o = 1 - (A'_f / A') (1 - eta_f).

    Parameters
    ----------
    eta_f : ndarray
        Fin efficiency [-].
    side_geom : dict
        Uses A_f and A_prime.

    Returns
    -------
    eta_o : ndarray, same shape as eta_f
        Overall surface efficiency [-].
    """
    return 1.0 - (side_geom["A_f"] / side_geom["A_prime"]) * (1.0 - eta_f)


def side_conductance(T, mdot, surr, side_geom, k_f):
    """Side conductance per unit length a = eta_o h A'.

    Combines Eq 21 and Eq 22 (CLAUDE.md: a = eta_o * h * A' [W/(m K)]).

    Parameters
    ----------
    T : float or ndarray, any shape
        Local bulk temperature of THIS stream [K].
    mdot : float
        Current mass flow per passage of this stream [kg/s].
    surr : dict
        Property surrogate of THIS stream (props["h"] with hot temperatures,
        props["c"] with cold temperatures).
    side_geom : dict
        Geometry of this passage.
    k_f : float
        Fin conductivity [W/(m K)].

    Returns
    -------
    a : ndarray, same shape as T
        Side conductance per unit length [W/(m K)].
    """
    cp = properties.evaluate(surr, "cp", T)
    mu = properties.evaluate(surr, "mu", T)
    k = properties.evaluate(surr, "k", T)

    h = heat_transfer_coefficient(mdot, cp, mu, k, side_geom)
    eta_f = fin_efficiency(h, k_f, side_geom)
    eta_o = surface_efficiency(eta_f, side_geom)
    return eta_o * h * side_geom["A_prime"]


def conductances(T_h, T_c, mdot_h, mdot_c, params, props, geom):
    """Hot- and cold-side conductances at local stream temperatures.

    One helper used by steady.py, transient.py and the quasi-steady model, so
    every caller evaluates a_h and a_c the same way.

    Parameters
    ----------
    T_h, T_c : ndarray, same shape
        Hot and cold stream temperatures [K].
    mdot_h, mdot_c : float
        Current mass flows per passage [kg/s].
    params : dict
        Uses k_f.
    props : dict
        {"h": hot surrogate, "c": cold surrogate}.
    geom : dict
        From exchanger_geometry(params).

    Returns
    -------
    a_h, a_c : ndarray, same shape as T_h
        Side conductances per unit length [W/(m K)].
    """

    k_f = params["k_f"]
    a_h = side_conductance(T_h, mdot_h, props["h"], geom["h"], k_f)
    a_c = side_conductance(T_c, mdot_c, props["c"], geom["c"], k_f)
    return a_h, a_c


def overall_conductance(a_h, a_c):
    """Overall conductance per unit length (two resistances in series).

    Eq 4: 1/U' = 1/a_h + 1/a_c.

    Parameters
    ----------
    a_h, a_c : ndarray, same shape
        Side conductances per unit length [W/(m K)].

    Returns
    -------
    U_prime : ndarray, same shape
        Overall conductance per unit length [W/(m K)].
    """
    
    return a_h * a_c / (a_h + a_c)
