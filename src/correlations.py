"""Closure relations: fin geometry, heat transfer coefficient and side conductances.

Math doc: Section 4.1 (offset strip fin geometry, Eq 17-18), Section 4.2
(Manglik-Bergles j and f, Eq 19-21), Section 4.3 (fin and surface efficiency,
Eq 22), Section 6.5 (laminar floor on h at low flow), Section 8.1 (fitted
Colburn factor, Eq 48) and Section 3.2 (overall conductance, Eq 4).

These functions turn local stream temperatures into the side conductances
a = eta_o h A' [W/(m K)] that drive both the steady BVP and the transient model.
Because h depends on the local properties, a_h and a_c are re-evaluated at every
collocation point and every cell; this is where the model becomes nonlinear.

Imports: numpy and src.properties (evaluate). Used by steady.py, transient.py,
sensitivity.py and regression.py.

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
    # 1. Ratios: alpha_mb = s/b, delta_mb = t/l, gamma_mb = t/s   (Eq 17).
    # 2. Hydraulic diameter D_h = 4 s b l / [2(s l + b l + t b) + t s]   (Eq 17).
    #    (The paper writes h for fin height; we write b.)
    # 3. Fins per unit width n = 1/(s + t).
    # 4. Eq 18, per passage:
    #      A_ff = W n s b            free-flow area (n W channels of s by b)
    #      A_p  = 2 W (1 - n t)      primary area per unit length; the factor 2
    #                                counts both parting sheets bounding the
    #                                passage, minus the fin footprints
    #      A_f  = 2 W n b            fin area per unit length (two faces per fin)
    #      A_prime = A_p + A_f
    # 5. Return everything (including the inputs s, b, t, l, W, which
    #    fin_efficiency needs) in one dict.
    # Check: hand-compute D_h for s = 1.525 mm, b = 6.25 mm, t = 0.102 mm,
    #    l = 3.175 mm (about 2.4 mm). Limit: for t -> 0 and l -> infinity,
    #    D_h -> 2 s b/(s + b), the parallel-channel value.
    raise NotImplementedError


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
    # 1. geom_h = osf_geometry(s_h, b_h, t_fin_h, l_h, W); same for the cold side.
    # 2. Metal cross-section of the repeating unit (two parting sheets plus the
    #    fins of both passages):
    #      A_m = W [2 t_sheet + n_h t_fin_h b_h + n_c t_fin_c b_c]
    # 3. C_m = rho_m c_m A_m.
    # Check: for the suggested fins and t_sheet = 1 mm, W = 1 m, the sheets
    #    dominate A_m (2e-3 m^2 against ~0.8e-3 m^2 of fins).
    raise NotImplementedError


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
    raise NotImplementedError


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
    # 1. Laminar part: 0.6522 Re^-0.5403 alpha^-0.1541 delta^0.1499 gamma^-0.0678.
    # 2. Turbulent bracket: [1 + 5.269e-5 Re^1.340 alpha^0.504 delta^0.456
    #    gamma^-1.055]^0.1.
    # 3. j = product. Use np.power / ** on arrays; no loops.
    # Pitfall: confirm the coefficients and the stated validity range (Re
    #    roughly 120 to 1e4, and ranges of alpha, delta, gamma) against the
    #    1995 paper before the final report (CLAUDE.md team note). Outside the
    #    Re range, warn or rely on the laminar floor in
    #    heat_transfer_coefficient.
    # Check: hand-compute j at Re = 3400 for the suggested fin
    #    (alpha = 0.244, delta = 0.032, gamma = 0.067); expect j ~ 0.009.
    #    j should fall smoothly with Re with no jump.
    raise NotImplementedError


def colburn_j_surface(Re, beta):
    """Colburn factor of one fitted surface (four-parameter form).

    Eq 48: ln j = ln beta1 + beta2 ln Re + 0.1 ln(1 + beta3 Re^beta4),
    i.e. j = beta1 Re^beta2 (1 + beta3 Re^beta4)^0.1.

    Parameters
    ----------
    Re : ndarray
        Reynolds number [-].
    beta : array_like, shape (4,)
        Fitted parameters (beta1, beta2, beta3, beta4), from regression.py.

    Returns
    -------
    j : ndarray, same shape as Re
        Colburn factor [-].
    """
    # 1. Unpack beta1..beta4 (beta1 > 0, beta3 > 0).
    # 2. j = beta1 * Re**beta2 * (1 + beta3 * Re**beta4)**0.1.
    # Check: with the Manglik-Bergles starting guess from
    #    regression.mb_initial_guess(alpha, delta, gamma), this equals
    #    colburn_j(Re, alpha, delta, gamma) to round-off.
    raise NotImplementedError


def fanning_f(Re, alpha_mb, delta_mb, gamma_mb):
    """Manglik-Bergles Fanning friction factor (pressure-drop extension).

    Eq 20:
      f = 9.6243 Re^-0.7422 alpha^-0.1856 delta^0.3053 gamma^-0.2659
          * [1 + 7.669e-8 Re^4.429 alpha^0.920 delta^3.767 gamma^0.236]^0.1
    Used only for the optional pressure-drop ODEs of Section 3.6 (Eq 15) and
    to report Delta p as a sanity check.

    Parameters
    ----------
    Re : ndarray
        Reynolds number [-].
    alpha_mb, delta_mb, gamma_mb : float
        Geometry ratios [-].

    Returns
    -------
    f : ndarray, same shape as Re
        Fanning friction factor [-].
    """
    # 1. Same structure as colburn_j with the Eq 20 coefficients.
    # 2. Pressure drop over the core (constant properties):
    #    Delta p = 4 f (L/D_h) G^2 / (2 rho)   (Section 3.6).
    # Check: f > j at the same Re (friction exceeds heat transfer for offset
    #    strip fins); Delta p for the base case should be a few kPa.
    raise NotImplementedError


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
    # 1. The math doc does not say which thermal boundary condition to use
    #    (flagged). Shah & London give polynomial fits in alpha for:
    #      constant wall temperature (T):
    #        Nu_T  = 7.541 (1 - 2.610 a + 4.970 a^2 - 5.119 a^3 + 2.702 a^4 - 0.548 a^5)
    #      axially uniform heat flux, peripherally uniform wall temperature (H1):
    #        Nu_H1 = 8.235 (1 - 2.0421 a + 3.0853 a^2 - 2.4765 a^3 + 1.0578 a^4 - 0.1861 a^5)
    #    Verify these coefficients in Shah & Sekulic (2003) before use; H1 is
    #    the usual choice for highly conductive aluminum fins.
    # 2. Evaluate with np.polynomial.polynomial.polyval(alpha, coeffs).
    # Check: a = 1 (square duct) gives Nu_T = 2.98, Nu_H1 = 3.61;
    #    a -> 0 (parallel plates) gives 7.54 and 8.24.
    raise NotImplementedError


def heat_transfer_coefficient(mdot, cp, mu, k, side_geom, zeta=1.0, beta=None):
    """Convective heat transfer coefficient on one side.

    Eq 21: h = j G cp Pr^(-2/3), Pr = cp mu / k, with j from Eq 19
    (Manglik-Bergles) or Eq 48 (fitted surface), times the multiplier zeta
    (Section 8.3), and bounded below by the laminar value (Section 6.5).

    Parameters
    ----------
    mdot : float or ndarray
        Mass flow per passage [kg/s].
    cp, mu, k : ndarray
        Local properties of THIS stream at its own temperature
        [J/(kg K)], [Pa s], [W/(m K)].
    side_geom : dict
        Geometry of this passage.
    zeta : float, optional
        Colburn multiplier j -> zeta j [-] (params["zeta_j"]).
    beta : array_like, shape (4,), optional
        Fitted Colburn parameters (params["colburn_beta"]). None means use
        Manglik-Bergles (Eq 19).

    Returns
    -------
    h : ndarray, broadcast shape of the inputs
        Heat transfer coefficient [W/(m^2 K)].
    """
    # 1. G = mdot / A_ff; Re = reynolds(mdot, mu, side_geom).
    # 2. j = colburn_j(Re, alpha_mb, delta_mb, gamma_mb) if beta is None,
    #    else colburn_j_surface(Re, beta). Multiply by zeta.
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
    raise NotImplementedError


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
    # Check: base case (h ~ 450, k_f = 180, t = 0.1 mm, b = 6.25 mm) gives
    #    m ell_f ~ 0.7 and eta_f ~ 0.86. Larger h or taller fins -> lower eta_f.
    raise NotImplementedError


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
    # 1. eta_o = 1 - (A_f / A_prime) (1 - eta_f).
    # Check: eta_f = 1 gives eta_o = 1; eta_f = 0 gives eta_o = A_p/A_prime.
    raise NotImplementedError


def side_conductance(T, mdot, surr, side_geom, k_f, zeta=1.0, beta=None):
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
    zeta : float, optional
        Colburn multiplier [-].
    beta : array_like, shape (4,), optional
        Fitted Colburn parameters, or None for Manglik-Bergles.

    Returns
    -------
    a : ndarray, same shape as T
        Side conductance per unit length [W/(m K)].
    """
    # 1. cp, mu, k = properties.evaluate(surr, name, T) for each name. This
    #    also enforces the fitted range.
    # 2. h = heat_transfer_coefficient(mdot, cp, mu, k, side_geom, zeta, beta).
    # 3. eta_f = fin_efficiency(h, k_f, side_geom);
    #    eta_o = surface_efficiency(eta_f, side_geom).
    # 4. a = eta_o h A_prime.
    # Pitfall: properties at this stream's temperature only. The cold-side
    #    conductance uses T_c and props["c"], never T_h.
    # Check: constant_surrogate input gives the same a for every T (a
    #    constant); base case a ~ 3-4 kW/(m K) per side.
    raise NotImplementedError


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
        Uses k_f, zeta_j, colburn_beta.
    props : dict
        {"h": hot surrogate, "c": cold surrogate}.
    geom : dict
        From exchanger_geometry(params).

    Returns
    -------
    a_h, a_c : ndarray, same shape as T_h
        Side conductances per unit length [W/(m K)].
    """
    # 1. a_h = side_conductance(T_h, mdot_h, props["h"], geom["h"], k_f,
    #    zeta_j, colburn_beta).
    # 2. a_c = side_conductance(T_c, mdot_c, props["c"], geom["c"], k_f,
    #    zeta_j, colburn_beta).
    # Check: swapping the hot and cold inputs (with identical surrogates and
    #    geometry) swaps the outputs.
    raise NotImplementedError


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
    # 1. U_prime = a_h a_c / (a_h + a_c) (same as 1/(1/a_h + 1/a_c), but
    #    with one division).
    # Check: a_h = a_c = a gives U' = a/2; a_c -> infinity gives U' -> a_h.
    raise NotImplementedError
