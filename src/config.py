"""Base-case inputs, scenario settings, numerical settings and ALPEMA limits.

This is the only place numbers live (CLAUDE.md: "Base-case inputs and ALPEMA
limits live in src/config.py. Never hard-code them anywhere else.").
Equation numbers refer to docs/bahx_model_math.tex.

How to use
----------
    from src import config
    params = dict(config.BASE_CASE)          # a copy; never edit BASE_CASE itself

and pass ``params`` to the solvers. For a study, change a copy:

    from src import sensitivity
    p = sensitivity.with_parameter(params, "b_h", 1.01 * params["b_h"])

Modules in src/ never import this file. Callers pass ``params`` and
``ALPEMA_LIMITS`` in, so every function can be run on a perturbed copy.

Conventions
-----------
- SI units everywhere: K, Pa, kg/s, W, m, s.
- A temperature *difference* of 1 K equals 1 degC, so ALPEMA limits on rates
  and differences are compared with model K values directly (never add 273.15
  to a difference).
- Flows are per passage (one hot passage and one cold passage form the
  repeating unit, assumption A1).

Status
------
Values marked ``None`` are TODO: the team fills them in together. Each comment
gives a suggested starting value. The suggestions for the operating point were
checked with CoolProp 8.0 (methane):
- cold side at 1.5 MPa: T_sat = 158.5 K, so 235 K inlet has ~76 K of superheat;
  cp varies < 2 % over 225-310 K.
- hot side at 6 MPa (supercritical): the pseudo-critical cp peak is at
  T_pc = 199.6 K (cp ~ 19.4 kJ/(kg K)). cp/cp(300 K) is 1.30 at 235 K, 1.50 at
  225 K, 1.67 at 220 K and 2.43 at 210 K. With T_c_in = 235 K and a 10 K fit
  margin, a degree-6 polynomial fits hot-side cp to < 1e-3 relative error;
  with T_c_in = 220 K even degree 8 only reaches ~4e-3 (spline territory).
- with equal flows the cold stream is C_min (C_r ~ 0.8).
"""

# Valid values of the ``scenario`` argument used by properties, transient and
# metrics. The numbers for each scenario live in BASE_CASE group 5.
SCENARIOS = ("none", "feed_ramp", "cold_trip")


BASE_CASE = {
    # ------------------------------------------------------------------
    # 1. Fluid and pressures (Section 1, assumption A5)
    # ------------------------------------------------------------------
    # CoolProp fluid name for both streams. Pure methane is fixed by the
    # project scope (single-phase gas/gas exchanger).
    "fluid": "Methane",
    # p_h [Pa]: hot (feed gas) pressure, held constant along the core (A5).
    # Enters only through the hot-side properties.
    # TODO: start ~6.0e6 (supercritical: methane P_crit = 4.599 MPa, T_crit = 190.6 K).
    "p_h": None,
    # p_c [Pa]: cold (residue gas) pressure, held constant along the core.
    # Sets the cold-side properties and the saturation temperature to stay above.
    # TODO: start ~1.5e6 (T_sat = 158.5 K).
    "p_c": None,

    # ------------------------------------------------------------------
    # 2. Inlet temperatures and flows (Eq 7, Eq 37-38 initial values)
    # ------------------------------------------------------------------
    # T_h,in [K]: hot inlet temperature at x = 0 (boundary condition, Eq 7).
    # TODO: start ~300.
    "T_h_in": None,
    # T_c,in [K]: cold inlet temperature at x = L (boundary condition, Eq 7).
    # The coldest temperature anywhere in the core, so it sets how close the
    # hot stream gets to its cp peak at 6 MPa (T_pc = 199.6 K).
    # TODO: start ~235 (cp_h(235)/cp_h(300) = 1.30). 220 K is a harder variant
    # for later: cp ratio 1.67 and a polynomial fit no longer reaches 1e-3.
    "T_c_in": None,
    # mdot_h [kg/s]: hot mass flow per passage. Sets C_h = mdot_h*cp_h,
    # G_h = mdot_h/A_ff,h and hence Re_h and h_h (Eq 19-21).
    # TODO: start ~0.1 (G ~ 17 kg/(m^2 s), Re ~ 3000-4500 with the fins below,
    # inside the Manglik-Bergles range).
    "mdot_h": None,
    # mdot_c [kg/s]: cold mass flow per passage (initial value in a trip, Eq 38).
    # The flow ratio mdot_c/mdot_h sets C_r.
    # TODO: start ~0.1 (equal flows; cold is then C_min with C_r ~ 0.8).
    "mdot_c": None,

    # ------------------------------------------------------------------
    # 3. Core and fin geometry (Section 4.1, Eq 17-18)
    # ------------------------------------------------------------------
    # L [m]: core length in the flow direction (domain 0 <= x <= L).
    # TODO: start ~0.7. A rough hand estimate with the suggested fins and flows
    # gives NTU ~ 7 per metre (U' ~ 1.7 kW/(m K), C_min ~ 230 W/K), so 0.7 m
    # gives NTU ~ 5. Re-check with steady.ntu_from_effectiveness once
    # correlations.py works.
    "L": None,
    # W [m]: core width. Scales A_ff, A' and A_m together, so it cancels out of
    # NTU at fixed G; it sets the absolute duty per passage.
    # TODO: start ~1.0.
    "W": None,
    # Hot-side offset strip fin. Suggested start: a serrated fin like Kays &
    # London surface 1/8-15.61 (1/8 in strip length, 15.61 fins per inch).
    # Confirm the dimensions against the Kays & London table before use.
    # s_h [m]: clear fin spacing s (Eq 17: alpha_mb = s/b, gamma_mb = t/s).
    # TODO: start ~1.525e-3 (fin pitch 25.4 mm/15.61 = 1.627 mm minus t).
    "s_h": None,
    # b_h [m]: clear fin height b (plate spacing b + t). Fin efficiency uses
    # ell_f = b/2 (Eq 22). Called b, not h, to avoid a clash with the heat
    # transfer coefficient.
    # TODO: start ~6.25e-3 (plate spacing 6.35 mm minus t).
    "b_h": None,
    # t_fin_h [m]: fin thickness t (Eq 17-18, 22). Named t_fin, not t, so it
    # cannot be confused with time t or with temperatures T.
    # TODO: start ~1.02e-4 (0.004 in).
    "t_fin_h": None,
    # l_h [m]: strip (serration) length l (Eq 17: delta_mb = t/l; Eq 22 edge term).
    # TODO: start ~3.175e-3 (1/8 in).
    "l_h": None,
    # Cold-side offset strip fin: same meaning as the hot side.
    # TODO: start with the same fin as the hot side.
    "s_c": None,
    "b_c": None,
    "t_fin_c": None,
    "l_c": None,

    # ------------------------------------------------------------------
    # 4. Parting sheet and aluminum properties (A4, Eq 29, 34, 53)
    # ------------------------------------------------------------------
    # t_w [m]: parting sheet thickness. Enters only the metal cross-section
    # A_m = W[2 t_w + n_h t_h b_h + n_c t_c b_c] (Section 4.1); its conduction
    # resistance is neglected (A4). Named t_sheet so it is not confused with T_w.
    # TODO: start ~1.0e-3.
    "t_sheet": None,
    # rho_m [kg/m^3]: aluminum density, in C'_m = rho_m c_m A_m (Eq 29, 34).
    # TODO: start ~2700.
    "rho_m": None,
    # c_m [J/(kg K)]: aluminum specific heat, in C'_m. It falls with
    # temperature (~800 at 200 K, ~900 at 300 K); one constant is fine here.
    # TODO: start ~880.
    "c_m": None,
    # k_f [W/(m K)]: fin conductivity in the fin parameter m (Eq 22).
    # TODO: start ~180 (3003-series aluminum; check a datasheet).
    "k_f": None,
    # k_m [W/(m K)]: metal conductivity for axial conduction (Eq 29, 34).
    # The math doc writes k_w in A4 and k_m in Eq 29; same quantity.
    # TODO: start ~180.
    "k_m": None,
    # E [Pa]: Young's modulus for the thermal stress estimate (Eq 53, stretch).
    # TODO: start ~70e9 (math doc Section 9.2).
    "E_m": None,
    # alpha_T [1/K]: thermal expansion coefficient (Eq 53).
    # TODO: start ~23e-6 (math doc Section 9.2).
    "alpha_T": None,
    # nu [-]: Poisson's ratio (Eq 53).
    # TODO: start ~0.33 (math doc Section 9.2).
    "nu_m": None,

    # ------------------------------------------------------------------
    # 5. Upset scenarios (Section 6.5, Eq 37-38)
    # ------------------------------------------------------------------
    # Feed temperature ramp, scenario "feed_ramp":
    #   T_h,in(t) = T_h,in,0 + Delta T_r * psi((t - t_start)/t_r)   (Eq 37)
    # Delta T_r [K]: size of the hot-inlet change (negative for a cooldown).
    # The property fit range is widened to cover T_h_in + ramp_dT.
    # TODO: start ~10.
    "ramp_dT": None,
    # t_r [s]: ramp duration. Nominal ramp rate r = ramp_dT/ramp_t; 300 s with
    # 10 K gives 2 K/min, between the 1 and 5 degC/min ALPEMA limits.
    # TODO: start ~300.
    "ramp_t": None,
    # t_start [s]: when the ramp begins. The smoothed ramp is not exactly zero
    # before t_start, so start several corner widths (ramp_t/ramp_sharpness)
    # after t = 0 to keep the initial state steady.
    # TODO: start ~120.
    "ramp_t_start": None,
    # omega [-]: ramp corner sharpness in psi (corner width ~ ramp_t/omega).
    # Peak inlet slope is r*tanh(omega/4), within 0.5 % of r for omega >= 12.
    # TODO: start ~20.
    "ramp_sharpness": None,
    # Cold-stream trip, scenario "cold_trip":
    #   mdot_c(t) = mdot_c,0 [phi + (1 - phi) exp(-(t - t_start)/tau_v)]  (Eq 38)
    # tau_v [s]: valve closing time constant.
    # TODO: start ~10.
    "trip_tau_v": None,
    # phi [-]: residual cold-flow fraction after the trip (must be > 0).
    # TODO: start ~0.05.
    "trip_phi": None,
    # t_start [s]: when the valve starts closing.
    # TODO: start ~60.
    "trip_t_start": None,

    # ------------------------------------------------------------------
    # 6. Model options (Sections 6.1, 8.1, 8.3)
    # ------------------------------------------------------------------
    # zeta [-]: Colburn-factor multiplier j -> zeta*j on both sides (Section 8.3).
    # 1.0 is the unmodified correlation; sensitivity studies perturb it.
    "zeta_j": 1.0,
    # beta [-], length 4: fitted Colburn parameters (Eq 48) from regression.py.
    # None (a final value, not a TODO) means use the general Manglik-Bergles
    # correlation (Eq 19) on both sides.
    "colburn_beta": None,
    # Include the k_m A_m d2T_w/dx2 term in the wall equation (Eq 29, 34).
    # TODO: start False (base model); turn on for the axial-conduction study.
    "axial_conduction": None,
    # Path (relative to the repo root) of the CSV of (Re, j) data for one Kays
    # & London surface, used by regression.py (Section 8.1).
    # TODO: e.g. "data/kays_london_1-8-15.61.csv" once digitized.
    "colburn_data_file": None,

    # ------------------------------------------------------------------
    # 7. Numerical settings (Sections 5, 6.3, 7.3, 8.3, 8.4)
    # ------------------------------------------------------------------
    # N [-]: number of finite-volume cells in the transient model (Eq 32-34).
    # Grid convergence uses N, 2N, 4N (Eq 35).
    # TODO: start ~100.
    "N": None,
    # Number of evenly spaced points at which steady solutions are reported.
    # TODO: start ~501.
    "n_out": None,
    # solve_bvp tolerance on the relative collocation residual (Section 5.1).
    # If the BVP fails, use continuation, never a looser tolerance (CLAUDE.md).
    # TODO: start ~1e-6 (tests tighten it to 1e-8).
    "bvp_tol": None,
    # solve_bvp maximum mesh nodes.
    # TODO: start ~10000.
    "bvp_max_nodes": None,
    # Continuation parameters lambda for Eq 23, used in order if the direct
    # variable-property solve fails.
    # TODO: start (0.0, 0.25, 0.5, 0.75, 1.0) as in Section 5.1.
    "continuation_lams": None,
    # solve_ivp relative tolerance for the transient (BDF).
    # TODO: start ~1e-6.
    "ivp_rtol": None,
    # solve_ivp absolute tolerance [K] for the transient.
    # TODO: start ~1e-6.
    "ivp_atol": None,
    # K [-]: polynomial degree of the property surrogates (Eq 51).
    # TODO: start ~6 (CoolProp check: hot cp error 8e-4 at degree 6 on 225-310 K).
    "fit_degree": None,
    # Number of CoolProp sample points per property fit (Section 8.4).
    # TODO: start ~200.
    "fit_n_points": None,
    # Margin [K] added below the coldest and above the hottest inlet temperature
    # when choosing the fit range. Protects against solver iterates that
    # overshoot slightly; do not "fix" range errors by loosening the range check.
    # TODO: start ~10.
    "fit_T_margin": None,
    # Target maximum relative error of every surrogate (Section 8.4).
    # build_surrogates raises if a fit misses it.
    # TODO: start ~1e-3.
    "fit_max_rel_err": None,
    # Solver relative tolerance to use when outputs are differenced
    # (Section 7.3: Delta p/p ~ eta^(1/3)).
    # TODO: start ~1e-8.
    "sens_rtol": None,
    # Number of Monte Carlo samples for uncertainty propagation (Section 8.3).
    # TODO: start ~300.
    "mc_samples": None,
    # Random seed for Monte Carlo, so results are reproducible.
    # TODO: start 0.
    "mc_seed": None,
}


# ALPEMA Integrity Operating Window limits for BAHX (ALPEMA IOW table, June 2025;
# math doc Section 6.6). The table applies them to temperatures measured
# "within 5 pipe diameters of the inlet and outlet nozzles". Differences and
# rates in degC equal those in K.
ALPEMA_LIMITS = {
    # Max |dT/dt| [degC/min] for frequent events (normal "steady-state" operation).
    "rate_frequent": 1.0,
    # Max |dT/dt| [degC/min] for infrequent events (startup, shutdown, trips).
    "rate_infrequent": 5.0,
    # Max temperature change [degC] over any 60-minute window for infrequent events.
    "change_per_hour_infrequent": 60.0,
    # Length [s] of the window used for the hourly limit.
    "hour_window": 3600.0,
    # Max stream-to-stream temperature difference [degC] ("< ~30 C" at the warm
    # end, the cold end and mid stream).
    "dT_stream_stream": 30.0,
    # Max stream-to-metal temperature difference [degC].
    "dT_stream_metal": 30.0,
}
