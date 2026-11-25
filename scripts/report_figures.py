"""Figures and tables for the project report notebook.

The report notebook shows results and short excerpts of the model code; the
code that draws its figures and builds its tables lives here, so each notebook
cell is a call or two. Everything here is post-processing and plotting: the
model itself is in src/.

Each figure function shows the figure, saves it to figures/ and returns
nothing unless stated. Each table function displays a markdown table (in a
notebook) or prints it as text (in a plain Python run).

Every number in a title, label or table comes from params (config.BASE_CASE)
or limits (config.ALPEMA_LIMITS), so the figures stay right if the base case
changes. The ALPEMA metrics, the grid-convergence order and the thermal stress
come from src/metrics.py and src/stress.py, not from copies written here.

Use from the notebook (after the setup cell puts the repo root on sys.path):
    from scripts import report_figures as rf
    rf.set_style()
or regenerate every report figure and table from the repo root:
    python -m scripts.make_report_figures
"""

import contextlib
import io
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
from IPython import get_ipython
from IPython.display import Image, Markdown, display

from src import config, correlations, metrics, properties, steady, stress, transient

# Each physical quantity keeps one color throughout the report: cold stream blue,
# hot stream orange, metal wall aqua. Grid levels N, 2N, 4N use light-to-dark blue,
# and ALPEMA limits are dashed gray lines with a text label.
COLORS = {"cold": "#2a78d6", "hot": "#eb6834", "wall": "#1baf7a"}
GRID_SHADES = ["#86b6ef", "#2a78d6", "#104281"]
INK, INK_2 = "#0b0b0b", "#52514e"


def set_style():
    """Set the report's matplotlib style. Inputs: none. Output: none."""
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
        "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.6,
        "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 10, "axes.titleweight": "bold", "axes.labelsize": 10,
        "font.size": 10, "legend.frameon": False, "legend.fontsize": 8.5,
        "lines.linewidth": 1.6, "lines.markersize": 5,
    })


def use_file_backend():
    """Draw to files only, for plain Python runs (python -m scripts.<name>).

    Inputs: none. Output: none. The figure functions end with plt.show() so that a notebook
    displays them; with this backend plt.show() opens no window and the run does not stop.
    """
    plt.switch_backend("Agg")
    warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def markdown_table(header, rows):
    """Display a table.

    Inputs: header, list of column titles (str); rows, list of lists of cell strings.
    Output: none (displays the table in a notebook, prints its markdown text otherwise; first
    column left-aligned, the rest right-aligned).
    """
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join([":--"] + ["--:"] * (len(header) - 1)) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    text = "\n".join(lines)
    if get_ipython() is None:
        # Plain Python (python -m scripts.make_report_figures): print the markdown text, with a
        # blank line on each side so it still renders as a table after other printed lines.
        print("\n" + text + "\n")
    else:
        display(Markdown(text))


def draw_limit(ax, value, label):
    """Draw a horizontal ALPEMA limit as a dashed gray line with a text label at its right end.

    Inputs: ax (matplotlib Axes), value (limit in the y-axis units), label (str).
    Output: none (draws on ax).
    """
    ax.axhline(value, color=INK_2, linestyle="--", linewidth=1.0)
    ax.annotate(label, xy=(1.0, value), xycoords=("axes fraction", "data"), xytext=(-3, 3),
                textcoords="offset points", ha="right", va="bottom", fontsize=8, color=INK_2)


def build_surrogates_quietly(params, scenario):
    """properties.build_surrogates without its progress printout (the fit errors are tabulated instead).

    Inputs: params (dict), scenario (str). Output: props, {"h": hot surrogate, "c": cold surrogate}.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        return properties.build_surrogates(params, scenario)


def alpema_histories(r):
    """ALPEMA comparison quantities versus time for a full-model transient result (math doc Eq 39-40).

    Input: r, result dict from transient.simulate(..., model="full").
    Output: the dict from metrics.alpema_metrics. The entries used here, each shape (n_t,):
        "R_w"          wall heating rate, max over cells of |dT_w/dt| [K/min]
        "R_end"        outlet-nozzle rate, max(|dT_h/dt| in cell N, |dT_c/dt| in cell 1) [K/min]
        "dT_ss"        stream-to-stream difference, largest over the cells and the two end faces [K]
        "dT_sm"        stream-to-metal difference, largest over the cells and the two end faces [K]
        "dT_ss_cells", "dT_sm_cells"  the same two over the cells only (they depend on the grid) [K]
        "dT_cold_end"  stream-to-stream difference at the cold-end nozzles, T_h,out - T_c,in [K]
        "dT_warm_end"  stream-to-stream difference at the warm-end nozzles, T_h,in - T_c,out [K]
    plus the scalar "max_dT_hour_wall", the largest metal temperature change in any 60 min [K].
    The rates come from the model right-hand side at the saved times (r["dTw_dt"] etc.),
    never from differencing the solution. A rate of 1 K/min equals 1 degC/min.
    """
    return metrics.alpema_metrics(r, window=config.ALPEMA_LIMITS["hour_window"])


def observed_order(y_N, y_2N, y_4N):
    """Observed order of grid convergence and Richardson-extrapolated value (math doc Eq 35).

    Inputs: y_N, y_2N, y_4N, one output on grids of N, 2N and 4N cells (floats).
    Outputs: p_obs [-] and y_star (units of y), from metrics.observed_order. Both are nan if
    the three values do not converge monotonically; p_obs is nan and y_star = y_4N if y_2N
    and y_4N already agree to round-off. (The tables show "n/a" for nan.)
    """
    if abs(y_2N - y_4N) < 1e-9 * max(abs(y_4N), 1.0):
        return np.nan, y_4N
    with warnings.catch_warnings():
        # metrics.observed_order warns for non-monotone values; the tables report them as n/a.
        warnings.simplefilter("ignore")
        p_obs, y_star = metrics.observed_order(y_N, y_2N, y_4N)
    if np.isnan(p_obs):
        return np.nan, np.nan
    return p_obs, y_star


def extrapolated(values):
    """Richardson-extrapolated value of an output from its N, 2N, 4N values (the 4N value if already converged)."""
    p_obs, y_star = observed_order(*values)
    return values[-1] if np.isnan(y_star) else y_star


def convergence_table(rows, grids):
    """Display N, 2N, 4N values of several outputs with their observed order and extrapolated value.

    Inputs: rows, list of (name, units, [y_N, y_2N, y_4N]); grids, the three cell counts.
    Output: none (displays the table).
    """
    table = []
    for name, units, y in rows:
        p_obs, y_star = observed_order(*y)
        table.append([f"{name} [{units}]", *(f"{v:.3f}" for v in y),
                      "n/a" if np.isnan(p_obs) else f"{p_obs:.2f}",
                      "n/a" if np.isnan(y_star) else f"{y_star:.3f}"])
    markdown_table(["Output", *(f"N = {N}" for N in grids), "$p_{obs}$", "extrapolated $y^\\star$"], table)


def run_grids(params, props, scenario, t_end, t_eval, grids):
    """Run one scenario on each grid and compute its ALPEMA histories.

    Inputs: params (dict), props (surrogates for the scenario), scenario (str), t_end [s],
    t_eval (saved times, s), grids (cell counts). Prints the run time per grid.
    Outputs: (runs, histories), dicts keyed by N: transient.simulate results and
    alpema_histories of them.
    """
    runs, histories = {}, {}
    for N in grids:
        start = time.perf_counter()
        runs[N] = transient.simulate(dict(params, N=N), props, scenario, t_end, t_eval=t_eval)
        histories[N] = alpema_histories(runs[N])
        print(f"N = {N}: {3 * N} equations, {time.perf_counter() - start:.2f} s")
    return runs, histories


# ---------------------------------------------------------------------------
# Section 2: methods figure
# ---------------------------------------------------------------------------

def fig_fv_grid(fig_dir, fig_label="Figure 1"):
    """Figure 1: one repeating unit (hot passage, metal, cold passage) divided into finite-volume cells."""
    fig, ax = plt.subplots(figsize=(10, 3.9))
    n_cells, i = 7, 3                                   # draw 7 cells and detail cell i (the 4th)
    bands = {"hot": (2.4, 3.4), "wall": (2.0, 2.4), "cold": (1.0, 2.0)}
    tints = {"hot": "#fbe1d6", "wall": "#cff0e3", "cold": "#dbe9fb"}
    names = {"hot": "hot gas", "wall": "metal", "cold": "cold gas"}
    for band, (y0, y1) in bands.items():
        ax.add_patch(plt.Rectangle((0, y0), n_cells, y1 - y0, facecolor=tints[band], edgecolor="none"))
        ax.text(n_cells - 0.06, (y0 + y1) / 2, names[band], ha="right", va="center", fontsize=8, color=INK_2)
    for k in range(n_cells + 1):                        # cell faces
        ax.plot([k, k], [1.0, 3.4], color="#c3c2b7", linewidth=0.8)
    ax.add_patch(plt.Rectangle((i, 1.0), 1, 2.4, facecolor="none", edgecolor=INK, linewidth=1.5))
    for k, label in [(0, "1"), (1, r"$\cdots$"), (i - 1, "$i-1$"), (i, "$i$"), (i + 1, "$i+1$"),
                     (n_cells - 2, r"$\cdots$"), (n_cells - 1, "$N$")]:
        ax.text(k + 0.5, 3.95, label, ha="center", va="center")
    ax.text(n_cells / 2, 4.3, "cell", ha="center", va="center", fontsize=8, color=INK_2)

    # Temperatures and metal exchange in cell i.
    ax.text(i + 0.3, 2.9, "$T_{h,i}$", ha="center", va="center")
    ax.text(i + 0.3, 2.2, "$T_{w,i}$", ha="center", va="center", fontsize=9)
    ax.text(i + 0.3, 1.5, "$T_{c,i}$", ha="center", va="center")
    arrow = dict(arrowstyle="-|>", color=INK, linewidth=1.1)
    ax.annotate("", xy=(i + 0.75, 2.27), xytext=(i + 0.75, 2.85), arrowprops=arrow)
    ax.annotate("", xy=(i + 0.75, 1.55), xytext=(i + 0.75, 2.13), arrowprops=arrow)
    ax.text(i + 1.08, 2.62, r"$a_{h,i}\Delta x\,(T_{h,i}-T_{w,i})$", va="center", fontsize=9)
    ax.text(i + 1.08, 1.78, r"$a_{c,i}\Delta x\,(T_{w,i}-T_{c,i})$", va="center", fontsize=9)

    # Enthalpy carried across the faces of cell i (hot above the core, cold below it).
    for x_face, y_arrow, direction, label, side in [
        (i, 3.62, +1, r"$\dot m_h\,\hat h_h(T_{h,i-1})$", "left"),
        (i + 1, 3.62, +1, r"$\dot m_h\,\hat h_h(T_{h,i})$", "right"),
        (i + 1, 0.78, -1, r"$\dot m_c\,\hat h_c(T_{c,i+1})$", "right"),
        (i, 0.78, -1, r"$\dot m_c\,\hat h_c(T_{c,i})$", "left"),
    ]:
        ax.annotate("", xy=(x_face + 0.3 * direction, y_arrow), xytext=(x_face - 0.3 * direction, y_arrow),
                    arrowprops=dict(arrowstyle="-|>", color=COLORS["hot" if direction > 0 else "cold"],
                                    linewidth=1.4))
        if side == "left":
            ax.text(x_face - 0.4, y_arrow, label, ha="right", va="center", fontsize=9)
        else:
            ax.text(x_face + 0.4, y_arrow, label, ha="left", va="center", fontsize=9)

    # Inlets and outlets at the two ends of the core.
    ends = [(-1.0, 0.0, 2.9, "hot", "hot in\n$T_{h,\\mathrm{in}}(t)$", "right"),
            (n_cells, n_cells + 1.0, 2.9, "hot", "hot out", "left"),
            (n_cells + 1.0, n_cells, 1.5, "cold", "cold in\n$T_{c,\\mathrm{in}}(t)$", "left"),
            (0.0, -1.0, 1.5, "cold", "cold out", "right")]
    for x_from, x_to, y, stream_name, label, side in ends:
        ax.annotate("", xy=(x_to, y), xytext=(x_from, y),
                    arrowprops=dict(arrowstyle="-|>", color=COLORS[stream_name], linewidth=1.8))
        x_text = min(x_from, x_to) - 0.08 if side == "right" else max(x_from, x_to) + 0.08
        ax.text(x_text, y, label, ha=side, va="center", fontsize=9)
    ax.text(0, 0.45, "$x = 0$", ha="center", va="center", fontsize=9)
    ax.text(n_cells, 0.45, "$x = L$", ha="center", va="center", fontsize=9)

    ax.set_xlim(-2.4, n_cells + 2.4)
    ax.set_ylim(0.3, 4.45)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"{fig_label}. Repeating unit split into N finite-volume cells (hot flows +x, cold flows -x)")
    fig.savefig(fig_dir / "report_fv_grid.png")
    plt.show()


# ---------------------------------------------------------------------------
# Sections 3.1-3.4: base case, surrogates, steady model
# ---------------------------------------------------------------------------

def base_case_tables(params, limits, grids):
    """Display the base-case inputs and the ALPEMA limits (both read from src/config.py).

    Inputs: params (config.BASE_CASE copy), limits (config.ALPEMA_LIMITS), grids (N, 2N, 4N).
    """
    p = params
    mm = 1e3  # m -> mm
    markdown_table(["Quantity", "Value"], [
        ["Fluid (both streams)", f"{p['fluid']}, single phase"],
        ["Hot (feed gas) pressure $p_h$", f"{p['p_h'] / 1e6:.1f} MPa"],
        ["Cold (residue gas) pressure $p_c$", f"{p['p_c'] / 1e6:.1f} MPa"],
        ["Inlet temperatures $T_{h,in}$ / $T_{c,in}$", f"{p['T_h_in']:.0f} K / {p['T_c_in']:.0f} K"],
        ["Mass flow per passage $\\dot m_h$ / $\\dot m_c$", f"{p['mdot_h']} kg/s / {p['mdot_c']} kg/s"],
        ["Core length $L$ / width $W$", f"{p['L']} m / {p['W']} m"],
        ["Fin spacing $s$, height $b$, thickness $t$, strip length $l$ (both sides)",
         f"{p['s_h'] * mm:.3f}, {p['b_h'] * mm:.2f}, {p['t_fin_h'] * mm:.3f}, {p['l_h'] * mm:.3f} mm"],
        ["Parting sheet thickness $t_w$", f"{p['t_sheet'] * mm:.1f} mm"],
        ["Aluminum density, heat capacity, conductivity",
         f"{p['rho_m']:.0f} kg/m$^3$, {p['c_m']:.0f} J/(kg K), {p['k_f']:.0f} W/(m K)"],
        ["Aluminum Young's modulus, expansion coefficient, Poisson's ratio",
         f"{p['E_m'] / 1e9:.0f} GPa, {p['alpha_T'] * 1e6:.0f}e-6 1/K, {p['nu_m']}"],
        ["Feed ramp: size, duration, start",
         f"+{p['ramp_dT']:.0f} K over {p['ramp_t']:.0f} s "
         f"({60 * p['ramp_dT'] / p['ramp_t']:.0f} K/min), starting at t = {p['ramp_t_start']:.0f} s"],
        ["Cold trip: valve time constant, residual flow, start",
         f"$\\tau_v$ = {p['trip_tau_v']:.0f} s, $\\varphi$ = {p['trip_phi']}, starting at t = {p['trip_t_start']:.0f} s"],
        ["Startup: idle core temperature, flow opening time, start",
         f"{p['startup_T0']:.0f} K, {p['startup_t_flow']:.0f} s, starting at t = {p['startup_t_start']:.0f} s"],
        ["Axial conduction in the metal", "off" if not p["axial_conduction"] else "on"],
        ["Cells $N$ (grid study)", f"{grids[0]} ({', '.join(str(N) for N in grids)})"],
        ["Tolerances: solve_bvp; BDF rtol / atol",
         f"{p['bvp_tol']:.0e}; {p['ivp_rtol']:.0e} / {p['ivp_atol']:.0e} K"],
        ["Property surrogates: degree, points, margin, max error",
         f"{p['fit_degree']}, {p['fit_n_points']}, {p['fit_T_margin']:.0f} K, {p['fit_max_rel_err']:.0e}"],
    ])
    markdown_table(["ALPEMA limit (ALPEMA, 2025)", "Value"], [
        ["Rate of change, frequent events", f"{limits['rate_frequent']:.0f} °C/min"],
        ["Rate of change, startup, shutdown and trips", f"{limits['rate_infrequent']:.0f} °C/min"],
        ["Change in any 60 min, startup, shutdown and trips", f"{limits['change_per_hour_infrequent']:.0f} °C"],
        ["Stream-to-stream difference", f"about {limits['dT_stream_stream']:.0f} °C"],
        ["Stream-to-metal difference", f"about {limits['dT_stream_metal']:.0f} °C"],
    ])


def surrogate_tables(params, props, props_ramp):
    """Display the surrogate fit errors and the distance of the fit range from the phase hazards.

    Inputs: params, props (steady/trip/startup surrogates), props_ramp (feed-ramp surrogates).
    """
    rows = []
    for label, pr in (("steady, trip and startup", props), ("feed ramp", props_ramp)):
        for side, stream_name in (("h", f"hot, {params['p_h'] / 1e6:g} MPa"),
                                  ("c", f"cold, {params['p_c'] / 1e6:g} MPa")):
            err = pr[side]["max_rel_err"]
            rows.append([f"{stream_name}, {label}", f"{pr[side]['T_lo']:.0f}-{pr[side]['T_hi']:.0f} K",
                         *(f"{err[name]:.1e}" for name in ("cp", "mu", "k", "rho", "hhat"))])
    markdown_table(["Surrogate", "Range", "$c_p$", "$\\mu$", "$k$", "$\\rho$", "$\\hat h$"], rows)

    # Distance from the bottom of the widest fit range to the nearest phase-related hazard.
    rows = []
    for side, key, stream_name in (("h", "p_h", "hot"), ("c", "p_c", "cold")):
        report = properties.check_single_phase(params["fluid"], params[key],
                                               props_ramp[side]["T_lo"], props_ramp[side]["T_hi"])
        hazard = (f"pseudo-critical $c_p$ peak at {report['T_pc']:.1f} K" if report["supercritical"]
                  else f"saturation at {report['T_sat']:.1f} K")
        rows.append([f"{stream_name}, {params[key] / 1e6:.1f} MPa", hazard,
                     f"{report['margin']:.1f} K", f"{report['cp_ratio']:.2f}"])
    markdown_table(["Stream", "Nearest hazard", "Margin below fit range", "max/min $c_p$ over range"], rows)


def fig_surrogates(params, props_ramp, fig_dir, fig_label="Figure 2"):
    """Figure 2: CoolProp heat capacity along both isobars, the fit window, and the surrogate error."""
    T_wide = np.linspace(165.0, 330.0, 400)
    cp_hot = properties.coolprop_table(params["fluid"], params["p_h"], T_wide)["cp"]
    cp_cold = properties.coolprop_table(params["fluid"], params["p_c"], T_wide)["cp"]

    fig, (ax_cp, ax_err) = plt.subplots(1, 2, figsize=(11, 3.8))
    T_lo, T_hi = props_ramp["h"]["T_lo"], props_ramp["h"]["T_hi"]
    ax_cp.axvspan(T_lo, T_hi, color="#f0efec", zorder=0)
    ax_cp.text((T_lo + T_hi) / 2, 17.5, "fit range", ha="center", fontsize=8, color=INK_2)
    ax_cp.plot(T_wide, cp_hot / 1e3, color=COLORS["hot"], label=f"hot side, {params['p_h'] / 1e6:g} MPa")
    ax_cp.plot(T_wide, cp_cold / 1e3, color=COLORS["cold"], label=f"cold side, {params['p_c'] / 1e6:g} MPa")
    i_peak = np.argmax(cp_hot)
    ax_cp.annotate(f"pseudo-critical peak\n{T_wide[i_peak]:.0f} K", (T_wide[i_peak], cp_hot[i_peak] / 1e3),
                   xytext=(12, -8), textcoords="offset points", fontsize=8, va="top")
    ax_cp.set_xlabel("T [K]")
    ax_cp.set_ylabel("$c_p$ [kJ/(kg K)]")
    ax_cp.set_title("(a) Methane heat capacity from CoolProp")
    ax_cp.legend(loc="center right")

    # Relative error of the c_p surrogate against fresh CoolProp values on a fine grid.
    for side, key, stream_name in (("h", "p_h", "hot side"), ("c", "p_c", "cold side")):
        surr = props_ramp[side]
        T_check = np.linspace(surr["T_lo"], surr["T_hi"], 500)
        cp_ref = properties.coolprop_table(params["fluid"], params[key], T_check)["cp"]
        rel_err = np.abs(properties.evaluate(surr, "cp", T_check) - cp_ref) / cp_ref
        ax_err.semilogy(T_check, rel_err, color=COLORS["hot" if side == "h" else "cold"], linewidth=1.2,
                        label=stream_name)
    draw_limit(ax_err, params["fit_max_rel_err"], f"acceptance limit {params['fit_max_rel_err']:g}")
    ax_err.set_ylim(1e-10, 1e-2)
    ax_err.set_xlabel("T [K]")
    ax_err.set_ylabel("relative error of $c_p$ surrogate")
    ax_err.set_title(f"(b) Surrogate error over the fit range (degree {params['fit_degree']})")
    ax_err.legend(loc="lower right")
    fig.suptitle(f"{fig_label}. Property surrogates", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(fig_dir / "report_surrogates.png")
    plt.show()


def steady_verification(params, fig_dir, fig_label="Figure 3"):
    """Figure 3 and its table: constant-property BVP solutions against the analytical solution.

    The same inputs as tests/test_steady.py: fixed mu, k and rho, and heat capacities and
    flows chosen to set C_r. solve_bvp controls the collocation residual, not the solution
    error, so its tolerance is tightened to 1e-8 for this comparison.
    """
    MU, K_GAS, RHO, CP_H = 1.2e-5, 0.035, 30.0, 2500.0
    cases = {
        "hot is $C_{min}$, $C_r$ = 0.5": dict(mdot_h=0.1, mdot_c=0.2, cp_c=2500.0, NTU=6.0),
        "cold is $C_{min}$, $C_r$ = 0.5": dict(mdot_h=0.2, mdot_c=0.1, cp_c=2500.0, NTU=6.0),
        "balanced, $C_r$ = 1": dict(mdot_h=0.1, mdot_c=0.125, cp_c=2000.0, NTU=4.0),
    }
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    rows = []
    for ax, (name, case) in zip(axes, cases.items()):
        p = dict(params, mdot_h=case["mdot_h"], mdot_c=case["mdot_c"], bvp_tol=1e-8)
        props_const = {"h": properties.constant_surrogate(CP_H, MU, K_GAS, RHO, 200.0, 320.0),
                       "c": properties.constant_surrogate(case["cp_c"], MU, K_GAS, RHO, 200.0, 320.0)}
        C_h, C_c = p["mdot_h"] * CP_H, p["mdot_c"] * case["cp_c"]
        C_min, C_r = min(C_h, C_c), min(C_h, C_c) / max(C_h, C_c)
        geom = correlations.exchanger_geometry(p)
        a_h, a_c = correlations.conductances(270.0, 270.0, p["mdot_h"], p["mdot_c"], p, props_const, geom)
        U_prime = float(correlations.overall_conductance(a_h, a_c))
        p["L"] = case["NTU"] * C_min / U_prime          # core length that gives the target NTU

        s_num = steady.solve_steady(p, props_const)
        T_h_ex, T_c_ex = steady.analytical_profiles(s_num["x"], p["L"], p["T_h_in"], p["T_c_in"], C_h, C_c, U_prime)
        dT_in = p["T_h_in"] - p["T_c_in"]
        theta_h, theta_c = (s_num["T_h"] - p["T_c_in"]) / dT_in, (s_num["T_c"] - p["T_c_in"]) / dT_in
        theta_h_ex, theta_c_ex = (T_h_ex - p["T_c_in"]) / dT_in, (T_c_ex - p["T_c_in"]) / dT_in

        eps_num = C_h * (p["T_h_in"] - s_num["T_h"][-1]) / (C_min * dT_in)
        eps_ex = steady.effectiveness(case["NTU"], C_r)
        max_err = max(np.max(np.abs(theta_h - theta_h_ex)), np.max(np.abs(theta_c - theta_c_ex)))
        rows.append([name, f"{case['NTU']:.0f}", f"{eps_ex:.7f}", f"{eps_num:.7f}", f"{max_err:.1e}"])

        xi = s_num["x"] / p["L"]
        ax.plot(xi, theta_h_ex, color=COLORS["hot"], label="hot, analytical")
        ax.plot(xi, theta_c_ex, color=COLORS["cold"], label="cold, analytical")
        ax.plot(xi[::25], theta_h[::25], "o", color=COLORS["hot"], mfc="white", label="hot, solve_bvp")
        ax.plot(xi[::25], theta_c[::25], "o", color=COLORS["cold"], mfc="white", label="cold, solve_bvp")
        ax.set_title(name)
        ax.set_xlabel("$x/L$")
    axes[0].set_ylabel(r"$\theta = (T - T_{c,in})/(T_{h,in} - T_{c,in})$")
    axes[1].legend(loc="lower left")
    fig.suptitle(f"{fig_label}. Constant-property BVP solution against the analytical solution",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(fig_dir / "report_steady_verification.png")
    plt.show()
    markdown_table(["Case", "NTU", "$\\varepsilon$ analytical", "$\\varepsilon$ solve_bvp",
                    "max $|\\Delta\\theta|$"], rows)


def steady_checks(params, props):
    """Solve the base case by collocation and by shooting and display the energy-closure checks.

    Inputs: params, props. Output: s, the collocation steady result dict.
    """
    s = steady.solve_steady(params, props)                 # collocation (solve_bvp)
    s_shoot = steady.solve_steady_shooting(params, props)  # shooting (solve_ivp + brentq)
    duty = steady.energy_balance(s, params, props)
    markdown_table(["Check", "Result"], [
        ["Hot-stream duty $Q_h$", f"{duty['Q_h'] / 1e3:.6f} kW"],
        ["Cold-stream duty $Q_c$", f"{duty['Q_c'] / 1e3:.6f} kW"],
        ["Integrated heat flow $\\int U'(T_h - T_c)\\,dx$", f"{duty['Q_int'] / 1e3:.6f} kW"],
        ["Closure $|Q_h - Q_c|/Q_h$", f"{duty['closure_hc']:.1e}"],
        ["Closure $|Q_h - Q_{int}|/Q_h$", f"{duty['closure_int']:.1e}"],
        ["max $|T_h|$ difference, collocation vs shooting", f"{np.max(np.abs(s['T_h'] - s_shoot['T_h'])):.1e} K"],
        ["max $|T_c|$ difference, collocation vs shooting", f"{np.max(np.abs(s['T_c'] - s_shoot['T_c'])):.1e} K"],
    ])
    return s


def steady_base_case(s, params, props, limits, fig_dir, fig_label="Figure 4"):
    """Figure 4 and the local heat transfer and exchanger tables for the steady base case."""
    duty = steady.energy_balance(s, params, props)
    geom = correlations.exchanger_geometry(params)
    x, T_h, T_c, T_w = s["x"], s["T_h"], s["T_c"], s["T_w"]

    fig, (ax_T, ax_d) = plt.subplots(1, 2, figsize=(11, 3.9))
    ax_T.plot(x, T_h, color=COLORS["hot"], label="hot gas $T_h$")
    ax_T.plot(x, T_w, color=COLORS["wall"], label="metal $T_w$")
    ax_T.plot(x, T_c, color=COLORS["cold"], label="cold gas $T_c$")
    for xi, Ti, stream_name, text in [(0, T_h[0], "hot", "hot in"), (-1, T_h[-1], "hot", "hot out"),
                                      (-1, T_c[-1], "cold", "cold in"), (0, T_c[0], "cold", "cold out")]:
        filled = text.endswith("in")
        ax_T.plot(x[xi], Ti, "o", color=COLORS[stream_name], mfc=COLORS[stream_name] if filled else "white", zorder=3)
        # Labels sit outside the two ends of the core so they never cross a curve.
        ax_T.annotate(f"{text} {Ti:.1f} K", (x[xi], Ti), xytext=(-7 if xi == 0 else 7, 0),
                      textcoords="offset points", ha="right" if xi == 0 else "left", va="center", fontsize=8)
    L = params["L"]
    ax_T.set_xlim(-0.34 * L, 1.34 * L)             # room for the inlet and outlet labels beside the core
    ax_T.set_xticks(np.linspace(0.0, L, 8))
    ax_T.set_xlabel("x [m]")
    ax_T.set_ylabel("T [K]")
    ax_T.set_title("(a) Temperature profiles")
    ax_T.legend(loc="upper right")

    ax_d.plot(x, T_h - T_c, color=INK, label="stream to stream $T_h - T_c$")
    ax_d.plot(x, T_h - T_w, color=COLORS["hot"], label="hot gas to metal $T_h - T_w$")
    ax_d.plot(x, T_w - T_c, color=COLORS["cold"], label="metal to cold gas $T_w - T_c$")
    draw_limit(ax_d, limits["dT_stream_stream"], f"ALPEMA limit, about {limits['dT_stream_stream']:g} °C")
    ax_d.set_ylim(0, limits["dT_stream_stream"] + 5.0)
    ax_d.set_xlabel("x [m]")
    ax_d.set_ylabel("temperature difference [K]")
    ax_d.set_title("(b) Local temperature differences")
    ax_d.legend(loc="upper left", bbox_to_anchor=(0.0, 0.83))
    fig.suptitle(f"{fig_label}. Steady base case", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(fig_dir / "report_steady_base_case.png")
    plt.show()

    # Local Reynolds number, heat transfer coefficient and efficiencies along each side.
    local = {}
    for side, T_side in (("h", T_h), ("c", T_c)):
        surr, mdot = props[side], params[f"mdot_{side}"]
        cp, mu, k = (properties.evaluate(surr, name, T_side) for name in ("cp", "mu", "k"))
        h = correlations.heat_transfer_coefficient(mdot, cp, mu, k, geom[side])
        eta_f = correlations.fin_efficiency(h, params["k_f"], geom[side])
        local[side] = {"Re": correlations.reynolds(mdot, mu, geom[side]), "h": h, "eta_f": eta_f,
                       "eta_o": correlations.surface_efficiency(eta_f, geom[side]), "C": mdot * cp}

    # NTU from the mean conductance and capacity rates; effectiveness from the enthalpy-based maximum duty.
    C_h, C_c = np.mean(local["h"]["C"]), np.mean(local["c"]["C"])
    NTU = np.mean(s["U_prime"]) * params["L"] / min(C_h, C_c)
    Q_max = min(params["mdot_h"] * (properties.enthalpy(props["h"], params["T_h_in"])
                                    - properties.enthalpy(props["h"], params["T_c_in"])),
                params["mdot_c"] * (properties.enthalpy(props["c"], params["T_h_in"])
                                    - properties.enthalpy(props["c"], params["T_c_in"])))

    def span(values, fmt):
        """Format the min-max range of an array, e.g. '3269-3591'. Inputs: values (array), fmt (format spec)."""
        return f"{np.min(values):{fmt}}-{np.max(values):{fmt}}"

    markdown_table(["Quantity", "Hot side", "Cold side"], [
        ["Reynolds number", span(local["h"]["Re"], ".0f"), span(local["c"]["Re"], ".0f")],
        ["Heat transfer coefficient $h$ [W/(m$^2$ K)]", span(local["h"]["h"], ".0f"), span(local["c"]["h"], ".0f")],
        ["Fin efficiency $\\eta_f$ / surface efficiency $\\eta_o$",
         f"{np.mean(local['h']['eta_f']):.3f} / {np.mean(local['h']['eta_o']):.3f}",
         f"{np.mean(local['c']['eta_f']):.3f} / {np.mean(local['c']['eta_o']):.3f}"],
        ["Side conductance $a$ [W/(m K)]", span(s["a_h"], ".0f"), span(s["a_c"], ".0f")],
        ["Mean capacity rate $C = \\dot m c_p$ [W/K]", f"{C_h:.1f}", f"{C_c:.1f}"],
    ])
    markdown_table(["Exchanger result", "Value"], [
        ["Duty per passage $Q$", f"{duty['Q_h'] / 1e3:.2f} kW"],
        ["Hot outlet / cold outlet", f"{T_h[-1]:.1f} K / {T_c[0]:.1f} K"],
        ["Overall conductance $U'$", f"{span(s['U_prime'], '.0f')} W/(m K)"],
        ["NTU (mean $U'$ and $C$) / $C_r$", f"{NTU:.2f} / {min(C_h, C_c) / max(C_h, C_c):.2f}"],
        ["Effectiveness $Q/Q_{max}$", f"{duty['Q_h'] / Q_max:.3f}"],
        ["Max stream-to-stream difference (location)",
         f"{np.max(T_h - T_c):.1f} K (x = {x[np.argmax(T_h - T_c)]:.2f} m)"],
        ["Max stream-to-metal difference", f"{max(np.max(T_h - T_w), np.max(T_w - T_c)):.1f} K"],
    ])


# ---------------------------------------------------------------------------
# Sections 3.5-3.6: time scales, Jacobian, transient verification
# ---------------------------------------------------------------------------

def time_scales_and_jacobian(params, props, fig_dir, fig_label="Figure 5"):
    """Time-scale tables and Figure 5 (Jacobian sparsity). Output: scales, from transient.time_scales."""
    scales = transient.time_scales(params, props)
    markdown_table(["Time scale", "Hot side", "Cold side"], [
        ["Gas residence time $\\tau_{res}$", f"{scales['tau_res_h']:.2f} s", f"{scales['tau_res_c']:.2f} s"],
        ["Gas-to-metal equilibration $\\tau_g$", f"{scales['tau_g_h']:.3f} s", f"{scales['tau_g_c']:.3f} s"],
    ])
    markdown_table(["Quantity", "Value"], [
        ["Metal relaxation time $\\tau_m$", f"{scales['tau_m']:.2f} s"],
        ["Metal-to-gas heat capacity ratio $\\kappa$", f"{scales['kappa']:.1f}"],
        ["Exchanger response time $(1 + \\kappa)\\max\\tau_{res}$", f"{scales['t_response']:.1f} s"],
        [f"Stiffness ratio estimate $N(1 + \\kappa)$, N = {params['N']}", f"{scales['stiffness']:.0f}"],
    ])

    N_demo = 6
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax, conduction in zip(axes, (False, True)):
        pattern = transient.jacobian_sparsity(N_demo, conduction).toarray()
        ax.spy(pattern, marker="s", markersize=6, color=INK)
        for k in range(1, N_demo):                    # cell boundaries every 3 states
            ax.axhline(3 * k - 0.5, color="#c3c2b7", linewidth=0.6)
            ax.axvline(3 * k - 0.5, color="#c3c2b7", linewidth=0.6)
        ax.grid(False)
        ax.set_title(f"axial conduction {'on' if conduction else 'off'}: {pattern.sum()} nonzeros")
        ax.set_xlabel("state $[T_{h,1}, T_{c,1}, T_{w,1}, T_{h,2}, \\dots]$")
    axes[0].set_ylabel("equation")
    fig.suptitle(f"{fig_label}. Jacobian sparsity pattern for N = {N_demo} cells (3N = {3 * N_demo} states)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(fig_dir / "report_jacobian_sparsity.png")
    plt.show()
    return scales


def transient_checks(params, props, props_ramp, s, scales):
    """Display the transient verification checks (a) hold, (b) BVP start, (c) relaxation after a ramp."""
    p = dict(params, ivp_rtol=1e-8, ivp_atol=1e-8)    # tight time integration for the checks
    geom = correlations.exchanger_geometry(p)

    # (a) Hold the inlets constant for five response times, starting from the discrete steady state.
    y_ss = transient.discrete_steady_state(p, props, "none")
    hold = transient.simulate(p, props, "none", 5.0 * scales["t_response"], y0=y_ss)
    drift = max(np.max(np.abs(hold[field] - hold[field][0])) for field in ("T_h", "T_c", "T_w"))

    # (b) Rates if the simulation were started from the BVP solution sampled at the cell centers.
    y_bvp = transient.steady_to_cells(s, params["N"])
    dTh_dt, dTc_dt, _ = transient.unpack_state(transient.transient_rhs(0.0, y_bvp, p, props, geom, "none"))

    # (c) Run the base feed ramp, wait 40 response times, and compare with the new discrete steady state.
    t_relax = p["ramp_t_start"] + p["ramp_t"] + 40.0 * scales["t_response"]
    relax = transient.simulate(p, props_ramp, "feed_ramp", t_relax, t_eval=np.linspace(0.0, t_relax, 401))
    y_final = transient.pack_state(relax["T_h"][-1], relax["T_c"][-1], relax["T_w"][-1])
    p_new = dict(p, T_h_in=p["T_h_in"] + p["ramp_dT"])
    y_new = transient.discrete_steady_state(p_new, props_ramp, "none")

    markdown_table(["Check", "Result"], [
        ["(a) max drift over 5 response times from the discrete steady state", f"{drift:.1e} K"],
        ["(a) max wall rate during the hold", f"{60 * np.max(np.abs(hold['dTw_dt'])):.1e} K/min"],
        ["(b) BVP start: hot inlet cell / interior hot cells, max $|dT_h/dt|$",
         f"{abs(dTh_dt[0]):.1f} K/s / {np.max(np.abs(dTh_dt[1:])):.3f} K/s"],
        ["(b) BVP start: cold inlet cell / interior cold cells, max $|dT_c/dt|$",
         f"{abs(dTc_dt[-1]):.1f} K/s / {np.max(np.abs(dTc_dt[:-1])):.3f} K/s"],
        ["(c) after the ramp: max difference from the new discrete steady state",
         f"{np.max(np.abs(y_final - y_new)):.1e} K"],
    ])


def fig_bvp_convergence(params, props, fig_dir, fig_label="Figure 6"):
    """Figure 6: the discrete steady state converges to the BVP solution as the grid is refined."""
    s_fine = steady.solve_steady(dict(params, bvp_tol=1e-8), props)
    N_list = [25, 50, 100, 200, 400]
    gap = []
    for N in N_list:
        y_N = transient.discrete_steady_state(dict(params, N=N), props, "none")
        gap.append(np.max(np.abs(y_N - transient.steady_to_cells(s_fine, N))))
    gap = np.array(gap)

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.loglog(N_list, gap, "o-", color=COLORS["cold"], label="max over cells and fields")
    ax.loglog(N_list, gap[0] * N_list[0] / np.array(N_list), color=INK_2, linewidth=1.0, label="slope $-1$ reference")
    ax.set_xticks(N_list, [str(N) for N in N_list])
    ax.minorticks_off()
    ax.set_xlabel("number of cells N")
    ax.set_ylabel(r"max $|y_{ss,N} - y_{BVP}|$ [K]")
    ax.set_title(f"{fig_label}. Discrete steady state converges to the BVP", loc="left")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "report_transient_verification.png")
    plt.show()
    print("Ratio of successive gaps (2 means first order):", np.round(gap[:-1] / gap[1:], 3))


# ---------------------------------------------------------------------------
# Sections 3.7-3.10: feed ramp, ramp sweep, cold trip, quasi-steady model
# ---------------------------------------------------------------------------

def fig_feed_ramp(runs, histories, params, limits, grids, t_end, fig_dir, fig_label="Figure 7"):
    """Figure 7 (feed-temperature ramp on the finest grid) and its grid-convergence table."""
    r, m = runs[grids[-1]], histories[grids[-1]]
    t_min = r["t"] / 60.0

    # Rate of the imposed inlet temperature (a known function of time), by a central difference.
    h_t = 1e-2
    p_ramp = dict(params, N=grids[-1])
    R_in = 60.0 * (transient.boundary_conditions(r["t"] + h_t, p_ramp, "feed_ramp")[0]
                   - transient.boundary_conditions(r["t"] - h_t, p_ramp, "feed_ramp")[0]) / (2.0 * h_t)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    ax = axes[0, 0]
    ax.plot(t_min, r["T_h_in"] - r["T_h_in"][0], color=COLORS["hot"], label="hot inlet (imposed)")
    ax.plot(t_min, r["T_c"][:, 0] - r["T_c"][0, 0], color=COLORS["cold"], label="cold outlet, x = 0")
    ax.plot(t_min, r["T_h"][:, -1] - r["T_h"][0, -1], color=COLORS["hot"], linewidth=1.0, alpha=0.6,
            label="hot outlet, x = L")
    ax.set_ylabel("change since t = 0 [K]")
    ax.set_title("(a) Nozzle temperatures")
    ax.legend(loc="upper left")

    ax = axes[0, 1]
    ax.plot(t_min, R_in, color=COLORS["hot"], label="hot inlet nozzle (imposed)")
    ax.plot(t_min, m["R_w"], color=COLORS["wall"], label="metal, $R_w$")
    ax.plot(t_min, m["R_end"], color=COLORS["cold"], label="outlet nozzles, $R_{end}$")
    draw_limit(ax, limits["rate_frequent"], f"frequent events, {limits['rate_frequent']:g} °C/min")
    draw_limit(ax, limits["rate_infrequent"], f"startup/trip, {limits['rate_infrequent']:g} °C/min")
    ax.set_ylim(0, limits["rate_infrequent"] + 1.0)
    ax.set_ylabel("rate of change [K/min]")
    ax.set_title("(b) Rates of change")
    ax.legend(loc="center left")

    ax = axes[1, 0]
    ax.plot(t_min, m["dT_ss"], color=INK, label="stream to stream, largest along the core")
    ax.plot(t_min, m["dT_sm"], color=COLORS["wall"], label="stream to metal, largest along the core")
    draw_limit(ax, limits["dT_stream_stream"], f"about {limits['dT_stream_stream']:g} °C")
    ax.set_ylim(0, limits["dT_stream_stream"] + 5.0)
    ax.set_xlabel("t [min]")
    ax.set_ylabel("temperature difference [K]")
    ax.set_title("(c) Temperature differences")
    ax.legend(loc="lower right")

    ax = axes[1, 1]
    t0, tr = params["ramp_t_start"], params["ramp_t"]
    snapshots = [t0 + 0.25 * tr, t0 + 0.5 * tr, t0 + 0.75 * tr, t0 + tr, t_end]
    shades = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
    for t_snap, shade in zip(snapshots, shades):
        k = np.argmin(np.abs(r["t"] - t_snap))
        ax.plot(r["x"], r["T_w"][k] - r["T_w"][0], color=shade, label=f"t = {r['t'][k] / 60:.1f} min")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("metal temperature change [K]")
    ax.set_title("(d) Metal temperature change along the core")
    ax.legend(loc="upper right")
    fig.suptitle(f"{fig_label}. Feed-temperature ramp, {params['ramp_dT']:+g} K at "
                 f"{60.0 * abs(params['ramp_dT']) / params['ramp_t']:g} K/min", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(fig_dir / "report_feed_ramp.png")
    plt.show()

    convergence_table([
        ("peak metal rate $R_w$", "K/min", [histories[N]["R_w"].max() for N in grids]),
        ("peak outlet-nozzle rate $R_{end}$", "K/min", [histories[N]["R_end"].max() for N in grids]),
        ("max stream-to-stream difference", "K", [histories[N]["dT_ss"].max() for N in grids]),
        ("max stream-to-metal difference", "K", [histories[N]["dT_sm"].max() for N in grids]),
        ("total metal temperature change", "K",
         [np.max(np.abs(runs[N]["T_w"] - runs[N]["T_w"][0])) for N in grids]),
    ], grids)


def ramp_sweep(params, props_ramp, limits, grids, fig_dir, fig_label="Figure 8"):
    """Figure 8 and its table: peak rates for the base feed ramp size at nominal rates from 0.5 to 300 K/min."""
    sweep_rates = np.array([0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 60.0, 120.0, 300.0])  # nominal feed ramp rates [K/min]
    sweep = {N: {"R_w": [], "R_end": []} for N in grids}
    start = time.perf_counter()
    for rate in sweep_rates:
        ramp_t = params["ramp_dT"] / (rate / 60.0)                      # duration of the ramp [s]
        # Start the ramp several corner widths after t = 0 so that the initial state is steady.
        t_start = max(params["ramp_t_start"], 6.0 * ramp_t / params["ramp_sharpness"])
        t_end = t_start + ramp_t + 120.0
        # Save every second, and more often during and just after the ramp to catch the peak rate.
        t_fine = np.arange(t_start, t_start + ramp_t + 30.0, min(0.5, ramp_t / 50.0))
        t_eval = np.union1d(np.append(np.arange(0.0, t_end, 1.0), t_end), t_fine)
        for N in grids:
            p = dict(params, N=N, ramp_t=ramp_t, ramp_t_start=t_start)
            mN = alpema_histories(transient.simulate(p, props_ramp, "feed_ramp", t_end, t_eval=t_eval))
            sweep[N]["R_w"].append(mN["R_w"].max())
            sweep[N]["R_end"].append(mN["R_end"].max())
    print(f"{len(sweep_rates) * len(grids)} simulations in {time.perf_counter() - start:.0f} s")

    R_w_fine = np.array(sweep[grids[-1]]["R_w"])
    R_end_fine = np.array(sweep[grids[-1]]["R_end"])

    fig, (ax_r, ax_ratio) = plt.subplots(1, 2, figsize=(11, 3.9))
    ax_r.loglog(sweep_rates, sweep_rates, color=COLORS["hot"], label="hot inlet nozzle (= ramp rate)")
    ax_r.loglog(sweep_rates, R_w_fine, "o-", color=COLORS["wall"], label="metal, peak $R_w$")
    ax_r.loglog(sweep_rates, R_end_fine, "o-", color=COLORS["cold"], label="outlet nozzles, peak $R_{end}$")
    draw_limit(ax_r, limits["rate_frequent"], f"frequent events, {limits['rate_frequent']:g} °C/min")
    draw_limit(ax_r, limits["rate_infrequent"], f"startup/trip, {limits['rate_infrequent']:g} °C/min")
    ax_r.set_xlabel("nominal feed ramp rate r [K/min]")
    ax_r.set_ylabel("peak rate of change [K/min]")
    ax_r.set_title("(a) Peak rates")
    ax_r.legend(loc="upper left")

    ax_ratio.semilogx(sweep_rates, R_w_fine / sweep_rates, "o-", color=COLORS["wall"], label="metal")
    ax_ratio.semilogx(sweep_rates, R_end_fine / sweep_rates, "o-", color=COLORS["cold"], label="outlet nozzles")
    ax_ratio.set_ylim(0, 1.05)
    ax_ratio.set_xlabel("nominal feed ramp rate r [K/min]")
    ax_ratio.set_ylabel("peak rate / ramp rate")
    ax_ratio.set_title("(b) Attenuation by the metal")
    ax_ratio.legend(loc="lower left")
    fig.suptitle(f"{fig_label}. Peak rates for a {params['ramp_dT']:g} K feed ramp at different rates "
                 f"(N = {grids[-1]})",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(fig_dir / "report_ramp_sweep.png")
    plt.show()

    # Grid convergence across the sweep.
    p_obs = [observed_order(*(sweep[N][key][j] for N in grids))[0]
             for key in ("R_w", "R_end") for j in range(len(sweep_rates))]
    change = max(np.max(np.abs(np.array(sweep[grids[-1]][key]) / np.array(sweep[grids[-2]][key]) - 1.0))
                 for key in ("R_w", "R_end"))
    print(f"Observed order across the sweep: {np.nanmin(p_obs):.2f} to {np.nanmax(p_obs):.2f}; "
          f"largest relative change from N = {grids[-2]} to {grids[-1]}: {100 * change:.1f} %")
    markdown_table(["Ramp rate [K/min]", "Peak $R_w$ [K/min]", "Peak $R_{end}$ [K/min]", "$R_w/r$", "$R_{end}/r$"],
                   [[f"{rate:g}", f"{rw:.3f}", f"{re:.3f}", f"{rw / rate:.3f}", f"{re / rate:.3f}"]
                    for rate, rw, re in zip(sweep_rates, R_w_fine, R_end_fine)])


def _upset_figure(runs, histories, params, limits, grids, t_show_min, title, filename, fig_dir,
                  temps_legend="lower right", diffs_legend="lower right"):
    """Four-panel figure for an upset that changes the flows (cold trip, startup), on the finest grid.

    Panels: (a) mass flows, (b) outlet and end-of-core metal temperatures, (c) rates of change,
    (d) stream-to-stream differences at the nozzles and over the cells on every grid. Panel (d)
    shows why the ALPEMA and stress checks use the end faces: the cells-only maximum creeps
    toward the cold-end nozzle difference as the grid is refined.
    Inputs: runs and histories keyed by N, params, limits, grids, t_show_min (end of the plotted
    window [min]), title (figure title), filename (saved in fig_dir), temps_legend and
    diffs_legend (legend positions in panels b and d).
    """
    r, m = runs[grids[-1]], histories[grids[-1]]
    t_min = r["t"] / 60.0

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    ax = axes[0, 0]
    flow_c, flow_h = r["mdot_c"] / params["mdot_c"], r["mdot_h"] / params["mdot_h"]
    ax.plot(t_min, flow_c, color=COLORS["cold"], label="cold stream")
    # Dash the hot curve when the two flows coincide (startup), so both stay visible.
    ax.plot(t_min, flow_h, color=COLORS["hot"], linestyle="--" if np.allclose(flow_c, flow_h) else "-",
            label="hot stream")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("flow / full flow")
    ax.set_title("(a) Mass flows")
    ax.legend(loc="center right")

    ax = axes[0, 1]
    ax.plot(t_min, r["T_h"][:, -1], color=COLORS["hot"], label="hot outlet, x = L")
    ax.plot(t_min, r["T_c"][:, 0], color=COLORS["cold"], label="cold outlet, x = 0")
    ax.plot(t_min, r["T_w"][:, -1], color=COLORS["wall"], label="metal at x = L")
    ax.plot(t_min, r["T_w"][:, 0], color=COLORS["wall"], linewidth=1.0, alpha=0.6, label="metal at x = 0")
    ax.set_ylabel("T [K]")
    ax.set_title("(b) Outlet and end-of-core metal temperatures")
    ax.legend(loc=temps_legend)

    ax = axes[1, 0]
    ax.plot(t_min, m["R_w"], color=COLORS["wall"], label="metal, $R_w$")
    ax.plot(t_min, m["R_end"], color=COLORS["cold"], label="outlet nozzles, $R_{end}$")
    draw_limit(ax, limits["rate_infrequent"], f"startup/trip, {limits['rate_infrequent']:g} °C/min")
    ax.set_xlabel("t [min]")
    ax.set_ylabel("rate of change [K/min]")
    ax.set_title("(c) Rates of change")
    ax.legend(loc="upper right")

    ax = axes[1, 1]
    ax.plot(t_min, m["dT_cold_end"], color=INK, label="cold-end nozzles, $T_{h,out} - T_{c,in}$")
    ax.plot(t_min, m["dT_warm_end"], color=INK_2, linewidth=1.0, label="warm-end nozzles, $T_{h,in} - T_{c,out}$")
    for N, shade in zip(grids, GRID_SHADES):
        ax.plot(runs[N]["t"] / 60.0, histories[N]["dT_ss_cells"], color=shade, linewidth=1.2,
                label=f"max over cells, N = {N}")
    draw_limit(ax, limits["dT_stream_stream"], f"about {limits['dT_stream_stream']:g} °C")
    ax.set_ylim(0, 70)
    ax.set_xlabel("t [min]")
    ax.set_ylabel("stream-to-stream difference [K]")
    ax.set_title("(d) Stream-to-stream differences")
    ax.legend(loc=diffs_legend, fontsize=7.5)
    for ax in axes.flat:
        ax.set_xlim(0, t_show_min)
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(fig_dir / filename)
    plt.show()

    convergence_table([
        ("peak metal rate $R_w$", "K/min", [histories[N]["R_w"].max() for N in grids]),
        ("peak outlet-nozzle rate $R_{end}$", "K/min", [histories[N]["R_end"].max() for N in grids]),
        ("max cold-end nozzle difference", "K", [histories[N]["dT_cold_end"].max() for N in grids]),
        ("max stream-to-metal difference, cells and end faces", "K", [histories[N]["dT_sm"].max() for N in grids]),
        ("max stream-to-stream difference over cells only", "K",
         [histories[N]["dT_ss_cells"].max() for N in grids]),
        ("max stream-to-metal difference over cells only", "K",
         [histories[N]["dT_sm_cells"].max() for N in grids]),
        ("total metal temperature change", "K",
         [np.max(np.abs(runs[N]["T_w"] - runs[N]["T_w"][0])) for N in grids]),
    ], grids)


def fig_cold_trip(runs, histories, params, limits, grids, fig_dir, fig_label="Figure 9"):
    """Figure 9 (cold-stream trip, first 5 minutes of the run) and its grid-convergence table."""
    t_show_min = 5.0
    t_run_min = runs[grids[-1]]["t"][-1] / 60.0
    _upset_figure(runs, histories, params, limits, grids, t_show_min,
                  f"{fig_label}. Cold-stream trip: {100.0 * (1.0 - params['trip_phi']):.0f} % loss of cold flow, "
                  f"hot flow unchanged (first {t_show_min:g} of {t_run_min:g} min)",
                  "report_cold_trip.png", fig_dir)


def trip_entrance_length(params, props, grids):
    """Display the cold-stream thermal entrance length at design and residual trip flow.

    The cold gas reaches the metal temperature within about C_c / a_c of its inlet (a decay
    length of the cold-stream ODE).
    """
    mdot_trip = params["trip_phi"] * params["mdot_c"]
    T_cin = params["T_c_in"]
    geom = correlations.exchanger_geometry(params)
    cp_c, mu_c, k_c = (properties.evaluate(props["c"], name, T_cin) for name in ("cp", "mu", "k"))
    a_c_trip = correlations.side_conductance(T_cin, mdot_trip, props["c"], geom["c"], params["k_f"])
    a_c_base = correlations.side_conductance(T_cin, params["mdot_c"], props["c"], geom["c"], params["k_f"])
    Re_trip = correlations.reynolds(mdot_trip, mu_c, geom["c"])
    G_trip = mdot_trip / geom["c"]["A_ff"]
    j_trip = correlations.colburn_j(Re_trip, geom["c"]["alpha_mb"], geom["c"]["delta_mb"], geom["c"]["gamma_mb"])
    h_mb = j_trip * G_trip * cp_c * (cp_c * mu_c / k_c) ** (-2.0 / 3.0)
    h_lam = correlations.laminar_nusselt_rect(geom["c"]["alpha_mb"]) * k_c / geom["c"]["D_h"]

    markdown_table(["Cold stream at its inlet", "Design flow", "Residual trip flow"], [
        ["Mass flow per passage", f"{params['mdot_c']:.3f} kg/s", f"{mdot_trip:.3f} kg/s"],
        ["Entrance length $\\dot m_c c_{p,c} / a_c$",
         f"{1e3 * params['mdot_c'] * cp_c / a_c_base:.0f} mm", f"{1e3 * mdot_trip * cp_c / a_c_trip:.0f} mm"],
    ])
    cell_widths = ", ".join(f"{1e3 * params['L'] / N:.2f} mm (N = {N})" for N in grids)
    print(f"Cell width: {cell_widths}")
    print(f"At the residual flow: Re = {float(Re_trip):.0f}, h from Manglik-Bergles = {float(h_mb):.0f} W/(m2 K), "
          f"laminar floor = {float(h_lam):.0f} W/(m2 K)")


def quasi_steady_comparison(params, props, props_ramp, t_end_ramp, t_end_trip):
    """Display the full model against the quasi-steady model on the base grid for the ramp and the trip."""
    rows = []
    for name, scenario, pr, t_end in (("feed ramp", "feed_ramp", props_ramp, t_end_ramp),
                                      ("cold trip", "cold_trip", props, t_end_trip)):
        t_eval = np.linspace(0.0, t_end, 601)
        runs, seconds = {}, {}
        for model in ("full", "quasi_steady"):
            start = time.perf_counter()
            runs[model] = transient.simulate(params, pr, scenario, t_end, t_eval=t_eval, model=model)
            seconds[model] = time.perf_counter() - start
        rows.append([name,
                     f"{np.max(np.abs(runs['full']['T_w'] - runs['quasi_steady']['T_w'])):.2f} K",
                     f"{60 * np.max(np.abs(runs['full']['dTw_dt'])):.2f} / "
                     f"{60 * np.max(np.abs(runs['quasi_steady']['dTw_dt'])):.2f}",
                     f"{seconds['full']:.1f} s / {seconds['quasi_steady']:.1f} s"])
    markdown_table([f"Scenario (N = {params['N']})", "max metal temperature difference",
                    "peak $R_w$ [K/min], full / quasi-steady", "run time, full / quasi-steady"], rows)


# ---------------------------------------------------------------------------
# Sections 3.11-3.13: startup, thermal stress, ALPEMA summary
# ---------------------------------------------------------------------------

def fig_startup(runs, histories, params, limits, grids, fig_dir, fig_label="Figure 10"):
    """Figure 10 (startup from rest, first 3 minutes) and its grid-convergence table."""
    _upset_figure(runs, histories, params, limits, grids, 3.0,
                  f"{fig_label}. Startup from rest: idle core at {params['startup_T0']:.0f} K, both flows open "
                  f"over {params['startup_t_flow']:.0f} s (first 3 min)",
                  "report_startup.png", fig_dir, temps_legend="center right", diffs_legend="upper right")


def startup_animation(fig_dir):
    """Figure 11: run scripts/plot_transient.py and show the startup animation and its six-frame page.

    Input: fig_dir, the figures/ folder that plot_transient.py saves into. The animation plays in
    Jupyter and Colab; a PDF export shows only the six-frame page.
    """
    from scripts import plot_transient
    with contextlib.redirect_stdout(io.StringIO()):   # hide the surrogate and run printout
        plot_transient.main()
    display(Image(filename=str(fig_dir / "startup.gif")))
    display(Image(filename=str(fig_dir / "startup_snapshots.png")))


def _largest_differences(histories, grids):
    """Largest stream-to-metal and stream-to-stream differences of a transient [K].

    Inputs: histories keyed by N (alpema_histories), grids. Both differences include the two
    end faces of the core (metrics.with_end_faces). The stream-to-metal value is extrapolated
    over the grids; the stream-to-stream value is a nozzle difference, the same on every grid.
    """
    dT_sm = extrapolated([histories[N]["dT_sm"].max() for N in grids])
    dT_ss = histories[grids[-1]]["dT_ss"].max()
    return dT_sm, dT_ss


def stress_table(s, scenarios, params, limits, grids):
    """Display the fully restrained thermal stress (report Section 2.9, src/stress.py) for each case.

    Inputs: s (steady result), scenarios (list of (name, histories keyed by N)), params, limits,
    grids. Both differences are the largest along the core, end faces included; for transients
    the stream-to-metal value is extrapolated over the grids. The largest stream-to-stream
    difference is always between the two nozzles at one end of the core.
    """
    E, alpha_T, nu = params["E_m"], params["alpha_T"], params["nu_m"]
    MPa = 1e6
    sigma_y = config.MATERIAL_LIMITS["sigma_y_3003_O"]
    steady_check = stress.stress_check(s["x"], s["T_h"], s["T_c"], s["T_w"], params, limits, sigma_y)["check"]
    cases = [("Steady base case", steady_check["stream-to-metal"]["value"],
              steady_check["stream-to-stream"]["value"])]
    cases += [(name, *_largest_differences(h, grids)) for name, h in scenarios]
    rows = [[name, f"{dT_sm:.1f}", f"{stress.thermal_stress(dT_sm, E, alpha_T, nu) / MPa:.0f}",
             f"{dT_ss:.1f}", f"{stress.thermal_stress(dT_ss, E, alpha_T, nu) / MPa:.0f}"]
            for name, dT_sm, dT_ss in cases]
    markdown_table(["Case", "Largest stream-to-metal $\\Delta T$ [K]", "$\\sigma_{th}$ [MPa]",
                    "Largest stream-to-stream $\\Delta T$ [K]", "$\\sigma_{th}$ [MPa]"], rows)
    per_K = stress.thermal_stress(1.0, E, alpha_T, nu)
    markdown_table(["Reference level", "Temperature difference", "Stress"], [
        ["Fully restrained stress per kelvin", "1 K", f"{per_K / MPa:.2f} MPa"],
        ["ALPEMA difference limit", f"{limits['dT_stream_metal']:.0f} K",
         f"{per_K * limits['dT_stream_metal'] / MPa:.0f} MPa"],
        ["Minimum yield strength of 3003-O aluminium (ASTM B209M)", f"{sigma_y / per_K:.1f} K",
         f"{sigma_y / MPa:.0f} MPa"],
    ])


def alpema_summary(s, ramp, trip, startup, params, limits, grids):
    """Display the ALPEMA checks for every case, using extrapolated values for grid-dependent outputs.

    Inputs: s (steady result); ramp, trip, startup, each (runs, histories) keyed by N; params,
    limits, grids. Temperature differences include the two end faces of the core.
    """
    (_, ramp_m), (_, trip_m), (_, start_m) = ramp, trip, startup
    T_h, T_c, T_w = s["T_h"], s["T_c"], s["T_w"]
    both = (limits["rate_frequent"], limits["rate_infrequent"])
    ramp_rate = 60.0 * abs(params["ramp_dT"]) / params["ramp_t"]
    ramp_name = f"Feed ramp, {ramp_rate:g} K/min"

    def hourly(histories):
        """Largest metal temperature change in any 60 min on the finest grid [K] (math doc Section 6.6)."""
        return histories[grids[-1]]["max_dT_hour_wall"]

    def ext(histories, key):
        """Peak of one history, extrapolated over the grids."""
        return extrapolated([histories[N][key].max() for N in grids])

    summary = [
        ("Steady base case", "max stream-to-stream difference", np.max(T_h - T_c), "K", limits["dT_stream_stream"]),
        ("Steady base case", "max stream-to-metal difference", max(np.max(T_h - T_w), np.max(T_w - T_c)), "K",
         limits["dT_stream_metal"]),
        (ramp_name, "rate at the hot inlet nozzle", ramp_rate, "K/min", both),
        (ramp_name, "peak metal rate $R_w$", ext(ramp_m, "R_w"), "K/min", both),
        (ramp_name, "peak outlet-nozzle rate $R_{end}$", ext(ramp_m, "R_end"), "K/min", both),
        (ramp_name, "max stream-to-stream difference", ext(ramp_m, "dT_ss"), "K", limits["dT_stream_stream"]),
        (ramp_name, "max stream-to-metal difference", ext(ramp_m, "dT_sm"), "K", limits["dT_stream_metal"]),
        (ramp_name, "metal change in any 60 min", hourly(ramp_m), "K",
         limits["change_per_hour_infrequent"]),
        ("Cold trip", "peak metal rate $R_w$", ext(trip_m, "R_w"), "K/min", limits["rate_infrequent"]),
        ("Cold trip", "peak outlet-nozzle rate $R_{end}$", ext(trip_m, "R_end"), "K/min", limits["rate_infrequent"]),
        ("Cold trip", "max cold-end nozzle difference", trip_m[grids[-1]]["dT_cold_end"].max(), "K",
         limits["dT_stream_stream"]),
        ("Cold trip", "max stream-to-metal difference", ext(trip_m, "dT_sm"), "K", limits["dT_stream_metal"]),
        ("Cold trip", "metal change in any 60 min", hourly(trip_m), "K", limits["change_per_hour_infrequent"]),
        ("Startup from rest", "peak metal rate $R_w$", ext(start_m, "R_w"), "K/min", limits["rate_infrequent"]),
        ("Startup from rest", "peak outlet-nozzle rate $R_{end}$", ext(start_m, "R_end"), "K/min",
         limits["rate_infrequent"]),
        ("Startup from rest", "max cold-end nozzle difference", start_m[grids[-1]]["dT_cold_end"].max(), "K",
         limits["dT_stream_stream"]),
        ("Startup from rest", "max stream-to-metal difference", ext(start_m, "dT_sm"), "K", limits["dT_stream_metal"]),
        ("Startup from rest", "metal change in any 60 min", hourly(start_m), "K",
         limits["change_per_hour_infrequent"]),
    ]
    rows = []
    for scenario, quantity, value, units, limit in summary:
        if isinstance(limit, tuple):   # rate limits for frequent and for infrequent events
            verdict = (f"{'yes' if value < limit[0] else 'no'} (frequent) / "
                       f"{'yes' if value < limit[1] else 'no'} (startup/trip)")
            limit_text = f"{limit[0]:g} / {limit[1]:g}"
        else:
            verdict = "yes" if value < limit else "no"
            limit_text = f"{limit:g}"
        rows.append([scenario, quantity, f"{value:.1f} {units}", f"{limit_text} {units.replace('K', '°C')}", verdict])
    markdown_table(["Scenario", "Quantity", "Model", "ALPEMA limit", "Within limit?"], rows)
