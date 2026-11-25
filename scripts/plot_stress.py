"""Thermal stress figures: the stress check of src/stress.py for every case.

The fully restrained stress (math doc Eq 53) is proportional to a temperature
difference, sigma_th = E alpha_T Delta T / (1 - nu), about 2.4 MPa per kelvin.
Every figure therefore shows the temperature difference in kelvin on one
scale and the same values as a stress in MPa on a second scale of the same
axis. Two reference levels appear in each figure: the ALPEMA difference limit
(config.ALPEMA_LIMITS) and the yield strength of the annealed core alloy
(config.MATERIAL_LIMITS).

Cases: the steady base case, the feed-temperature ramp, the cold-stream trip
and the startup from rest. Transients use stress.stress_check_transient, which
includes the two end faces of the core.

Saves four files in figures/:
  report_stress_steady.png      local differences and stress along the core at steady state
  report_stress_transients.png  largest difference and stress along the core against time
  report_stress_profiles.png    where the stress sits: profiles along the core at chosen times
  report_stress_summary.png     peak of every case, and its convergence with the grid

Run from the repo root with either of:
    python -m scripts.plot_stress
    python scripts/plot_stress.py
The report notebook can call the figure functions with runs it already has.
"""

import sys
from pathlib import Path

# Running `python scripts/plot_stress.py` puts scripts/ (not the repo root) on
# sys.path, so `from src import ...` would fail. Add the repo root so both run
# styles work.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np

from scripts import report_figures as rf
from src import config, steady, stress, transient

FIG_DIR = ROOT / "figures"
MPa = 1e6  # Pa per MPa

# The three local differences keep the colors of the other report figures:
# the hot side orange, the cold side blue, stream to stream black.
DIFF_STYLE = {
    "hot-to-metal": dict(color=rf.COLORS["hot"], label="hot gas to metal, $T_h - T_w$"),
    "metal-to-cold": dict(color=rf.COLORS["cold"], label="metal to cold gas, $T_w - T_c$"),
    "stream-to-stream": dict(color=rf.INK, label="stream to stream, $T_h - T_c$"),
}
# Light-to-dark blues for profiles at successive times (same ramp as the feed-ramp figure).
TIME_SHADES = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
GRAY = "#898781"  # de-emphasised series (cells-only values)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def add_stress_scale(ax, per_K, axis="y"):
    """Add a second scale that shows the temperature difference as a stress (Eq 53).

    Inputs: ax (matplotlib Axes whose y or x scale is a temperature difference [K]),
    per_K (stress per kelvin of difference [Pa/K], from stress.thermal_stress(1.0, ...)),
    axis ("y": scale on the right; "x": scale on the top).
    Output: the secondary axis. Both scales show the same data: Eq 53 is linear, so the
    MPa scale is the K scale multiplied by per_K.
    """
    def to_MPa(dT):
        return dT * per_K / MPa

    def to_K(sigma_MPa):
        return sigma_MPa * MPa / per_K

    label = r"fully restrained stress $\sigma_{th}$ [MPa]"
    if axis == "y":
        scale = ax.secondary_yaxis("right", functions=(to_MPa, to_K))
        scale.set_ylabel(label)
    else:
        scale = ax.secondary_xaxis("top", functions=(to_MPa, to_K))
        scale.set_xlabel(label)
    return scale


def draw_reference_levels(ax, check, limits, axis="y"):
    """Draw the ALPEMA difference limit (dashed) and the yield strength (dotted) as lines.

    Inputs: ax, check (from stress.stress_check: uses "per_K", "sigma_yield", "dT_yield"),
    limits (config.ALPEMA_LIMITS), axis ("y": horizontal lines; "x": vertical lines).
    Output: none. The two lines carry legend labels with both units.
    """
    dT_alpema = limits["dT_stream_metal"]
    levels = [
        (dT_alpema, "--", f"ALPEMA limit, about {dT_alpema:g} °C "
                          f"({check['per_K'] * dT_alpema / MPa:.0f} MPa)"),
        (check["dT_yield"], ":", f"3003-O yield, {check['sigma_yield'] / MPa:.0f} MPa "
                                 f"({check['dT_yield']:.1f} K)"),
    ]
    for value, linestyle, label in levels:
        if axis == "y":
            ax.axhline(value, color=rf.INK_2, linestyle=linestyle, linewidth=1.1, label=label)
        else:
            ax.axvline(value, color=rf.INK_2, linestyle=linestyle, linewidth=1.1, label=label)


def scenario_titles(params):
    """Short description of each transient scenario, with its numbers read from params.

    Input: params (config.BASE_CASE copy). Output: dict keyed by scenario name.
    """
    ramp_rate = 60.0 * abs(params["ramp_dT"]) / params["ramp_t"]
    return {
        "feed_ramp": f"Feed ramp, {params['ramp_dT']:+g} K at {ramp_rate:g} K/min",
        "cold_trip": f"Cold trip, {100.0 * (1.0 - params['trip_phi']):.0f} % loss of cold flow",
        "startup": f"Startup from rest, flows open over {params['startup_t_flow']:g} s",
    }


def scenario_end_times(params, props):
    """Run length of each transient scenario [s], long enough for it to settle.

    Inputs: params, props (surrogates for scenario "none"; only used for the time scales).
    Output: dict keyed by scenario name. Feed ramp: 5 min after the ramp ends. Cold trip:
    10 min. Startup: 15 whole-exchanger response times (math doc Section 6.2) after the
    flows are fully open, as in scripts/plot_transient.py.
    """
    t_response = transient.time_scales(params, props)["t_response"]
    return {
        "feed_ramp": params["ramp_t_start"] + params["ramp_t"] + 300.0,
        "cold_trip": 600.0,
        "startup": params["startup_t_start"] + params["startup_t_flow"] + 15.0 * t_response,
    }


def run_scenarios(params, grids, props_none):
    """Simulate the three transient scenarios on every grid.

    Inputs: params (config.BASE_CASE copy), grids (cell counts, coarse to fine), props_none
    (surrogates for scenario "none").
    Output: runs, dict keyed by scenario name, each a dict keyed by N of transient.simulate
    results. Tight time integration (rtol = atol = 1e-8) so the grid error dominates the
    time error (metrics.observed_order pitfall).
    """
    t_ends = scenario_end_times(params, props_none)
    runs = {}
    for scenario, t_end in t_ends.items():
        props = rf.build_surrogates_quietly(params, scenario)
        t_eval = np.linspace(0.0, t_end, 1201)
        runs[scenario] = {}
        for N in grids:
            p = dict(params, N=N, ivp_rtol=1e-8, ivp_atol=1e-8)
            runs[scenario][N] = transient.simulate(p, props, scenario, t_end, t_eval=t_eval)
    return runs


def peak_differences(runs_by_N, params, limits, sigma_y):
    """Peak temperature differences of one scenario on every grid, with and without the end faces.

    Inputs: runs_by_N (dict keyed by N of transient.simulate results), params, limits
    (config.ALPEMA_LIMITS), sigma_y (yield strength [Pa]).
    Output: dict keyed by N; each entry is a dict with the peak differences [K]
        "sm", "ss"              stream-to-metal and stream-to-stream, cells and end faces
        "sm_cells", "ss_cells"  the same over the cells only
    and "check", the full stress.stress_check_transient result for that grid.
    """
    peaks = {}
    for N, r in runs_by_N.items():
        check = stress.stress_check_transient(r, params, limits, sigma_y)
        cells = stress.alpema_difference_check(r["T_h"], r["T_c"], r["T_w"], limits)
        peaks[N] = {
            "sm": check["check"]["stream-to-metal"]["value"],
            "ss": check["check"]["stream-to-stream"]["value"],
            "sm_cells": cells["stream-to-metal"]["value"],
            "ss_cells": cells["stream-to-stream"]["value"],
            "check": check,
        }
    return peaks


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_stress_steady(s, params, limits, sigma_y, fig_dir, fig_label="Figure 12"):
    """Local temperature differences and fully restrained stress along the core at steady state.

    Inputs: s (steady.solve_steady result), params, limits (config.ALPEMA_LIMITS), sigma_y
    (yield strength [Pa]), fig_dir (folder to save into), fig_label (figure number text).
    Output: the stress.stress_check result. Saves report_stress_steady.png.
    """
    check = stress.stress_check(s["x"], s["T_h"], s["T_c"], s["T_w"], params, limits, sigma_y)

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    for name in stress.DIFFERENCES:
        ax.plot(s["x"], check["dT"][name], **DIFF_STYLE[name])
        # Mark the largest value of each difference (all three are at the cold end).
        peak = check["peaks"][name]
        ax.plot(peak["x"], peak["dT"], "o", color=DIFF_STYLE[name]["color"], markeredgecolor="white",
                markeredgewidth=1.2, zorder=3, clip_on=False)
    # Label the two peaks ALPEMA limits: stream to stream, and the larger stream-to-metal side.
    worse_side = max(("hot-to-metal", "metal-to-cold"), key=lambda name: abs(check["peaks"][name]["dT"]))
    for name in ("stream-to-stream", worse_side):
        peak = check["peaks"][name]
        ax.annotate(f"{peak['dT']:.1f} K = {peak['sigma'] / MPa:.0f} MPa", (peak["x"], peak["dT"]),
                    xytext=(-8, 5), textcoords="offset points", ha="right", va="bottom", fontsize=8.5,
                    color=rf.INK)
    draw_reference_levels(ax, check, limits)
    ax.set_xlim(0.0, params["L"])
    ax.set_ylim(0.0, limits["dT_stream_metal"] + 13.0)   # room for the legend above the limit line
    ax.set_xlabel("x [m]   (hot inlet at 0, cold inlet at L)")
    ax.set_ylabel(r"temperature difference $\Delta T$ [K]")
    add_stress_scale(ax, check["per_K"])
    ax.legend(loc="upper left", ncol=2, columnspacing=1.5)
    ax.set_title(f"{fig_label}. Thermal stress check, steady base case "
                 f"({check['per_K'] / MPa:.2f} MPa per kelvin)", loc="left")
    fig.tight_layout()
    fig.savefig(fig_dir / "report_stress_steady.png")
    plt.show()
    return check


def fig_stress_transients(runs, params, limits, sigma_y, grids, fig_dir, fig_label="Figure 13"):
    """Largest temperature difference and stress along the core against time, one panel per scenario.

    Inputs: runs (dict keyed by scenario, each keyed by N, from run_scenarios), params, limits,
    sigma_y [Pa], grids, fig_dir, fig_label. The finest grid is plotted; the end faces of the
    core are included (stress.stress_check_transient).
    Output: none. Saves report_stress_transients.png.
    """
    titles = scenario_titles(params)
    per_K = stress.thermal_stress(1.0, params["E_m"], params["alpha_T"], params["nu_m"])
    # Plotted window [min]: the whole ramp; the first 5 min of the trip; the first 3 min of the startup.
    t_show = {"feed_ramp": runs["feed_ramp"][grids[-1]]["t"][-1] / 60.0, "cold_trip": 5.0, "startup": 3.0}

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.3), sharey=True)
    for ax, letter, scenario in zip(axes, "abc", ("feed_ramp", "cold_trip", "startup")):
        check = stress.stress_check_transient(runs[scenario][grids[-1]], params, limits, sigma_y)
        t_min = check["t"] / 60.0
        for name in stress.DIFFERENCES:
            ax.plot(t_min, check["dT_max"][name], **DIFF_STYLE[name])
        draw_reference_levels(ax, check, limits)
        # Label the peak of the stream-to-stream difference, the largest of the three. The text
        # starts at the peak if it is early in the window and ends at it otherwise.
        peak = check["peaks"]["stream-to-stream"]
        t_peak = min(peak["t"] / 60.0, t_show[scenario])
        early = t_peak < 0.5 * t_show[scenario]
        ax.annotate(f"peak {abs(peak['dT']):.1f} K = {abs(peak['sigma']) / MPa:.0f} MPa",
                    (t_peak, abs(peak["dT"])), xytext=(4 if early else -4, 6), textcoords="offset points",
                    ha="left" if early else "right", va="bottom", fontsize=8.5, color=rf.INK)
        ax.set_xlim(0.0, t_show[scenario])
        ax.set_xlabel("t [min]")
        ax.set_title(f"({letter}) {titles[scenario]}")
    axes[0].set_ylim(0.0, 80.0)
    axes[0].set_ylabel(r"largest $\Delta T$ along the core [K]")
    add_stress_scale(axes[-1], per_K)
    axes[0].legend(loc="upper left")
    fig.suptitle(f"{fig_label}. Thermal stress during the transients (N = {grids[-1]}, end faces included)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(fig_dir / "report_stress_transients.png")
    plt.show()


def fig_stress_profiles(runs, params, limits, sigma_y, grids, fig_dir, fig_label="Figure 14"):
    """Where the stress sits: the local stream-to-metal difference along the core at chosen times.

    Inputs: runs (from run_scenarios), params, limits, sigma_y [Pa], grids, fig_dir, fig_label.
    Panels: the cold trip and the startup on the finest grid. Each curve is the larger of
    |T_h - T_w| and |T_w - T_c| at one time, from x = 0 to x = L including the two end faces.
    Output: none. Saves report_stress_profiles.png.
    """
    titles = scenario_titles(params)
    per_K = stress.thermal_stress(1.0, params["E_m"], params["alpha_T"], params["nu_m"])
    # Times after the start of each upset [s]; the last curve is the end of the run.
    after = {"cold_trip": (0.0, 10.0, 20.0, 40.0), "startup": (0.0, 5.0, 10.0, 20.0)}
    t_start = {"cold_trip": params["trip_t_start"], "startup": params["startup_t_start"]}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, letter, scenario in zip(axes, "ab", ("cold_trip", "startup")):
        r = runs[scenario][grids[-1]]
        check = stress.stress_check_transient(r, params, limits, sigma_y)
        # Stream-to-metal difference at every position and time: the larger of the two sides.
        dT_sm = np.maximum(np.abs(check["dT"]["hot-to-metal"]), np.abs(check["dT"]["metal-to-cold"]))
        times = [t_start[scenario] + dt for dt in after[scenario]] + [r["t"][-1]]
        for t_snap, shade in zip(times, TIME_SHADES):
            k = np.argmin(np.abs(r["t"] - t_snap))
            ax.plot(check["x"], dT_sm[k], color=shade,
                    label=f"t = {r['t'][k]:.0f} s" + (" (end of run)" if t_snap == times[-1] else ""))
        draw_reference_levels(ax, check, limits)
        ax.set_xlim(0.0, params["L"])
        ax.set_xlabel("x [m]   (hot inlet at 0, cold inlet at L)")
        ax.set_title(f"({letter}) {titles[scenario]}")
        ax.legend(loc="upper left", ncol=2, columnspacing=1.5)
    axes[0].set_ylim(0.0, 80.0)
    axes[0].set_ylabel(r"stream-to-metal $\Delta T$ [K]")
    add_stress_scale(axes[-1], per_K)
    fig.suptitle(f"{fig_label}. The stress concentrates at the cold inlet (N = {grids[-1]}, end faces included)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(fig_dir / "report_stress_profiles.png")
    plt.show()


def fig_stress_summary(steady_check, peaks, params, limits, grids, fig_dir, fig_label="Figure 15"):
    """Peak temperature difference and stress of every case, and their convergence with the grid.

    Inputs: steady_check (stress.stress_check result of the steady base case), peaks (dict keyed
    by scenario of peak_differences results; the trip and the startup may hold more grids than
    `grids`), params, limits, grids (the N, 2N, 4N used for the extrapolated values), fig_dir,
    fig_label.
    Panel (a): bars of the peak stream-to-metal difference (extrapolated over N, 2N, 4N) and
    the peak stream-to-stream difference (a nozzle difference, the same on every grid).
    Panel (b): peak stream-to-metal difference against N, over the cells only and with the end
    faces, for the two scenarios with a temperature front at the cold inlet.
    Output: rows, list of (case name, stream-to-metal [K], stream-to-stream [K]) as plotted.
    Saves report_stress_summary.png.
    """
    titles = scenario_titles(params)
    per_K = steady_check["per_K"]
    rows = [("Steady base case", steady_check["check"]["stream-to-metal"]["value"],
             steady_check["check"]["stream-to-stream"]["value"])]
    for scenario in ("feed_ramp", "cold_trip", "startup"):
        dT_sm = rf.extrapolated([peaks[scenario][N]["sm"] for N in grids])
        rows.append((titles[scenario], dT_sm, peaks[scenario][grids[-1]]["ss"]))

    fig, (ax_bar, ax_grid) = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw={"width_ratios": [1.25, 1]})

    # (a) Two thin bars per case, read against the two reference levels.
    y = np.arange(len(rows))[::-1]                 # first case at the top
    bar_height = 0.2
    series = [("stream to metal", [row[1] for row in rows], rf.COLORS["wall"], +0.13),
              ("stream to stream", [row[2] for row in rows], rf.INK, -0.13)]
    for label, values, color, offset in series:
        ax_bar.barh(y + offset, values, height=bar_height, color=color, label=label)
        for yi, value in zip(y + offset, values):
            # White box behind each value so a reference line never strikes through the text.
            ax_bar.annotate(f"{value:.1f} K · {per_K * value / MPa:.0f} MPa", (value, yi), xytext=(5, 0),
                            textcoords="offset points", va="center", fontsize=8.5, color=rf.INK,
                            bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    draw_reference_levels(ax_bar, steady_check, limits, axis="x")
    ax_bar.set_axisbelow(True)                     # grid lines behind the bars
    ax_bar.set_yticks(y, [row[0].replace(", ", ",\n", 1) for row in rows])
    ax_bar.set_xlim(0.0, 95.0)
    ax_bar.set_xlabel(r"peak temperature difference $\Delta T$ [K]")
    add_stress_scale(ax_bar, per_K, axis="x")
    ax_bar.grid(axis="y", visible=False)
    # The two steady-like cases at the top have short bars, so the legend goes beside them.
    ax_bar.legend(loc="upper right")
    ax_bar.set_title("(a) Peak of every case (stream to metal extrapolated over the grids)", pad=34)

    # (b) Grid convergence of the peak stream-to-metal difference.
    for scenario, marker in (("cold_trip", "o"), ("startup", "s")):
        N_list = sorted(peaks[scenario])
        name = "cold trip" if scenario == "cold_trip" else "startup"
        ax_grid.semilogx(N_list, [peaks[scenario][N]["sm"] for N in N_list], marker=marker,
                         color=rf.COLORS["wall"], markeredgecolor="white", markersize=7,
                         label=f"{name}, cells and end faces")
        ax_grid.semilogx(N_list, [peaks[scenario][N]["sm_cells"] for N in N_list], marker=marker,
                         color=GRAY, markeredgecolor="white", markersize=7, label=f"{name}, cells only")
    ax_grid.axhline(limits["dT_stream_metal"], color=rf.INK_2, linestyle="--", linewidth=1.1)
    ax_grid.annotate(f"ALPEMA limit, about {limits['dT_stream_metal']:g} °C", xy=(1.0, limits["dT_stream_metal"]),
                     xycoords=("axes fraction", "data"), xytext=(-3, -4), textcoords="offset points",
                     ha="right", va="top", fontsize=8, color=rf.INK_2)
    N_all = sorted(peaks["cold_trip"])
    ax_grid.set_xticks(N_all, [str(N) for N in N_all])
    ax_grid.minorticks_off()
    ax_grid.set_ylim(0.0, 80.0)
    ax_grid.set_xlabel("number of cells N")
    ax_grid.set_ylabel(r"peak stream-to-metal $\Delta T$ [K]")
    ax_grid.legend(loc="lower right")
    ax_grid.set_title("(b) Peak stream-to-metal difference against the grid", pad=34)

    fig.suptitle(f"{fig_label}. Thermal stress summary", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(fig_dir / "report_stress_summary.png")
    plt.show()
    return rows


def stress_convergence_table(peaks, params, grids):
    """Display the grid convergence of the peak stress of each transient (CLAUDE.md verification 4).

    Inputs: peaks (dict keyed by scenario of peak_differences results), params, grids (N, 2N, 4N).
    Output: none (displays N, 2N, 4N values, the observed order and the extrapolated value of
    the peak stream-to-metal and stream-to-stream stresses [MPa]).
    """
    titles = scenario_titles(params)
    per_K = stress.thermal_stress(1.0, params["E_m"], params["alpha_T"], params["nu_m"])
    rows = []
    for scenario in ("feed_ramp", "cold_trip", "startup"):
        for key, name in (("sm", "stream to metal"), ("ss", "stream to stream"),
                          ("sm_cells", "stream to metal, cells only")):
            rows.append((f"{titles[scenario]}: peak stress, {name}", "MPa",
                         [per_K * peaks[scenario][N][key] / MPa for N in grids]))
    rf.convergence_table(rows, grids)


def main():
    """Run every case and save the four stress figures.

    Inputs: none (uses config.BASE_CASE, config.ALPEMA_LIMITS and config.MATERIAL_LIMITS).
    Outputs: prints the stress check of each case and the grid-convergence table; saves the
    four report_stress_*.png files in figures/.
    """
    rf.use_file_backend()   # save files only; no windows when run as a script
    rf.set_style()
    params = dict(config.BASE_CASE)
    limits = config.ALPEMA_LIMITS
    sigma_y = config.MATERIAL_LIMITS["sigma_y_3003_O"]
    N = params["N"]
    grids = (N, 2 * N, 4 * N)                      # grid study of the extrapolated values
    grids_wide = (N // 2, N, 2 * N, 4 * N, 8 * N)  # wider range for the convergence panel
    FIG_DIR.mkdir(exist_ok=True)

    props = rf.build_surrogates_quietly(params, "none")
    s = steady.solve_steady(params, props)
    runs = run_scenarios(params, grids_wide, props)
    peaks = {scenario: peak_differences(runs[scenario], params, limits, sigma_y) for scenario in runs}

    steady_check = fig_stress_steady(s, params, limits, sigma_y, FIG_DIR)
    print("Steady base case\n" + stress.summary_text(steady_check) + "\n")
    for scenario, title in scenario_titles(params).items():
        print(f"{title} (N = {grids[-1]})\n" + stress.summary_text(peaks[scenario][grids[-1]]["check"]) + "\n")

    fig_stress_transients(runs, params, limits, sigma_y, grids, FIG_DIR)
    fig_stress_profiles(runs, params, limits, sigma_y, grids, FIG_DIR)
    fig_stress_summary(steady_check, peaks, params, limits, grids, FIG_DIR)
    stress_convergence_table(peaks, params, grids)
    print("saved figures/report_stress_steady.png, report_stress_transients.png, "
          "report_stress_profiles.png and report_stress_summary.png")


if __name__ == "__main__":
    main()
