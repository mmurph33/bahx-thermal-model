"""Regenerate every figure and table of the report in one run.

Calls the functions of scripts/report_figures.py and scripts/plot_stress.py in
report order, with the base case from src/config.py. It does what the report
notebook does, without a notebook, so anyone can rebuild the figures after a
change to the model:

    python -m scripts.make_report_figures

Saves in figures/:
  report_fv_grid.png                 finite-volume cells of one repeating unit
  report_surrogates.png              methane heat capacity and the surrogate fit error
  report_steady_verification.png     constant-property BVP against the analytical solution
  report_steady_base_case.png        steady temperature profiles and differences
  report_jacobian_sparsity.png       sparsity pattern of the transient Jacobian
  report_transient_verification.png  discrete steady state against the BVP
  report_feed_ramp.png               feed-temperature ramp
  report_ramp_sweep.png              peak rates against the ramp rate
  report_cold_trip.png               cold-stream trip
  report_startup.png                 startup from rest
  report_stress_steady.png, report_stress_transients.png,
  report_stress_profiles.png, report_stress_summary.png   thermal stress check
  report_tables.md                   every table and printed number of the report

The startup animation (startup.gif, startup_snapshots.png) is not rebuilt here;
run python -m scripts.plot_transient for it.

The run takes a few minutes: the grid studies use N, 2N and 4N cells with tight
time integration (rtol = atol = 1e-8), and the ramp sweep is 27 simulations.
"""

import contextlib
import io
import sys
import time
from pathlib import Path

# Running `python scripts/make_report_figures.py` puts scripts/ (not the repo
# root) on sys.path, so `from src import ...` would fail. Add the repo root so
# both run styles work.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np

from scripts import plot_stress
from scripts import report_figures as rf
from src import config, stress, transient

FIG_DIR = ROOT / "figures"


def section(title):
    """Start a section of the report output.

    Input: title (str). Output: none. Writes a markdown heading to standard output (which
    main collects into report_tables.md) and a progress line to standard error, and closes
    the figures of the previous section.
    """
    plt.close("all")
    print(f"\n## {title}\n")
    print(f"[make_report_figures] {title}", file=sys.stderr)


def build_report(params, limits, sigma_y, grids):
    """Make every report figure and print every report table, in report order.

    Inputs: params (config.BASE_CASE copy), limits (config.ALPEMA_LIMITS), sigma_y (yield
    strength of the core metal [Pa]), grids (cell counts N, 2N, 4N of the grid studies).
    Output: none (figures are saved in figures/; tables go to standard output).
    """
    props = rf.build_surrogates_quietly(params, "none")           # steady state and cold trip
    props_ramp = rf.build_surrogates_quietly(params, "feed_ramp")
    props_startup = rf.build_surrogates_quietly(params, "startup")
    # Grid studies: tight time integration, so the grid error dominates the time error.
    params_tight = dict(params, ivp_rtol=1e-8, ivp_atol=1e-8)
    t_ends = plot_stress.scenario_end_times(params, props)        # run length of each scenario [s]

    section("Finite-volume grid (Figure 1)")
    rf.fig_fv_grid(FIG_DIR)

    section("Base case and ALPEMA limits")
    rf.base_case_tables(params, limits, grids)

    section("Property surrogates (Figure 2)")
    rf.surrogate_tables(params, props, props_ramp)
    rf.fig_surrogates(params, props_ramp, FIG_DIR)

    section("Steady model: verification against the analytical solution (Figure 3)")
    rf.steady_verification(params, FIG_DIR)

    section("Steady base case (Figure 4)")
    s = rf.steady_checks(params, props)
    rf.steady_base_case(s, params, props, limits, FIG_DIR)

    section("Time scales and Jacobian sparsity (Figure 5)")
    scales = rf.time_scales_and_jacobian(params, props, FIG_DIR)

    section("Transient model: verification (Figure 6)")
    rf.transient_checks(params, props, props_ramp, s, scales)
    rf.fig_bvp_convergence(params, props, FIG_DIR)

    section("Feed-temperature ramp (Figure 7)")
    t_end = t_ends["feed_ramp"]
    ramp = rf.run_grids(params_tight, props_ramp, "feed_ramp", t_end, np.linspace(0.0, t_end, 1201), grids)
    rf.fig_feed_ramp(*ramp, params, limits, grids, t_end, FIG_DIR)

    section("Ramp-rate sweep (Figure 8)")
    rf.ramp_sweep(params_tight, props_ramp, limits, grids, FIG_DIR)

    section("Cold-stream trip (Figure 9)")
    t_end = t_ends["cold_trip"]
    trip = rf.run_grids(params_tight, props, "cold_trip", t_end, np.linspace(0.0, t_end, 1201), grids)
    rf.fig_cold_trip(*trip, params, limits, grids, FIG_DIR)
    rf.trip_entrance_length(params, props, grids)

    section("Full model against the quasi-steady model")
    rf.quasi_steady_comparison(params, props, props_ramp, t_ends["feed_ramp"], t_ends["cold_trip"])

    section("Startup from rest (Figure 10)")
    t_end = t_ends["startup"]
    startup = rf.run_grids(params_tight, props_startup, "startup", t_end, np.linspace(0.0, t_end, 1201), grids)
    rf.fig_startup(*startup, params, limits, grids, FIG_DIR)

    section("Thermal stress (Figures 12-15)")
    rf.stress_table(s, [("Feed ramp", ramp[1]), ("Cold trip", trip[1]), ("Startup from rest", startup[1])],
                    params, limits, grids)
    # The stress figures reuse the runs above; the grid panel of the summary adds a coarser
    # and a finer grid for the two scenarios with a temperature front at the cold inlet.
    runs = {"feed_ramp": dict(ramp[0]), "cold_trip": dict(trip[0]), "startup": dict(startup[0])}
    for scenario, props_s in (("cold_trip", props), ("startup", props_startup)):
        for N in (grids[0] // 2, 2 * grids[-1]):
            runs[scenario][N] = transient.simulate(
                dict(params_tight, N=N), props_s, scenario, t_ends[scenario],
                t_eval=np.linspace(0.0, t_ends[scenario], 1201))
    peaks = {scenario: plot_stress.peak_differences(runs[scenario], params, limits, sigma_y) for scenario in runs}
    steady_check = plot_stress.fig_stress_steady(s, params, limits, sigma_y, FIG_DIR)
    print("Steady base case\n\n```\n" + stress.summary_text(steady_check) + "\n```\n")
    for scenario, title in plot_stress.scenario_titles(params).items():
        check = peaks[scenario][grids[-1]]["check"]
        print(f"{title} (N = {grids[-1]})\n\n```\n" + stress.summary_text(check) + "\n```\n")
    plot_stress.fig_stress_transients(runs, params, limits, sigma_y, grids, FIG_DIR)
    plot_stress.fig_stress_profiles(runs, params, limits, sigma_y, grids, FIG_DIR)
    plot_stress.fig_stress_summary(steady_check, peaks, params, limits, grids, FIG_DIR)
    plot_stress.stress_convergence_table(peaks, params, grids)

    section("ALPEMA summary")
    rf.alpema_summary(s, ramp, trip, startup, params, limits, grids)
    plt.close("all")


def main():
    """Rebuild all report figures and write the tables to figures/report_tables.md.

    Inputs: none (uses config.BASE_CASE, config.ALPEMA_LIMITS and config.MATERIAL_LIMITS).
    Outputs: saves the figures listed in the module docstring and figures/report_tables.md;
    prints progress to standard error and the tables to standard output.
    """
    rf.use_file_backend()   # save files only; no windows when run as a script
    rf.set_style()
    params = dict(config.BASE_CASE)
    limits = config.ALPEMA_LIMITS
    sigma_y = config.MATERIAL_LIMITS["sigma_y_3003_O"]
    N = params["N"]
    grids = (N, 2 * N, 4 * N)
    FIG_DIR.mkdir(exist_ok=True)

    # Everything the report functions print (tables, run times, checks) is collected here,
    # then written to report_tables.md and to the terminal.
    start = time.perf_counter()
    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output):
            print("# Report tables\n")
            print("Made by `python -m scripts.make_report_figures` from the base case in `src/config.py`.")
            print(f"Grid studies use N = {', '.join(str(n) for n in grids)}; temperature differences "
                  "include the two end faces of the core.")
            build_report(params, limits, sigma_y, grids)
    finally:
        # Keep whatever was produced, also if a step failed part-way.
        text = output.getvalue()
        (FIG_DIR / "report_tables.md").write_text(text)
        print(text)
    print(f"[make_report_figures] done in {time.perf_counter() - start:.0f} s; "
          f"figures and report_tables.md are in {FIG_DIR}", file=sys.stderr)


if __name__ == "__main__":
    main()
