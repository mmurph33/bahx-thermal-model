"""Animated startup of the exchanger from rest.

Scenario "startup": the idle core and the gas in it start at one uniform
temperature (config startup_T0) with no flow. Both flows then open together
over startup_t_flow seconds, with the hot gas entering at x = 0 at T_h_in and
the cold gas entering at x = L at T_c_in, and the exchanger settles to its
normal steady state.

Saves two files in figures/:
  startup.gif            animation of the hot gas, wall and cold gas temperature
                         profiles along the core, with the outlet temperatures
                         over time above it (for the notebook);
  startup_snapshots.png  six frames of the same animation on one page (for the
                         PDF, which cannot show an animation).

Run from the repo root with either of:
    python -m scripts.plot_transient
    python scripts/plot_transient.py
"""

import sys
import time
from pathlib import Path

# Running `python scripts/plot_transient.py` puts scripts/ (not the repo root) on
# sys.path, so `from src import ...` would fail. Add the repo root so both run
# styles work.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from src import config, properties, transient

FIG_DIR = ROOT / "figures"
SCENARIO = "startup"
FPS = 12             # animation frame rate [frames/s]
N_FRAMES = 110       # frames from the start of the flow until the core has settled
HOLD_S = 1.5         # how long the final frame is held at the end of the loop [s]
STYLE = {"hot gas": dict(color="C1"), "wall": dict(color="C2", linewidth=2.2),
         "cold gas": dict(color="C0")}


def run_startup(params):
    """Simulate the startup and return the result with a few derived times.

    Inputs: params (base-case dict).
    Outputs: (r, t_open, t_settle). r is the transient.simulate result dict;
    t_open is when the flows start to open [s]; t_settle is when the wall is
    within 0.1 K (mean over the cells) of its final profile [s].
    The run lasts 15 whole-exchanger response times (Section 6.2, ~4 min for
    the base case) after the flows are fully open, so it ends at steady state.
    """
    # Surrogates fitted over the inlet temperatures and the idle temperature.
    props = properties.build_surrogates(params, SCENARIO)
    t_open = params["startup_t_start"]
    t_end = t_open + params["startup_t_flow"] + 15.0 * transient.time_scales(params, props)["t_response"]

    start = time.perf_counter()
    r = transient.simulate(params, props, SCENARIO, t_end)
    run_time = time.perf_counter() - start

    distance = np.mean(np.abs(r["T_w"] - r["T_w"][-1]), axis=-1)  # mean |T_w - final| [K]
    t_settle = r["t"][np.nonzero(distance <= 0.1)[0][0]]
    print(f"startup: simulated {t_end:.0f} s in {run_time:.1f} s ({params['N']} cells); "
          f"wall within 0.1 K of its final profile after {t_settle:.0f} s")
    print(f"  hot outlet  {r['T_h'][0, -1]:.1f} K -> {r['T_h'][-1, -1]:.1f} K   "
          f"cold outlet {r['T_c'][0, 0]:.1f} K -> {r['T_c'][-1, 0]:.1f} K")
    return r, t_open, t_settle


def frame_indices(t, t_open, t_settle):
    """Saved-time indices to animate.

    Inputs: t, saved times (n_t,) [s]; t_open, when the flows start [s];
    t_settle, when the core has settled [s].
    Outputs: index array. A few frames of the idle core, then N_FRAMES frames
    from t_open to t_settle, closer together at the start where the
    temperatures change fastest (spacing grows like k^1.5).
    """
    idle = np.linspace(0.0, t_open, 4, endpoint=False)
    k = np.linspace(0.0, 1.0, N_FRAMES)
    running = t_open + (t_settle - t_open) * k**1.5
    return np.unique([np.argmin(np.abs(t - tf)) for tf in np.concatenate((idle, running))])


def draw_profiles(ax, r, n, params, final):
    """Draw the three temperature profiles at saved time index n on ax.

    Inputs: ax (matplotlib axes), r (result dict), n (time index), params,
    final (whether to draw the final steady profiles as a dotted reference).
    Outputs: dict of the three Line2D objects, keyed like STYLE.
    """
    x = r["x"]
    if final:
        for name, T in (("hot gas", r["T_h"][-1]), ("wall", r["T_w"][-1]), ("cold gas", r["T_c"][-1])):
            ax.plot(x, T, ":", color=STYLE[name]["color"], linewidth=1, alpha=0.7)
    lines = {name: ax.plot(x, T, label=name, **STYLE[name])[0]
             for name, T in (("hot gas", r["T_h"][n]), ("wall", r["T_w"][n]), ("cold gas", r["T_c"][n]))}
    L = params["L"]
    ax.set_xlim(0.0, L)
    ax.set_ylim(params["T_c_in"] - 6.0, max(params["T_h_in"], params["startup_T0"]) + 6.0)
    # Inlet labels: hot gas enters at x = 0 and flows +x; cold gas enters at x = L and flows -x.
    ax.annotate("hot in →", (0.0, params["T_h_in"]), xytext=(4, 6), textcoords="offset points",
                color=STYLE["hot gas"]["color"], fontsize=8)
    ax.annotate("← cold in", (L, params["T_c_in"]), xytext=(-4, -12), textcoords="offset points",
                color=STYLE["cold gas"]["color"], fontsize=8, ha="right")
    return lines


def status(r, n, params):
    """Caption for time index n: time and flow as a percentage of full flow."""
    return f"t = {r['t'][n]:5.1f} s    flow = {100 * r['mdot_c'][n] / params['mdot_c']:3.0f} % of full"


def make_animation(r, t_open, t_settle, params):
    """Build and save figures/startup.gif.

    Inputs: r (result dict), t_open and t_settle [s], params.
    Outputs: saves the GIF. Top panel: outlet temperatures over time with a
    moving cursor; bottom panel: hot gas, wall and cold gas profiles along the
    core at the current time, with the final steady profiles dotted.
    """
    t = r["t"]
    frames = frame_indices(t, t_open, t_settle)
    shown = t <= t[frames[-1]]

    fig, (ax_t, ax_x) = plt.subplots(2, 1, figsize=(7.5, 7.5),
                                     gridspec_kw={"height_ratios": [1, 2.2]})
    fig.suptitle(f"Startup from rest: idle core at {params['startup_T0']:.0f} K, "
                 f"then both flows open over {params['startup_t_flow']:.0f} s", fontsize=11)

    # Top: outlet temperatures over the animated period, with a cursor at the current time.
    ax_t.plot(t[shown], r["T_h"][shown, -1], color=STYLE["hot gas"]["color"], label="hot outlet (x = L)")
    ax_t.plot(t[shown], r["T_c"][shown, 0], color=STYLE["cold gas"]["color"], label="cold outlet (x = 0)")
    cursor = ax_t.axvline(t[frames[0]], color="k", linewidth=1)
    ax_t.set_xlim(0.0, t[frames[-1]])
    ax_t.set_xlabel("time [s]")
    ax_t.set_ylabel("T [K]")
    ax_t.legend(fontsize=8, loc="center right")

    # Bottom: the profiles along the core.
    lines = draw_profiles(ax_x, r, frames[0], params, final=True)
    caption = ax_x.text(0.02, 0.04, status(r, frames[0], params), transform=ax_x.transAxes,
                        family="monospace", fontsize=9)
    ax_x.set_xlabel("x along the core [m]")
    ax_x.set_ylabel("temperature [K]")
    # Legend in the lower left, just above the caption: no profile ever goes there.
    ax_x.legend(fontsize=8, loc="lower left", bbox_to_anchor=(0.0, 0.08),
                title="dotted: final steady state", title_fontsize=7)
    fig.tight_layout()

    def update(n):
        for name, T in (("hot gas", r["T_h"][n]), ("wall", r["T_w"][n]), ("cold gas", r["T_c"][n])):
            lines[name].set_ydata(T)
        cursor.set_xdata([t[n], t[n]])
        caption.set_text(status(r, n, params))
        return [*lines.values(), cursor, caption]

    # Hold the last frame for HOLD_S seconds before the GIF loops.
    sequence = list(frames) + [frames[-1]] * int(HOLD_S * FPS)
    animation = FuncAnimation(fig, update, frames=sequence, blit=False)
    FIG_DIR.mkdir(exist_ok=True)
    animation.save(FIG_DIR / "startup.gif", writer=PillowWriter(fps=FPS), dpi=90)
    plt.close(fig)


def make_snapshots(r, params):
    """Save six frames of the startup on one page, figures/startup_snapshots.png.

    Inputs: r (result dict), params.
    Outputs: saves the PNG. Frames: the idle core at t = 0, the times at which
    the wall has made 10 %, 30 %, 60 % and 90 % of its total change, and the end.
    """
    change = np.mean(np.abs(r["T_w"] - r["T_w"][0]), axis=-1)
    picks = [0]
    picks += [int(np.nonzero(change >= f * change[-1])[0][0]) for f in (0.1, 0.3, 0.6, 0.9)]
    picks.append(len(r["t"]) - 1)

    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5), sharex=True, sharey=True)
    for ax, n in zip(axes.flat, picks):
        draw_profiles(ax, r, n, params, final=(n != picks[-1]))
        ax.set_title(status(r, n, params), fontsize=9, family="monospace")
    for ax in axes[1]:
        ax.set_xlabel("x along the core [m]")
    for ax in axes[:, 0]:
        ax.set_ylabel("temperature [K]")
    axes.flat[0].legend(fontsize=8, loc="center left")
    fig.suptitle(f"Startup from rest: idle core at {params['startup_T0']:.0f} K, then both flows "
                 f"open over {params['startup_t_flow']:.0f} s (dotted: final steady state)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "startup_snapshots.png", dpi=150)
    plt.close(fig)


def main():
    """Simulate the startup and save the animation and the snapshot page.

    Inputs: none (uses config.BASE_CASE).
    Outputs: prints a short summary; saves figures/startup.gif and
    figures/startup_snapshots.png.
    """
    params = dict(config.BASE_CASE)
    r, t_open, t_settle = run_startup(params)
    make_animation(r, t_open, t_settle, params)
    make_snapshots(r, params)
    print(f"saved {FIG_DIR / 'startup.gif'} and {FIG_DIR / 'startup_snapshots.png'}")


if __name__ == "__main__":
    main()
