"""Quick look at the steady-state model: solve the base case and plot the profiles.

Run from the repo root with either of:
    python -m scripts.run_steady
    python scripts/run_steady.py
"""

import sys
from pathlib import Path

# Running `python scripts/run_steady.py` puts scripts/ (not the repo root) on sys.path,
# so `from src import ...` would fail. Add the repo root so both run styles work.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt

from src import config, properties, steady

FIG_DIR = ROOT / "figures"


def main():
    """Solve the base-case steady state, print the key numbers and save the profile plot.

    Inputs: none (uses config.BASE_CASE).
    Outputs: prints duty [kW], outlet and wall end temperatures [K], energy
    closure and the max stream-to-stream difference [K]; saves figures/steady_profiles.png
    (top: T_h, T_c, T_w with inlet/outlet markers; bottom: T_h - T_c vs x).
    """
    params = dict(config.BASE_CASE)
    props = properties.build_surrogates(params)
    s = steady.solve_steady(params, props)

    duty = steady.energy_balance(s, params, props)
    print(f"Duty per passage: {duty['Q_h'] / 1e3:.2f} kW")
    print(f"Hot outlet:  {s['T_h'][-1]:.2f} K   Cold outlet: {s['T_c'][0]:.2f} K")
    print(f"Wall at x = 0: {s['T_w'][0]:.2f} K   Wall at x = L: {s['T_w'][-1]:.2f} K")
    print(f"Energy closure: hot-vs-cold {duty['closure_hc']:.1e}, integral {duty['closure_int']:.1e}")
    print(f"Max stream-to-stream difference: {(s['T_h'] - s['T_c']).max():.1f} K")

    # Top panel: the three temperature profiles. Hot enters at x = 0 and flows +x;
    # cold enters at x = L and flows -x, so each stream's inlet is at opposite ends.
    # All three arrays are indexed by x (T_c[0] is the cold OUTLET at x = 0), so
    # they are plotted as they are; reversing T_c would mirror the cold profile.
    x, T_h, T_c, T_w = s["x"], s["T_h"], s["T_c"], s["T_w"]
    fig, (ax_T, ax_gap) = plt.subplots(2, 1, sharex=True, figsize=(7, 7),
                                       gridspec_kw={"height_ratios": [2, 1]})
    ax_T.plot(x, T_h, color="C1", label="hot")
    ax_T.plot(x, T_c, color="C0", label="cold")
    ax_T.plot(x, T_w, "--", color="C2", label="wall")

    # Filled dot = inlet, open dot = outlet (the label says which end each stream enters).
    ax_T.plot(x[0], T_h[0], "o", color="C1")
    ax_T.plot(x[-1], T_h[-1], "o", color="C1", mfc="white")
    ax_T.plot(x[-1], T_c[-1], "o", color="C0")
    ax_T.plot(x[0], T_c[0], "o", color="C0", mfc="white")
    # Labels sit in the left/right margins (xlim is widened below) so they never overlap a curve.
    left = dict(textcoords="offset points", xytext=(-9, 0), ha="right", va="center", fontsize=9)
    right = dict(textcoords="offset points", xytext=(9, 0), ha="left", va="center", fontsize=9)
    ax_T.annotate(f"hot in\n{T_h[0]:.1f} K", (x[0], T_h[0]), color="C1", **left)
    ax_T.annotate(f"cold out\n{T_c[0]:.1f} K", (x[0], T_c[0]), color="C0", **left)
    ax_T.annotate(f"hot out\n{T_h[-1]:.1f} K", (x[-1], T_h[-1]), color="C1", **right)
    ax_T.annotate(f"cold in\n{T_c[-1]:.1f} K", (x[-1], T_c[-1]), color="C0", **right)
    # No wall end labels: at x = 0 the wall sits only a few K from both streams, so
    # its label would overlap "hot in" and "cold out". The values are printed above.
    ax_T.set_ylabel("T [K]")
    ax_T.legend()

    # Bottom panel: stream-to-stream difference. In counterflow this is smallest where the
    # lower-C stream (here the cold gas) leaves and grows toward the other end when C_h > C_c.
    gap = T_h - T_c
    ax_gap.plot(x, gap, color="k")
    i_min, i_max = gap.argmin(), gap.argmax()
    ax_gap.plot(x[[i_min, i_max]], gap[[i_min, i_max]], "ko")
    ax_gap.annotate(f"{gap[i_min]:.1f} K", (x[i_min], gap[i_min]), **left)
    ax_gap.annotate(f"{gap[i_max]:.1f} K", (x[i_max], gap[i_max]), **right)
    ax_gap.margins(y=0.12)
    # Widen the shared x-axis for the margin labels, but only tick the real core length 0..L.
    ax_gap.set_xlim(x[0] - 0.16, x[-1] + 0.16)
    ticks = [tick for tick in ax_gap.get_xticks() if 0 <= tick <= x[-1] - 0.03]
    ax_gap.set_xticks(ticks + [x[-1]])  # include x = L so the cold inlet end is labeled
    ax_gap.set_xlabel("x [m]")
    ax_gap.set_ylabel("T_h - T_c [K]")
    fig.tight_layout()

    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / "steady_profiles.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
