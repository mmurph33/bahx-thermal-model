"""Image-style view of the steady-state result: temperature along the exchanger.

The model is 1D (one temperature per stream at each x), so this is the 1D result
drawn as a picture: hot passage on top, metal wall in the middle, cold passage
on the bottom, each colored by its temperature at that x.

Run from the repo root with either of:
    python -m model.plot_heatmap
    python model/plot_heatmap.py
"""

import sys
from pathlib import Path

# Running `python model/plot_heatmap.py` puts model/ (not the repo root) on sys.path,
# so `from src import ...` would fail. Add the repo root so both run styles work.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np

from src import config, properties, steady

FIG_DIR = ROOT / "figures"

params = dict(config.BASE_CASE)
props = properties.build_surrogates(params)
s = steady.solve_steady(params, props)
L = params["L"]

# Build the image row by row: hot passage, thin wall, cold passage.
rows_hot, rows_wall, rows_cold = 40, 8, 40
image = np.vstack(
    [np.tile(s["T_h"], (rows_hot, 1)),
     np.tile(s["T_w"], (rows_wall, 1)),
     np.tile(s["T_c"], (rows_cold, 1))]
)
total_rows = rows_hot + rows_wall + rows_cold

fig, ax = plt.subplots(figsize=(10, 4))
im = ax.imshow(image, aspect="auto", cmap="coolwarm", extent=[0, L, total_rows, 0])
fig.colorbar(im, ax=ax, label="Temperature [K]")

# Passage labels and flow directions (hot enters at x = 0, cold enters at x = L).
ax.text(0.01 * L, rows_hot / 2, "hot gas  \u2192", va="center", ha="left", color="black")
ax.text(0.99 * L, rows_hot + rows_wall + rows_cold / 2, "\u2190  cold gas",
        va="center", ha="right", color="black")
ax.text(0.5 * L, rows_hot + rows_wall / 2, "metal wall", va="center", ha="center",
        color="black", fontsize=8)

ax.set_xlabel("x along the core [m]")
ax.set_yticks([])
ax.set_title("Steady-state temperature field (1D result drawn as a picture)")
fig.tight_layout()
FIG_DIR.mkdir(exist_ok=True)
fig.savefig(FIG_DIR / "steady_heatmap.png", dpi=150)
plt.show()
