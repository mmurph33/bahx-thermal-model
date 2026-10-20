# Thermal Model of a Counterflow Brazed Aluminum Plate-Fin Heat Exchanger

CMU 06-623 (Mathematical Modeling of Chemical Engineering Processes), Fall 2026
Team: Ryan Kim, Matt Murphy, Calvin Wang

Brazed aluminum plate-fin heat exchangers (BAHX) are the compact cores used in cryogenic gas processing. They are efficient, but they fail by thermal fatigue: fast temperature swings during startup, shutdown or a trip, and large local temperature differences between streams and metal, crack the brazed joints. The ALPEMA Integrity Operating Window (IOW) table sets the guidance operators follow: rates of change below 1 °C/min for frequent events and below 5 °C/min and 60 °C/h for startups, shutdowns and trips, and stream-to-stream and stream-to-metal differences below about 30 °C. This project builds a 1D model of a two-stream counterflow BAHX with offset strip fins (pure methane on both sides, single phase), first at steady state and then in time, to identify the operating conditions that approach those limits during startup and upset conditions.

## Model overview

- **Steady state (minimum viable product):** two coupled energy-balance ODEs with inlet conditions at opposite ends (a boundary value problem), solved with `scipy.integrate.solve_bvp`. It is verified against the analytical ε-NTU solution and cross-checked by shooting.
- **Closure relations:** Manglik–Bergles offset-strip-fin correlations for the heat transfer coefficient, plus fin and surface efficiencies. Properties come from CoolProp through smooth polynomial surrogates fitted once, so CoolProp is never called inside an ODE right-hand side.
- **Transient (primary extension):** a method-of-lines finite-volume model (hot gas, cold gas and metal in each of N cells) integrated with the stiff solver `solve_ivp(method="BDF")` and a sparse Jacobian pattern, for a feed-temperature ramp and a cold-stream trip. A quasi-steady gas model is the fallback if the full model proves too slow.
- **Thermal-fatigue metrics:** stream-to-stream and stream-to-metal temperature differences, the peak wall heating rate dT_w/dt and the peak wall gradient dT_w/dx, compared with the ALPEMA limits.
- **Stretch goals (if time permits):** the fastest admissible startup ramp by constrained optimization, and a first-order thermal stress estimate.

Scope and deliverables follow the submitted proposal, [docs/project_proposal.pdf](docs/project_proposal.pdf). The equations, with numbers we cite in the code, are in [docs/bahx_model_math.tex](docs/bahx_model_math.tex) (PDF alongside). Project rules and physics conventions are in [CLAUDE.md](CLAUDE.md).

## Repository layout

```text
bahx-thermal-model/
├── docs/                 math formulation (bahx_model_math.tex/.pdf, source of truth for equations) and the submitted proposal
├── src/                  all model code (plain functions on NumPy arrays)
│   ├── __init__.py       makes src importable as a package
│   ├── config.py         base-case inputs, scenario settings, numerics and ALPEMA limits (the only place numbers live)
│   ├── properties.py     CoolProp tables, polynomial property surrogates, range checks
│   ├── correlations.py   fin geometry, Manglik-Bergles j, fin/surface efficiency, side conductances
│   ├── steady.py         analytical ε-NTU solution, steady BVP (collocation + shooting), duty and energy closure
│   ├── transient.py      method-of-lines transient, upset scenarios, discrete steady state, quasi-steady fallback
│   └── metrics.py        thermal-fatigue and ALPEMA metrics, grid-convergence order, smooth max, thermal stress
├── tests/                pytest verification checks (see CLAUDE.md "Verification")
├── notebooks/            analysis notebooks; import from src/ and only make plots
├── figures/              saved figures from notebooks
├── data/                 small input data (e.g. NIST WebBook spot-check values)
├── pytest.ini            puts the repo root on sys.path for tests
├── requirements.txt      dependencies with minimum versions
└── CLAUDE.md             project rules, conventions and verification list
```

## Setup

Supported Python: **3.11–3.14** (developed on 3.14). The minimum versions in `requirements.txt` are the first releases with Python 3.14 wheels.

```bash
git clone <repo-url> bahx-thermal-model
cd bahx-thermal-model
python3 -m venv .venv
```

Activate the environment:

```bash
# macOS / Linux
source .venv/bin/activate
```

```powershell
# Windows (PowerShell or cmd)
.venv\Scripts\activate
```

Install and run the tests:

```bash
pip install -r requirements.txt
pytest -q
```

On a fresh clone, `test_setup.py` should pass and every other test should be reported as skipped until its module is implemented.

### Notebooks

Every notebook in `notebooks/` should start with this cell so that `src` is importable and edits to `src/` are picked up without restarting the kernel:

```python
import sys
from pathlib import Path

ROOT = Path.cwd().resolve().parent  # notebooks/ -> repo root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

%load_ext autoreload
%autoreload 2

from src import config, properties, correlations, steady, transient, metrics

FIG_DIR = ROOT / "figures"  # save with plt.savefig(FIG_DIR / "name.png")
```

Clear outputs before committing a notebook: `jupyter nbconvert --clear-output --inplace notebooks/*.ipynb`.

## Code contract

The function signatures in `src/` are the agreement between teammates. The full details are in each module's docstring.

- **`params`**: one flat dict of every input, `params = dict(config.BASE_CASE)`. To change something for a study (a ramp-rate sweep, a grid-convergence run), use a copy: `p = dict(params, ramp_t=600.0)`. Never edit `config.BASE_CASE` in place. Modules in `src/` never import `config`; callers pass `params` and `config.ALPEMA_LIMITS` in.
- **`scenario`**: a name, `"none"`, `"feed_ramp"` or `"cold_trip"`. The numbers for each scenario live in `params`.
- **`props`**: `{"h": surrogate, "c": surrogate}` from `properties.build_surrogates(params, scenario)`. It is built once with CoolProp and then passed into the solvers. Rebuild it when a pressure or the temperature range changes.
- **`geom`**: `{"h": side_geom, "c": side_geom, "A_m": ..., "C_m": ...}` from `correlations.exchanger_geometry(params)`. Public solvers build it themselves from `params`; only right-hand-side functions and helpers take it as an argument.
- **Shapes and units:** SI units throughout (K, Pa, kg/s, W, m). Steady profiles are `(n,)`, the `solve_bvp` state is `(2, m)`, the transient state is `(3N,)` ordered cell by cell `[T_h1, T_c1, T_w1, T_h2, ...]`, and transient histories are `(n_t, N)`.
- **Directions:** x runs from 0 to L. The hot stream enters at x = 0 and flows +x; the cold stream enters at x = L and flows −x.

## Where to start

Each function in `src/` has a docstring, numbered pseudocode and a `# Check:` comment that says how we will know it works. Each `# Check:` is a ready-made unit test. Implement a function, turn its check into a test, then remove the `raise NotImplementedError`.

Suggested build order:

1. **`properties.py`**. Depends on CoolProp and numpy. Start with `constant_surrogate`, `check_range`, `evaluate` and `enthalpy`: the constant-property steady verification needs only these, not CoolProp fits. Then add `coolprop_table`, `check_single_phase`, `fit_surrogate` and `build_surrogates`.
   Owner: TBD
2. **`correlations.py`**. Depends on `properties`. Geometry first (`osf_geometry`, `exchanger_geometry`), then `colburn_j`, `heat_transfer_coefficient`, fin efficiency, `side_conductance` and `conductances`.
   Owner: TBD
3. **`steady.py`**. Depends on `properties` and `correlations`. Analytical functions first (`effectiveness`, `analytical_profiles`), then `steady_rhs`, `solve_steady`, `energy_balance`, then shooting (the proposal's root-finding backup and our cross-check on collocation). This is the minimum viable product.
   Owner: TBD
4. **`transient.py`**. Depends on `properties`, `correlations` and `steady` (initial guess). State packing and the Jacobian pattern first, then `transient_rhs`, `discrete_steady_state` and `simulate`; the quasi-steady functions only if the full model is too slow.
   Owner: TBD
5. **`metrics.py`**. Depends only on numpy/scipy, so it can be written at any time in parallel with the others.
   Owner: TBD

`config.py` is shared: fill in the `TODO` values together before running anything beyond the constant-property tests.
Owner: TBD

## Roadmap

- [ ] Steady model (minimum viable product) + ε-NTU verification (constant properties, including C_r = 1; energy closure; collocation vs shooting)
- [ ] Property surrogates (CoolProp fits with reported max relative error, single-phase range checks, NIST spot-checks)
- [ ] Transient model (primary extension) + thermal-fatigue and ALPEMA metrics (feed ramp and cold trip; grid convergence N, 2N, 4N)
- [ ] Fallback only if the full transient stalls: quasi-steady gas model, or a parametric sweep of startup ramp rates checked against the ALPEMA limits
- [ ] Stretch: fastest admissible startup ramp (constrained optimization)
- [ ] Stretch: thermal stress estimate from the temperature differences

## References

1. G. F. Nellis, "A heat exchanger model that includes axial conduction, parasitic heat loads, and property variations," *Cryogenics*, vol. 43, no. 9, pp. 523–538, 2003.
2. J. C. Pacio and C. A. Dorao, "A review on heat exchanger thermal hydraulic models for cryogenic applications," *Cryogenics*, vol. 51, no. 7, pp. 366–379, 2011. <https://doi.org/10.1016/j.cryogenics.2011.04.005>
3. R. M. Manglik and A. E. Bergles, "Heat transfer and pressure drop correlations for the rectangular offset strip fin compact heat exchanger," *Experimental Thermal and Fluid Science*, vol. 10, no. 2, pp. 171–180, 1995.
4. W. M. Kays and A. L. London, *Compact Heat Exchangers*, 3rd ed. New York: McGraw-Hill, 1984.
5. R. K. Shah and D. P. Sekulić, *Fundamentals of Heat Exchanger Design*. Hoboken, NJ: Wiley, 2003.
6. ALPEMA, "Integrity Operating Windows for Brazed Aluminium Plate-Fin Heat Exchangers," June 2025. <https://www.alpema.org/pdf/ALPEMA_Integrity_Operating_Window_Table.pdf>
7. I. H. Bell, J. Wronski, S. Quoilin, and V. Lemort, "Pure and pseudo-pure fluid thermophysical property evaluation and the open-source thermophysical property library CoolProp," *Industrial & Engineering Chemistry Research*, vol. 53, no. 6, pp. 2498–2508, 2014. <https://doi.org/10.1021/ie4033999>
8. O. Levenspiel, *Chemical Reaction Engineering*, 3rd ed. New York: Wiley, 1999.
9. P. J. Linstrom and W. G. Mallard, Eds., *NIST Chemistry WebBook*, NIST Standard Reference Database Number 69. Gaithersburg, MD: National Institute of Standards and Technology. <https://doi.org/10.18434/T4D303> (methane property spot-checks)
