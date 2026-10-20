# BAHX thermal model (CMU 06-623 course project)

1D thermal model of a two-stream counterflow brazed aluminum plate-fin heat exchanger (BAHX): a steady-state BVP for the temperature profiles, then a method-of-lines transient to check startup and upset conditions against ALPEMA thermal limits. Team: Ryan Kim, Matt Murphy, Calvin Wang. This is graded coursework: every line must be something the team can read and explain, so favor clear, explicit code over clever abstractions.

## Source of truth

- Equations live in `docs/bahx_model_math.tex`. Read the relevant section before implementing or changing any equation, and cite the equation number in the docstring.
- The submitted proposal is in `docs/` for scope and deliverables.
- If the code, the math doc and these notes disagree, stop and flag it. Do not silently pick one.

## Commands

- Activate environment: `source .venv/bin/activate` (Windows: `.venv\Scripts\activate`)
- Install: `pip install -r requirements.txt`
- Test: `pytest -q` (run after every change under `src/` and before any commit)

## Physics conventions (do not change without asking)

- SI units in all code: K, Pa, kg/s, W, m. Convert to degC only for ALPEMA checks and plot labels.
- x runs from 0 to L. Hot stream enters at x = 0 and flows +x. Cold stream enters at x = L and flows -x.
- `a = eta_o * h * A'` is the side conductance per unit length [W/(m K)]; `1/U' = 1/a_h + 1/a_c`.
- Steady balances (both dT/dx are negative in normal operation):
  - hot: `mdot_h * cp_h(T_h) * dT_h/dx = -a_h * (T_h - T_w)`
  - cold: `mdot_c * cp_c(T_c) * dT_c/dx = -a_c * (T_w - T_c)` (cp_c at T_c, not T_h)
  - wall: `T_w = (a_h*T_h + a_c*T_c) / (a_h + a_c)`
- Transient balances (the cold-stream sign has been written wrong before; check it):
  - hot: `rho_h*A_ff_h*cp_h * dT_h/dt + mdot_h*cp_h * dT_h/dx = -a_h * (T_h - T_w)`
  - cold: `rho_c*A_ff_c*cp_c * dT_c/dt - mdot_c*cp_c * dT_c/dx = +a_c * (T_w - T_c)`
  - wall: `C_m' * dT_w/dt = a_h*(T_h - T_w) - a_c*(T_w - T_c)` (plus optional axial conduction)
- Advection uses upwind differences in the direction of flow. Hot: `(T[i] - T[i-1])/dx` with `T[0] = T_h_in`. Cold: `(T[i+1] - T[i])/dx` with `T[N+1] = T_c_in`. Never use central or downwind differences for advection.
- Naming: fin height is `b` (not `h`, which is the heat transfer coefficient); thermal expansion is `alpha_T`; Manglik-Bergles ratios are `alpha_mb = s/b`, `delta_mb = t/l`, `gamma_mb = t/s`.
- Manglik-Bergles Colburn factor:
  `j = 0.6522 Re^-0.5403 alpha_mb^-0.1541 delta_mb^0.1499 gamma_mb^-0.0678 * [1 + 5.269e-5 Re^1.340 alpha_mb^0.504 delta_mb^0.456 gamma_mb^-1.055]^0.1`
  then `h = j * G * cp * Pr^(-2/3)` with `G = mdot / A_ff`.
- Fin efficiency uses half the fin height: `ell_f = b/2`.
<!-- Team note: confirm the Manglik-Bergles coefficients against the 1995 paper before the final report. -->

## Base case and limits

- Pure methane on both sides, single phase only. Hot: high-pressure feed gas. Cold: low-pressure residue gas.
- Base-case inputs and ALPEMA limits live in `src/config.py`. Never hard-code them anywhere else.
- ALPEMA limits: temperature rate of change < 1 degC/min for frequent events; < 5 degC/min and < 60 degC/h for startup, shutdown and trips; stream-to-stream and stream-to-metal differences < about 30 degC.

## Numerics

- Steady state: `scipy.integrate.solve_bvp`. Initial guess = constant-property analytical (epsilon-NTU) profiles. If it fails to converge, use property continuation (math doc Section 5.1), not looser tolerances.
- Transient: `solve_ivp(method="BDF")` with `jac_sparsity`. No explicit methods (RK45) for the transient; the system is stiff.
- Never call CoolProp inside an ODE or BVP right-hand side. Use the surrogate fits in `src/properties.py`, which must raise an error outside their fitted temperature range.
- Methane condenses at low temperature on the low-pressure side, and cp spikes near the pseudo-critical point on the high-pressure side. Keep fit ranges inside the single-phase operating window and report each surrogate's max relative error.
- Take dT_w/dt from the right-hand-side function at saved times, not from `np.gradient` of the solution.

## Verification (tests/)

- Constant-property steady solution matches analytical epsilon-NTU, including C_r = 1.
- Energy closure: hot duty = cold duty = integral of U'(T_h - T_c) dx.
- A transient started from steady state stays there; after a disturbance it relaxes to the new steady solution.
- Report grid convergence (N, 2N, 4N) for any transient metric.
- Never weaken a test or loosen a tolerance to make it pass. Report the failure and its likely cause.

## Code style

- Plain functions on NumPy arrays; avoid classes unless clearly needed.
- Variable names follow the math doc: `T_h`, `T_c`, `T_w`, `a_h`, `U_prime`, `NTU`, `C_r`, `mdot_h`.
- Every function docstring states inputs, outputs and units.
- Comments explain the physics or the equation being implemented, not the Python.
- Real logic goes in `src/`. Notebooks import from `src/` and only make plots; save figures to `figures/`.
- Dependencies: numpy, scipy, matplotlib, CoolProp, jupyter, pytest. Ask before adding anything else (jax only if the team agrees).

## Git

- Small, focused commits with clear messages, with tests passing.
- Do not commit, push, force-push, rebase or amend unless asked.
- Each teammate owns modules; do not edit files outside the current task without asking.
- Never commit `.venv/`, notebook outputs or large data files.
