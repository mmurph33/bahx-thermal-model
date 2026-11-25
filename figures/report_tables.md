# Report tables

Made by `python -m scripts.make_report_figures` from the base case in `src/config.py`.
Grid studies use N = 100, 200, 400; temperature differences include the two end faces of the core.

## Finite-volume grid (Figure 1)


## Base case and ALPEMA limits


| Quantity | Value |
|:--|--:|
| Fluid (both streams) | Methane, single phase |
| Hot (feed gas) pressure $p_h$ | 6.0 MPa |
| Cold (residue gas) pressure $p_c$ | 1.5 MPa |
| Inlet temperatures $T_{h,in}$ / $T_{c,in}$ | 300 K / 235 K |
| Mass flow per passage $\dot m_h$ / $\dot m_c$ | 0.1 kg/s / 0.1 kg/s |
| Core length $L$ / width $W$ | 0.7 m / 1.0 m |
| Fin spacing $s$, height $b$, thickness $t$, strip length $l$ (both sides) | 1.525, 6.25, 0.102, 3.175 mm |
| Parting sheet thickness $t_w$ | 1.0 mm |
| Aluminum density, heat capacity, conductivity | 2700 kg/m$^3$, 880 J/(kg K), 180 W/(m K) |
| Aluminum Young's modulus, expansion coefficient, Poisson's ratio | 70 GPa, 23e-6 1/K, 0.33 |
| Feed ramp: size, duration, start | +10 K over 300 s (2 K/min), starting at t = 120 s |
| Cold trip: valve time constant, residual flow, start | $\tau_v$ = 10 s, $\varphi$ = 0.05, starting at t = 60 s |
| Startup: idle core temperature, flow opening time, start | 300 K, 20 s, starting at t = 10 s |
| Axial conduction in the metal | off |
| Cells $N$ (grid study) | 100 (100, 200, 400) |
| Tolerances: solve_bvp; BDF rtol / atol | 1e-06; 1e-06 / 1e-06 K |
| Property surrogates: degree, points, margin, max error | 7, 200, 10 K, 1e-03 |


| ALPEMA limit (ALPEMA, 2025) | Value |
|:--|--:|
| Rate of change, frequent events | 1 °C/min |
| Rate of change, startup, shutdown and trips | 5 °C/min |
| Change in any 60 min, startup, shutdown and trips | 60 °C |
| Stream-to-stream difference | about 30 °C |
| Stream-to-metal difference | about 30 °C |


## Property surrogates (Figure 2)


| Surrogate | Range | $c_p$ | $\mu$ | $k$ | $\rho$ | $\hat h$ |
|:--|--:|--:|--:|--:|--:|--:|
| hot, 6 MPa, steady, trip and startup | 225-310 K | 2.9e-04 | 3.6e-05 | 7.9e-05 | 7.4e-05 | 6.1e-06 |
| cold, 1.5 MPa, steady, trip and startup | 225-310 K | 3.0e-07 | 8.6e-09 | 2.4e-07 | 2.2e-07 | 5.8e-09 |
| hot, 6 MPa, feed ramp | 225-320 K | 4.4e-04 | 5.6e-05 | 1.2e-04 | 1.1e-04 | 9.0e-06 |
| cold, 1.5 MPa, feed ramp | 225-320 K | 5.6e-07 | 1.6e-08 | 4.0e-07 | 4.3e-07 | 1.1e-08 |


| Stream | Nearest hazard | Margin below fit range | max/min $c_p$ over range |
|:--|--:|--:|--:|
| hot, 6.0 MPa | pseudo-critical $c_p$ peak at 199.7 K | 25.3 K | 1.52 |
| cold, 1.5 MPa | saturation at 158.5 K | 66.5 K | 1.03 |


## Steady model: verification against the analytical solution (Figure 3)


| Case | NTU | $\varepsilon$ analytical | $\varepsilon$ solve_bvp | max $|\Delta\theta|$ |
|:--|--:|--:|--:|--:|
| hot is $C_{min}$, $C_r$ = 0.5 | 6 | 0.9744710 | 0.9744710 | 2.7e-11 |
| cold is $C_{min}$, $C_r$ = 0.5 | 6 | 0.9744710 | 0.9744710 | 2.7e-11 |
| balanced, $C_r$ = 1 | 4 | 0.8000000 | 0.8000000 | 1.8e-15 |


## Steady base case (Figure 4)


| Check | Result |
|:--|--:|
| Hot-stream duty $Q_h$ | 13.302706 kW |
| Cold-stream duty $Q_c$ | 13.302706 kW |
| Integrated heat flow $\int U'(T_h - T_c)\,dx$ | 13.302706 kW |
| Closure $|Q_h - Q_c|/Q_h$ | 5.6e-10 |
| Closure $|Q_h - Q_{int}|/Q_h$ | 4.1e-10 |
| max $|T_h|$ difference, collocation vs shooting | 6.4e-07 K |
| max $|T_c|$ difference, collocation vs shooting | 7.8e-07 K |


| Quantity | Hot side | Cold side |
|:--|--:|--:|
| Reynolds number | 3269-3591 | 3621-4366 |
| Heat transfer coefficient $h$ [W/(m$^2$ K)] | 439-446 | 357-399 |
| Fin efficiency $\eta_f$ / surface efficiency $\eta_o$ | 0.865 / 0.891 | 0.881 / 0.904 |
| Side conductance $a$ [W/(m K)] | 3738-3792 | 3102-3430 |
| Mean capacity rate $C = \dot m c_p$ [W/K] | 276.5 | 229.5 |


| Exchanger result | Value |
|:--|--:|
| Duty per passage $Q$ | 13.30 kW |
| Hot outlet / cold outlet | 252.2 K / 293.0 K |
| Overall conductance $U'$ | 1704-1801 W/(m K) |
| NTU (mean $U'$ and $C$) / $C_r$ | 5.34 / 0.83 |
| Effectiveness $Q/Q_{max}$ | 0.891 |
| Max stream-to-stream difference (location) | 17.2 K (x = 0.70 m) |
| Max stream-to-metal difference | 9.5 K |


## Time scales and Jacobian sparsity (Figure 5)


| Time scale | Hot side | Cold side |
|:--|--:|--:|
| Gas residence time $\tau_{res}$ | 2.09 s | 0.46 s |
| Gas-to-metal equilibration $\tau_g$ | 0.231 s | 0.046 s |


| Quantity | Value |
|:--|--:|
| Metal relaxation time $\tau_m$ | 0.93 s |
| Metal-to-gas heat capacity ratio $\kappa$ | 6.4 |
| Exchanger response time $(1 + \kappa)\max\tau_{res}$ | 15.4 s |
| Stiffness ratio estimate $N(1 + \kappa)$, N = 100 | 738 |


## Transient model: verification (Figure 6)


| Check | Result |
|:--|--:|
| (a) max drift over 5 response times from the discrete steady state | 1.1e-13 K |
| (a) max wall rate during the hold | 6.3e-12 K/min |
| (b) BVP start: hot inlet cell / interior hot cells, max $|dT_h/dt|$ | 9.6 K/s / 0.173 K/s |
| (b) BVP start: cold inlet cell / interior cold cells, max $|dT_c/dt|$ | 82.6 K/s / 0.977 K/s |
| (c) after the ramp: max difference from the new discrete steady state | 1.1e-13 K |

Ratio of successive gaps (2 means first order): [1.797 1.891 1.943 1.97 ]

## Feed-temperature ramp (Figure 7)

N = 100: 300 equations, 0.54 s
N = 200: 600 equations, 0.94 s
N = 400: 1200 equations, 2.89 s

| Output | N = 100 | N = 200 | N = 400 | $p_{obs}$ | extrapolated $y^\star$ |
|:--|--:|--:|--:|--:|--:|
| peak metal rate $R_w$ [K/min] | 1.860 | 1.868 | 1.871 | 1.00 | 1.875 |
| peak outlet-nozzle rate $R_{end}$ [K/min] | 1.728 | 1.735 | 1.738 | 0.99 | 1.742 |
| max stream-to-stream difference [K] | 19.705 | 19.505 | 19.404 | 1.00 | 19.303 |
| max stream-to-metal difference [K] | 11.240 | 10.919 | 10.756 | 0.98 | 10.588 |
| total metal temperature change [K] | 9.304 | 9.339 | 9.356 | 1.00 | 9.374 |


## Ramp-rate sweep (Figure 8)

27 simulations in 44 s
Observed order across the sweep: 0.96 to 1.00; largest relative change from N = 200 to 400: 1.3 %

| Ramp rate [K/min] | Peak $R_w$ [K/min] | Peak $R_{end}$ [K/min] | $R_w/r$ | $R_{end}/r$ |
|:--|--:|--:|--:|--:|
| 0.5 | 0.468 | 0.435 | 0.936 | 0.869 |
| 1 | 0.936 | 0.869 | 0.936 | 0.869 |
| 2 | 1.871 | 1.738 | 0.936 | 0.869 |
| 5 | 4.670 | 4.335 | 0.934 | 0.867 |
| 10 | 9.252 | 8.522 | 0.925 | 0.852 |
| 20 | 17.973 | 16.096 | 0.899 | 0.805 |
| 60 | 48.894 | 39.465 | 0.815 | 0.658 |
| 120 | 87.036 | 63.119 | 0.725 | 0.526 |
| 300 | 160.433 | 98.888 | 0.535 | 0.330 |


## Cold-stream trip (Figure 9)

N = 100: 300 equations, 0.61 s
N = 200: 600 equations, 1.01 s
N = 400: 1200 equations, 2.90 s

| Output | N = 100 | N = 200 | N = 400 | $p_{obs}$ | extrapolated $y^\star$ |
|:--|--:|--:|--:|--:|--:|
| peak metal rate $R_w$ [K/min] | 101.762 | 103.110 | 103.795 | 0.98 | 104.502 |
| peak outlet-nozzle rate $R_{end}$ [K/min] | 92.394 | 93.351 | 93.841 | 0.97 | 94.352 |
| max cold-end nozzle difference [K] | 62.190 | 62.190 | 62.190 | n/a | n/a |
| max stream-to-metal difference, cells and end faces [K] | 54.948 | 53.911 | 53.285 | 0.73 | 52.331 |
| max stream-to-stream difference over cells only [K] | 45.814 | 52.689 | 57.065 | 0.65 | 64.727 |
| max stream-to-metal difference over cells only [K] | 38.237 | 44.372 | 48.161 | 0.70 | 54.278 |
| total metal temperature change [K] | 49.069 | 49.333 | 49.473 | 0.90 | 49.635 |


| Cold stream at its inlet | Design flow | Residual trip flow |
|:--|--:|--:|
| Mass flow per passage | 0.100 kg/s | 0.005 kg/s |
| Entrance length $\dot m_c c_{p,c} / a_c$ | 74 mm | 17 mm |

Cell width: 7.00 mm (N = 100), 3.50 mm (N = 200), 1.75 mm (N = 400)
At the residual flow: Re = 218, h from Manglik-Bergles = 74 W/(m2 K), laminar floor = 62 W/(m2 K)

## Full model against the quasi-steady model


| Scenario (N = 100) | max metal temperature difference | peak $R_w$ [K/min], full / quasi-steady | run time, full / quasi-steady |
|:--|--:|--:|--:|
| feed ramp | 0.04 K | 1.86 / 1.86 | 0.3 s / 12.2 s |
| cold trip | 1.68 K | 101.76 / 105.90 | 0.4 s / 12.1 s |


## Startup from rest (Figure 10)

N = 100: 300 equations, 0.55 s
N = 200: 600 equations, 0.66 s
N = 400: 1200 equations, 0.84 s

| Output | N = 100 | N = 200 | N = 400 | $p_{obs}$ | extrapolated $y^\star$ |
|:--|--:|--:|--:|--:|--:|
| peak metal rate $R_w$ [K/min] | 276.684 | 286.883 | 291.699 | 1.08 | 296.008 |
| peak outlet-nozzle rate $R_{end}$ [K/min] | 168.437 | 168.592 | 168.521 | n/a | n/a |
| max cold-end nozzle difference [K] | 65.000 | 65.000 | 65.000 | n/a | 65.000 |
| max stream-to-metal difference, cells and end faces [K] | 65.000 | 65.000 | 65.000 | n/a | 65.000 |
| max stream-to-stream difference over cells only [K] | 47.557 | 52.633 | 56.264 | 0.48 | 65.390 |
| max stream-to-metal difference over cells only [K] | 40.077 | 44.749 | 47.636 | 0.69 | 52.307 |
| total metal temperature change [K] | 54.964 | 55.246 | 55.389 | 0.98 | 55.537 |


## Thermal stress (Figures 12-15)


| Case | Largest stream-to-metal $\Delta T$ [K] | $\sigma_{th}$ [MPa] | Largest stream-to-stream $\Delta T$ [K] | $\sigma_{th}$ [MPa] |
|:--|--:|--:|--:|--:|
| Steady base case | 9.5 | 23 | 17.2 | 41 |
| Feed ramp | 10.6 | 25 | 19.4 | 47 |
| Cold trip | 52.3 | 126 | 62.2 | 149 |
| Startup from rest | 65.0 | 156 | 65.0 | 156 |


| Reference level | Temperature difference | Stress |
|:--|--:|--:|
| Fully restrained stress per kelvin | 1 K | 2.40 MPa |
| ALPEMA difference limit | 30 K | 72 MPa |
| Minimum yield strength of 3003-O aluminium (ASTM B209M) | 14.6 K | 35 MPa |

Steady base case

```
ALPEMA check on the temperature differences:
  stream-to-metal   max   9.5 K   limit 30 K   margin  20.5 K   OK
  stream-to-stream  max  17.2 K   limit 30 K   margin  12.8 K   OK
Fully restrained thermal stress (Eq 53): 2.40 MPa per K of difference
  hot-to-metal      peak  18.7 MPa at x = 0.70 m (Delta T = 7.8 K)
  metal-to-cold     peak  22.7 MPa at x = 0.70 m (Delta T = 9.5 K)
  stream-to-stream  peak  41.4 MPa at x = 0.70 m (Delta T = 17.2 K)
  ALPEMA limit as a stress: 72.1 MPa
  Yield strength 35 MPa is reached at Delta T = 14.6 K
```

Feed ramp, +10 K at 2 K/min (N = 400)

```
ALPEMA check on the temperature differences:
  stream-to-metal   max  10.8 K   limit 30 K   margin  19.2 K   OK
  stream-to-stream  max  19.4 K   limit 30 K   margin  10.6 K   OK
Fully restrained thermal stress (Eq 53): 2.40 MPa per K of difference
  hot-to-metal      peak  20.8 MPa at x = 0.70 m, t = 720 s (Delta T = 8.6 K)
  metal-to-cold     peak  25.8 MPa at x = 0.70 m, t = 720 s (Delta T = 10.8 K)
  stream-to-stream  peak  46.6 MPa at x = 0.70 m, t = 720 s (Delta T = 19.4 K)
  ALPEMA limit as a stress: 72.1 MPa
  Yield strength 35 MPa is reached at Delta T = 14.6 K
```

Cold trip, 95 % loss of cold flow (N = 400)

```
ALPEMA check on the temperature differences:
  stream-to-metal   max  53.3 K   limit 30 K   margin -23.3 K   EXCEEDED
  stream-to-stream  max  62.2 K   limit 30 K   margin -32.2 K   EXCEEDED
Fully restrained thermal stress (Eq 53): 2.40 MPa per K of difference
  hot-to-metal      peak  28.4 MPa at x = 0.70 m, t = 85 s (Delta T = 11.8 K)
  metal-to-cold     peak 128.0 MPa at x = 0.70 m, t = 262 s (Delta T = 53.3 K)
  stream-to-stream  peak 149.4 MPa at x = 0.70 m, t = 262 s (Delta T = 62.2 K)
  ALPEMA limit as a stress: 72.1 MPa
  Yield strength 35 MPa is reached at Delta T = 14.6 K
```

Startup from rest, flows open over 20 s (N = 400)

```
ALPEMA check on the temperature differences:
  stream-to-metal   max  65.0 K   limit 30 K   margin -35.0 K   EXCEEDED
  stream-to-stream  max  65.0 K   limit 30 K   margin -35.0 K   EXCEEDED
Fully restrained thermal stress (Eq 53): 2.40 MPa per K of difference
  hot-to-metal      peak  38.1 MPa at x = 0.70 m, t = 17 s (Delta T = 15.9 K)
  metal-to-cold     peak 156.2 MPa at x = 0.70 m, t = 0 s (Delta T = 65.0 K)
  stream-to-stream  peak 156.2 MPa at x = 0.70 m, t = 0 s (Delta T = 65.0 K)
  ALPEMA limit as a stress: 72.1 MPa
  Yield strength 35 MPa is reached at Delta T = 14.6 K
```


| Output | N = 100 | N = 200 | N = 400 | $p_{obs}$ | extrapolated $y^\star$ |
|:--|--:|--:|--:|--:|--:|
| Feed ramp, +10 K at 2 K/min: peak stress, stream to metal [MPa] | 27.009 | 26.238 | 25.846 | 0.98 | 25.442 |
| Feed ramp, +10 K at 2 K/min: peak stress, stream to stream [MPa] | 47.350 | 46.869 | 46.628 | 1.00 | 46.385 |
| Feed ramp, +10 K at 2 K/min: peak stress, stream to metal, cells only [MPa] | 24.679 | 25.056 | 25.251 | 0.95 | 25.460 |
| Cold trip, 95 % loss of cold flow: peak stress, stream to metal [MPa] | 132.040 | 129.548 | 128.044 | 0.73 | 125.752 |
| Cold trip, 95 % loss of cold flow: peak stress, stream to stream [MPa] | 149.442 | 149.442 | 149.442 | n/a | n/a |
| Cold trip, 95 % loss of cold flow: peak stress, stream to metal, cells only [MPa] | 91.883 | 106.625 | 115.729 | 0.70 | 130.430 |
| Startup from rest, flows open over 20 s: peak stress, stream to metal [MPa] | 156.194 | 156.194 | 156.194 | n/a | 156.194 |
| Startup from rest, flows open over 20 s: peak stress, stream to stream [MPa] | 156.194 | 156.194 | 156.194 | n/a | 156.194 |
| Startup from rest, flows open over 20 s: peak stress, stream to metal, cells only [MPa] | 96.304 | 107.531 | 114.469 | 0.69 | 125.693 |


## ALPEMA summary


| Scenario | Quantity | Model | ALPEMA limit | Within limit? |
|:--|--:|--:|--:|--:|
| Steady base case | max stream-to-stream difference | 17.2 K | 30 °C | yes |
| Steady base case | max stream-to-metal difference | 9.5 K | 30 °C | yes |
| Feed ramp, 2 K/min | rate at the hot inlet nozzle | 2.0 K/min | 1 / 5 °C/min | no (frequent) / yes (startup/trip) |
| Feed ramp, 2 K/min | peak metal rate $R_w$ | 1.9 K/min | 1 / 5 °C/min | no (frequent) / yes (startup/trip) |
| Feed ramp, 2 K/min | peak outlet-nozzle rate $R_{end}$ | 1.7 K/min | 1 / 5 °C/min | no (frequent) / yes (startup/trip) |
| Feed ramp, 2 K/min | max stream-to-stream difference | 19.3 K | 30 °C | yes |
| Feed ramp, 2 K/min | max stream-to-metal difference | 10.6 K | 30 °C | yes |
| Feed ramp, 2 K/min | metal change in any 60 min | 9.4 K | 60 °C | yes |
| Cold trip | peak metal rate $R_w$ | 104.5 K/min | 5 °C/min | no |
| Cold trip | peak outlet-nozzle rate $R_{end}$ | 94.4 K/min | 5 °C/min | no |
| Cold trip | max cold-end nozzle difference | 62.2 K | 30 °C | no |
| Cold trip | max stream-to-metal difference | 52.3 K | 30 °C | no |
| Cold trip | metal change in any 60 min | 49.5 K | 60 °C | yes |
| Startup from rest | peak metal rate $R_w$ | 296.0 K/min | 5 °C/min | no |
| Startup from rest | peak outlet-nozzle rate $R_{end}$ | 168.5 K/min | 5 °C/min | no |
| Startup from rest | max cold-end nozzle difference | 65.0 K | 30 °C | no |
| Startup from rest | max stream-to-metal difference | 65.0 K | 30 °C | no |
| Startup from rest | metal change in any 60 min | 55.4 K | 60 °C | yes |

