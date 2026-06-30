# sim/ — acrylic heat-bend-quench feasibility models

Physics-first feasibility study for the CNC acrylic bender. Answers: *can a
moving heat → bend → quench zone form arbitrary curves in acrylic strip, under
computer control?* Verdict and synthesis live in `../FEASIBILITY.md`.

## Run

```bash
make            # runs all three models, writes plots to out/
# or individually:
python thermal.py      # material properties sanity check (alpha)
python run_thermal.py  # heat-up / quench times, thickness sweep, feed rate
python control.py      # PID loop: loop-speed test + narrow-window challenge
python bending.py      # min radius, forming force, springback, temp->shape
python stations.py     # continuous pre-heat/bend/set balance -> feed vs thickness
python strip_sim.py    # MuJoCo co-sim: hot-zone sag sweep + forming/springback
python machine_sim.py  # full-machine animation (feed/heat/bend/quench) -> machine.gif
python machine_interactive.py            # LIVE full machine (feed + curvature sliders)
python machine_interactive.py --selftest # headless check
python strip_interactive.py              # LIVE single-bend forming bench
```

From the repo root:
- `make interactive` — **live full machine** (feed + curvature sliders)
- `make machine` — batch machine animation → `out/machine.gif`
- `make bench` — live single-bend forming bench
- `make sim` — batch sag + forming benches
- `make models` — analytic feasibility models · `make selftest` — headless viewer checks

Requires `numpy`, `scipy`, `matplotlib`.

## Files

| file | what it models |
|---|---|
| `material.py` | Cast PMMA properties (thermal + process temps). One place to edit material. |
| `thermal.py` | 1-D transient conduction solver (implicit FD, Robin BCs) + BC presets. The engine. |
| `run_thermal.py` | Drives the thermal engine: heat-up, quench, thickness sweep, feed rate. |
| `control.py` | Heater→temperature PID. Plant ID (FOPDT), loop-rate sweep, scorch-margin/soak-drift. |
| `bending.py` | Bend mechanics: E(T), stress relaxation, springback, **temp→shape accuracy**. |
| `stations.py` | Continuous pre-heat→bend→set balance; **feed vs thickness**, the bottleneck station, pre-heat payoff. Reuses `thermal.py`. |
| `strip_sim.py` | **MuJoCo thermo-mechanical co-sim.** Strip = serial chain of box segments + hinges; temperature (reduced model, calibrated to `thermal.py`) drives each hinge's stiffness `E(T)·I/seg` and a viscoelastic rest-angle relaxation → hot-zone gravity sag + forming/springback emerge. Validates vs Euler-Bernoulli and `bending.py`. |
| `strip_interactive.py` | Live `glfw` viewer of the co-sim with `ui_heat`/`ui_bend` sliders; forces a windowed GL backend before importing `strip_sim`. `--selftest` runs the physics headless. |
| `machine_sim.py` | **Full-machine animation.** Strip on a FEED carriage driven through fixed stations (pre-heat zone → bend rollers + former → chilled set roller); temperature follows position-based zones, curvature `kappa(t)` is programmable → a varying-radius curve exits. Segments tinted by temperature. Visual rollers/zone markers (forming = the validated former+relaxation, not roller contact). Times compressed for watchability. |
| `machine_interactive.py` | Live `glfw` viewer of the full machine with `ui_feed`/`ui_kappa` sliders; strip recycles continuously. `--selftest` headless. |
| `seg_thermal.py` | **Real through-thickness conduction, batched over segments.** Replaces the lumped thermal surrogate; gives correct heat/cool times for thickness (validates vs `thermal.py` to ~3–4%) so forming can be gated on the **core** temperature. |
| `rollers.py` | **Roller material model** (friction μ, contact conductance, tack) + grip/slip + **will-it-stick** maps. Conclusion: non-stick on the hot bend roller, chilled steel on the cold set roller. |
| `roller_contact.py` | **REAL measured roller contact force** (MuJoCo contact, not commanded) validated vs beam theory and shown collapsing with temperature; friction drag by material. |

## Key modelling assumptions (where the bodies are buried)

- Temperature-independent thermal properties (cp/k vary mildly with T; cp rises
  through Tg — we use representative averages). Fine for order-of-magnitude.
- Springback uses a **first-order viscoelastic relaxation** model with assumed
  `tau_ref`/`D` — directionally right, but the absolute springback % needs real
  PMMA stress-relaxation data. Idealizes to ~0 above ~140 °C; reality will show a
  few %, but small and (the load-bearing claim) temperature-insensitive.
- Crazing is **not** modelled — it's a residual-stress/solvent failure the
  thermal model can't predict. Treated as a risk requiring a physical coupon test.
- The MuJoCo co-sim uses a **lumped beam** (rigid links + torsional hinges): it
  reproduces continuum sag to ~0.93 at N=40 (discretization, converges up with N),
  and a **reduced lumped-temperature** model per segment (a 1st-order lag, not the
  full through-thickness PDE) calibrated to `thermal.py`'s dwell numbers. Right
  fidelity for mechanism dynamics; not a substitute for the 1-D thermal solver.

Edit constants at the top of each module (or `material.py`) to re-scope.
