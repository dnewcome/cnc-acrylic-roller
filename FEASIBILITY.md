# Acrylic heat-bend-quench — feasibility verdict (v1)

_First-slice physics study. Reference strip: cast PMMA, 25 mm wide × 3 mm thick.
All numbers from the parametric models in `sim/` — re-run with `make` (see
`sim/README.md`). Treat them as **order-of-magnitude engineering estimates**;
the springback and crazing models need real material data to firm up._

## Verdict

**Feasible — for slow, one-off, thin-strip sculpture — and the thing that makes
or breaks it is THERMAL THROUGHPUT, not control and not mechanics.**

Your instinct ("heat, roll, cool to set, maybe water") is right in shape, with
three corrections the model is fairly confident about:

1. It will be **slow** (~0.5 mm/s of strip), because acrylic is a thermal
   insulator. For sculpture that's fine; for production it isn't.
2. **Water cooling is the risky choice, not the obvious one** — air quench is
   slower but much safer (see crazing, below). Start with strong chilled air.
3. The control loop is the *easy* part. The hard parts are the narrow
   heat-vs-scorch window and getting heat through the thickness at all.

## The three findings that matter

### 1. Mechanically trivial — forces are a rounding error
At forming temperature the strip is soft (modulus drops from ~3 GPa cold to
~6 MPa at 155 °C). Bending a 3 mm strip to a 100 mm radius needs a roller force
of **~0.16 N**. Motors, frame, bearings: non-issues. Don't over-build the
mechanism. _(`sim/bending.py`)_

### 2. Control-loop *speed* is a non-issue — physics is the bottleneck
The strip is a slow thermal plant (time constant **τ ≈ 21 s**). A PID sampling
at **0.05 Hz** rejects a draft just as well as one at 10 Hz (identical ~9 °C dip).
Any microcontroller is wild overkill. _(`sim/control.py`)_

The *real* control difficulties are physical:
- **Narrow window:** forming 155 °C ↔ scorch 180 °C = only **25 °C** to play in.
- **Soak drift:** holding the surface near the ceiling, it *drifts upward* as the
  core soaks heat (less conducts inward) — a simple PID needs **≥ 20 °C of margin**
  below scorch, or feedforward of the declining power demand.
- **You can't measure the core** directly; you infer it from surface history +
  the thermal model (an observer).

### 3. Form HOT or don't bother — it's what makes open-loop viable
Springback collapses as you form further above Tg (105 °C):

| Forming temp | springback (5 s dwell) | radius error from ±5 °C wobble |
|---|---|---|
| 125 °C | ~16 % | **±25 %**  ← open-loop hopeless |
| 140 °C | ~0 % | ±2 % |
| 155 °C | ~0 % | **~0 %**  ← open-loop fine |

Form near Tg and a few degrees of thermal noise wreck the radius. Form at
≥150–155 °C and the polymer fully stress-relaxes: springback nearly vanishes
*and* stops caring about temperature. This is the single most important design
rule. _(`sim/bending.py`, `out/temp_to_shape.png`)_

## The numbers (3 mm strip, 25 mm heat zone)

| quench method | heat | quench | cycle | feed rate | note |
|---|---|---|---|---|---|
| forced air (h≈120) | 22 s | 26 s | **48 s** | 0.52 mm/s | safest |
| chilled air (h≈180) | 22 s | 18 s | 40 s | 0.63 mm/s | good balance |
| water mist (h≈1500) | 22 s | 10 s | 32 s | 0.78 mm/s | **craze risk** |

Throughput scales as **thickness²** (diffusion): a 6 mm strip is ~150 s/zone.
→ **stay at 2–3 mm.** A 600 mm strip at ~0.5 mm/s ≈ **20 minutes** — acceptable
for sculpture, not for volume. _(`sim/run_thermal.py`, `out/thickness_sweep.png`)_

## The continuous, feed-matched architecture (pre-heat → bend → set)

Treating it as a **continuous flow** of three stations — a long gentle pre-heat
tunnel (20 → ~95 °C, *just under Tg* so the strip stays stiff and handleable), a
short intense bend-heat at the forming roller (95 → 155 °C), and a set/quench
(chilled contact roller, mist, or air) — beats batch indexing, because the
stations **pipeline**: heating one section while quenching another. That roughly
**doubles** the effective feed vs the single-zone cycle above.
_(`sim/stations.py`)_

Two findings worth pinning:

- **Pre-heating to just under Tg is a real win.** At the same scorch-safe surface
  temp (170 °C), pre-warming the core cuts the bend-heat dwell **~27%** (3 mm:
  24 → 18 s) and shrinks the surface-core gradient (more scorch margin). Diffusion
  *time* is geometry-set, so you can't speed it up — but you can arrive at the bend
  with the core already most of the way there. _(`out/preheat_payoff.png`)_

- **The bottleneck is the floppy zone, not the cooling.** With the fully-soft
  (bend-heat) zone capped at 20 mm for handling, *that* station caps feed — the
  quench isn't the limiter. Feed is a **handling ↔ speed trade**: a longer,
  better-supported hot zone (or thinner stock) goes faster. The quench choice then
  only sets how much *set-zone length* you need (mist ~11 mm, chilled roller
  ~30 mm, plain air ~30–50 mm — air can run out of roller-wrap on thin fast strips).

Resulting feed envelope (pre-heated, 20 mm floppy limit, 120° roller wrap):

| thickness | max feed | bend dwell | note |
|---|---|---|---|
| 2 mm | ~2.4 mm/s | 8 s | plain air *just* fits the roller wrap |
| 3 mm | ~1.1 mm/s | 18 s | comfortable for any quench |
| 4 mm | ~0.65 mm/s | 31 s | getting slow; thin is better |

_(`sim/stations.py`, `out/feed_vs_thickness.png`)_

## Dynamic co-sim (MuJoCo): sag limit + forming both check out

A thermo-mechanical co-simulation (`sim/strip_sim.py`) models the strip as a
serial chain of segments whose hinge stiffness is driven by `E(T)` and whose
rest angle follows a viscoelastic relaxation ODE — so gravity sag and
forming/springback **emerge** rather than being scripted. Two payoffs:

- **The floppy-zone limit is now physical, not guessed.** A free, fully-hot
  (155 °C) cantilever band — the worst case (the formed part hanging off the
  bend) — sags **0.5 mm at ~26 mm** band length, **2 mm at ~47 mm**. This
  confirms the 20 mm handling limit used in the station model was sound (mildly
  conservative), so the feed envelope above holds. The all-cold case validates
  against Euler-Bernoulli self-weight sag to within ~8% (pure discretization,
  converges up with segment count). _(`out/sag_vs_band.png`, `out/strip_sag.gif`)_

- **Forming + springback reproduce `bending.py` independently.** The co-sim's
  heat→bend→cool→release cycle yields ~0% springback at 140–155 °C and large
  springback near Tg — matching the analytic model to <1% in the hot regime
  (two independent implementations agreeing). _(`out/forming_springback.png`,
  `out/strip_forming.gif`)_

- **Full-machine animation** (`sim/machine_sim.py`): the strip rides a feed
  carriage through fixed pre-heat → bend-roller → chilled-set stations; segments
  are tinted by temperature and a programmable curvature `kappa(t)` makes a
  varying-radius curve exit the machine. The temperature trace along the strip
  shows the clean heat→peak→quench cycle, and the centerline shows the curve
  forming right at the bend and holding as it cools. _(`out/machine.gif`,
  `out/machine_curve.png`)_

Run it live: `make interactive` (full machine — drag `ui_feed` / `ui_kappa` to
feed and draw a curve), or `make bench` (single-bend forming: heat, bend, cool,
release, and watch springback). `make machine` / `make sim` write the gifs.

## Roller forces, materials, and "will it stick" (engineering-grade)

Three upgrades toward a real machine model (MuJoCo engineering-grade, literature
material data → trends solid, absolutes ±2×):

- **Heat/cool times are now correct for thickness** — a batched through-thickness
  conduction model (`sim/seg_thermal.py`) reproduces `thermal.py` to ~3–4% and
  lets forming be gated on the **core** temperature (feed too fast → cold core →
  won't form). Real coupling, not a lumped surrogate.

- **Real roller contact forces** (`sim/roller_contact.py`): a collidable roller
  pressed into the strip; the **measured** contact force matches beam theory at the
  cold end (~0.83×, the rest is contact compliance) and **collapses ~3 orders of
  magnitude with temperature** — ~0.84 N cold → sub-mN at forming temp. So roller /
  bearing / motor loads are set by the *cold* end of any accidental under-heating,
  not the hot design point. _(`out/roller_force.png`)_

- **Roller material + will-it-stick** (`sim/rollers.py`): friction (grip/slip),
  contact conductance (quench), and tack (adhesion) trade off. **Polished steel
  sticks to hot acrylic (150–165 °C) and can wrap the roller; PTFE/silicone
  release; cold acrylic never sticks.** Design rule that's robust to the ±2×
  numbers: **non-stick surface on the hot bend roller, chilled steel on the cold
  set roller** (cold acrylic doesn't tack, and steel quenches best). Friction
  matters at the cooler pinch/drive rollers (grip), not the near-zero-force hot
  bend. _(`out/stick_map.png`, `out/roller_tradeoff.png`)_

**Still approximate:** these are coupled into analyses, not yet into one unified
contact-driven machine sim (dynamic friction-feed grip/slip and in-sim adhesion
injection remain analytic). The continuous cable-rod representation was tested but
its built-in elasticity fought clean per-element stiffness control, so the
validated hinge-chain beam is used for force measurement.

## Where it breaks (the honest risk list)

- **Crazing from the quench** — *the biggest unmodeled risk.* Fast surface
  cooling (water, Bi≈12) freezes steep gradients → residual tensile stress →
  crazing, possibly delayed, and acrylic+water+stress is the classic craze
  recipe. The thermal model can't predict craze onset without material data.
  **This needs a physical coupon test before anything else.** Mitigation:
  chilled *air* (Bi≈1), gentler gradients.
- **Thick strips (>~4–5 mm):** diffusion time explodes; through-heating becomes
  impractical for a moving zone — those want oven/drape forming, not this machine.
- **3D twist (your actual goal):** not yet modeled. Adds a *second* springback
  mode, and a fully-soft hot zone will **sag/untwist under gravity** until quenched —
  the part likely needs support and the quench must complete *while still held*.
- **Curve resolution vs speed:** the length of strip that's hot at once sets the
  smallest curvature feature you can form. Fine detail ⇒ short hot zone ⇒ slow.
  Smooth flowing sculpture (large radii) relaxes this; sharp features fight it.

## Recommended process envelope (v1)

- Strip **2–3 mm** thick (thin wins, quadratically).
- **Two-sided radiant** heating, modulated to hold surface **≤ 160–165 °C**
  (keeps ≥15–20 °C scorch margin against soak-drift).
- Through-heat to **≥ 150 °C core**, ~20–25 s/zone.
- **Strong chilled-air quench** while the bend is held; water only if air can't
  set it fast enough, and then gently.
- **Feed ~0.5 mm/s.** Open-loop on temperature is fine *because* you form hot;
  add a surface pyrometer for the loop.

## Next slices (in priority order)

1. **Physical crazing/springback coupon test** — the one thing the model can't
   settle. Bend hot scrap at several temps + quench methods; look for craze and
   measure actual springback. This de-risks the whole project.
2. **3D twist model** — second springback mode + gravity sag of the hot zone.
3. **Toolpath/geometry sim** — target space-curve → feed + bend + twist + thermal
   schedule, with the springback map baked in.
4. Only then: **machine concept CAD.**
