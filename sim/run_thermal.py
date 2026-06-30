"""
Drive the thermal model and answer the make-or-break questions:

  1. How long to through-heat a strip to forming temp (the DIFFUSION FLOOR --
     best case, surface clamped just below scorch)?
  2. How long to quench the core back below set temp -- air vs water?
  3. How does that scale with thickness (the killer is t ~ thickness^2)?
  4. What feed rate / throughput does that imply?

Run:  python run_thermal.py
Outputs: printed summary + PNGs in sim/out/.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from material import PMMA
import thermal as th

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)
m = PMMA

# Heat-zone geometry: how long (along the feed direction) the heated band is.
HEAT_ZONE_LEN_MM = 25.0   # length of strip exposed to the heater at once

THICKNESSES_MM = [2.0, 3.0, 4.5, 6.0]
SURFACE_HOLD_C = 175.0    # clamp surface just under the 180 C scorch ceiling


def dt_for(thk_m):
    # implicit solver is stable for any dt; pick dt for time resolution.
    return min(0.05, (thk_m**2 / m.alpha) / 2000)


def heat_then_quench(thk_mm, quench_surface, quench_label):
    """Full cycle for one thickness + quench method. Returns a dict of results."""
    thk = thk_mm / 1000.0
    dt = dt_for(thk)
    diff_time = thk**2 / m.alpha            # gross diffusion time scale

    # --- HEAT: both faces clamped near scorch ceiling (diffusion-floor case) ---
    hot = th.contact_shoe(SURFACE_HOLD_C)
    heat = th.solve(m, thk, hot, hot, T_init=20.0,
                    t_end=4 * diff_time, dt=dt, nodes=81)
    t_heat, scorched, peak_surf = th.time_to_through_heat(heat, m.T_form, m.T_scorch)
    idx = np.argmax(heat.T.min(axis=1) >= m.T_form)
    field_at_heat = heat.T[idx]             # carry this profile into the quench

    # --- QUENCH: from that profile, both faces to the coolant ---
    quench = th.solve(m, thk, quench_surface, quench_surface,
                      T_init=field_at_heat, t_end=6 * diff_time, dt=dt, nodes=81)
    t_quench = th.time_to_set(quench, m.T_set)

    # Biot number for the quench (lumped check)
    Bi = quench_surface.h_conv * (thk / 2) / m.k

    cycle = (t_heat or np.nan) + (t_quench or np.nan)
    feed_mm_s = HEAT_ZONE_LEN_MM / cycle if cycle == cycle else np.nan

    return dict(thk_mm=thk_mm, diff_time=diff_time, t_heat=t_heat,
                scorched=scorched, peak_surf=peak_surf, t_quench=t_quench,
                Bi=Bi, cycle=cycle, feed_mm_s=feed_mm_s,
                quench_label=quench_label,
                heat=heat, quench=quench, idx_heat=idx)


def main():
    print("=" * 72)
    print(f"THERMAL FEASIBILITY  --  {m.name}")
    print(f"  alpha = {m.alpha:.3e} m^2/s   (steel ~ {1.2e-5/m.alpha:.0f}x faster)")
    print(f"  forming {m.T_form:.0f}C  set<{m.T_set:.0f}C  scorch>{m.T_scorch:.0f}C")
    print(f"  surface held at {SURFACE_HOLD_C:.0f}C during heat (diffusion floor)")
    print(f"  heat-zone length {HEAT_ZONE_LEN_MM:.0f} mm")
    print("=" * 72)

    methods = {
        "air":   th.air_quench(air_C=20.0, h=120.0),
        "chilled_air": th.air_quench(air_C=5.0, h=180.0),
        "water_mist": th.mist_or_water_quench(water_C=18.0, h=1500.0),
    }

    # Run full sweep for the air-quench baseline (for the thickness chart),
    # plus all three methods at the reference 3 mm strip.
    rows = []
    for thk in THICKNESSES_MM:
        r = heat_then_quench(thk, methods["air"], "air")
        rows.append(r)

    print("\n  THICKNESS SWEEP  (air quench, h=120):")
    print(f"  {'thk':>5} {'heat s':>8} {'quench s':>9} {'cycle s':>8} "
          f"{'feed mm/s':>9} {'scorch?':>8}")
    for r in rows:
        print(f"  {r['thk_mm']:>4.1f}m {fmt(r['t_heat']):>8} "
              f"{fmt(r['t_quench']):>9} {fmt(r['cycle']):>8} "
              f"{r['feed_mm_s']:>9.3f} {'YES' if r['scorched'] else 'no':>8}")

    print("\n  QUENCH METHOD COMPARISON  (3.0 mm strip):")
    print(f"  {'method':>12} {'Bi':>6} {'quench s':>9} {'cycle s':>8} {'feed mm/s':>9}")
    ref = {}
    for name, surf in methods.items():
        r = heat_then_quench(3.0, surf, name)
        ref[name] = r
        print(f"  {name:>12} {r['Bi']:>6.1f} {fmt(r['t_quench']):>9} "
              f"{fmt(r['cycle']):>8} {r['feed_mm_s']:>9.3f}")

    # ---- plots ----
    plot_cycle(ref["air"], OUT / "cycle_3mm_air.png")
    plot_thickness(rows, OUT / "thickness_sweep.png")
    plot_quench_methods(ref, OUT / "quench_methods_3mm.png")

    # ---- verdict numbers ----
    r3 = rows[1]
    print("\n  --> 3 mm strip, air quench: heat {:.0f}s + quench {:.0f}s "
          "= {:.0f}s/zone, feed ~{:.2f} mm/s"
          .format(r3['t_heat'], r3['t_quench'], r3['cycle'], r3['feed_mm_s']))
    print("  --> diffusion floor scales as thickness^2: doubling thickness ~4x slower")
    print(f"\n  plots written to {OUT}/")


def fmt(v):
    return "n/a" if v is None or v != v else f"{v:.1f}"


def plot_cycle(r, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    h, q = r["heat"], r["quench"]
    idx = r["idx_heat"]
    th_mm = r["thk_mm"]
    # heat phase: core & surface
    ax[0].plot(h.t[:idx + 1], h.T.min(axis=1)[:idx + 1], label="core (coldest)")
    ax[0].plot(h.t[:idx + 1], h.T.max(axis=1)[:idx + 1], label="surface")
    ax[0].axhline(m.T_form, ls="--", c="r", lw=0.8, label=f"form {m.T_form:.0f}C")
    ax[0].axhline(m.T_scorch, ls=":", c="k", lw=0.8, label=f"scorch {m.T_scorch:.0f}C")
    ax[0].set(title=f"HEAT ({th_mm:.0f} mm)", xlabel="time (s)", ylabel="T (C)")
    ax[0].legend(fontsize=7); ax[0].grid(alpha=0.3)
    # quench phase
    tq = th.time_to_set(q, m.T_set)
    ax[1].plot(q.t, q.T.max(axis=1), label="core (hottest)")
    ax[1].plot(q.t, q.T.min(axis=1), label="surface")
    ax[1].axhline(m.T_set, ls="--", c="b", lw=0.8, label=f"set {m.T_set:.0f}C")
    if tq: ax[1].axvline(tq, c="g", lw=0.8)
    ax[1].set(title=f"QUENCH (air) {th_mm:.0f} mm", xlabel="time (s)", ylabel="T (C)")
    ax[1].legend(fontsize=7); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def plot_thickness(rows, path):
    thk = [r["thk_mm"] for r in rows]
    heat = [r["t_heat"] for r in rows]
    quench = [r["t_quench"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.bar(thk, heat, width=0.5, label="heat", color="#d9772b")
    ax.bar(thk, quench, width=0.5, bottom=heat, label="quench (air)", color="#3b78b0")
    # ideal thickness^2 reference through the 3 mm point
    ref = rows[1]["cycle"] / rows[1]["thk_mm"]**2
    xs = np.linspace(min(thk), max(thk), 50)
    ax.plot(xs, ref * xs**2, "k--", lw=0.8, label="~ thickness^2")
    ax.set(xlabel="strip thickness (mm)", ylabel="time per zone (s)",
           title="Cycle time vs thickness (diffusion-limited)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def plot_quench_methods(ref, path):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for name, r in ref.items():
        q = r["quench"]
        ax.plot(q.t, q.T.max(axis=1), label=f"{name} (Bi={r['Bi']:.1f})")
    ax.axhline(m.T_set, ls="--", c="k", lw=0.8, label=f"set {m.T_set:.0f}C")
    ax.set(xlabel="time (s)", ylabel="core temp (C)",
           title="Quench: core cooldown by method (3 mm)", xlim=(0, 80))
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


if __name__ == "__main__":
    main()
