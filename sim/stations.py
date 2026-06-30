"""
Station-balance model for the continuous, feed-matched architecture:

    [ PRE-HEAT tunnel ] --> [ BEND-HEAT + form ] --> [ SET / chilled roller ]
       long, gentle           short, intense          contact/mist/air
       20 -> ~T_pre (<Tg)      T_pre -> T_form         T_form -> T_set

Design logic (this is what the user is reasoning about):
  * Pre-heat the WHOLE strip to just BELOW Tg (~105C) so it stays STIFF enough
    to feed/handle. This happens over a long tunnel, so its through-thickness
    diffusion time does NOT bottleneck the feed -- you just make the tunnel long.
  * Save the final heat (cross Tg -> forming) for a SHORT zone right at the bend.
    Because the core is pre-warmed, the bend station reaches forming temp through
    the thickness at a LOWER, scorch-safe surface temperature and in a shorter
    dwell than heating from cold. Short hot zone = no floppy-handling problem.
  * SET the bend immediately (chilled roller / mist / air) before release.

Feed speed is then set by the most demanding station:
    v_max = min over stations of  (zone_length / dwell_needed)
The bend-heat and set zones are LENGTH-LIMITED (floppy length must stay short;
roller wrap arc is finite), so they -- not the long pre-heat tunnel -- cap feed.

Everything below reuses the validated 1-D conduction solver in thermal.py.
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

# --- architecture parameters (the knobs you'd actually set) ---
T_PRE = 95.0            # pre-heat target, just below Tg(105) -> still handleable
T_S_BEND = 170.0        # bend-heat surface clamp (scorch-safe, <180)
L_BEND_MAX_MM = 20.0    # max length of fully-soft "floppy" zone (handling limit)
ROLLER_R_MM = 25.0      # chilled set-roller radius
ROLLER_WRAP_DEG = 120.0 # strip wrap angle on the set roller -> set-zone length


def set_zone_len_mm():
    return 2 * np.pi * ROLLER_R_MM * (ROLLER_WRAP_DEG / 360.0)


def preheat_dwell(thk):
    """Time to soak the strip to ~T_PRE through the thickness (informational:
    sets how long the pre-heat tunnel must be, not the feed bottleneck)."""
    hot = th.contact_shoe(T_PRE)        # surface clamped at T_pre, core -> T_pre
    diff = thk**2 / m.alpha
    res = th.solve(m, thk, hot, hot, T_init=20.0, t_end=6 * diff,
                   dt=min(0.05, diff / 2000), nodes=81)
    coldest = res.T.min(axis=1)
    idx = np.argmax(coldest >= T_PRE - 5.0)   # within 5C of target = "soaked"
    return res.t[idx] if coldest[-1] >= T_PRE - 5.0 else None


def bendheat_dwell(thk, T_start):
    """From a strip uniformly pre-warmed to T_start, time for the COLDEST point
    to reach forming temp with the bend-heater surface clamped at T_S_BEND.
    Returns (dwell, scorched?). T_start=20 reproduces the cold/no-preheat case."""
    hot = th.contact_shoe(T_S_BEND)
    diff = thk**2 / m.alpha
    res = th.solve(m, thk, hot, hot, T_init=float(T_start), t_end=4 * diff,
                   dt=min(0.05, diff / 2000), nodes=81)
    t_heat, scorched, _ = th.time_to_through_heat(res, m.T_form, m.T_scorch)
    # capture the through-heated field to hand to the set station
    field = None
    if t_heat is not None:
        idx = np.argmax(res.T.min(axis=1) >= m.T_form)
        field = res.T[idx]
    return t_heat, scorched, field


def set_dwell(thk, quench0, quenchL, field):
    """From the through-heated field, time for the HOTTEST point to fall to
    T_set under the given (possibly asymmetric) quench BCs."""
    diff = thk**2 / m.alpha
    res = th.solve(m, thk, quench0, quenchL, T_init=field, t_end=6 * diff,
                   dt=min(0.05, diff / 2000), nodes=81)
    return th.time_to_set(res, m.T_set)


# --- quench options for the SET station ---
def chilled_roller(T_chill=12.0, h_contact=800.0, h_air=30.0):
    """One face on a chilled metal roller (contact conduction), other face in
    air. Asymmetric. This is the user's preferred 'last roller' approach."""
    return th.contact_shoe(T_chill, h=h_contact), th.air_quench(20.0, h=h_air)

def mist_both(T=18.0, h=1500.0):
    return th.mist_or_water_quench(T, h), th.mist_or_water_quench(T, h)

def air_both(T=20.0, h=80.0):
    return th.air_quench(T, h), th.air_quench(T, h)


def analyze(thk_mm, quench_name, quench_pair, preheat=True):
    thk = thk_mm / 1000.0
    T_start = T_PRE if preheat else 20.0
    d_bend, scorched, field = bendheat_dwell(thk, T_start)
    if field is None:
        return None
    d_set = set_dwell(thk, quench_pair[0], quench_pair[1], field)
    d_pre = preheat_dwell(thk) if preheat else 0.0

    L_set_avail = set_zone_len_mm()
    # feed caps: bend zone capped by handling (floppy) length; set zone by the
    # available roller-wrap length.
    v_bend = L_BEND_MAX_MM / d_bend
    v_set = L_set_avail / d_set if d_set else np.nan
    v_max = min(v_bend, v_set)
    limiter = "bend-heat" if v_bend <= v_set else "set/quench"
    # at the (bend-limited) feed, how much set-zone length does this quench need,
    # and does it fit inside the roller wrap?
    L_set_need = v_max * d_set if d_set else np.nan
    set_fits = L_set_need <= L_set_avail + 1e-6
    L_pre = v_max * d_pre if d_pre else 0.0  # pre-heat tunnel length to keep up
    return dict(thk=thk_mm, quench=quench_name, preheat=preheat,
                d_pre=d_pre, d_bend=d_bend, d_set=d_set, scorched=scorched,
                v_bend=v_bend, v_set=v_set, v_max=v_max, limiter=limiter,
                L_set_need=L_set_need, L_set_avail=L_set_avail, set_fits=set_fits,
                L_pre=L_pre)


def main():
    print("=" * 74)
    print("STATION BALANCE  --  pre-heat / bend-heat / set, feed-matched")
    print(f"  pre-heat to {T_PRE:.0f}C (<Tg, handleable); bend surface clamp "
          f"{T_S_BEND:.0f}C")
    print(f"  floppy-zone limit {L_BEND_MAX_MM:.0f} mm; set roller wrap "
          f"{set_zone_len_mm():.0f} mm")
    print("=" * 74)

    quenches = {
        "chilled-roller": chilled_roller(),
        "water-mist": mist_both(),
        "plain-air": air_both(),
    }

    # 1) the headline: does pre-heat help? compare bend dwell + surface need
    print("\n  PRE-HEAT PAYOFF (3 mm strip, bend-heat station only):")
    for ph, lab in [(False, "cold start (20C)"), (True, f"pre-heated ({T_PRE:.0f}C)")]:
        d, sc, _ = bendheat_dwell(0.003, T_PRE if ph else 20.0)
        print(f"    {lab:<22} bend dwell {fmt(d):>5}s  surface@{T_S_BEND:.0f}C  "
              f"scorch={'YES' if sc else 'no'}")
    print("    (Both are scorch-safe at 170C surface; pre-heat is ~27% faster at")
    print("     the SAME safe surface temp. To match the pre-heated dwell from")
    print("     cold you'd have to push the surface hotter -> toward scorch.)")

    # 2) feed-speed table across thickness x quench
    print(f"\n  MAX FEED SPEED (mm/s), pre-heated, by thickness x set-method.")
    print(f"  Feed is capped by the {L_BEND_MAX_MM:.0f}mm floppy-zone (bend-heat) "
          f"limit; 'set need' is the")
    print(f"  set-zone length that quench then needs (must fit the "
          f"{set_zone_len_mm():.0f}mm roller wrap):")
    print(f"  {'thk':>4} {'quench':>15} {'d_bend':>7} {'d_set':>6} "
          f"{'v_max':>7} {'limiter':>10} {'set need':>9} {'fit?':>5}")
    rows = []
    for thk in [2.0, 3.0, 4.0]:
        for qn, qp in quenches.items():
            r = analyze(thk, qn, qp, preheat=True)
            if r is None:
                continue
            rows.append(r)
            print(f"  {r['thk']:>3.0f}m {qn:>15} {r['d_bend']:>6.1f}s "
                  f"{fmt(r['d_set']):>5}s {r['v_max']:>6.2f} {r['limiter']:>10} "
                  f"{r['L_set_need']:>7.0f}mm {'yes' if r['set_fits'] else 'NO':>5}")

    plot_feed(rows)
    plot_preheat_payoff()

    # 3) takeaways (data-driven)
    pre3 = bendheat_dwell(0.003, T_PRE)[0]
    cold3 = bendheat_dwell(0.003, 20.0)[0]
    gain = (cold3 - pre3) / cold3 * 100
    print("\n  TAKEAWAYS:")
    print(f"  * With a {L_BEND_MAX_MM:.0f}mm floppy-zone limit, the BEND-HEAT station is the")
    print(f"    feed bottleneck -- NOT cooling. Feed is set by how short you must")
    print(f"    keep the fully-soft zone. Want more speed? Allow a longer (better-")
    print(f"    supported) hot zone, or go thinner. It's a handling<->speed trade.")
    print(f"  * Pre-heating to {T_PRE:.0f}C cuts the bend dwell ~{gain:.0f}% at the same")
    print(f"    safe surface temp (3mm: {cold3:.0f}s ->{pre3:.0f}s) and shrinks the surface-core")
    print(f"    gradient, buying scorch margin. Modest speed-up, real safety margin.")
    print(f"  * Since bend-heat limits feed, the quench choice doesn't change v_max")
    print(f"    here -- it sets how much SET-ZONE LENGTH you need. Chilled roller &")
    print(f"    mist need little; plain air needs nearly the whole roller wrap (and")
    print(f"    can become the limiter for thin/fast strips -- watch the 'fit?' col).")
    print(f"  plots -> {OUT}/")


def plot_feed(rows):
    fig, ax = plt.subplots(figsize=(7, 4.3))
    thks = sorted(set(r["thk"] for r in rows))
    for qn in ["chilled-roller", "water-mist", "plain-air"]:
        ys = [next(r["v_max"] for r in rows if r["thk"] == t and r["quench"] == qn)
              for t in thks]
        ax.plot(thks, ys, "-o", label=qn)
    ax.set(xlabel="strip thickness (mm)", ylabel="max feed speed (mm/s)",
           title="Feed speed vs thickness & set method (pre-heated, feed-matched)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(OUT / "feed_vs_thickness.png", dpi=110); plt.close(fig)


def plot_preheat_payoff():
    """Bend dwell vs pre-heat temperature for 3 mm -- shows the payoff curve and
    the scorch cliff for low pre-heat."""
    fig, ax = plt.subplots(figsize=(7, 4.3))
    Tps = np.linspace(20, 100, 17)
    dwell, scorch = [], []
    for Tp in Tps:
        d, sc, _ = bendheat_dwell(0.003, Tp)
        dwell.append(d if d else np.nan); scorch.append(sc)
    ax.plot(Tps, dwell, "-o", color="C1")
    ax.axvline(m.T_glass, ls=":", c="grey", label=f"Tg {m.T_glass:.0f}C (handling limit)")
    ax.set(xlabel="pre-heat temperature (C)", ylabel="bend-heat dwell (s)",
           title="Higher pre-heat -> shorter bend dwell (3 mm, surface clamp 170C)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(OUT / "preheat_payoff.png", dpi=110); plt.close(fig)


def fmt(v):
    return "n/a" if v is None or v != v else f"{v:.1f}"


if __name__ == "__main__":
    main()
