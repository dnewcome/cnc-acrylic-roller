"""
Bend mechanics: min radius, forming force, springback, and -- the punchline --
how temperature-control accuracy turns into curve-SHAPE accuracy.

Why this matters for an OPEN-LOOP machine: you command a die/roller curvature
and the strip springs back when released. If springback is large and very
temperature-sensitive, you can't hit a target radius without closed-loop shape
feedback. If you form HOT enough that the polymer fully stress-relaxes,
springback collapses toward zero and becomes insensitive to temperature --
which is exactly the regime that makes open-loop control viable.

Models (first-order engineering approximations; real PMMA wants measured
WLF / stress-relaxation data, flagged as the key validation step):

  Modulus E(T):  logistic drop from glassy ~3 GPa to rubbery ~5 MPa across Tg.
  Relaxation time tau_relax(T): Arrhenius-like collapse above Tg
        tau_relax = tau_ref * exp(-(T - Tg)/D)
     -> below Tg the chains are frozen (huge tau, no relaxation),
        well above Tg they flow in well under a second.
  Springback fraction f_sb = exp(-t_hold / tau_relax(T_form))
     = the un-relaxed (still-elastic) fraction of the imposed bend that
       recovers on release.  Final curvature = die curvature * (1 - f_sb).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from material import PMMA

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)
m = PMMA

# Mechanical constants
E_GLASS = 3.0e9          # Pa, glassy modulus (room temp)
E_RUBBER = 5.0e6         # Pa, rubbery plateau modulus (well above Tg)
E_WIDTH = 6.0            # deg C, width of the glass-transition softening

TAU_REF = 20.0          # s, stress-relaxation time AT Tg
TAU_D = 10.0            # deg C, decade-ish collapse scale above Tg

EPS_CRAZE_COLD = 0.010  # ~1% surface strain crazes cold acrylic
EPS_MAX_HOT = 0.40      # rubbery acrylic tolerates large strain (thinning limit)


def E_of_T(T):
    """Young's modulus vs temperature (Pa)."""
    return E_RUBBER + (E_GLASS - E_RUBBER) / (1.0 + np.exp((T - m.T_glass) / E_WIDTH))


def tau_relax(T):
    """Stress-relaxation time (s). Frozen below Tg, fast above."""
    T = np.asarray(T, dtype=float)
    return np.where(T < m.T_glass,
                    1e9,                                   # effectively frozen
                    TAU_REF * np.exp(-(T - m.T_glass) / TAU_D))


def surface_strain(thk, R):
    """Peak fibre strain bending thickness `thk` to neutral radius R."""
    return (thk / 2.0) / R


def min_radius(thk, eps_allow):
    return (thk / 2.0) / eps_allow


def forming_moment(T, width, thk, R):
    """Bending moment to hold curvature 1/R at temperature T (N*m)."""
    I = width * thk**3 / 12.0
    return E_of_T(T) * I / R


def springback_fraction(T_form, t_hold):
    """Fraction of imposed curvature recovered on release."""
    return np.exp(-t_hold / tau_relax(T_form))


def main():
    print("=" * 72)
    print("BEND MECHANICS  --  springback, force, and temp->shape accuracy")
    print("=" * 72)

    w = 0.025          # 25 mm strip width
    thks = [0.002, 0.003, 0.0045, 0.006]

    # --- min radius: cold vs hot ---
    print(f"\n  MIN BEND RADIUS (before crazing/thinning):")
    print(f"  {'thk mm':>7} {'cold (1% strain)':>18} {'hot (40% strain)':>18}")
    for t in thks:
        rc = min_radius(t, EPS_CRAZE_COLD) * 1000
        rh = min_radius(t, EPS_MAX_HOT) * 1000
        print(f"  {t*1000:>7.1f} {rc:>15.0f} mm {rh:>15.1f} mm")
    print("  -> cold bending needs gentle radii (50x thickness); HOT bending")
    print("     reaches a few x thickness. Tight sculptural curves REQUIRE heat.")

    # --- forming force at temperature (is the mechanism hard?) ---
    R = 0.10           # 100 mm target radius
    t = 0.003
    print(f"\n  FORMING MOMENT for 3mm strip to R=100mm:")
    for T in [20, 120, 155]:
        M = forming_moment(T, w, t, R)
        F = M / 0.02                       # roller at 20 mm lever
        print(f"    at {T:>3}C:  E={E_of_T(T)/1e6:8.1f} MPa  "
              f"M={M*1e3:7.2f} mN*m  roller force ~{F:6.2f} N")
    print("  -> at forming temp the strip is soft: forces are a few N. The")
    print("     mechanism/motors are trivial. The hard part is thermal, not force.")

    # --- springback vs forming temperature ---
    print(f"\n  SPRINGBACK vs forming temperature (dwell = 5 s):")
    print(f"  {'T_form':>7} {'tau_relax':>10} {'springback':>11} {'overbend':>9}")
    for T in [115, 125, 140, 155, 165]:
        f = springback_fraction(T, 5.0)
        ob = 1.0 / (1.0 - f) if f < 0.999 else float('inf')
        print(f"  {T:>6}C {tau_relax(T):>9.2f}s {f*100:>9.1f}% "
              f"{ob:>8.2f}x")
    print("  -> form COLD-ish (near Tg) = big, twitchy springback; form HOT")
    print("     (>=155C) = springback collapses to ~0 and stops caring about T.")

    # --- THE punchline: temp accuracy -> radius accuracy ---
    print(f"\n  TEMP-CONTROL ACCURACY -> RADIUS ACCURACY (target R=100mm):")
    print(f"  {'form @':>8} {'+/-5C ->':>10} {'radius error':>13}")
    for T_nom in [125.0, 155.0]:
        err = radius_error_for_dT(T_nom, dT=5.0, t_hold=5.0, R_target=0.10)
        print(f"  {T_nom:>6.0f}C  {'+/-5C':>10} {err*100:>11.1f} %")
    print("  -> at 125C a +/-5C thermal wobble = big radius error (open-loop")
    print("     hopeless); at 155C the SAME wobble barely moves the radius.")
    print("     CONCLUSION: forming hot is what makes open-loop control feasible.")

    make_plots(w)
    print(f"\n  plots -> {OUT}/")


def radius_error_for_dT(T_nom, dT, t_hold, R_target):
    """If you calibrate overbend at T_nom but the real temp is T_nom +/- dT,
    what fractional radius error results? Returns max |dR/R|."""
    f_nom = springback_fraction(T_nom, t_hold)
    kappa_die = (1.0 / R_target) / (1.0 - f_nom)      # overbend to hit target
    errs = []
    for T in (T_nom - dT, T_nom + dT):
        f = springback_fraction(T, t_hold)
        kappa_final = kappa_die * (1.0 - f)
        R_final = 1.0 / kappa_final
        errs.append(abs(R_final - R_target) / R_target)
    return max(errs)


def make_plots(w):
    T = np.linspace(20, 180, 400)
    # (1) modulus + relaxation time
    fig, ax1 = plt.subplots(figsize=(6.8, 4.3))
    ax1.semilogy(T, E_of_T(T) / 1e6, 'C0', label="modulus E (MPa)")
    ax1.set(xlabel="temperature (C)", ylabel="E (MPa)")
    ax1.axvline(m.T_glass, ls=":", c="grey"); ax1.axvline(m.T_form, ls="--", c="g")
    ax2 = ax1.twinx()
    ax2.semilogy(T, tau_relax(T), 'C3', label="relaxation time (s)")
    ax2.set_ylabel("tau_relax (s)")
    ax1.set_title("Soft & fast-relaxing above Tg (105C) -> form here")
    ax1.legend(loc="upper right", fontsize=8); ax2.legend(loc="lower left", fontsize=8)
    ax1.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(OUT / "modulus_relaxation.png", dpi=110); plt.close(fig)

    # (2) springback vs forming temp, several dwells
    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    Tf = np.linspace(110, 170, 300)
    for th in [1.0, 3.0, 10.0]:
        ax.plot(Tf, springback_fraction(Tf, th) * 100, label=f"dwell {th:.0f}s")
    ax.axvline(m.T_form, ls="--", c="g", label=f"form {m.T_form:.0f}C")
    ax.set(xlabel="forming temperature (C)", ylabel="springback (%)",
           title="Springback collapses as you form further above Tg")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(OUT / "springback.png", dpi=110); plt.close(fig)

    # (3) radius error vs temperature error -- the open-loop viability chart
    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    dTs = np.linspace(0, 12, 50)
    for T_nom, c in [(125.0, 'C3'), (140.0, 'C1'), (155.0, 'C2')]:
        errs = [radius_error_for_dT(T_nom, d, 5.0, 0.10) * 100 for d in dTs]
        ax.plot(dTs, errs, c, label=f"form @ {T_nom:.0f}C")
    ax.set(xlabel="temperature error +/- (C)", ylabel="resulting radius error (%)",
           title="Temp accuracy -> shape accuracy (why hot forming wins)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(OUT / "temp_to_shape.png", dpi=110); plt.close(fig)


if __name__ == "__main__":
    main()
