"""rollers.py -- roller MATERIAL model + grip/slip + will-it-stick analysis.

Three things a roller material decides, and they trade off against each other:

  1. FRICTION (mu)            -> feed grip vs slip at the drive/pinch rollers
  2. CONTACT CONDUCTANCE (h)  -> how well a chilled roller pulls heat out
  3. TACK / ADHESION          -> whether HOT acrylic sticks to and wraps the roller

The cruel triangle: polished steel grips well and quenches well but is TACKY to
hot acrylic; PTFE is non-stick but slippery (poor grip) and insulating (poor
quench); silicone grips well and is fairly non-stick but insulates. So the
likely answer is material-by-station: non-stick on the HOT bend roller, chilled
steel on the COLD set roller (cold acrylic doesn't tack).

All numbers are LITERATURE ESTIMATES (uncalibrated, ~+/-2x). The rigorous
version of "will it stick" is a coupled thermo-visco-adhesive peel problem
(FEM / coupon test); this is a defensible engineering force-balance to rank
materials and find where the margin is thin. Flagged accordingly.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import dataclass
from pathlib import Path

from material import PMMA
import bending

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)
m = PMMA


@dataclass(frozen=True)
class RollerMaterial:
    name: str
    mu: float            # static friction coeff vs acrylic
    h_contact: float     # W/(m^2 K) contact conductance to the strip (for quench)
    tack0: float         # Pa, tack stress to MELT/rubbery acrylic at full tack
    note: str


MATERIALS = {
    # mu (vs PMMA), contact conductance, tack stress to hot PMMA
    "polished_steel": RollerMaterial("polished_steel", 0.40, 1200.0, 6.0e4,
        "grips + quenches well, but TACKY to hot acrylic"),
    "ptfe_coated":    RollerMaterial("ptfe_coated",    0.07,  150.0, 1.5e3,
        "non-stick, but slippery (poor grip) + insulating (poor quench)"),
    "silicone":       RollerMaterial("silicone",       0.90,  220.0, 8.0e3,
        "great grip, fairly non-stick, but insulating"),
}


def tack_factor(T):
    """Fraction of full tack vs temperature: ~0 below Tg (glassy, not sticky),
    rising to 1 by forming temp (rubbery/tacky). Acrylic only sticks when hot."""
    return float(np.clip((T - m.T_glass) / (m.T_form - m.T_glass), 0.0, 1.0))


def hertz_halfwidth(F_per_len, R, E_strip, nu=0.35):
    """Hertzian line-contact half-width (m) of a cylinder (radius R) pressed onto
    the strip with load F per unit length. Softer (hot) strip -> wider patch."""
    Estar = E_strip / (1 - nu**2)            # roller assumed rigid vs hot strip
    if F_per_len <= 0:
        return 0.0
    return np.sqrt(4 * F_per_len * R / (np.pi * Estar))


# ---------- grip / slip at the drive (pinch) rollers ----------

def bend_force(T, width, thk, R_bend):
    """Force the bend roller must exert to curve the strip to radius R_bend at
    temperature T (N). M = E(T)*I*kappa, acting on a ~roller-radius lever."""
    I = width * thk**3 / 12.0
    M = bending.E_of_T(T) * I / R_bend
    return M / R_bend                         # force ~ moment / radius arm


def grip_analysis(clamp_N=20.0, width=0.025, thk=0.003, R_bend=0.05,
                  T_bend=155.0, feed_drag=0.5):
    """Can the pinch rollers feed the strip without slipping? Needed feed force =
    bend force + drag; capacity = mu * clamp * (2 contacts)."""
    F_need = bend_force(T_bend, width, thk, R_bend) + feed_drag
    rows = []
    for mat in MATERIALS.values():
        cap = mat.mu * clamp_N * 2
        rows.append((mat.name, mat.mu, F_need, cap, cap > F_need))
    return F_need, rows


# ---------- will it stick? force balance at the bend roller ----------

def stick_analysis(T, mat: RollerMaterial, width=0.025, thk=0.003, R_roll=0.010,
                   feed_tension=0.8, wrap_load_per_len=400.0):
    """Engineering stick criterion at a HOT roller.
    HOLD  = tack stress * contact-patch area  (adhesion gluing strip to roller)
    PEEL  = feed/takeup tension that lifts the strip at the exit tangent
          + the strip's own elastic straightening (small when well-relaxed/hot)
    Sticks (wraps the roller) if HOLD > PEEL. Returns (sticks, hold_N, peel_N)."""
    E = bending.E_of_T(T)
    b = hertz_halfwidth(wrap_load_per_len, R_roll, E)      # contact half-width
    A_c = 2 * b * width                                    # contact patch area
    hold = mat.tack0 * tack_factor(T) * A_c                # adhesive hold force
    # elastic straightening (peel) force from residual curvature ~ wrapping kappa;
    # when hot the strip relaxes to the roller so residual is small -> use a small
    # fraction; the dominant peeler is the feed/takeup tension.
    I = width * thk**3 / 12.0
    straighten = 0.1 * E * I / R_roll**2                   # weak when hot/relaxed
    peel = feed_tension + straighten
    return (hold > peel), hold, peel


def main():
    print("=" * 74)
    print("ROLLER MATERIAL MODEL  --  grip/slip + will-it-stick (literature est.)")
    print("=" * 74)

    print("\n  MATERIAL PROPERTIES (vs acrylic):")
    print(f"  {'material':>16} {'mu':>5} {'h_contact':>10} {'tack0(kPa)':>11}  note")
    for mat in MATERIALS.values():
        print(f"  {mat.name:>16} {mat.mu:>5.2f} {mat.h_contact:>8.0f}   "
              f"{mat.tack0/1e3:>9.0f}  {mat.note}")

    print("\n  FEED GRIP (pinch clamp 20 N each, feed a 3mm strip, bend R=50mm):")
    F_need, rows = grip_analysis()
    print(f"  force needed to feed+bend ~ {F_need:.2f} N")
    print(f"  {'material':>16} {'mu':>5} {'grip capacity N':>15} {'verdict':>9}")
    for name, mu, need, cap, ok in rows:
        print(f"  {name:>16} {mu:>5.2f} {cap:>15.1f} {'GRIP' if ok else 'SLIP':>9}")
    print("  -> forces are small when hot, so even PTFE usually grips; but PTFE's")
    print("     margin is thin -- a stiffer/cooler strip or lighter clamp -> SLIP.")

    print("\n  WILL IT STICK? (hot bend roller R=10mm, 3mm strip, feed tension 0.8N)")
    print(f"  {'material':>16} " + " ".join(f"{int(T):>5}C" for T in (110, 130, 150, 165)))
    for mat in MATERIALS.values():
        cells = []
        for T in (110, 130, 150, 165):
            sticks, hold, peel = stick_analysis(T, mat)
            cells.append("STICK" if sticks else " ok  ")
        print(f"  {mat.name:>16} " + " ".join(f"{c:>6}" for c in cells))
    print("  -> steel tacks to hot acrylic and can wrap the roller as it softens;")
    print("     PTFE/silicone release. Cold acrylic (<Tg) never sticks to anything.")

    plot_stick(); plot_tradeoff()
    print("\n  DESIGN CONCLUSION (robust to the +/-2x numbers):")
    print("  * HOT bend roller: use a NON-STICK surface (PTFE/silicone) -- steel")
    print("    risks wrapping once the strip is rubbery. Trade some grip for release.")
    print("  * COLD set roller: chilled STEEL is fine AND quenches best -- cold")
    print("    acrylic doesn't tack, and steel's contact conductance pulls heat fast.")
    print("  * Drive grip: ensure mu*clamp comfortably exceeds the (small) feed")
    print("    force; silicone drive rollers give the most grip margin.")
    print(f"  plots -> {OUT}/")


def plot_stick():
    Ts = np.linspace(95, 170, 120)
    fig, ax = plt.subplots(figsize=(7, 4.3))
    for mat in MATERIALS.values():
        margin = [stick_analysis(T, mat)[1] / max(stick_analysis(T, mat)[2], 1e-9)
                  for T in Ts]                       # hold/peel ratio
        ax.plot(Ts, margin, label=mat.name)
    ax.axhline(1.0, ls="--", c="k", lw=0.8, label="stick threshold")
    ax.axvline(m.T_glass, ls=":", c="grey", lw=0.8)
    ax.set(xlabel="strip temperature (C)", ylabel="stick number (hold/peel)",
           title="Will it stick? >1 = wraps the roller (hot bend roller)",
           ylim=(0, 3))
    ax.legend(fontsize=8); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(OUT / "stick_map.png", dpi=110); plt.close(fig)


def plot_tradeoff():
    names = list(MATERIALS)
    mu = [MATERIALS[n].mu for n in names]
    h = [MATERIALS[n].h_contact for n in names]
    tack = [MATERIALS[n].tack0 / 1e3 for n in names]
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.6))
    for a, vals, title, c in [(ax[0], mu, "grip (mu) - higher better", "C0"),
                              (ax[1], h, "quench h - higher better", "C2"),
                              (ax[2], tack, "tack0 kPa - LOWER better", "C3")]:
        a.bar(names, vals, color=c); a.set_title(title, fontsize=9)
        a.tick_params(axis="x", labelrotation=20, labelsize=7); a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "roller_tradeoff.png", dpi=110); plt.close(fig)


if __name__ == "__main__":
    main()
