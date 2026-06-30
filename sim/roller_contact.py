"""roller_contact.py -- REAL roller contact forces (measured, not commanded).

The cartoon machine imposed the bend with a "former" spring. Here a real,
collidable roller is physically pressed into the strip and we MEASURE the
contact reaction force MuJoCo computes -- then validate it against beam theory
and show how it scales with temperature (the E(T) coupling).

Setup: the strip is the validated hinge-chain beam (per-joint stiffness
k = E(T)*I/seg, matches Euler-Bernoulli to ~0.9). It's a cantilever; a roller
on a slide joint is pressed down a known indentation delta at distance a from
the root. Measured roller force is compared to the cantilever point-load result
F = 3*E*I*delta / a^3.

Why it matters: cold acrylic needs large forming force; hot acrylic needs almost
none. That sets roller/bearing/motor loads AND feeds the grip & stick analysis
in rollers.py (the force pressing the strip onto the roller).

Friction at the contact uses mu from the chosen roller material (rollers.py).
The rod is the round-stiffness 1-D beam; contact is line-ish (box on cylinder).
"""
import os
os.environ.setdefault("MUJOCO_GL", "osmesa")
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from material import PMMA
import bending
from rollers import MATERIALS

OUT = Path(__file__).resolve().parent / "out"
m = PMMA
G = 9.81

N = 28
L = 0.14
WIDTH = 0.025
THK = 0.003
SEG = L / N
I_STRIP = WIDTH * THK**3 / 12.0
R_ROLL = 0.010
A_PRESS = 0.9 * L                       # press point distance from root


def build_xml(mu):
    seg, hw, ht = SEG, WIDTH / 2, THK / 2
    s = [f'<mujoco model="roller_contact">',
         '<option timestep="2e-4" integrator="implicitfast" gravity="0 0 -9.81">'
         '<flag contact="enable"/></option>',
         '<visual><global offwidth="1200" offheight="600"/></visual>',
         '<default><geom solref="0.003 1" solimp="0.95 0.99 0.001"/></default>',
         '<worldbody>',
         '<light pos="0.07 -0.2 0.4" dir="0 0.4 -1"/>',
         # roller on a vertical slide joint, pressed kinematically. Collidable.
         f'<body name="roller" pos="{A_PRESS} 0 {ht + R_ROLL + 0.02}">',
         '  <joint name="press" type="slide" axis="0 0 1"/>',
         f'  <geom name="rollg" type="cylinder" euler="90 0 0" size="{R_ROLL} {hw}" '
         f'rgba="0.3 0.5 0.9 1" contype="1" conaffinity="2" '
         f'friction="{mu} 0.005 0.0001"/>',
         '</body>',
         # cantilever hinge-chain strip; rod-rod don't collide, rod-roller do
         f'<body name="seg0" pos="0 0 0">',
         f'  <geom type="box" size="{seg/2} {hw} {ht}" pos="{seg/2} 0 0" '
         f'density="{m.rho}" contype="2" conaffinity="1" '
         f'friction="{mu} 0.005 0.0001" rgba="0.8 0.4 0.2 1"/>']
    depth = 1
    for i in range(1, N):
        s.append('  ' * depth + f'<body name="seg{i}" pos="{seg} 0 0">')
        s.append('  ' * depth + f'  <joint name="h{i-1}" type="hinge" axis="0 1 0" '
                 f'pos="0 0 0" stiffness="1" damping="0.001"/>')
        s.append('  ' * depth + f'  <geom type="box" size="{seg/2} {hw} {ht}" '
                 f'pos="{seg/2} 0 0" density="{m.rho}" contype="2" conaffinity="1" '
                 f'friction="{mu} 0.005 0.0001" rgba="0.8 0.45 0.25 1"/>')
        depth += 1
    s.append('  ' * depth + '</body>' * (N - 1))
    s.append('</body></worldbody></mujoco>')
    return "\n".join(s)


class Rig:
    def __init__(self, mu=0.4):
        self.model = mujoco.MjModel.from_xml_string(build_xml(mu))
        self.data = mujoco.MjData(self.model)
        self.jid = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"h{i}")
                    for i in range(N - 1)]
        self.dadr = [self.model.jnt_dofadr[j] for j in self.jid]
        self.press = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "press")
        self.press_q = self.model.jnt_qposadr[self.press]
        self.press_d = self.model.jnt_dofadr[self.press]
        self.rollg = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "rollg")

    def set_stiffness(self, T):
        k = bending.E_of_T(T) * I_STRIP / SEG
        for i, j in enumerate(self.jid):
            self.model.jnt_stiffness[j] = k
            self.model.dof_damping[self.dadr[i]] = 0.15 * k + 2e-4

    def roller_force(self):
        """Sum the normal contact force on the roller geom (N)."""
        f = np.zeros(6); total = 0.0
        for c in range(self.data.ncon):
            con = self.data.contact[c]
            if self.rollg in (con.geom1, con.geom2):
                mujoco.mj_contactForce(self.model, self.data, c, f)
                total += abs(f[0])             # normal component
        return total


def measure_force(T, delta, mu=0.4, t_end=3.0):
    """Press the roller down by `delta` at temperature T; return measured force."""
    rig = Rig(mu)
    rig.set_stiffness(T)
    z_touch = THK / 2                          # roller bottom just grazes top face
    z_start = z_touch + R_ROLL + 0.02          # body z so geom bottom starts above
    mujoco.mj_forward(rig.model, rig.data)
    nsteps = int(t_end / 2e-4)
    forces = []
    for k in range(nsteps):
        frac = min(1.0, k / (nsteps * 0.4))    # ramp press over first 40%
        target_drop = (0.02 + delta) * frac    # lower the roller by up to 0.02+delta
        rig.data.qpos[rig.press_q] = -target_drop
        rig.data.qvel[rig.press_d] = 0.0
        mujoco.mj_step(rig.model, rig.data)
        if k > nsteps * 0.7:                    # average once settled
            forces.append(rig.roller_force())
    return float(np.mean(forces)) if forces else 0.0


def main():
    print("=" * 72)
    print("REAL ROLLER CONTACT FORCE  --  measured vs beam theory, vs temperature")
    print(f"  strip {WIDTH*1e3:.0f}x{THK*1e3:.0f}mm, press at a={A_PRESS*1e3:.0f}mm, "
          f"roller R={R_ROLL*1e3:.0f}mm")
    print("=" * 72)

    delta = 0.004                               # 4 mm indentation
    print(f"\n  VALIDATION at {delta*1e3:.0f}mm indent: measured vs F=3EI*delta/a^3")
    print(f"  {'T':>6} {'E(MPa)':>8} {'measured N':>11} {'beam N':>9} {'ratio':>6}")
    Ts = [20, 80, 120, 140, 155]
    meas = []
    for T in Ts:
        F = measure_force(T, delta)
        EI = bending.E_of_T(T) * I_STRIP
        beam = 3 * EI * delta / A_PRESS**3
        meas.append(F)
        print(f"  {T:>5}C {bending.E_of_T(T)/1e6:>8.1f} {F:>11.3f} {beam:>9.3f} "
              f"{(F/beam if beam>1e-9 else 0):>6.2f}")
    print("  -> cold acrylic needs newtons to bend; at forming temp it's "
          "milli-newtons.")
    print("     The roller/bearing/motor loads are set by the COLD end of any "
          "accidental")
    print("     under-heating, not the hot design point.")

    # Friction (roller material) -> tangential drag = mu * normal. The normal
    # force barely depends on mu, so measure it once and scale. NOTE: at the HOT
    # bend roller the normal force is ~0, so friction there is moot -- mu matters
    # at the cooler PINCH/DRIVE rollers (grip, see rollers.py), not the hot bend.
    Tdrag = 110                                  # a cooler/under-heated condition
    N_drag = measure_force(Tdrag, delta)
    print(f"\n  ROLLER MATERIAL friction -> tangential drag (= mu*N) at {Tdrag}C, "
          f"normal {N_drag*1e3:.0f} mN:")
    for name, mat in MATERIALS.items():
        print(f"    {name:>16} mu={mat.mu:>4.2f}  drag <= {mat.mu*N_drag*1e3:>6.1f} mN")
    print("    -> at the HOT bend roller normal force ~0, so friction is moot there;")
    print("       mu matters at the cooler pinch/drive rollers (grip) -- see rollers.py.")

    plot_force(Ts, meas, delta)
    print(f"\n  plots -> {OUT}/")


def plot_force(Ts, meas, delta):
    beam = [3 * bending.E_of_T(T) * I_STRIP * delta / A_PRESS**3 for T in Ts]
    floor = 1e-4                                  # measurement resolution floor (N)
    meas = [max(f, floor) for f in meas]
    fig, ax = plt.subplots(figsize=(7, 4.3))
    ax.axhline(floor, ls=":", c="0.6", lw=0.8, label="~measurement floor (1 mN)")
    ax.semilogy(Ts, meas, "-o", label="MuJoCo measured contact force")
    ax.semilogy(Ts, beam, "--s", label="beam theory 3EI*delta/a^3")
    ax.axvline(m.T_glass, ls=":", c="grey", lw=0.8, label="Tg")
    ax.set(xlabel="strip temperature (C)", ylabel="roller force (N), log",
           title=f"Roller forming force collapses with temperature ({delta*1e3:.0f}mm indent)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both"); fig.tight_layout()
    fig.savefig(OUT / "roller_force.png", dpi=110); plt.close(fig)


if __name__ == "__main__":
    main()
