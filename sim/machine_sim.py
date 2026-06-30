"""machine_sim.py -- full-machine MuJoCo animation of the acrylic bender.

The strip (the validated lumped-beam chain from strip_sim) rides a FEED carriage
and is driven through three fixed stations:

   feeder --> [ PRE-HEAT zone ] --> [ BEND rollers + former ] --> [ chilled SET roller ]
              20->~95C (stiff)       ~95->155C, curved to          155->20C, curve frozen
                                     commanded curvature kappa(t)

As each segment feeds through, its temperature follows a position-based reduced
thermal model and its hinge is curved by the bend station's former while hot,
then frozen as it cools past the set roller -> a continuous formed curve EXITS.
kappa(t) is programmable, so the machine draws an arbitrary (varying-radius)
curve -- the whole point of the project.

Bending is in the vertical (x-z) plane so gravity acts on the hot zone. Rollers
and zone markers are VISUAL (forming is imposed by the validated former+relaxation
model, not by finicky roller contact). Times are COMPRESSED for watchability
(thermal lags shortened); the spatial sequencing and forming physics are the point.

   python machine_sim.py            # animation gif + curve/temperature plots -> out/
   python machine_sim.py --selftest # headless: feed a strip through, check it forms
"""
import os
os.environ.setdefault("MUJOCO_GL", "osmesa")
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import mujoco
from dataclasses import dataclass, field
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import strip_sim as S
import bending
from material import PMMA

OUT = Path(__file__).resolve().parent / "out"
m = PMMA


@dataclass
class Machine:
    # strip
    N: int = 48
    strip_len: float = 0.20
    # station x-ranges (m), world frame
    preheat: tuple = (0.00, 0.060)
    bend: tuple = (0.070, 0.095)
    quench: tuple = (0.105, 0.160)
    # temperatures
    T_pre: float = 95.0
    T_form: float = 155.0
    T_cold: float = 20.0
    # forming
    K_former: float = 10.0
    kappa_max: float = 16.0        # 1/m, peak commanded curvature (R ~ 60 mm)
    # feed
    feed_speed: float = 0.009      # m/s
    # roller radius (small enough that the pinch + bend + set rollers don't overlap)
    r_roll: float = 0.006

    def params(self):
        return S.Params(N=self.N, strip_len=self.strip_len, bend_axis="0 1 0",
                        relax=True, tau_heat=1.5, tau_cool=1.5,
                        damp_beta=0.25, damp_floor=0.02, T_hot=self.T_form)

    @property
    def x0(self):                  # carriage start: leading end just before pre-heat
        return self.preheat[0] - self.strip_len - 0.005

    def zone_of(self, x):
        if self.preheat[0] <= x < self.preheat[1]: return "pre"
        if self.bend[0]    <= x < self.bend[1]:    return "bend"
        if self.quench[0]  <= x < self.quench[1]:  return "quench"
        return "ambient"


def build_machine_xml(g: Machine, ui=False):
    p = g.params()
    seg, hw, ht = p.seg, p.strip_w / 2, p.strip_t / 2
    zl = 0.0                                    # strip line height
    def roller(name, x, z, rgba):
        return (f'<geom name="{name}" type="cylinder" euler="90 0 0" '
                f'pos="{x} 0 {z}" size="{g.r_roll} {hw*1.2}" '
                f'rgba="{rgba}" contype="0" conaffinity="0"/>')
    def zone(name, x0, x1, rgba):
        cx = (x0 + x1) / 2
        return (f'<geom name="{name}" type="box" pos="{cx} 0 {zl}" '
                f'size="{(x1-x0)/2} {hw*1.6} {ht*4}" rgba="{rgba}" '
                f'contype="0" conaffinity="0"/>')
    s = [f'<mujoco model="acrylic_bender">',
         f'<option timestep="{p.dt}" integrator="implicitfast" gravity="0 0 -9.81"/>',
         '<visual><global offwidth="1280" offheight="720"/>'
         '<headlight diffuse="0.6 0.6 0.6"/></visual>',
         '<worldbody>',
         '<light pos="0.06 -0.3 0.4" dir="0 0.6 -1"/>',
         f'<geom name="ref" type="plane" pos="0.08 0 -0.12" size="0.4 0.2 0.01" '
         'rgba="0.92 0.92 0.92 1" contype="0" conaffinity="0"/>',
         zone("z_pre", *g.preheat, "0.95 0.55 0.2 0.18"),
         zone("z_bend", *g.bend, "0.95 0.2 0.1 0.22"),
         zone("z_quench", *g.quench, "0.2 0.5 0.95 0.18"),
         # Roller layout (spaced so none overlap): a pinch pair at the bend
         # entry, the bend roller well downstream of it (its z is set live to the
         # convex side of the bend in machine_step), a chilled set roller in the
         # quench zone. x-spacing >= 2*r_roll keeps them clear.
         roller("roll_top", g.bend[0], zl + ht + g.r_roll, "0.35 0.35 0.4 1"),
         roller("roll_bot", g.bend[0], zl - ht - g.r_roll, "0.35 0.35 0.4 1"),
         roller("roll_bend", g.bend[1] - 0.005, zl + ht + g.r_roll + 0.002, "0.3 0.5 0.9 1"),
         roller("roll_set", (g.quench[0]+g.quench[1])/2, zl - ht - g.r_roll, "0.2 0.5 0.95 1"),
         # feed carriage carrying the chain
         f'<body name="carriage" pos="{g.x0} 0 {zl}">',
         '  <joint name="feed" type="slide" axis="1 0 0"/>',
         f'  <body name="seg0" pos="0 0 0">',
         f'    <geom type="box" size="{seg/2} {hw} {ht}" pos="{seg/2} 0 0" '
         f'density="{m.rho}" rgba="0.4 0.35 0.85 1"/>']
    depth = 2
    for i in range(1, p.N):
        s.append('  ' * depth + f'<body name="seg{i}" pos="{seg} 0 0">')
        s.append('  ' * depth + f'  <joint name="h{i-1}" type="hinge" axis="0 1 0" '
                 f'pos="0 0 0" stiffness="1" damping="0.001" springref="0"/>')
        s.append('  ' * depth + f'  <geom type="box" size="{seg/2} {hw} {ht}" '
                 f'pos="{seg/2} 0 0" density="{m.rho}" rgba="0.4 0.35 0.85 1"/>')
        depth += 1
    s.append('  ' * depth + '</body>' * (p.N - 1))
    s.append('  </body>')        # seg0
    s.append('</body>')          # carriage
    s.append('</worldbody>')
    if ui:   # gear=0 proxy actuators -> live sliders (no mechanical effect)
        s.append('<actuator>')
        s.append('  <motor name="ui_feed" joint="feed" gear="0" ctrllimited="true" '
                 'ctrlrange="0 0.020"/>')
        s.append('  <motor name="ui_kappa" joint="feed" gear="0" ctrllimited="true" '
                 f'ctrlrange="{-g.kappa_max} {g.kappa_max}"/>')
        s.append('</actuator>')
    s.append('</mujoco>')
    return "\n".join(s)


def machine_step(strip, g: Machine, kappa, dt):
    """One physics step of the machine given the current commanded curvature.
    Feed is set by the caller (batch uses time; interactive accumulates). Shared
    by run_machine and the interactive viewer so they can't drift."""
    p = strip.p
    zt = {"pre": g.T_pre, "bend": g.T_form, "quench": g.T_cold, "ambient": g.T_cold}
    xb = strip.seg_world()[:, 0]
    strip.step_thermal(np.array([zt[g.zone_of(x)] for x in xb]), dt)
    hinge_x = xb[1:]                                  # hinge i ~ body i+1
    in_bend = np.where((hinge_x >= g.bend[0]) & (hinge_x < g.bend[1]))[0]
    former = None
    if in_bend.size:
        iz0, iz1 = int(in_bend[0]), int(in_bend[-1]) + 1
        per = kappa * p.seg
        former = (iz0, iz1, per * (iz1 - iz0), g.K_former)
    # Place the bend roller on the CONVEX (outer) side of the bend and push IN --
    # like a 3-roll bender. +kappa bends the strip DOWN, so its convex side is UP
    # -> roller above; -kappa -> roller below. Displace more for tighter curves.
    if not hasattr(strip, "_bend_gid"):
        strip._bend_gid = mujoco.mj_name2id(strip.model, mujoco.mjtObj.mjOBJ_GEOM,
                                            "roll_bend")
    if strip._bend_gid >= 0:
        sgn = 1.0 if kappa >= 0 else -1.0
        clr = p.strip_t / 2 + g.r_roll + 0.002
        strip.model.geom_pos[strip._bend_gid, 2] = sgn * (clr + 0.0006 * abs(kappa))
    strip.apply_thermo(former=former)
    strip.step_relax(dt)
    mujoco.mj_step(strip.model, strip.data)


def run_machine(g: Machine, kappa_fn, t_end=22.0, render_to=None):
    p = g.params()
    strip = S.Strip(p, xml=build_machine_xml(g))
    strip.T[:] = g.T_cold
    seg = p.seg

    frames = []
    renderer = mujoco.Renderer(strip.model, 480, 960) if render_to else None
    cam = mujoco.MjvCamera()
    cam.azimuth, cam.elevation, cam.distance = 90, -12, 0.34
    cam.lookat[:] = [0.085, 0, -0.01]
    every = max(1, int((1 / 50) / p.dt))
    nsteps = int(t_end / p.dt)
    mujoco.mj_forward(strip.model, strip.data)       # valid xpos before first read

    for k in range(nsteps):
        t = k * p.dt
        # carriage body is already at x0; feed joint adds only the displacement
        strip.set_feed(t, g.feed_speed)
        machine_step(strip, g, kappa_fn(t), p.dt)
        if k % every == 0:
            strip.color_by_temp()
            if renderer is not None:
                renderer.update_scene(strip.data, cam); frames.append(renderer.render())
    if render_to and frames:
        import imageio.v2 as iio
        iio.mimsave(render_to, frames, fps=30)
    return strip


def kappa_scurve(t, g: Machine, period=10.0):
    """Programmed curvature: a smooth sign-changing curvature -> an S-curve."""
    return g.kappa_max * np.sin(2 * np.pi * t / period)


def main():
    g = Machine()
    selftest = "--selftest" in sys.argv
    p = g.params()
    print("=" * 72)
    print("MuJoCo MACHINE sim -- feed / pre-heat / bend / chilled-set")
    strip0 = S.Strip(p, xml=build_machine_xml(g))
    total_mass = sum(strip0.model.body_mass[strip0.bid]) * 1e3
    print(f"  strip {p.N} seg, {p.strip_len*1000:.0f}mm, {total_mass:.1f} g; "
          f"feed {g.feed_speed*1000:.0f} mm/s")
    print(f"  zones (mm): pre {tuple(int(v*1000) for v in g.preheat)}, "
          f"bend {tuple(int(v*1000) for v in g.bend)}, "
          f"quench {tuple(int(v*1000) for v in g.quench)}")

    kfn = lambda t: kappa_scurve(t, g)
    if selftest:
        strip = run_machine(g, kfn, t_end=15.0)   # long enough to reach the bend zone
        ok = np.isfinite(strip.data.qpos).all()
        ang = np.degrees(np.array(strip.hinge_angles()))
        total = abs(ang).sum()                 # integrated curve (deg) is the real metric
        print(f"  selftest {'OK' if ok and total > 20 else 'FAILED'}: finite={ok}, "
              f"max hinge {abs(ang).max():.1f} deg, total formed curve {total:.0f} deg")
        sys.exit(0 if ok and total > 20 else 1)

    strip = run_machine(g, kfn, t_end=22.0, render_to=str(OUT / "machine.gif"))
    plot_curve_and_temp(strip, g)
    print(f"  wrote out/machine.gif, out/machine_curve.png")


def plot_curve_and_temp(strip, g: Machine):
    w = strip.seg_world()
    x, z = w[:, 0] * 1000, w[:, 2] * 1000
    fig, ax = plt.subplots(2, 1, figsize=(8, 6.5))
    ax[0].plot(x, z, "-o", ms=3, color="C0")
    for (lo, hi), c, lab in [(g.preheat, "#f0a060", "pre-heat"),
                             (g.bend, "#e05040", "bend"),
                             (g.quench, "#5090e0", "quench")]:
        ax[0].axvspan(lo * 1000, hi * 1000, color=c, alpha=0.2, label=lab)
    ax[0].set(xlabel="world x (mm)", ylabel="z (mm)", title="Formed strip centerline (side view)")
    ax[0].legend(fontsize=7); ax[0].grid(alpha=0.3); ax[0].set_aspect("equal", "box")
    ax[1].plot(x, strip.T, "-o", ms=3, color="C3")
    for (lo, hi), c in [(g.preheat, "#f0a060"), (g.bend, "#e05040"), (g.quench, "#5090e0")]:
        ax[1].axvspan(lo * 1000, hi * 1000, color=c, alpha=0.2)
    for yv, lab in [(m.T_glass, "Tg"), (g.T_form, "form")]:
        ax[1].axhline(yv, ls=":", lw=0.8, c="grey");
    ax[1].set(xlabel="world x (mm)", ylabel="segment temp (C)", title="Temperature along the strip")
    ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "machine_curve.png", dpi=110); plt.close(fig)


if __name__ == "__main__":
    main()
