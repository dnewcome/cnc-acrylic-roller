"""machine_contact.py -- the rollers are PART OF THE PHYSICS.

No "former" spring. The bend is produced by REAL roller contact: the strip is a
capsule chain fed (kinematically) through a pinch pair that guides it and stops
it buckling, then forced down by an offset BEND ROLLER whose bottom dips below
the feed line. The bend-roller offset (a CNC axis) sets the curvature.

It becomes PERMANENT through the thermo-viscoelastic coupling:
  * per-segment through-thickness temperature (seg_thermal) gates each hinge's
    stiffness k = E(core)*I/seg  -> the bend zone is hot & soft (bends under a few
    mN of roller force), downstream is cold & stiff (holds the shape);
  * each hinge's rest angle relaxes toward its current (contact-bent) angle with
    tau_relax(core T) -> while hot it "forgets" straight; once cooled past the set
    roller the rest angle is frozen and the curve is set.

So: roller contact decides the SHAPE, the thermal field decides whether it STICKS.
Feed too fast for the thickness -> core never heats -> it springs back. Real.

  python machine_contact.py            # batch run: animation + curve/force/temp plots
  python machine_contact.py --selftest # headless: feed through, check it forms & holds
"""
import os
os.environ.setdefault("MUJOCO_GL", "osmesa")
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import mujoco
from dataclasses import dataclass
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from material import PMMA
import bending
from seg_thermal import SegThermal

OUT = Path(__file__).resolve().parent / "out"
m = PMMA


@dataclass
class CMachine:
    N: int = 24
    L: float = 0.11
    width: float = 0.025
    thk: float = 0.002           # 2 mm: thermal time ~thk^2, so feasible feed in real time
    rs: float = 0.001            # capsule radius (~half thickness)
    r_roll: float = 0.005
    # stations (world x, m) -- sized so the CORE actually reaches forming at this feed
    preheat: tuple = (0.00, 0.040)
    bend: tuple = (0.045, 0.075)
    quench: tuple = (0.080, 0.140)
    x_pinch: float = 0.045
    x_bend: float = 0.060
    x_set: float = 0.100
    dip: float = 0.004           # bend-roller bottom dips this far below z=0 (CONTROL)
    # temperatures
    T_pre_surf: float = 105.0
    T_bend_surf: float = 172.0
    T_cold: float = 20.0
    feed: float = 0.002          # m/s  (feed-matched to the 2 mm thermal time)
    dt: float = 2.0e-4
    therm_sub: int = 15          # thermal step every N mech steps

    @property
    def seg(self):
        return self.L / self.N
    @property
    def I(self):
        return self.width * self.thk**3 / 12.0
    @property
    def x0(self):
        # PRE-THREADED: the strip starts straight through all stations (leading end
        # already in the quench zone) so it never has to self-thread under the bend
        # roller. The bend roller starts retracted and engages after warm-up.
        return self.quench[0] - self.L + 0.010

    def zone(self, x):
        if self.preheat[0] <= x < self.preheat[1]: return "pre"
        if self.bend[0] <= x < self.bend[1]: return "bend"
        if self.quench[0] <= x < self.quench[1]: return "quench"
        return "amb"


def build_xml(g: CMachine):
    seg, rs, R, hw = g.seg, g.rs, g.r_roll, g.width / 2
    def roller(name, x, z, rgba="0.3 0.5 0.9 1"):
        return (f'<geom name="{name}" type="cylinder" euler="90 0 0" pos="{x} 0 {z}" '
                f'size="{R} {hw*0.7}" rgba="{rgba}" contype="1" conaffinity="2" '
                f'friction="0.3 0.005 0.0001"/>')
    def zonebox(name, x0, x1, rgba):
        return (f'<geom name="{name}" type="box" pos="{(x0+x1)/2} 0 0" '
                f'size="{(x1-x0)/2} {hw*1.4} {g.thk*2}" rgba="{rgba}" '
                f'contype="0" conaffinity="0"/>')
    s = [f'<mujoco model="contact_bender">',
         f'<option timestep="{g.dt}" integrator="implicitfast" gravity="0 0 -9.81"/>',
         '<visual><global offwidth="1100" offheight="520"/>'
         '<headlight diffuse="0.6 0.6 0.6"/></visual>',
         '<default><geom solref="0.01 1" solimp="0.9 0.95 0.001"/></default>',
         '<worldbody>',
         '<light pos="0.08 -0.25 0.4" dir="0 0.4 -1"/>',
         f'<geom name="ref" type="plane" pos="0.08 0 -0.10" size="0.4 0.2 0.01" '
         'rgba="0.93 0.93 0.93 1" contype="0" conaffinity="0"/>',
         zonebox("z_pre", *g.preheat, "0.95 0.55 0.2 0.15"),
         zonebox("z_bend", *g.bend, "0.95 0.2 0.1 0.18"),
         zonebox("z_quench", *g.quench, "0.2 0.5 0.95 0.15"),
         roller("pinch_top", g.x_pinch, +(rs + R - 0.0002), "0.35 0.35 0.4 1"),
         roller("pinch_bot", g.x_pinch, -(rs + R - 0.0002), "0.35 0.35 0.4 1"),
         roller("bend", g.x_bend, rs + 0.002 + R),          # starts RETRACTED (clear)
         roller("set", g.x_set, +(rs + R), "0.2 0.5 0.95 1"),
         f'<body name="carriage" pos="{g.x0} 0 0">',
         '  <joint name="feed" type="slide" axis="1 0 0"/>',
         f'  <body name="seg0" pos="0 0 0">',
         f'    <geom type="capsule" fromto="0 0 0 {seg} 0 0" size="{rs}" '
         f'density="{m.rho}" contype="2" conaffinity="1" '
         f'friction="0.3 0.005 0.0001" rgba="0.4 0.35 0.85 1"/>']
    depth = 2
    for i in range(1, g.N):
        s.append('  ' * depth + f'<body name="seg{i}" pos="{seg} 0 0">')
        s.append('  ' * depth + f'  <joint name="h{i-1}" type="hinge" axis="0 1 0" '
                 f'pos="0 0 0" stiffness="1" damping="0.001"/>')
        s.append('  ' * depth + f'  <geom type="capsule" fromto="0 0 0 {seg} 0 0" '
                 f'size="{rs}" density="{m.rho}" contype="2" conaffinity="1" '
                 f'friction="0.3 0.005 0.0001" rgba="0.45 0.4 0.85 1"/>')
        depth += 1
    s.append('  ' * depth + '</body>' * (g.N - 1))
    s.append('  </body></body></worldbody></mujoco>')
    return "\n".join(s)


class ContactStrip:
    def __init__(self, g: CMachine):
        self.g = g
        self.model = mujoco.MjModel.from_xml_string(build_xml(g))
        self.data = mujoco.MjData(self.model)
        self.nh = g.N - 1
        self.jid = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"h{i}")
                    for i in range(self.nh)]
        self.qadr = [self.model.jnt_qposadr[j] for j in self.jid]
        self.dadr = [self.model.jnt_dofadr[j] for j in self.jid]
        self.bid = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"seg{i}")
                    for i in range(g.N)]
        self.gid = [self.model.body_geomadr[b] for b in self.bid]
        self.fj = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "feed")
        self.fq = self.model.jnt_qposadr[self.fj]
        self.fd = self.model.jnt_dofadr[self.fj]
        self.bendg = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "bend")
        self.therm = SegThermal(g.N, g.thk, nz=11, T_init=g.T_cold)
        self.rest = np.zeros(self.nh)

    # ---- thermal BC by zone ----
    def _bc(self, xb):
        g = self.g
        H = 2500.0                                   # heater clamp conductance
        h0 = np.full(g.N, 20.0); T0 = np.full(g.N, 20.0)
        hL = np.full(g.N, 20.0); TL = np.full(g.N, 20.0)
        for i, x in enumerate(xb):
            zn = g.zone(x)
            if zn == "pre":
                h0[i] = hL[i] = H; T0[i] = TL[i] = g.T_pre_surf
            elif zn == "bend":
                h0[i] = hL[i] = H; T0[i] = TL[i] = g.T_bend_surf
            elif zn == "quench":
                h0[i] = hL[i] = 350.0; T0[i] = TL[i] = g.T_cold   # forced air both faces
            else:
                h0[i] = hL[i] = 20.0; T0[i] = TL[i] = 20.0
        return h0, T0, hL, TL

    def feed_step(self, t):
        self.data.qpos[self.fq] = self.g.feed * t
        self.data.qvel[self.fd] = self.g.feed

    def update_material(self):
        """Per-hinge stiffness from CORE temp; rest angle relaxes toward current
        angle with tau_relax(core) -> contact-bent shape becomes permanent on cooling."""
        g = self.g
        core = self.therm.core                       # per-segment core temp
        for i in range(self.nh):
            Tc = max(core[i], core[i + 1])
            k = max(bending.E_of_T(Tc) * g.I / g.seg, 1e-4)
            self.model.jnt_stiffness[self.jid[i]] = k
            self.model.dof_damping[self.dadr[i]] = 0.3 * k + 5e-4
            theta = self.data.qpos[self.qadr[i]]
            tr = float(bending.tau_relax(Tc))
            self.rest[i] += (theta - self.rest[i]) * (g.dt / max(tr, g.dt))
            self.model.qpos_spring[self.qadr[i]] = self.rest[i]

    def set_dip(self, dip):
        self.model.geom_pos[self.bendg, 2] = -dip + self.g.r_roll

    def bend_force(self):
        f = np.zeros(6); tot = 0.0
        for c in range(self.data.ncon):
            con = self.data.contact[c]
            if self.bendg in (con.geom1, con.geom2):
                mujoco.mj_contactForce(self.model, self.data, c, f)
                tot += abs(f[0])
        return tot

    def seg_world(self):
        return self.data.xpos[np.array(self.bid)].copy()

    def color_by_temp(self):
        mean = self.therm.mean
        for i in range(self.g.N):
            fr = np.clip((mean[i] - 20) / 140.0, 0, 1)
            self.model.geom_rgba[self.gid[i]] = [0.25 + 0.7 * fr, 0.35,
                                                 0.85 - 0.7 * fr, 1.0]


def dip_schedule(t, g: CMachine, t_engage=18.0, ramp=5.0):
    """Thread straight first (roller retracted), then ramp the bend-roller dip in
    after the bend zone has warmed up -- how you'd actually run the machine."""
    lo = -(g.rs + 0.002)                         # retracted: roller clear of strip
    if t < t_engage:
        return lo
    return lo + (g.dip - lo) * min(1.0, (t - t_engage) / ramp)


def run(g: CMachine, t_end=40.0, render_to=None, dip_fn=None):
    s = ContactStrip(g)
    mujoco.mj_forward(s.model, s.data)
    frames = []
    ren = mujoco.Renderer(s.model, 480, 1040) if render_to else None
    cam = mujoco.MjvCamera()
    cam.azimuth, cam.elevation, cam.distance = 90, -10, 0.30
    cam.lookat[:] = [0.07, 0, -0.02]
    every = max(1, int((1 / 40) / g.dt))
    nsteps = int(t_end / g.dt)
    log = {k: [] for k in ("t", "fbend", "core_bend")}
    for k in range(nsteps):
        t = k * g.dt
        s.feed_step(t)
        if dip_fn:
            s.set_dip(dip_fn(t))
        if k % g.therm_sub == 0:
            xb = s.seg_world()[:, 0]
            s.therm.step(g.therm_sub * g.dt, *s._bc(xb))
        s.update_material()
        mujoco.mj_step(s.model, s.data)
        if k % every == 0:
            xb = s.seg_world()[:, 0]
            ib = np.where((xb >= g.bend[0]) & (xb < g.bend[1]))[0]
            log["t"].append(t); log["fbend"].append(s.bend_force())
            log["core_bend"].append(s.therm.core[ib].max() if ib.size else np.nan)
            if ren is not None:
                s.color_by_temp(); ren.update_scene(s.data, cam); frames.append(ren.render())
    if render_to and frames:
        import imageio.v2 as iio
        iio.mimsave(render_to, frames, fps=30)
    return s, {k: np.array(v) for k, v in log.items()}


def main():
    g = CMachine()
    selftest = "--selftest" in sys.argv
    print("=" * 72)
    print("CONTACT-DRIVEN BENDER  --  rollers form the strip by real contact")
    print(f"  strip {g.N} seg {g.L*1e3:.0f}mm, feed {g.feed*1e3:.1f}mm/s, "
          f"bend-roller dip {g.dip*1e3:.1f}mm")
    print("=" * 72)
    t_end = 60.0 if selftest else 65.0
    dip_fn = lambda t: dip_schedule(t, g)
    s, log = run(g, t_end=t_end, dip_fn=dip_fn,
                 render_to=None if selftest else str(OUT / "machine_contact.gif"))
    w = s.seg_world()
    # FORMED-AND-SET region = downstream of the bend zone AND cooled below Tg
    setseg = [i for i in range(s.nh)
              if w[i + 1, 0] > g.bend[1] and s.therm.core[i + 1] < m.T_glass]
    held = np.degrees(abs(np.array([s.data.qpos[s.qadr[i]] for i in setseg]))).sum() \
        if setseg else 0.0
    set_drop = (w[[i + 1 for i in setseg], 2].max() - w[[i + 1 for i in setseg], 2].min()) \
        * 1000 if setseg else 0
    peak_core = np.nanmax(log["core_bend"]) if log["core_bend"].size else 0
    peak_force = np.nanmax(log["fbend"]) if log["fbend"].size else 0
    ok = np.isfinite(s.data.qpos).all()
    print(f"  finite={ok}  peak core-in-bend={peak_core:.0f}C  "
          f"peak bend-roller force={peak_force*1e3:.1f} mN")
    print(f"  formed+SET region: {len(setseg)} cooled segments hold {held:.0f} deg "
          f"of curve (z-span {set_drop:.0f} mm)")
    if selftest:
        good = ok and peak_core > 145 and held > 12
        print(f"  selftest {'OK' if good else 'CHECK'}: core heats, contact forms a "
              f"curve that holds after cooling")
        sys.exit(0 if good else 1)
    plot(s, log, g)
    print(f"  wrote out/machine_contact.gif, out/machine_contact.png")


def plot(s, log, g):
    w = s.seg_world()
    fig, ax = plt.subplots(3, 1, figsize=(8, 8))
    ax[0].plot(w[:, 0] * 1e3, w[:, 2] * 1e3, "-o", ms=3)
    for (lo, hi), c in [(g.preheat, "#f0a060"), (g.bend, "#e05040"), (g.quench, "#5090e0")]:
        ax[0].axvspan(lo * 1e3, hi * 1e3, color=c, alpha=0.2)
    ax[0].set(title="Formed centerline (contact-bent)", xlabel="x (mm)", ylabel="z (mm)")
    ax[0].set_aspect("equal", "box"); ax[0].grid(alpha=0.3)
    ax[1].plot(log["t"], np.array(log["fbend"]) * 1e3, color="C3")
    ax[1].set(title="Bend-roller contact force", xlabel="t (s)", ylabel="force (mN)")
    ax[1].grid(alpha=0.3)
    ax[2].plot(log["t"], log["core_bend"], color="C1")
    ax[2].axhline(m.T_form, ls="--", c="grey", lw=0.8)
    ax[2].set(title="Core temp of strip in the bend zone", xlabel="t (s)", ylabel="T (C)")
    ax[2].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "machine_contact.png", dpi=110); plt.close(fig)


if __name__ == "__main__":
    main()
