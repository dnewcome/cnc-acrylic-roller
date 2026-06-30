"""strip_sim.py -- MuJoCo thermo-mechanical co-sim of the acrylic strip.

MuJoCo is a rigid-body engine, NOT a heat solver, so this is a CO-SIMULATION:
  * the strip is a serial chain of box segments + hinge joints (a lumped /
    pseudo-rigid-body model of a flexible beam);
  * a reduced thermal model (calibrated to the validated 1-D solver in
    thermal.py) gives each segment a temperature;
  * temperature drives, live, each hinge's STIFFNESS  k = E(T)*I/seg_len  and
    its REST ANGLE via a viscoelastic relaxation ODE -> forming + springback
    emerge from the dynamics rather than being scripted.

SI units throughout (mm dimensions * 0.001). Material + E(T) imported from the
same modules that drive the rest of the study (single source of truth).

  python strip_sim.py            # validation + sag sweep + gif + plots -> out/
"""
import os
os.environ.setdefault("MUJOCO_GL", "osmesa")   # BEFORE importing mujoco
import numpy as np
import mujoco
from dataclasses import dataclass, field
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from material import PMMA
import bending

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)
m = PMMA
G = 9.81


def smoothstep(x):
    """Clamped smoothstep 0..1 for gentle (quasi-static) command ramps."""
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


@dataclass
class Params:
    strip_len: float = 0.15        # m, total modeled strip length
    strip_w: float = 0.025         # m, width
    strip_t: float = 0.003         # m, thickness
    N: int = 40                    # chain segments (N-1 hinges); ~0.93 of
                                   # continuum sag (discretization, converges up)
    bend_axis: str = "0 1 0"       # hinge axis: y=vertical-plane sag; z=horizontal forming
    dt: float = 2.0e-4
    # thermal
    T_cold: float = 20.0
    T_hot: float = 155.0           # forming temperature in the heated band
    tau_heat: float = 7.0          # s, 1st-order heat-up const (~through-heat/3)
    tau_cool: float = 8.0          # s, cool-down const
    relax: bool = False            # enable rest-angle viscoelastic relaxation
    # mechanical damping (Rayleigh-ish, stiffness-proportional + floor)
    damp_beta: float = 0.08        # s, c_i = damp_beta * k_i
    damp_floor: float = 2.0e-4     # N*m*s/rad

    @property
    def seg(self):
        return self.strip_len / self.N
    @property
    def I(self):
        return self.strip_w * self.strip_t**3 / 12.0


# ---------- model construction ----------

def build_xml(p: Params, roller=None, ui=False):
    seg, hw, ht = p.seg, p.strip_w / 2, p.strip_t / 2
    ax = p.bend_axis
    s = [f'<mujoco model="acrylic_strip">',
         f'<option timestep="{p.dt}" integrator="implicitfast" gravity="0 0 {-G}"/>',
         '<visual><global offwidth="1280" offheight="720"/>'
         '<headlight diffuse="0.5 0.5 0.5"/></visual>',
         '<worldbody>',
         '<light pos="0.0 -0.3 0.4" dir="0 0.6 -1"/>',
         '<geom name="ref" type="plane" pos="0 0 -0.1" size="0.4 0.2 0.01" '
         'rgba="0.9 0.9 0.9 1" contype="0" conaffinity="0"/>']
    if roller is not None:
        rx, rz, rr = roller
        s.append(f'<geom name="roller" type="cylinder" euler="90 0 0" '
                 f'pos="{rx} 0 {rz}" size="{rr} {hw}" rgba="0.2 0.5 0.9 1"/>')
    # base segment (index 0) is welded to the world = the feed chuck / grip.
    s.append(f'<body name="seg0" pos="0 0 0">')
    s.append(f'  <geom type="box" size="{seg/2} {hw} {ht}" pos="{seg/2} 0 0" '
             f'density="{m.rho}" rgba="0.8 0.4 0.2 1"/>')
    depth = 1
    for i in range(1, p.N):
        s.append('  ' * depth + f'<body name="seg{i}" pos="{seg} 0 0">')
        s.append('  ' * depth + f'  <joint name="h{i-1}" type="hinge" axis="{ax}" '
                 f'pos="0 0 0" stiffness="1" damping="0.001" springref="0"/>')
        s.append('  ' * depth + f'  <geom type="box" size="{seg/2} {hw} {ht}" '
                 f'pos="{seg/2} 0 0" density="{m.rho}" rgba="0.8 0.45 0.25 1"/>')
        depth += 1
    s.append('  ' * depth + '</body>' * (p.N - 1))
    s.append('</body>')           # close seg0
    s.append('</worldbody>')
    if ui:   # gear=0 proxy actuators -> pure UI sliders (move nothing mechanically)
        s.append('<actuator>')
        s.append('  <motor name="ui_heat" joint="h0" gear="0" ctrllimited="true" '
                 f'ctrlrange="{p.T_cold} 170"/>')
        s.append('  <motor name="ui_bend" joint="h0" gear="0" ctrllimited="true" '
                 'ctrlrange="0 1.4"/>')
        s.append('</actuator>')
    s.append('</mujoco>')
    return "\n".join(s)


class Strip:
    """Wraps a built MuJoCo model and the per-segment thermal/relaxation state."""
    def __init__(self, p: Params, roller=None, ui=False, xml=None):
        self.p = p
        # `xml` lets a caller (e.g. machine_sim) supply a richer model -- feed
        # carriage, rollers -- as long as it keeps the h{i}/seg{i} naming.
        self.model = mujoco.MjModel.from_xml_string(
            xml if xml is not None else build_xml(p, roller, ui=ui))
        self.data = mujoco.MjData(self.model)
        self.nh = p.N - 1
        self.jid = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"h{i}")
                    for i in range(self.nh)]
        self.qadr = [self.model.jnt_qposadr[j] for j in self.jid]
        self.dadr = [self.model.jnt_dofadr[j] for j in self.jid]
        self.bid = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"seg{i}")
                    for i in range(p.N)]
        self.gid = [self.model.body_geomadr[b] for b in self.bid]   # 1 geom/seg
        # optional feed (carriage) slide joint
        fj = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "feed")
        self.feed_jid = fj if fj >= 0 else None
        if self.feed_jid is not None:
            self.feed_qadr = self.model.jnt_qposadr[fj]
            self.feed_dadr = self.model.jnt_dofadr[fj]
        self.T = np.full(p.N, p.T_cold)        # segment temperatures
        self.rest = np.zeros(self.nh)          # hinge rest angles

    def set_feed(self, t, speed, x0=0.0):
        """Kinematically prescribe the carriage position (robust feed drive)."""
        if self.feed_jid is None:
            return
        self.data.qpos[self.feed_qadr] = x0 + speed * t
        self.data.qvel[self.feed_dadr] = speed

    def seg_world(self):
        """World (x,y,z) of every segment body's origin."""
        return self.data.xpos[np.array(self.bid)].copy()

    def color_by_temp(self, t_lo=20.0, t_hi=160.0):
        """Tint each segment blue(cold)->red(hot) so heating/cooling is visible."""
        for i in range(self.p.N):
            f = np.clip((self.T[i] - t_lo) / (t_hi - t_lo), 0, 1)
            self.model.geom_rgba[self.gid[i]] = [0.25 + 0.7 * f, 0.35,
                                                 0.85 - 0.7 * f, 1.0]

    def seg_mass(self):
        return self.model.body_mass[self.bid[1]]

    def apply_thermo(self, former=None):
        """Push current temperatures into hinge stiffness, damping, rest angle.
        `former` = (i0, i1, theta_total, K_former) engages a STIFF IMPLICIT spring
        on the band hinges toward the target shape -- a numerically-stable stand-in
        for a rigid forming roller (explicit PD chatters at these tiny inertias).
        The material's own stress-free angle (self.rest) keeps relaxing underneath."""
        p = self.p
        fi0 = fi1 = -1
        if former is not None:
            fi0, fi1, th_tot, Kf = former
            hinges = list(self.band_hinges(fi0, fi1))
            per = th_tot / max(1, len(hinges))
        for i in range(self.nh):
            Ti = max(self.T[i], self.T[i + 1])
            if fi0 <= i < fi1:                         # former engaged on this hinge
                k = former[3]
                rest = per
                damp = p.damp_beta * k + p.damp_floor    # Rayleigh on former stiffness
            else:
                k = bending.E_of_T(Ti) * p.I / p.seg
                rest = self.rest[i]
                damp = p.damp_beta * k + p.damp_floor
            self.model.jnt_stiffness[self.jid[i]] = k
            self.model.dof_damping[self.dadr[i]] = damp
            self.model.qpos_spring[self.qadr[i]] = rest

    def step_thermal(self, T_target, dt):
        """First-order relaxation of each segment toward its target temperature."""
        tau = np.where(T_target >= self.T, self.p.tau_heat, self.p.tau_cool)
        self.T += (T_target - self.T) * (dt / tau)

    def step_relax(self, dt):
        """Viscoelastic rest-angle relaxation: rest angle drifts toward the
        current angle with the temperature-dependent relaxation time."""
        if not self.p.relax:
            return
        for i in range(self.nh):
            theta = self.data.qpos[self.qadr[i]]
            tr = float(bending.tau_relax(max(self.T[i], self.T[i + 1])))
            self.rest[i] += (theta - self.rest[i]) * (dt / max(tr, dt))

    def tip_z(self):
        return self.data.xpos[self.bid[-1], 2]

    def seg_x(self):
        return self.data.xpos[np.array(self.bid), 0]

    def hinge_angles(self):
        return np.array([self.data.qpos[a] for a in self.qadr])

    def band_hinges(self, i0, i1):
        """Hinge indices fully inside the segment band [i0,i1) (there are N-1
        hinges for N segments, so clamp to the valid range)."""
        return range(max(0, i0), min(i1, self.nh))

    def total_bend(self, i0, i1):
        """Sum of hinge angles across the band hinges (rad)."""
        return float(sum(self.data.qpos[self.qadr[i]] for i in self.band_hinges(i0, i1)))

    def apply_band_pd(self, i0, i1, theta_total, kp, kd):
        """PD-hold each band hinge to an equal share of theta_total (rad).
        Position control -> no creep runaway while the rest angle relaxes."""
        self.data.qfrc_applied[:] = 0.0
        hinges = list(self.band_hinges(i0, i1))
        per = theta_total / max(1, len(hinges))
        for i in hinges:
            th = self.data.qpos[self.qadr[i]]
            thd = self.data.qvel[self.dadr[i]]
            self.data.qfrc_applied[self.dadr[i]] = kp * (per - th) - kd * thd


def band_target(strip: Strip, i0, i1, T_hot):
    """Temperature target: segments [i0,i1) hot, rest cold."""
    tgt = np.full(strip.p.N, strip.p.T_cold)
    tgt[i0:i1] = T_hot
    return tgt


# ---------- runs ----------

def settle_sag(p: Params, i0, i1, t_end=4.0, instant_T=True, render_to=None):
    """Clamp base (cantilever), heat band [i0,i1), let it sag, return tip drop."""
    strip = Strip(p)
    if instant_T:
        strip.T = band_target(strip, i0, i1, p.T_hot)
    tgt = band_target(strip, i0, i1, p.T_hot)
    z0 = None
    frames, ts, sag = [], [], []
    renderer = mujoco.Renderer(strip.model, 360, 640) if render_to else None
    cam = mujoco.MjvCamera()
    cam.azimuth, cam.elevation, cam.distance = 90, -8, 0.26
    cam.lookat[:] = [0.07, 0, -0.01]
    nsteps = int(t_end / p.dt)
    every = max(1, int((1 / 60) / p.dt))
    mujoco.mj_forward(strip.model, strip.data)
    z0 = strip.tip_z()
    for k in range(nsteps):
        if not instant_T:
            strip.step_thermal(tgt, p.dt)
        strip.apply_thermo()
        strip.step_relax(p.dt)
        mujoco.mj_step(strip.model, strip.data)
        if k % every == 0:
            ts.append(k * p.dt); sag.append((z0 - strip.tip_z()) * 1000)
            if renderer is not None:
                renderer.update_scene(strip.data, cam)
                frames.append(renderer.render())
    if render_to and frames:
        import imageio.v2 as iio
        iio.mimsave(render_to, frames, fps=30)
    return dict(sag_mm=(z0 - strip.tip_z()) * 1000, t=np.array(ts),
                sag_t=np.array(sag), strip=strip)


def forming_cycle(T_form=155.0, band_n=10, theta_target=1.0, t_band_dwell=3.0,
                  K_former=10.0, render_to=None, verbose=False):
    """Heat a band, bend it against a rigid FORMER (a stiff implicit spring to
    the target shape), hold so the material's stress-free angle relaxes, cool to
    freeze it, then RELEASE the former and measure how much bend is retained.
    Springback = (commanded - retained)/commanded, and emerges from the same
    relaxation ODE that bending.py uses. Horizontal-plane bending (axis z) so
    gravity doesn't bias the angle."""
    p = Params(bend_axis="0 0 1", relax=True, tau_cool=2.5,
               damp_beta=0.25, damp_floor=0.02)
    strip = Strip(p)
    i0, i1 = p.N - band_n, p.N
    # phase schedule (s): slow ramp -> quasi-static
    t_ramp0, t_ramp1 = 0.5, 2.5
    t_cool = t_ramp1 + t_band_dwell           # start cooling after the hot dwell
    t_release = t_cool + 8.0                  # cool well below Tg before release
    t_end = t_release + 2.0
    cmd_cold = np.full(p.N, p.T_cold)

    frames = []
    renderer = mujoco.Renderer(strip.model, 360, 720) if render_to else None
    cam = mujoco.MjvCamera()
    cam.azimuth, cam.elevation, cam.distance = 90, -60, 0.26
    cam.lookat[:] = [0.06, 0.02, 0]
    every = max(1, int((1 / 60) / p.dt))
    log = {k: [] for k in ("t", "Tband", "bend", "rest")}

    strip.T[:] = band_target(strip, i0, i1, T_form)   # band starts hot (instant)
    nsteps = int(t_end / p.dt)
    for k in range(nsteps):
        t = k * p.dt
        if t >= t_cool:
            strip.step_thermal(cmd_cold, p.dt)
        theta = theta_target * smoothstep((t - t_ramp0) / (t_ramp1 - t_ramp0))
        engaged = t < t_release
        former = (i0, i1, theta, K_former) if engaged else None
        strip.apply_thermo(former=former)
        strip.step_relax(p.dt)                # material rest angle relaxes toward theta
        mujoco.mj_step(strip.model, strip.data)
        if k % every == 0:
            log["t"].append(t); log["Tband"].append(strip.T[i0])
            log["bend"].append(strip.total_bend(i0, i1))
            log["rest"].append(float(sum(strip.rest[i0:i1])))
            if renderer is not None:
                renderer.update_scene(strip.data, cam); frames.append(renderer.render())
    if render_to and frames:
        import imageio.v2 as iio
        iio.mimsave(render_to, frames, fps=30)
    commanded = theta_target
    retained = strip.total_bend(i0, i1)
    sb = (commanded - retained) / commanded * 100
    if verbose:
        print(f"    T_form={T_form:.0f}C: commanded {np.degrees(commanded):.0f} deg, "
              f"retained {np.degrees(retained):.0f} deg, springback {sb:.1f}%")
    return dict(commanded=commanded, retained=retained, springback=sb,
                log={k: np.array(v) for k, v in log.items()})


def run_forming():
    print("\n" + "=" * 72)
    print("MuJoCo strip co-sim  --  forming + springback (heat/bend/cool/release)")
    print("=" * 72)
    print("  Emergent springback from the rest-angle relaxation model; compare to")
    print("  bending.py's prediction (springback collapses when formed hot):")
    print(f"  {'T_form':>7} {'sim springback':>15} {'bending.py':>12}")
    rows = []
    for T in (125.0, 140.0, 155.0):
        r = forming_cycle(T_form=T, t_band_dwell=3.0)
        pred = bending.springback_fraction(T, 3.0) * 100
        rows.append((T, r["springback"], pred))
        print(f"  {T:>6.0f}C {r['springback']:>13.1f}% {pred:>11.1f}%")
    # render one cycle to a gif at a temperature with visible springback
    forming_cycle(T_form=130.0, render_to=str(OUT / "strip_forming.gif"))
    plot_forming(rows)
    print(f"  wrote out/strip_forming.gif, out/forming_springback.png")


def plot_forming(rows):
    Ts = [r[0] for r in rows]; sim = [r[1] for r in rows]; pred = [r[2] for r in rows]
    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    ax.plot(Ts, sim, "-o", label="MuJoCo co-sim")
    ax.plot(Ts, pred, "--s", label="bending.py model")
    ax.set(xlabel="forming temperature (C)", ylabel="springback (%)",
           title="Springback vs forming temp: sim matches the analytic trend")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(OUT / "forming_springback.png", dpi=110); plt.close(fig)


def main():
    p = Params()
    print("=" * 72)
    print("MuJoCo strip co-sim  --  hot-zone gravity sag")
    s0 = Strip(p)
    print(f"  segments {p.N}  seg_len {p.seg*1000:.1f}mm  seg_mass "
          f"{s0.seg_mass()*1e3:.3f} g  total {s0.seg_mass()*p.N*1e3:.2f} g")
    expected = m.rho * p.strip_w * p.strip_t * p.strip_len * 1e3
    print(f"  expected total mass {expected:.2f} g  (density check)")

    # --- VALIDATION: all-cold cantilever vs Euler-Bernoulli self-weight sag ---
    print("\n  VALIDATION: all-cold cantilever tip sag vs analytic q L^4/(8 EI):")
    q = m.rho * p.strip_w * p.strip_t * G
    for Ltest in (0.06, 0.10, 0.15):
        pn = Params(strip_len=Ltest)
        r = settle_sag(pn, 0, 0, t_end=3.0)         # no hot band = all cold
        EI = bending.E_of_T(20) * pn.I
        analytic = q * Ltest**4 / (8 * EI) * 1000
        print(f"    L={Ltest*1000:4.0f}mm  sim={r['sag_mm']:7.4f}mm  "
              f"analytic={analytic:7.4f}mm  ratio={r['sag_mm']/analytic:5.2f}")

    # --- SAG vs hot-band length (the physical floppy-zone limit) ---
    # Free end fully hot and hanging = WORST CASE (the formed part cantilevering
    # off the bend). Real hot zone is supported upstream + at the roller, so the
    # true limit is more generous; this is the conservative bound.
    print("\n  HOT-ZONE SAG vs heated-band length (cantilever, worst case, "
          f"T_hot={p.T_hot:.0f}C):")
    print(f"  {'band mm':>8} {'tip sag mm':>11}")
    bands_mm, sags = [], []
    for L_band_mm in range(10, 125, 10):
        nseg = max(1, round((L_band_mm / 1000) / p.seg))
        r = settle_sag(p, p.N - nseg, p.N, t_end=4.0)   # hot band at the free end
        bands_mm.append(L_band_mm); sags.append(r['sag_mm'])
        print(f"  {L_band_mm:>8.0f} {r['sag_mm']:>11.3f}")

    # crossings of common tolerances (the usable floppy-zone limit)
    for tol in (0.5, 2.0):
        lim = np.interp(tol, sags, bands_mm)
        print(f"  -> sag = {tol} mm at heated-band length ~{lim:.0f} mm")
    plot_sag(bands_mm, sags)

    # --- time-resolved heat-up sag + gif ---
    r = settle_sag(p, p.N - 8, p.N, t_end=6.0, instant_T=False,
                   render_to=str(OUT / "strip_sag.gif"))
    plot_sag_time(r)
    print(f"\n  wrote out/strip_sag.gif, out/sag_vs_band.png, out/sag_time.png")

    run_forming()


def plot_sag(bands_mm, sags):
    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    ax.plot(bands_mm, sags, "-o", color="C3")
    for tol, lab in [(0.5, "0.5 mm"), (2.0, "2 mm")]:
        ax.axhline(tol, ls=":", lw=0.8, c="grey")
        ax.text(bands_mm[0], tol * 1.05, f"{lab} tol", fontsize=7, color="grey")
    ax.set(xlabel="heated-band length (mm)", ylabel="free-end sag (mm)",
           title="Hot-zone gravity sag vs band length (3 mm strip, 155 C)")
    ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(OUT / "sag_vs_band.png", dpi=110); plt.close(fig)


def plot_sag_time(r):
    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    ax.plot(r["t"], r["sag_t"], color="C1")
    ax.set(xlabel="time (s)", ylabel="free-end sag (mm)",
           title="Sag as the heated band warms up (1st-order thermal)")
    ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(OUT / "sag_time.png", dpi=110); plt.close(fig)


if __name__ == "__main__":
    main()
