"""machine_interactive.py -- live MuJoCo viewer of the whole acrylic bender.

Watch the strip feed through the machine in real time: it warms blue->red across
the pre-heat zone, gets curved at the bend rollers, and freezes its curve as it
cools past the chilled set roller. Two sliders in the viewer's Control pane
(gear=0 proxy actuators -> pure UI):

    ui_feed  : feed speed (m/s)
    ui_kappa : commanded curvature 1/m  (sign = bend up/down; magnitude = 1/radius)

Drag ui_kappa WHILE it feeds to draw an arbitrary varying-radius curve -- the
whole point of the machine. The strip recycles automatically once it fully exits.

Needs a windowed GL backend (glfw); forced before importing the sim modules.
Run `python machine_interactive.py --selftest` to verify the path headless.
"""
import os
os.environ.setdefault("MUJOCO_GL", "glfw")
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import mujoco
import mujoco.viewer
import machine_sim as MS
import strip_sim as S

SUBSTEPS = 40


def aid(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)


def main():
    selftest = "--selftest" in sys.argv
    g = MS.Machine()
    p = g.params()
    strip = S.Strip(p, xml=MS.build_machine_xml(g, ui=True))
    strip.T[:] = g.T_cold
    a_feed = aid(strip.model, "ui_feed")
    a_kappa = aid(strip.model, "ui_kappa")

    state = {"disp": 0.0}
    recycle_at = g.quench[1] + 0.01 - g.x0            # trailing end fully exits

    def reset():
        state["disp"] = 0.0
        strip.data.qpos[:] = 0.0
        strip.data.qvel[:] = 0.0
        strip.rest[:] = 0.0
        strip.T[:] = g.T_cold
        mujoco.mj_forward(strip.model, strip.data)

    def seed():
        strip.data.ctrl[a_feed] = g.feed_speed
        strip.data.ctrl[a_kappa] = 0.6 * g.kappa_max

    def step():
        speed = float(strip.data.ctrl[a_feed])
        kappa = float(strip.data.ctrl[a_kappa])
        state["disp"] += speed * p.dt
        strip.data.qpos[strip.feed_qadr] = state["disp"]
        strip.data.qvel[strip.feed_dadr] = speed
        MS.machine_step(strip, g, kappa, p.dt)
        if state["disp"] > recycle_at:
            reset()

    seed()
    reset()
    seed()

    if selftest:
        for _ in range(int(20.0 / p.dt)):            # feed a full strip through
            step()
        ok = np.isfinite(strip.data.qpos).all()
        total = np.degrees(np.abs(np.array(strip.hinge_angles()))).sum()
        print(f"selftest {'OK' if ok else 'FAILED'}: finite={ok}, "
              f"formed curve currently in machine = {total:.0f} deg")
        sys.exit(0 if ok else 1)

    print(__doc__)
    with mujoco.viewer.launch_passive(strip.model, strip.data) as v:
        seed()
        v.cam.azimuth, v.cam.elevation, v.cam.distance = 90, -14, 0.36
        v.cam.lookat[:] = [0.085, 0, -0.01]
        while v.is_running():
            for _ in range(SUBSTEPS):
                step()
            strip.color_by_temp()
            v.sync()


if __name__ == "__main__":
    main()
