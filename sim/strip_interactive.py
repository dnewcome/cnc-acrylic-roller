"""strip_interactive.py -- interactive MuJoCo viewer for the acrylic strip.

Drag two sliders in the viewer's Control pane (they are gear=0 proxy actuators
-> pure UI, they move nothing mechanically):

    ui_heat  : heated-band target temperature (C)
    ui_bend  : forming-roller target bend angle (rad); 0 = roller retracted

Workflow to feel the whole process:
    1. raise ui_heat toward ~155 C  -> the band goes soft (rubbery above Tg)
    2. raise ui_bend                -> the rigid former curls the hot band
    3. lower ui_heat back to ~20 C  -> the curve FREEZES in (rest angle locks)
    4. lower ui_bend to 0 (release) -> SPRINGBACK: it holds the formed curve if
                                       it cooled, springs flat if it was still hot

The interactive viewer needs a windowed GL backend (glfw), so this module forces
it BEFORE importing the shared sim (which otherwise defaults to headless osmesa).
Run `python strip_interactive.py --selftest` to verify the physics path headless.
"""
import os
os.environ.setdefault("MUJOCO_GL", "glfw")     # windowed; BEFORE importing strip_sim
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import mujoco
import mujoco.viewer        # module-level so it doesn't shadow `mujoco` inside main()
import strip_sim as S

K_FORMER = 10.0
BAND_N = 12
SUBSTEPS = 60                                   # physics steps per rendered frame


def build():
    # fast heat-up (tau_heat small) so the slider feels responsive
    p = S.Params(bend_axis="0 0 1", relax=True, tau_cool=2.0, tau_heat=2.0,
                 damp_beta=0.25, damp_floor=0.02)
    strip = S.Strip(p, ui=True)
    strip.T[:] = p.T_cold
    return p, strip


def aid(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)


def main():
    selftest = "--selftest" in sys.argv
    p, strip = build()
    i0, i1 = p.N - BAND_N, p.N
    a_heat = aid(strip.model, "ui_heat")
    a_bend = aid(strip.model, "ui_bend")

    def seed():
        strip.data.ctrl[a_heat] = p.T_cold
        strip.data.ctrl[a_bend] = 0.0

    def step():
        T_band = float(strip.data.ctrl[a_heat])
        bend = float(strip.data.ctrl[a_bend])
        strip.step_thermal(S.band_target(strip, i0, i1, T_band), p.dt)
        former = (i0, i1, bend, K_FORMER) if bend > 0.02 else None
        strip.apply_thermo(former=former)
        strip.step_relax(p.dt)
        mujoco.mj_step(strip.model, strip.data)

    seed()
    mujoco.mj_forward(strip.model, strip.data)

    if selftest:
        # exercise heat -> bend -> cool -> release without opening a window.
        # Force the band hot instantly so the short test budget still forms it.
        strip.T[i0:i1] = p.T_hot
        strip.data.ctrl[a_heat] = p.T_hot
        strip.data.ctrl[a_bend] = 1.0
        for _ in range(15000):            # bend + hot dwell -> rest angle relaxes
            step()
        strip.data.ctrl[a_heat] = p.T_cold
        for _ in range(40000):            # cool below Tg -> freeze the shape
            step()
        strip.data.ctrl[a_bend] = 0.0
        for _ in range(8000):             # release the former
            step()
        ok = np.isfinite(strip.data.qpos).all()
        retained = np.degrees(strip.total_bend(i0, i1))
        held = retained > 40                      # formed-then-cooled should hold
        print(f"selftest {'OK' if ok and held else 'FAILED'}: finite={ok}, "
              f"retained bend after release = {retained:.0f} deg "
              f"(formed hot then cooled -> expect a held curve ~57 deg)")
        sys.exit(0 if ok and held else 1)

    print(__doc__)
    with mujoco.viewer.launch_passive(strip.model, strip.data) as v:
        seed()                                  # sliders open at sensible defaults
        v.cam.azimuth, v.cam.elevation, v.cam.distance = 90, -55, 0.28
        v.cam.lookat[:] = [0.06, 0.02, 0]
        while v.is_running():
            for _ in range(SUBSTEPS):
                step()
            v.sync()


if __name__ == "__main__":
    main()
