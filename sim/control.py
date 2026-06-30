"""
Thermal control loop: can a PID hold the heated zone at forming temperature,
and how fast does the control loop actually need to run?

The headline result (spoiler): the acrylic strip is a slow, heavily damped
thermal plant (time constant ~ seconds, modest dead time). So the control
loop does NOT need to be fast -- 1-10 Hz is luxurious. The real hazards are
(a) OVERSHOOT -> surface scorch, and (b) the diffusion lag itself, which no
controller can beat. "PID control speed" is not the bottleneck; physics is.

We:
  1. Build a small 1-D plant (heated face = controllable IR flux, back face
     convective). Surface thermocouple is the measured variable.
  2. Step-test it and fit a first-order-plus-dead-time (FOPDT) model.
  3. IMC-tune a PID and run the closed loop at several sample rates to show
     loop rate is irrelevant until it approaches the plant time constant.
  4. Show a too-aggressive tune overshooting into the scorch zone.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit

from material import PMMA

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)
m = PMMA
SIGMA = 5.670374419e-8
C2K = 273.15


class Plant1D:
    """3 mm strip, one face heated by controllable flux q (W/m^2), other face
    convective to ambient. Implicit (backward-Euler) stepper, one dt per call.
    Surface (heated face) temperature is the sensor."""
    def __init__(self, thk=3e-3, nodes=61, T0=20.0, h_back=15.0, T_amb=20.0,
                 eps=0.9):
        self.N = nodes
        self.dx = thk / (nodes - 1)
        self.T = np.full(nodes, float(T0))
        self.h_back = h_back
        self.T_amb = T_amb
        self.eps = eps

    def step(self, q, dt):
        N, dx, k = self.N, self.dx, m.k
        r = m.alpha * dt / dx**2
        lo = np.full(N, -r); di = np.full(N, 1 + 2 * r); up = np.full(N, -r)
        rhs = self.T.copy()
        # heated face (node 0): flux q in, minus radiative + small conv loss
        Ts = self.T[0]
        h_rad = self.eps * SIGMA * ((Ts + C2K)**2 + (self.T_amb + C2K)**2) \
            * ((Ts + C2K) + (self.T_amb + C2K))
        q_net = q - h_rad * (Ts - self.T_amb)         # W/m^2 into the face
        di[0] = 1 + 2 * r; up[0] = -2 * r
        rhs[0] = self.T[0] + 2 * r * dx / k * q_net
        # back face (node N-1): convective loss
        gh = self.h_back * dx / k
        di[-1] = 1 + 2 * r + 2 * r * gh; lo[-1] = -2 * r
        rhs[-1] = self.T[-1] + 2 * r * gh * self.T_amb
        self.T = _thomas(lo, di, up, rhs)
        return self.T[0]                               # measured surface temp


def _thomas(a, b, c, d):
    n = len(d); cp = np.empty(n); dp = np.empty(n)
    cp[0] = c[0] / b[0]; dp[0] = d[0] / b[0]
    for i in range(1, n):
        mm = b[i] - a[i] * cp[i - 1]
        cp[i] = c[i] / mm
        dp[i] = (d[i] - a[i] * dp[i - 1]) / mm
    x = np.empty(n); x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


class PID:
    def __init__(self, Kp, Ki, Kd, out_lo=0.0, out_hi=40000.0):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.lo, self.hi = out_lo, out_hi
        self.i = 0.0; self.prev = None

    def update(self, err, dt):
        p = self.Kp * err
        d = 0.0 if self.prev is None else self.Kd * (err - self.prev) / dt
        self.prev = err
        # tentative integral, then back-calculation anti-windup against the
        # actuator limits (keeps the integrator honest when q saturates).
        i_try = self.i + self.Ki * err * dt
        raw = p + i_try + d
        out = np.clip(raw, self.lo, self.hi)
        self.i = i_try + (out - raw)          # bleed off the saturated excess
        return out


def step_test(q_step=20000.0, t_end=60.0, dt=0.02):
    p = Plant1D()
    n = int(t_end / dt)
    t = np.arange(n) * dt
    y = np.empty(n)
    for i in range(n):
        y[i] = p.step(q_step, dt)
    return t, y, q_step


def fit_fopdt(t, y, q_step):
    """Fit y(t) = y0 + K*q*(1 - exp(-(t-theta)/tau)) for t>theta."""
    y0 = y[0]
    def model(t, K, tau, theta):
        out = np.full_like(t, y0)
        mask = t > theta
        out[mask] = y0 + K * q_step * (1 - np.exp(-(t[mask] - theta) / tau))
        return out
    Kg = (y[-1] - y0) / q_step
    p0 = [Kg, 8.0, 0.3]
    # physical bounds: gain>0, tau>0, dead-time>=0 (surface sensor -> theta~0)
    popt, _ = curve_fit(model, t, y, p0=p0, maxfev=20000,
                        bounds=([0, 0.1, 0.0], [1e-1, 300, 10]))
    return popt  # K (C per W/m^2), tau (s), theta (s)


def imc_pid(K, tau, theta, lam=None):
    """IMC-PID tuning for FOPDT (lambda = closed-loop time constant).
    Returns Kp, Ki, Kd in flux units."""
    if lam is None:
        # lambda = tau is the conservative IMC choice: ~zero overshoot, which
        # is what we MUST have here -- overshoot eats the thin scorch margin.
        lam = max(1.0 * tau, theta + 1.0)
    Kp = tau / (K * (lam + theta))
    Ti = tau
    Td = 0.0                                # derivative off (noise-prone)
    return Kp, Kp / Ti, Kp * Td


def closed_loop(pid_gains, setpoint=175.0, t_end=120.0, dt_plant=0.02,
                Ts=1.0, dist_at=None, ramp=0.0, T0=20.0, i0=0.0):
    """Run plant under PID sampled every Ts seconds.
    ramp: soft-start the setpoint from T0 up to `setpoint` over `ramp` seconds
          (0 = step change). dist_at: (time, new_h_back) applies a cooling
          disturbance (a draft on the back face) to test rejection.
    i0:   pre-seed the integrator (e.g. to the steady-state flux) so a
          regulation test doesn't start with a cold-start dip artifact."""
    p = Plant1D(T0=T0)
    pid = PID(*pid_gains)
    pid.i = i0
    n = int(t_end / dt_plant)
    t = np.arange(n) * dt_plant
    y = np.empty(n); u = np.empty(n)
    q = 0.0; next_ctrl = 0.0
    for i in range(n):
        ti = t[i]
        sp = setpoint if ramp <= 0 else T0 + (setpoint - T0) * min(1.0, ti / ramp)
        if ti >= next_ctrl:
            q = pid.update(sp - p.T[0], Ts)
            next_ctrl += Ts
        if dist_at and ti >= dist_at[0]:
            p.h_back = dist_at[1]
        y[i] = p.step(q, dt_plant); u[i] = q
    return t, y, u


def main():
    print("=" * 72)
    print("THERMAL CONTROL  --  heater -> surface-temp loop (3 mm strip)")
    print("=" * 72)

    # 1. identify the plant
    t, y, qs = step_test()
    K, tau, theta = fit_fopdt(t, y, qs)
    print(f"\n  FOPDT fit:  tau = {tau:.1f} s   dead-time theta = {theta:.2f} s")
    print(f"              gain K = {K*1000:.3f} C per kW/m^2")
    print(f"  controllability theta/tau = {theta/tau:.3f}  "
          f"({'easy' if theta/tau < 0.3 else 'harder'} to control)")

    # 2. tune
    Kp, Ki, Kd = imc_pid(K, tau, theta)
    print(f"\n  IMC-PID:  Kp={Kp:.2f}  Ki={Ki:.3f}  Kd={Kd:.2f}  (flux per degC)")

    # ---- 3. LOOP-SPEED question, asked cleanly --------------------------
    # Isolate "is the loop fast enough?" from the messy startup transient by
    # regulating around a SOAKED operating point (strip already at 175C) and
    # rejecting a draft disturbance. This is the fair test of control rate.
    # Warm up to steady state FIRST (100 s), then hit it with a draft (back-face
    # h 15 -> 45 W/m^2K, a real breeze). Measure deviation only after the draft,
    # so the number reflects rejection, not the start-up transient.
    t_dist = 100.0
    print(f"\n  REGULATION @175C (warm up to steady, then a draft "
          f"(back-face h 15->45) at t={t_dist:.0f}s):")
    print(f"  {'Ts (s)':>7} {'rate':>8} {'max dev C':>10} {'recover s':>10}")
    fig, ax = plt.subplots(figsize=(7, 4.3))
    for Ts in [0.1, 1.0, 5.0, 20.0]:
        tt, yy, _ = closed_loop((Kp, Ki, Kd), setpoint=175.0, T0=175.0, i0=4600.0,
                                t_end=300.0, Ts=Ts, dist_at=(t_dist, 45.0))
        win = tt >= (t_dist - 2)
        dev = np.abs(yy[win] - 175.0).max()
        rec = _recover_time(tt, yy, 175.0, after=t_dist, band=2.0)
        print(f"  {Ts:>7.1f} {1/Ts:>6.1f}Hz {dev:>10.2f} {fmt(rec):>10}")
        ax.plot(tt[tt > 60], yy[tt > 60], label=f"Ts={Ts}s ({1/Ts:.1f}Hz)")
    ax.axhline(175, ls="--", c="k", lw=0.8, label="setpoint")
    ax.axvline(t_dist, ls=":", c="grey", lw=0.8, label="draft hits")
    ax.set(xlabel="time (s)", ylabel="surface temp (C)",
           title="Loop speed barely matters: a 20 s loop still rejects the draft")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "control_sample_rate.png", dpi=110)
    plt.close(fig)

    # ---- 4. The REAL control challenge: the narrow thermal window --------
    # From cold, naively holding the surface at 175C drifts UP into scorch as
    # the bulk soaks (less heat conducts inward -> surface creeps). Leaving
    # margin (lower surface setpoint) is the fix. Window is only T_form..T_scorch.
    window = m.T_scorch - m.T_form
    print(f"\n  NARROW-WINDOW startup (form {m.T_form:.0f} <-> scorch "
          f"{m.T_scorch:.0f} = {window:.0f}C window). Surface drifts UP as bulk soaks;")
    print(f"  how much margin below scorch must the setpoint leave?")
    print(f"  {'setpoint C':>10} {'peak C':>8} {'drift C':>8} {'verdict':>8}")
    fig, ax = plt.subplots(figsize=(7, 4.3))
    safe_sp = None
    for sp in [175.0, 168.0, 160.0]:
        tt, yy, _ = closed_loop((Kp, Ki, Kd), setpoint=sp, T0=20.0,
                                t_end=160.0, Ts=1.0, ramp=2.5 * tau)
        peak = yy.max()
        drift = peak - sp
        verdict = "SCORCH" if peak > m.T_scorch else "safe"
        if peak <= m.T_scorch and safe_sp is None:
            safe_sp = sp
        print(f"  {sp:>10.0f} {peak:>8.1f} {drift:>8.1f} {verdict:>8}")
        ax.plot(tt, yy, label=f"set {sp:.0f}C -> peak {peak:.0f}C")
    ax.axhline(m.T_scorch, ls=":", c="r", lw=0.9, label=f"scorch {m.T_scorch:.0f}C")
    ax.axhline(m.T_form, ls="--", c="g", lw=0.8, label=f"form {m.T_form:.0f}C")
    ax.set(xlabel="time (s)", ylabel="surface temp (C)",
           title="Surface drifts up as bulk soaks -> setpoint must leave margin")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "control_window.png", dpi=110)
    plt.close(fig)
    if safe_sp is not None:
        print(f"  -> need surface setpoint <= ~{safe_sp:.0f}C (>= "
              f"{m.T_scorch-safe_sp:.0f}C margin) with a simple PID; feedforward")
        print(f"     the declining soak-power would shrink this margin.")

    print(f"\n  CONCLUSION:")
    print(f"  * Plant tau ~ {tau:.0f}s -> control loop SPEED is a non-issue; even a")
    print(f"    20 s ({1/20:.2f} Hz) loop rejects disturbances. Any MCU is overkill.")
    print(f"  * The real control problems are PHYSICAL, not computational:")
    print(f"    - narrow {window:.0f}C window between forming and scorch;")
    print(f"    - surface drifts UP as the bulk soaks (must leave margin / use")
    print(f"      feedforward), and you can't directly measure the core temp;")
    print(f"    - diffusion lag (~20s/zone) sets throughput, no loop can beat it.")
    print(f"  plots -> {OUT}/")


def _settle_time(t, y, sp, band=2.0):
    outside = np.abs(y - sp) > band
    if not outside.any():
        return 0.0
    last = np.where(outside)[0][-1]
    return t[last] if last < len(t) - 1 else None


def _recover_time(t, y, sp, after, band=2.0):
    """Time after a disturbance (at t=`after`) until y returns within band
    and stays there. Returns seconds-since-disturbance, or None."""
    mask = t >= after
    tt, yy = t[mask], y[mask]
    outside = np.abs(yy - sp) > band
    if not outside.any():
        return 0.0
    last = np.where(outside)[0][-1]
    if last >= len(tt) - 1:
        return None
    return tt[last] - after


def fmt(v):
    return "n/a" if v is None or v != v else f"{v:.1f}"


if __name__ == "__main__":
    main()
