"""seg_thermal.py -- REAL through-thickness conduction for every strip segment.

Replaces the lumped 1st-order temperature surrogate used in the cartoon machine.
Each segment carries a through-thickness temperature profile (Nz nodes) and is
advanced with the same validated implicit conduction physics as thermal.py, but
BATCHED over all segments so the whole strip's thermal state evolves at once.

Why it matters: acrylic's surface heats fast while the CORE lags by the
diffusion time (~thickness^2). Forming must be gated on the CORE reaching
forming temperature, or you "form" a hot skin over a cold stiff core -> it
springs back. With this model, feeding too fast for the thickness leaves the
core cold and the strip refuses to form -- the honest coupling.

Boundary condition per face is Robin: -k dT/dx = h*(T_inf - Ts). A large h with
a surface setpoint approximates a well-controlled (surface-clamped) heater; a
moderate h to coolant models air/contact quench. Radiation can be folded into
an effective h by the caller.
"""
import numpy as np
from material import PMMA

m = PMMA


class SegThermal:
    """Batched 1-D through-thickness conduction for Nseg segments.

    T has shape (Nseg, Nz). Faces are node 0 and node Nz-1. step() takes
    per-segment, per-face (h, T_inf) arrays and advances one implicit step."""

    def __init__(self, nseg, thickness, nz=11, T_init=20.0):
        self.nseg = nseg
        self.nz = nz
        self.L = thickness
        self.dx = thickness / (nz - 1)
        self.T = np.full((nseg, nz), float(T_init))

    @property
    def core(self):                       # coldest-on-heating / hottest-on-cooling
        return self.T[:, self.nz // 2]
    @property
    def surf(self):
        return np.maximum(self.T[:, 0], self.T[:, -1])
    @property
    def mean(self):
        return self.T.mean(axis=1)

    def step(self, dt, h0, Tinf0, hL, TinfL):
        """One backward-Euler step. h0/Tinf0 = face at x=0 (per segment),
        hL/TinfL = face at x=L. All arrays shape (nseg,) or scalars."""
        nz, dx, k = self.nz, self.dx, m.k
        r = m.alpha * dt / dx**2
        ns = self.nseg
        h0 = np.broadcast_to(np.asarray(h0, float), (ns,))
        hL = np.broadcast_to(np.asarray(hL, float), (ns,))
        Tinf0 = np.broadcast_to(np.asarray(Tinf0, float), (ns,))
        TinfL = np.broadcast_to(np.asarray(TinfL, float), (ns,))

        lo = np.full((ns, nz), -r)
        di = np.full((ns, nz), 1 + 2 * r)
        up = np.full((ns, nz), -r)
        rhs = self.T.copy()
        # x=0 face half-cell balance
        g0 = h0 * dx / k
        di[:, 0] = 1 + 2 * r + 2 * r * g0
        up[:, 0] = -2 * r
        rhs[:, 0] = self.T[:, 0] + 2 * r * g0 * Tinf0
        # x=L face
        gL = hL * dx / k
        di[:, -1] = 1 + 2 * r + 2 * r * gL
        lo[:, -1] = -2 * r
        rhs[:, -1] = self.T[:, -1] + 2 * r * gL * TinfL
        self.T = _batched_thomas(lo, di, up, rhs)


def _batched_thomas(a, b, c, d):
    """Solve nseg independent tridiagonal systems of size nz. Arrays (nseg,nz).
    Vectorized over segments; loops only over the (small) nz axis."""
    n = b.shape[1]
    cp = np.empty_like(b); dp = np.empty_like(b)
    cp[:, 0] = c[:, 0] / b[:, 0]
    dp[:, 0] = d[:, 0] / b[:, 0]
    for i in range(1, n):
        mden = b[:, i] - a[:, i] * cp[:, i - 1]
        cp[:, i] = c[:, i] / mden
        dp[:, i] = (d[:, i] - a[:, i] * dp[:, i - 1]) / mden
    x = np.empty_like(b)
    x[:, -1] = dp[:, -1]
    for i in range(n - 2, -1, -1):
        x[:, i] = dp[:, i] - cp[:, i] * x[:, i + 1]
    return x


# ---- validation: does it reproduce thermal.py's thickness-dependent times? ----
if __name__ == "__main__":
    import thermal as th
    print("SegThermal validation -- through-heat (core 20->155C, surface clamped"
          " 170C) vs thermal.py:")
    print(f"  {'thk':>5} {'segThermal s':>13} {'thermal.py s':>13}")
    H_CLAMP = 3000.0      # large h -> surface ~ setpoint (well-controlled heater)
    for thk_mm in (2.0, 3.0, 4.0):
        thk = thk_mm / 1000.0
        st = SegThermal(1, thk, nz=15, T_init=20.0)
        dt = min(0.02, (thk**2 / m.alpha) / 3000)
        t = 0.0
        while st.core[0] < m.T_form and t < 400:
            st.step(dt, H_CLAMP, 170.0, H_CLAMP, 170.0)
            t += dt
        # reference from thermal.py
        hot = th.contact_shoe(170.0)
        res = th.solve(m, thk, hot, hot, T_init=20.0,
                       t_end=4 * thk**2 / m.alpha, dt=dt, nodes=81)
        tref, _, _ = th.time_to_through_heat(res, m.T_form, 999)
        print(f"  {thk_mm:>4.1f}m {t:>12.1f} {tref:>13.1f}")
    print("  (should agree within a few %, and scale ~thickness^2)")
