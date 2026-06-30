"""
1-D transient conduction across the thickness of an acrylic strip.

This is the heart of the feasibility study. Acrylic is a thermal *insulator*
(alpha ~ 1e-7 m^2/s), so heating or cooling a strip THROUGH ITS THICKNESS is
slow and is what limits the whole machine -- not the motors, not the heaters.

Model
-----
Solve   rho*cp dT/dt = d/dx( k dT/dx )   on  0 <= x <= L  (the thickness).
Backward-Euler (implicit) finite difference -> unconditionally stable, so we
can use a sane dt even with a fine mesh. Tridiagonal solve each step.

Surface boundary conditions are Robin (mixed):
    -k dT/dx |surface = h_conv*(T_drive - Ts) + eps*sigma*(T_rad^4 - Ts^4)
which covers, with one form:
    * forced hot/cold air   -> h_conv, T_drive = air temp,   T_rad = ambient
    * radiant IR heater     -> T_rad = emitter temp (+ small h_conv)
    * contact shoe/roller   -> very large h_conv, T_drive = shoe temp (~Dirichlet)
The radiative term is linearised about the previous step's surface temp
(semi-implicit) -- standard and accurate for these temperature swings.

Two surfaces (x=0 and x=L) get independent BCs, so we can model two-sided
heating (symmetric) or one-sided (other face adiabatic / against a roller).
"""

from dataclasses import dataclass, field
import numpy as np

SIGMA = 5.670374419e-8  # Stefan-Boltzmann, W/(m^2 K^4)
C2K = 273.15


@dataclass
class Surface:
    """One face's boundary condition. Temperatures in deg C."""
    h_conv: float          # convective coefficient W/(m^2 K)
    T_drive: float         # convective driving temp (air/shoe), deg C
    T_rad: float = 20.0    # radiative environment/emitter temp, deg C
    eps: float = 0.0       # radiative exchange effectiveness (0 disables)
    adiabatic: bool = False


# ---- Boundary-condition presets (engineering ballparks) --------------------

def radiant_heat(emitter_C: float, eps_eff: float = 0.5, h_air: float = 10.0):
    """Ceramic/quartz IR emitter facing the strip. eps_eff folds in emissivity
    and view factor (how much of the emitter's radiosity actually lands)."""
    return Surface(h_conv=h_air, T_drive=20.0, T_rad=emitter_C, eps=eps_eff)

def hot_air(air_C: float, h: float = 120.0):
    """Forced hot-air convective heating (heat gun / blower)."""
    return Surface(h_conv=h, T_drive=air_C, T_rad=air_C, eps=0.0)

def contact_shoe(shoe_C: float, h: float = 2000.0):
    """Heated contact shoe/roller -> near-Dirichlet at the surface."""
    return Surface(h_conv=h, T_drive=shoe_C, T_rad=20.0, eps=0.0)

def air_quench(air_C: float = 20.0, h: float = 120.0):
    """Forced ambient (or chilled) air blast."""
    return Surface(h_conv=h, T_drive=air_C, T_rad=air_C, eps=0.0)

def mist_or_water_quench(water_C: float = 18.0, h: float = 1500.0):
    """Water mist / spray. Big h, but craze/condensation risk (flagged)."""
    return Surface(h_conv=h, T_drive=water_C, T_rad=water_C, eps=0.0)

def adiabatic():
    return Surface(h_conv=0.0, T_drive=20.0, adiabatic=True)


@dataclass
class ThermalResult:
    t: np.ndarray              # time vector, s
    x: np.ndarray              # node positions across thickness, m
    T: np.ndarray              # T[i, j] temp at time i, node j, deg C
    @property
    def T_core(self):  # slowest-moving interior point (min during heat)
        return self.T.min(axis=1)
    @property
    def T_surf_max(self):
        return self.T.max(axis=1)
    @property
    def T_center(self):
        return self.T[:, self.T.shape[1] // 2]


def _h_eff(s: Surface, Ts_C: float):
    """Effective surface conductance and driving temp, linearising radiation
    about Ts. Returns (h, T_inf) so that q = h*(T_inf - Ts)."""
    if s.adiabatic:
        return 0.0, Ts_C
    h = s.h_conv
    T_inf_num = s.h_conv * s.T_drive
    if s.eps > 0.0:
        Ts_K = Ts_C + C2K
        Tr_K = s.T_rad + C2K
        h_rad = s.eps * SIGMA * (Tr_K**2 + Ts_K**2) * (Tr_K + Ts_K)
        h += h_rad
        T_inf_num += h_rad * s.T_rad
    return h, (T_inf_num / h if h > 0 else Ts_C)


def solve(material, thickness_m, surf0: Surface, surfL: Surface,
          T_init=20.0, t_end=120.0, dt=0.05, nodes=61) -> ThermalResult:
    """Integrate the 1-D heat equation. surf0 is the x=0 face, surfL is x=L."""
    L = thickness_m
    N = nodes
    dx = L / (N - 1)
    x = np.linspace(0.0, L, N)
    a = material.alpha
    r = a * dt / dx**2                     # Fourier number per step
    rho_cp = material.rho * material.cp

    nsteps = int(round(t_end / dt))
    T = np.asarray(T_init, dtype=float)
    if T.ndim == 0:                        # scalar -> uniform field
        T = np.full(N, float(T_init))
    else:                                  # carry over a profile (e.g. heat->quench)
        assert T.shape == (N,), "T_init field must match node count"
        T = T.copy()
    out = np.empty((nsteps + 1, N))
    out[0] = T
    tvec = np.arange(nsteps + 1) * dt

    # Interior tridiagonal coefficients (backward Euler): constant.
    lower = np.full(N, -r)
    diag = np.full(N, 1.0 + 2.0 * r)
    upper = np.full(N, -r)

    for n in range(nsteps):
        lo = lower.copy(); di = diag.copy(); up = upper.copy()
        rhs = T.copy()

        # --- x=0 face (node 0): half-cell energy balance ---
        h0, Tinf0 = _h_eff(surf0, T[0])
        # half control volume: rho_cp*dx/2 dT/dt = k/dx (T1-T0) + h0(Tinf0-T0)
        ghost = h0 * dx / material.k       # Biot-like for this cell
        di[0] = 1.0 + 2.0 * r + 2.0 * r * ghost
        up[0] = -2.0 * r
        rhs[0] = T[0] + 2.0 * r * ghost * Tinf0

        # --- x=L face (node N-1) ---
        hL, TinfL = _h_eff(surfL, T[-1])
        ghostL = hL * dx / material.k
        di[-1] = 1.0 + 2.0 * r + 2.0 * r * ghostL
        lo[-1] = -2.0 * r
        rhs[-1] = T[-1] + 2.0 * r * ghostL * TinfL

        T = _thomas(lo, di, up, rhs)
        out[n + 1] = T

    return ThermalResult(t=tvec, x=x, T=out)


def _thomas(a, b, c, d):
    """Tridiagonal (Thomas) solver. a=sub, b=diag, c=super, d=rhs."""
    n = len(d)
    cp = np.empty(n); dp = np.empty(n)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n):
        m = b[i] - a[i] * cp[i - 1]
        cp[i] = c[i] / m
        dp[i] = (d[i] - a[i] * dp[i - 1]) / m
    xsol = np.empty(n)
    xsol[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        xsol[i] = dp[i] - cp[i] * xsol[i + 1]
    return xsol


# ---- analysis helpers ------------------------------------------------------

def time_to_through_heat(res: ThermalResult, T_form, T_scorch):
    """First time the COLDEST point reaches T_form. Also reports whether the
    surface ever exceeds the scorch ceiling before that (a process failure)."""
    coldest = res.T.min(axis=1)
    hottest = res.T.max(axis=1)
    idx = np.argmax(coldest >= T_form)
    if coldest[-1] < T_form:
        return None, hottest.max() > T_scorch, hottest.max()
    t_heat = res.t[idx]
    scorched = hottest[:idx + 1].max() > T_scorch
    return t_heat, scorched, hottest[:idx + 1].max()


def time_to_set(res: ThermalResult, T_set):
    """First time the HOTTEST point drops below T_set (curve frozen in)."""
    hottest = res.T.max(axis=1)
    idx = np.argmax(hottest <= T_set)
    if hottest[-1] > T_set:
        return None
    return res.t[idx]
