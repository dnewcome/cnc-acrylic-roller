"""
Material properties for cast PMMA (acrylic) used in the feasibility study.

All values are representative engineering figures for CAST acrylic at the
temperatures relevant to thermoforming. Properties are mildly temperature
dependent (notably c_p rises through the glass transition); we use
representative averages and flag this as a modelling simplification.

Sources (typical handbook ranges):
  density            1170-1200 kg/m^3
  specific heat      1400-1500 J/(kg K) solid, rising toward ~2000 above Tg
  thermal cond.      0.17-0.21 W/(m K)
  glass transition   ~105 C
  forming temp       150-165 C (rubbery, line-bend / drape forming range)
  heat deflection    ~95 C (cast, 1.8 MPa) -- below this the shape is "set"
  thermal degrade    starts ~200 C; surface bubbling/scorch risk above ~180 C
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Material:
    name: str
    rho: float          # kg/m^3   density
    cp: float           # J/(kg K) specific heat (representative average)
    k: float            # W/(m K)  thermal conductivity
    emissivity: float   # -        thermal-IR emissivity of the surface

    # Process temperatures (deg C)
    T_glass: float      # glass transition Tg
    T_form: float       # target forming temperature (formable, low stress)
    T_set: float        # below this the curve is "frozen in" (~ HDT)
    T_scorch: float     # surface ceiling before bubbling / degradation

    @property
    def alpha(self) -> float:
        """Thermal diffusivity alpha = k / (rho * cp)  [m^2/s]."""
        return self.k / (self.rho * self.cp)


# Cast PMMA, the workhorse for optical / sculptural acrylic.
PMMA = Material(
    name="Cast PMMA (acrylic)",
    rho=1180.0,
    cp=1500.0,          # solid-to-rubbery representative average
    k=0.19,
    emissivity=0.90,
    T_glass=105.0,
    T_form=155.0,
    T_set=85.0,         # conservative: comfortably below HDT before release
    T_scorch=180.0,
)


if __name__ == "__main__":
    m = PMMA
    print(f"{m.name}")
    print(f"  thermal diffusivity alpha = {m.alpha:.3e} m^2/s")
    print(f"  (for reference, alpha_steel ~ 1.2e-5, alpha_water ~ 1.4e-7,")
    print(f"   so acrylic ~ {1.2e-5/m.alpha:.0f}x slower than steel to heat through)")
