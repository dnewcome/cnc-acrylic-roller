# cnc-acrylic-roller — kickoff brief

- **Problem:** Make sculpture from acrylic strips bent into arbitrary **3D space curves**
  (curve + twist), under computer control — but it's unknown whether the
  heat → bend → cool process is even physically feasible.

- **Done looks like:** A feasibility model that, given a strip spec (e.g. 25×3 mm PMMA),
  outputs the governing numbers — forming temp, heat/quench dwell times, achievable min
  radius before crazing, springback %, and resulting feed rate — and a clear **verdict on
  where the process breaks**.

- **Not now:** Machine CAD, the physical rig, control/toolpath software, and the twist /
  3rd axis hardware. Twist-mode springback is *modeled later*, not in the first pass.

- **First slice:** A physics-first feasibility study of a single moving
  **heat → bend → quench** zone: thermal model (local heat-up + cool-below-HDT timing),
  single-axis bend mechanics + springback → derive feed rate and min radius → feasibility
  verdict. Parametric so strip size / material are inputs.

- **Open question (riskiest):** Can a local zone be heated to ~155 °C and quenched back
  below ~85 °C fast enough to "freeze in" a *varying* curve **without crazing or scorch** —
  and is springback predictable enough to control open-loop? (Water-quench crazing is the
  prime suspect.)

---
_Kickoff: 2026-06-29. Decisions: feasibility model first; ultimate target is 3D space curves._
