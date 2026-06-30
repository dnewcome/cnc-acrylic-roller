# Re-run the acrylic bender feasibility models and regenerate plots in out/.
PY ?= python

.PHONY: all thermal control bending stations rollers roller-contact seg-thermal \
        sim interactive selftest clean
all: thermal control bending stations rollers

thermal:
	$(PY) run_thermal.py

control:
	$(PY) control.py

bending:
	$(PY) bending.py

stations:
	$(PY) stations.py

# Roller / contact / material physics (the forces + grip + stick + thickness work)
rollers:                   # roller material model: grip/slip + will-it-stick maps
	$(PY) rollers.py

roller-contact:            # REAL measured roller contact force vs beam theory & temp
	$(PY) roller_contact.py

seg-thermal:               # validate batched through-thickness heat/cool vs thermal.py
	$(PY) seg_thermal.py

# MuJoCo thermo-mechanical co-sim
.PHONY: machine bench machine-selftest bench-selftest
sim:                       # batch: strip sag sweep + forming bench, plots + gifs
	$(PY) strip_sim.py

machine:                   # batch: full-machine animation (feed/heat/bend/quench)
	$(PY) machine_sim.py

interactive:               # LIVE full machine, feed + curvature sliders (needs display)
	$(PY) machine_interactive.py

bench:                     # LIVE single-bend forming bench, heat + bend sliders
	$(PY) strip_interactive.py

selftest: machine-selftest bench-selftest   # headless checks of both viewers
machine-selftest:
	MUJOCO_GL=osmesa $(PY) machine_interactive.py --selftest
bench-selftest:
	MUJOCO_GL=osmesa $(PY) strip_interactive.py --selftest

clean:
	rm -f out/*.png out/*.gif
