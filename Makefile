# CNC acrylic bender -- feasibility study.
# Analytic models + MuJoCo thermo-mechanical co-sim all live in sim/; this
# root Makefile just delegates so you can drive everything from the repo root.
#
#   make interactive   # <- open the live MuJoCo viewer (heat / bend sliders)
#
.PHONY: help interactive selftest sim models thermal control bending stations clean

help:
	@echo "make interactive  - LIVE full machine: feed the strip through, drag the"
	@echo "                    feed-speed & curvature sliders to draw a curve"
	@echo "make machine      - batch full-machine animation -> sim/out/machine.gif"
	@echo "make bench        - LIVE single-bend forming bench (heat & bend sliders)"
	@echo "make sim          - batch sag sweep + forming bench (plots + gifs)"
	@echo "make selftest     - headless check of both interactive viewers"
	@echo "make models       - run all analytic feasibility models"
	@echo "make rollers      - roller material: grip/slip + will-it-stick maps"
	@echo "make roller-contact - REAL measured roller force vs beam theory & temp"
	@echo "make {thermal|control|bending|stations|seg-thermal} - one analytic model"
	@echo "make clean        - remove generated plots/gifs in sim/out/"

interactive machine bench selftest sim thermal control bending stations \
rollers roller-contact seg-thermal clean:
	$(MAKE) -C sim $@

models:
	$(MAKE) -C sim all
