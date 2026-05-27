# Latent Human Dynamics Lab — common workflows
#
# Quick reference:
#   make help        — show this list
#   make install     — pip install in development mode
#   make smoke       — run the 10-stage smoke test (~1 second)
#   make health      — run the health check
#   make test        — run the test suite
#   make pipeline    — generate synthetic cohort + engineer features
#   make train       — train the dynamics model
#   make figures     — regenerate all paper figures
#   make demo        — run the perturbation demo
#   make all         — pipeline → train → figures → demo
#   make ingest-studentlife DIR=/path/to/studentlife_1.1.0/
#                    — ingest a real StudentLife cohort
#   make compare CSV=studentlife_daily.csv
#                    — compare a real cohort against the generator
#   make clean       — wipe generated artifacts

PYTHON ?= python3
PARTICIPANTS ?= 500
DAYS ?= 180
SEED ?= 17

.PHONY: help install smoke health test pipeline train figures demo all clean \
        ingest-studentlife compare lint check

help:
	@echo "Latent Human Dynamics Lab — available targets:"
	@echo ""
	@echo "  make smoke       Run the 10-stage smoke test (~1 second)"
	@echo "  make health      Run the install health check"
	@echo "  make test        Run the test suite (uses pytest if available)"
	@echo "  make pipeline    Generate synthetic cohort + engineer features"
	@echo "  make train       Train the dynamics model"
	@echo "  make figures     Regenerate all paper figures"
	@echo "  make demo        Run the perturbation demo"
	@echo "  make all         Pipeline → train → figures → demo"
	@echo "  make check       Health + smoke + test (CI-style)"
	@echo "  make clean       Wipe generated artifacts"
	@echo ""
	@echo "Real-cohort workflow:"
	@echo "  make ingest-studentlife DIR=/path/to/studentlife_1.1.0/"
	@echo "  make compare CSV=studentlife_daily.csv"
	@echo ""
	@echo "Variables:"
	@echo "  PARTICIPANTS=$(PARTICIPANTS)  DAYS=$(DAYS)  SEED=$(SEED)"
	@echo "  Override via: make pipeline PARTICIPANTS=100 DAYS=60"

install:
	$(PYTHON) -m pip install -e .

smoke:
	$(PYTHON) scripts/smoke_test.py

health:
	PYTHONPATH=src $(PYTHON) -m utils.health_check

test:
	@command -v pytest >/dev/null 2>&1 && pytest tests/ -v || \
		(echo "pytest not installed; running fallback verifier..." && \
		 $(PYTHON) scripts/smoke_test.py)

pipeline:
	$(PYTHON) scripts/run_pipeline.py --participants $(PARTICIPANTS) --days $(DAYS) --seed $(SEED)

train: pipeline
	$(PYTHON) scripts/train_dynamics_model.py

figures: train
	$(PYTHON) scripts/generate_figures.py

demo: pipeline
	$(PYTHON) scripts/run_perturbation_demo.py

all: figures demo

ingest-studentlife:
	@if [ -z "$(DIR)" ]; then \
		echo "Usage: make ingest-studentlife DIR=/path/to/studentlife_1.1.0/"; exit 1; \
	fi
	$(PYTHON) scripts/ingest_studentlife.py $(DIR)

compare:
	@if [ -z "$(CSV)" ]; then \
		echo "Usage: make compare CSV=studentlife_daily.csv"; exit 1; \
	fi
	$(PYTHON) scripts/compare_to_synthetic.py $(CSV)

clean:
	rm -rf data/synthetic/* data/processed/* \
	       results/figures/* results/tables/* results/checkpoints/* \
	       __pycache__ */__pycache__ */*/__pycache__ \
	       .pytest_cache build/ dist/ *.egg-info
	@find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true

check: health smoke test
	@echo "All checks passed."
