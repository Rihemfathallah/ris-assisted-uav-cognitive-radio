# Reproduce everything:  make all   (most of the wall-clock is TD3 training; see README)
PY ?= python -W ignore

.PHONY: all install test preflight bank train eval sensing octave figures report untrained clean-results
all: test preflight bank train eval sensing octave figures report

untrained:               # E14: where the untrained action heads start
	$(PY) scripts/untrained_policies.py

install:
	pip install -e . && pip install -r requirements.txt
test:
	$(PY) -m pytest
preflight:               # E00 / E02: is the operating point meaningful?
	$(PY) scripts/link_budget.py --blockage 0 10 15
	$(PY) scripts/regime_check.py --blockage 0 5 10
bank:                    # frozen validation / test flights for every N
	$(PY) -c "from ris_cr.config import load_config; from ris_cr.scenarios import load_bank; c=load_config(); [load_bank(s, N, c, rebuild=True) for N in (8,16,32,64) for s in ('val','test')]"
train:                   # resumable; JOBS=<n> limits concurrency
	./scripts/run_training_queue.sh
eval:                    # run 'runtime' on an idle machine: it measures time per decision
	$(PY) -m ris_cr.evaluate --part fig1 val_refs runtime sweeps main td3
sensing:
	$(PY) -m ris_cr.sensing
octave:                  # needs results/octave/fig1_input_N*.mat from `make eval`
	octave-cli octave/rate_vs_N.m
figures:
	$(PY) scripts/make_figures.py
report:
	cd report && latexmk -pdf -interaction=nonstopmode main.tex
