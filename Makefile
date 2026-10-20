ifeq ($(OS),Windows_NT)
PYTHON ?= py -3
VENV_PYTHON := .venv/Scripts/python.exe
else
PYTHON ?= python3
VENV_PYTHON := .venv/bin/python
endif

.PHONY: setup models test lint typecheck eval eval-live eval-attacks-live eval-classifier-live eval-benign-live badges demo

setup:
	$(PYTHON) -m venv .venv
	$(VENV_PYTHON) -m pip install --upgrade pip
	$(VENV_PYTHON) -m pip install -e ".[dev]"
	$(VENV_PYTHON) -m repopulse.cli models prepare

models:
	$(VENV_PYTHON) -m repopulse.cli models prepare

test:
	$(VENV_PYTHON) -m pytest

lint:
	$(VENV_PYTHON) -m ruff check .

typecheck:
	$(VENV_PYTHON) -m mypy src evals

eval:
	$(VENV_PYTHON) -m evals.run_suite

eval-live:
	$(VENV_PYTHON) -m evals.run_live

eval-attacks-live:
	$(VENV_PYTHON) -m evals.run_attacks_live

eval-classifier-live:
	$(VENV_PYTHON) -m evals.run_classifier_eval

eval-benign-live:
	$(VENV_PYTHON) -m evals.run_benign_fp

badges:
	$(VENV_PYTHON) -m evals.generate_badges

demo:
	@echo "Scripted demo is scheduled for Phase 8."
