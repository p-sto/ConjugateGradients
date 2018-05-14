# basic:
SHELL:=/bin/bash
PROJECT_NAME=ConjugateGradients
PWD:=$(shell pwd)
PYTHONPATH=$(PWD)/../
TEST_DIR=tests/
VENV=venv/bin
PIP=$(VENV)/pip3
PIP_FLAGS=--trusted-host=http://pypi.python.org/simple/
PYTEST=$(VENV)/py.test
PYLINT=$(VENV)/pylint
COVERAGE=$(VENV)/coverage
MYPY=$(VENV)/mypy
MYPYFLAGS=--ignore-missing-imports --follow-imports=skip
HOST_PYTHON_VER=/usr/local/bin/python3.5

CONDA_VENV_NAME=$(PROJECT_NAME)
CONDA_ROOT=/home/anaconda3
CONDA_PYTHON=$(CONDA_ROOT)/bin/python3.5
CONDA_PYLINT=$(CONDA_ROOT)/bin/pylint
CONDA_COVERAGE=coverage
CONDA_PYTEST=py.test
CONDA_MYPY=mypy


.PHONY: all venv clean test test_pytest test_gen_coverage_rep test_pylint test_mypy git-status commit-id

all: venv test clean

conda:
	$(CONDA_ROOT)/bin/conda create --name $(CONDA_VENV_NAME)
	$(CONDA_ROOT)/bin/conda install --name $(CONDA_VENV_NAME) pip
	@echo anaconda virtual env create, perform calls:
	@echo source $(CONDA_ROOT)/bin/activate $(CONDA_VENV_NAME)
	@echo pip install -Ur requirements.txt

conda_remove:
	$(CONDA_ROOT)/bin/conda remove --name $(PROJECT_NAME) --all

venv: venv/bin/activate

venv/bin/activate: requirements.txt
	test -d venv || virtualenv -p $(HOST_PYTHON_VER) venv
	$(PIP) $(PIP_FLAGS) install -Ur requirements.txt
	touch venv/bin/activate

test_pytest:
	PYTHONPATH=$(PYTHONPATH) $(PYTEST) --verbose --color=yes --cov=$(PROJECT_NAME) --cov-report html --cov-config .coveragerc --tb=short $(TEST_DIR)

test_pylint:
	find $(PROJECT_NAME) -name '*.py' | xargs $(PYLINT) --rcfile=$(PWD)/.pylintrc

test_gen_coverage_rep:
	$(COVERAGE) report

test_mypy:
	find $(PROJECT_NAME) -name '*.py' | xargs $(MYPY) $(MYPYFLAGS)

conda_test_pytest:
	PYTHONPATH=$(PYTHONPATH) $(CONDA_PYTEST) --verbose --color=yes --cov=$(PROJECT_NAME) --cov-report html --cov-config .coveragerc --tb=short $(TEST_DIR)

conda_test_pylint:
	find $(PROJECT_NAME) -name '*.py' | xargs $(CONDA_PYLINT) --rcfile=$(PWD)/.pylintrc

conda_test_gen_coverage_rep:
	$(CONDA_COVERAGE) report

conda_test_mypy:
	find $(PROJECT_NAME) -name '*.py' | xargs $(CONDA_MYPY) $(MYPYFLAGS)

conda_test: conda_test_pylint conda_test_mypy conda_test_pytest conda_test_gen_coverage_rep clean

test: test_pylint test_mypy test_pytest test_gen_coverage_rep clean

clean:
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf .cache
	rm -rf .mypy_cache
	find -name '$(PROJECT_NAME).log' | xargs rm -rf
	find $(PROJECT_NAME) -name '*.pyc' | xargs rm -rf
	find $(PROJECT_NAME) -name '__pycache__' -type d | xargs rm -rf
	find $(TEST_DIR) -name '__pycache__' -type d | xargs rm -rf
