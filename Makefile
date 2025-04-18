#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = pilling-attribute-recognition
PYTHON_VERSION = 3.12.9
VENV = .venv
PIP = $(VENV)/bin/pip
PYTHON = $(VENV)/bin/python 
BIN = $(VENV)/bin

#################################################################################
# COMMANDS                                                                      #
#################################################################################

.PHONY: run
run: $(VENV)/bin/activate
	echo 'TODO RUN THE THING'

.PHONY: run-notebook
run-notebook:
	cd $(PROJECT_NAME)/notebook.ipynb && jupyter notebook

.PHONY: train
train: $(VENV)/bin/activate
	$(PYTHON) sympany_textile_classification/modeling/train.py


$(VENV)/bin/activate: requirements.txt
	python3 -m venv $(VENV)
	$(PIP) install -r requirements.txt

## Delete all compiled Python files
.PHONY: clean
clean:
	rm -rf __pycache__
	rm -rf $(VENV)

## Lint using flake8 and black (use `make format` to do formatting)
.PHONY: lint
lint:
	$(BIN)/flake8 sympany_textile_classification
	$(BIN)/isort --check --diff --profile black sympany_textile_classification
	$(BIN)/black --check --config pyproject.toml sympany_textile_classification

## Format source code with black
.PHONY: format
format: 
	$(BIN)/black --config pyproject.toml sympany_textile_classification

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Make Dataset
.PHONY: data
data: 
	$(PYTHON) sympany_textile_classification/dataset.py


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)