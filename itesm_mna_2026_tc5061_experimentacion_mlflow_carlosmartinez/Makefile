#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = itesm_mna_2026_tc5061_experimentacion_mlflow_carlosmartinez
PYTHON_VERSION = 3.14
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format



## Run tests
.PHONY: test
test:
	python -m pytest tests


## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	
	conda create --name $(PROJECT_NAME) python=$(PYTHON_VERSION) -y
	
	@echo ">>> conda env created. Activate with:\nconda activate $(PROJECT_NAME)"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Make dataset
.PHONY: data
data:
	$(PYTHON_INTERPRETER) -m imdb_rating.dataset


## Train model with default hyperparameters and log the run to MLflow
.PHONY: train
train:
	$(PYTHON_INTERPRETER) -m imdb_rating.modeling.train


## Run the 5 experiment configurations documented in the README
.PHONY: experiments
experiments:
	$(PYTHON_INTERPRETER) -m imdb_rating.modeling.train --run-name rf_base
	$(PYTHON_INTERPRETER) -m imdb_rating.modeling.train --run-name rf_poco_profundo --n-estimators 100 --max-depth 5
	$(PYTHON_INTERPRETER) -m imdb_rating.modeling.train --run-name rf_sin_limite --max-depth 0 --min-samples-leaf 1
	$(PYTHON_INTERPRETER) -m imdb_rating.modeling.train --run-name rf_hojas_grandes --min-samples-leaf 20
	$(PYTHON_INTERPRETER) -m imdb_rating.modeling.train --run-name rf_sqrt --max-features sqrt
	$(PYTHON_INTERPRETER) -m imdb_rating.modeling.train --run-name rf_intermedio --n-estimators 500 --max-depth 20 --min-samples-leaf 3 --max-features 0.5


## Open the MLflow UI on http://127.0.0.1:8080
.PHONY: mlflow-ui
mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db --port 8080


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
