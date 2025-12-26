.PHONY: clean data lint requirements sync_data_to_s3 sync_data_from_s3

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')
PROFILE = default
PROJECT_NAME = epml_project
PYTHON_INTERPRETER = python3

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
requirements: test_environment
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

## Make Dataset
data: requirements
	$(PYTHON_INTERPRETER) src/data/make_dataset.py data/raw data/processed

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8
lint:
	flake8 src

## Upload Data to S3
sync_data_to_s3:
ifeq (default,$(PROFILE))
	aws s3 sync data/ s3://$(BUCKET)/data/
else
	aws s3 sync data/ s3://$(BUCKET)/data/ --profile $(PROFILE)
endif

## Download Data from S3
sync_data_from_s3:
ifeq (default,$(PROFILE))
	aws s3 sync s3://$(BUCKET)/data/ data/
else
	aws s3 sync s3://$(BUCKET)/data/ data/ --profile $(PROFILE)
endif

## Set up python interpreter environment
create_environment:
ifeq (True,$(HAS_CONDA))
		@echo ">>> Detected conda, creating conda environment."
ifeq (3,$(findstring 3,$(PYTHON_INTERPRETER)))
	conda create --name $(PROJECT_NAME) python=3
else
	conda create --name $(PROJECT_NAME) python=2.7
endif
		@echo ">>> New conda env created. Activate with:\nsource activate $(PROJECT_NAME)"
else
	$(PYTHON_INTERPRETER) -m pip install -q virtualenv virtualenvwrapper
	@echo ">>> Installing virtualenvwrapper if not already installed.\nMake sure the following lines are in shell startup file\n\
	export WORKON_HOME=$$HOME/.virtualenvs\nexport PROJECT_HOME=$$HOME/Devel\nsource /usr/local/bin/virtualenvwrapper.sh\n"
	@bash -c "source `which virtualenvwrapper.sh`;mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER)"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
endif

## Test python environment is setup correctly
test_environment:
	$(PYTHON_INTERPRETER) test_environment.py

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

#################################################################################
# HW4: ML Pipeline Automation                                                   #
#################################################################################

## Prepare data using Hydra config
prepare:
	$(PYTHON_INTERPRETER) -m src.pipelines.prepare_data

## Train single model (usage: make train MODEL=random_forest)
train:
	$(PYTHON_INTERPRETER) -m src.pipelines.train_pipeline model=$(MODEL)

## Train Random Forest model
train_rf:
	$(PYTHON_INTERPRETER) -m src.pipelines.train_pipeline model=random_forest

## Train Gradient Boosting model
train_gb:
	$(PYTHON_INTERPRETER) -m src.pipelines.train_pipeline model=gradient_boosting

## Train all models sequentially
train_all:
	$(PYTHON_INTERPRETER) -m src.pipelines.run_all_models

## Evaluate and compare all models
evaluate:
	$(PYTHON_INTERPRETER) -m src.pipelines.evaluate_models

## Run full DVC pipeline (prepare + all models + evaluate)
pipeline:
	dvc repro

## Run DVC pipeline for specific stage
pipeline_stage:
	dvc repro $(STAGE)

## Show DVC pipeline DAG
dag:
	dvc dag

## Show DVC metrics
metrics:
	dvc metrics show

## Compare DVC metrics with previous runs
metrics_diff:
	dvc metrics diff

## Show DVC params
params:
	dvc params diff

## Clean output directories
clean_outputs:
	rm -rf outputs/
	rm -rf multirun/

## Run full pipeline from scratch
run_full: clean_outputs pipeline evaluate
	@echo "Full pipeline completed!"


#################################################################################
# HW5: ClearML MLOps                                                            #
#################################################################################

## Start ClearML Server (Docker)
clearml_server_start:
	cd clearml && docker-compose up -d
	@echo "ClearML Server started!"
	@echo "Web UI: http://localhost:8080"
	@echo "API: http://localhost:8008"
	@echo "Files: http://localhost:8081"

## Stop ClearML Server
clearml_server_stop:
	cd clearml && docker-compose down
	@echo "ClearML Server stopped"

## Show ClearML Server status
clearml_server_status:
	cd clearml && docker-compose ps

## Setup ClearML configuration
clearml_setup:
	$(PYTHON_INTERPRETER) clearml/setup_clearml.py --status

## Test ClearML connection
clearml_test:
	$(PYTHON_INTERPRETER) clearml/setup_clearml.py --test

## Create ClearML project structure
clearml_create_project:
	$(PYTHON_INTERPRETER) clearml/setup_clearml.py --create-project

## Run single experiment with ClearML (usage: make clearml_experiment MODEL=RandomForest)
clearml_experiment:
	$(PYTHON_INTERPRETER) -m src.clearml_integration.run_experiments --model $(MODEL)

## Run all experiments with ClearML tracking
clearml_experiments_all:
	$(PYTHON_INTERPRETER) -m src.clearml_integration.run_experiments --all

## Run experiments in offline mode (no server required)
clearml_experiments_offline:
	$(PYTHON_INTERPRETER) -m src.clearml_integration.run_experiments --all --offline

## Compare ClearML experiments
clearml_compare:
	$(PYTHON_INTERPRETER) -m src.clearml_integration.run_experiments --compare

## Compare registered models
clearml_compare_models:
	$(PYTHON_INTERPRETER) -m src.clearml_integration.run_experiments --compare-models

## Run ClearML pipeline for single model (usage: make clearml_pipeline MODEL=RandomForest)
clearml_pipeline:
	$(PYTHON_INTERPRETER) -m src.clearml_integration.pipeline --model $(MODEL)

## Run ClearML pipeline for all models
clearml_pipeline_all:
	$(PYTHON_INTERPRETER) -m src.clearml_integration.pipeline --all

## Generate ClearML dashboard report
clearml_dashboard:
	$(PYTHON_INTERPRETER) -m src.clearml_integration.dashboard --summary

## Generate full ClearML report
clearml_report:
	$(PYTHON_INTERPRETER) -m src.clearml_integration.dashboard --report

## Export ClearML metrics and reports
clearml_export:
	$(PYTHON_INTERPRETER) -m src.clearml_integration.dashboard --all

## Full ClearML workflow: experiments + comparison + report
clearml_full: clearml_experiments_all clearml_compare_models clearml_report
	@echo "Full ClearML workflow completed!"

## Clean ClearML outputs
clearml_clean:
	rm -rf outputs/clearml/
	@echo "ClearML outputs cleaned"


#################################################################################
# HW6: Documentation and Reports                                                #
#################################################################################

## Build documentation with MkDocs
docs_build:
	mkdocs build --strict
	@echo "Documentation built in site/"

## Serve documentation locally
docs_serve:
	mkdocs serve
	@echo "Documentation available at http://127.0.0.1:8000"

## Deploy documentation to GitHub Pages
docs_deploy:
	mkdocs gh-deploy --force
	@echo "Documentation deployed to GitHub Pages"

## Generate experiment reports
generate_reports:
	$(PYTHON_INTERPRETER) -m src.reports.generate_reports
	@echo "Reports generated in outputs/reports/"

## Generate all reports (experiments + docs update)
reports_all: generate_reports docs_build
	@echo "All reports generated!"

## Verify reproducibility (run experiments twice and compare)
verify_reproducibility:
	@echo "Running first experiment set..."
	$(PYTHON_INTERPRETER) -m src.clearml_integration.run_experiments --all --offline
	cp -r outputs/clearml/models outputs/clearml/models_run1
	@echo "Running second experiment set..."
	$(PYTHON_INTERPRETER) -m src.clearml_integration.run_experiments --all --offline
	@echo "Comparing results..."
	@echo "Results saved in outputs/clearml/models and outputs/clearml/models_run1"

## Full documentation workflow
docs_full: generate_reports docs_build
	@echo "Full documentation workflow completed!"


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
