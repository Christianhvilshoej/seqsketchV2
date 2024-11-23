.PHONY: help create_environment install_requirements download_quickdraw clean train

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = seqsketch
PYTHON_VERSION = 3.11

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Display this help message
help:
	@echo Project name $(PROJECT_NAME)
	@echo Python version $(PYTHON_VERSION)

## Set up python interpreter environment
create_conda_environment:
	conda env remove --name $(PROJECT_NAME) --yes --quiet || true
	conda create --name $(PROJECT_NAME) python=$(PYTHON_VERSION) --no-default-packages -y

create_venv:
	python3 -m venv cs492d
	source cs492d/bin/activate

## Install Python Dependencies
install_requirements:
	python -m pip install -U pip setuptools wheel --no-cache-dir
	python -m pip install -r requirements.txt --no-cache-dir
	python -m pip install -e . --no-cache-dir

## Download data
download_quickdraw:
	for category in "cat" "garden" "helicopter"; do \
		echo "Processing category: $$category" ; \
		gsutil -m cp "gs://quickdraw_dataset/full/simplified/$$category.ndjson" "data/quickdraw/processed/$$category/"; \
	done

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Train the model
train:
	train --config baseline.yaml
	inference --model_folder baseline/v0_20241101_195027