.PHONY: scrape_report sync_reports conda

scrape_report :
	@poetry run python -m jupytext --to notebook --execute reports/status_reports/scrape_report.md;
	poetry run python -m nbconvert --no-input --to html reports/status_reports/scrape_report.ipynb;

sync_reports :
	poetry run python -m jupytext --sync reports/*/*.md;

# Need to specify bash in order for conda activate to work.
SHELL=/bin/bash
# Note that the extra activate is needed to ensure that the activate floats env to the front of PATH
CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

conda :
	conda create -n pb-buddy python=3.9 -y; \
	($(CONDA_ACTIVATE) pb-buddy; \
	pip install poetry; \
	poetry install; \
	python -m ipykernel install --user --name pb-buddy;)

add_modelling : 
	($(CONDA_ACTIVATE) pb-buddy; \
	poetry install --with modelling; \
	# Install torch + autogluon separate to get different indexes \
	pip3 install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116; \
	pip3 install autogluon==0.7.0; \
	); # For some reason unknown symbol in installed environment, not needed explicitly