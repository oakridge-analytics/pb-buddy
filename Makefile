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
	conda create -n pb-buddy python=3.11 -y; \
	($(CONDA_ACTIVATE) pb-buddy; \
	pip install poetry; \
	poetry install; \
	python -m ipykernel install --user --name pb-buddy;)

add_modelling : 
	($(CONDA_ACTIVATE) pb-buddy; \
	pip3 install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2+cu117 --extra-index-url https://download.pytorch.org/whl/cu117;\
	poetry install --with modelling; \
	# Install torch + autogluon separate to get different indexes \
	); # For some reason unknown symbol in installed environment, not needed explicitly