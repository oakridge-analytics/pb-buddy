.PHONY: scrape_report

scrape_report :
	@poetry run python -m jupytext --to notebook --execute reports/scrape_report.md;
	poetry run python -m nbconvert --no-input --to html reports/scrape_report.ipynb;