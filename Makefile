.PHONY: scrape_report sync_reports

scrape_report :
	@poetry run python -m jupytext --to notebook --execute reports/scrape_report.md;
	poetry run python -m nbconvert --no-input --to html reports/scrape_report.ipynb;

sync_reports :
	poetry run python -m jupytext --sync reports/*.md;