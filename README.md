# pb-buddy

Find your next bike with recommendations and price intelligence: [https://bike.broker](https://bike.broker)

Or price a used bike here: [https://dbandrews--bike-buddy-ui-flask-app.modal.run](https://dbandrews--bike-buddy-ui-flask-app.modal.run)

## Development Notes

### Environment Setup

For normal web scraping work, use the following to create the `pb-buddy` `conda` environment:

```bash
make conda
```

### Scraping Configuration

#### Used Ads

#### Bike Specs

**Warning: very rough process, needs refactoring and likely reading code to use**. 

To build the bike spec dataset, first get links to bike spec pages with:

```
python scripts/scrape_bike_links.py --links_path_out=<folder_name_here>
```

Then, use this folder of csv files per manufacturer to scrape the specs of each model with:

```
python scripts/scrape_bike_specs.py --csv_folder_path=<path_to_folder_of_links_above> --specs_folder_path=<desired_output_folder> --existing_specs_folder=<desired_output_folder>
```

Progressively larger files of scraped bike specs will be saved to `--specs_folder_path`, dropping duplicates on `spec_url` across these gives the final dataset.

Refreshing package mapping of year, make, model, after confirming which version of specs data you want to use in `pb_buddy.constants`:

```
python -m pb_buddy.data.specs > pb_buddy/resources/year_make_model_mapping.json
```

#### Modelling

For running modelling with GPU+CUDA after initial environment setup is done, first ensure that CUDA 11.6 is installed.

For working with WSL - follow the guide here: https://docs.nvidia.com/cuda/wsl-user-guide/index.html to ensure:

- Nvidia driver is only installed in Windows side
- Specific CUDA toolkit is installed for WSL, to not overwrite Windows side libraries that are linked into WSL

```bash
make add_modelling
```

#### UMAP Embedding Browser

For running the UMAP browser, you need to install `cuml`, using `mamba` - *work in progress*:

```bash
conda install mamba -n base -c conda-forge -c defaults # Mamba solver appears to be recommended by RapidsAI
conda create --name pb-buddy-umap python=3.9 -y && conda activate pb-buddy-umap
mamba install -c rapidsai -c conda-forge -c nvidia cuml -y
pip install dash-bootstrap-components dash
```

### Refreshing PAT for self hosted Github Actions Runner

When the personal access token (PAT) expires to register the runner:

- Stop and uninstall existing runner (assuming runner setup in default named `runner` folder) on the VM:
- `cd runner && sudo ./svc.sh uninstall`
- `cd .. && rm -rf runner`
- Remove runner under Github interface (if there are running jobs stop/cancel those first).
- Get PAT token (classic, with admin:org, workflow, repo permissions), then run (based on [here](https://github.com/actions/runner/blob/main/docs/automate.md#automate-configuring-self-hosted-runners)):
- ` export RUNNER_CFG_PAT=<PAT_HERE> && curl -s https://raw.githubusercontent.com/actions/runner/main/scripts/create-latest-svc.sh | bash -s yourorg/yourrepo`
- Note the space in front of command so it's not in shell history with your PAT

