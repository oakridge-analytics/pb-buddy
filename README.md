# pb-buddy

## Development Notes

### Environment Setup

For normal web scraping work, use the following to create the `pb-buddy` `conda` environment:

```bash
make conda
```

For running modelling with GPU+CUDA after initial environment setup is done, can use:

```bash
make add_modelling
```

For running the UMAP browser, you need to install `cuml`, using `mamba` - *work in progress*:

```bash
conda install mamba -n base -c conda-forge -c defaults # Mamba solver appears to be recommended by RapidsAI
mamba install -c rapidsai -c conda-forge -c nvidia cuml
```

### Refreshing PAT for self hosted Github Actions Runner

When the personal access token (PAT) expires to register the runner:

- Stop and uninstall existing runner (assuming runner setup in default named `runner` folder) on the VM:
- `cd runner && sudo ./svc.sh uninstall`
- `cd .. && rm -rf runner`
- Remove runner under Github interface (if there are running jobs stop/cancel those first).
- Get PAT token, then run (based on [here](https://github.com/actions/runner/blob/main/docs/automate.md#automate-configuring-self-hosted-runners)):
- `RUNNER_CFG_PAT=<PAT_HERE> curl -s https://raw.githubusercontent.com/actions/runner/main/scripts/create-latest-svc.sh | bash -s yourorg/yourrepo`

