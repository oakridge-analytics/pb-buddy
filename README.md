# pb-buddy

## Development Notes

### Environment Setup

For normal web scraping work, use the following to create the `pb-buddy` `conda` environment:

```bash
make conda
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
- Get PAT token, then run (based on [here](https://github.com/actions/runner/blob/main/docs/automate.md#automate-configuring-self-hosted-runners)):
- `RUNNER_CFG_PAT=<PAT_HERE> curl -s https://raw.githubusercontent.com/actions/runner/main/scripts/create-latest-svc.sh | bash -s yourorg/yourrepo`

