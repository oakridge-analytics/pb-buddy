# pb-buddy

## Development Notes

### Environment Setup

For normal web scraping work, use:

```bash
conda create -n pb-buddy python=3.9
conda activate pb-buddy
pip install poetry
poetry install
```

For running modelling with GPU+CUDA after initial environment setup is done, can use:

```bash
conda activate pb-buddy
poetry install --with modelling
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
pip uninstall torchtext # For some reason unknown symbol in installed environment, not needed explicitly
conda install mamba -n base -c conda-forge -c defaults # Mamba solver appears to be recommended by RapidsAI
mamba install -c rapidsai -c conda-forge -c nvidia cuml
```

### Refreshing PAT for self hosted Github Actions Runner

When the personal access token (PAT) expires to register the runner:

- Stop and uninstall existing runner (assuming runner setup in default named `runner` folder):
- `cd runner && sudo ./svc.sh uninstall`
- Get PAT token, then run (based on [here](https://github.com/actions/runner/blob/main/docs/automate.md#automate-configuring-self-hosted-runners)):
- `RUNNER_CFG_PAT=<PAT_HERE> curl -s https://raw.githubusercontent.com/actions/runner/main/scripts/create-latest-svc.sh | bash -s yourorg/yourrepo`

