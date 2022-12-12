# pb-buddy

## Development Notes

### Refreshing PAT for self hosted Github Actions Runner

When the personal access token (PAT) expires to register the runner:

- Stop and uninstall existing runner (assuming runner setup in default named `runner` folder):
- `cd runner && sudo ./svc.sh uninstall`
- Get PAT token, then run (based on [here](https://github.com/actions/runner/blob/main/docs/automate.md#automate-configuring-self-hosted-runners)):
- `RUNNER_CFG_PAT=<PAT_HERE> curl -s https://raw.githubusercontent.com/actions/runner/main/scripts/create-latest-svc.sh | bash -s yourorg/yourrepo`

### Installing Modelling Dependencies

To install all packages needed for creating the models:

`poetry install --with modelling`