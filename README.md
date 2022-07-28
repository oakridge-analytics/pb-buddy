# pb-buddy

## Development Notes

### Refreshing PAT for self hosted Github Actions Runner

When the personal access token (PAT) expires to register the runner:

- Stop and uninstall existing runner (assuming runner setup in default named `runner` folder) on the VM:
- `cd runner && sudo ./svc.sh uninstall`
- `cd .. && rm -rf runner`
- Remove runner under Github interface (if there are running jobs stop/cancel those first).
- Get PAT token, then run (based on [here](https://github.com/actions/runner/blob/main/docs/automate.md#automate-configuring-self-hosted-runners)):
- `export RUNNER_CFG_PAT=<PAT_HERE> && curl -s https://raw.githubusercontent.com/actions/runner/main/scripts/create-latest-svc.sh | bash -s -- -s yourorg/yourrepo -n runner_name`

Notes on OVH setup 2022:
- SSH Port on 49999
- Default user/password from OVH.
