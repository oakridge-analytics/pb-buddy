import shlex
import subprocess
from pathlib import Path, PurePosixPath

import modal
from modal import App, Image, Secret

image = (
    Image.debian_slim(python_version="3.9")
    .pip_install(
        [
            "dash==2.9.1",
            "dash-bootstrap-components==1.4.1",
        ],
        # force_build=True,
    )
    .pip_install_private_repos(
        "github.com/pb-buddy/pb-buddy@feat/add_other_ad_parsing",
        git_user="dbandrews",
        secrets=[Secret.from_name("pb-buddy-github")],
        # force_build=True,
    )
)

app = App("bike-buddy-ui", image=image)


dash_script_local_path = Path(__file__).parent / "app.py"
dash_script_remote_path = PurePosixPath("/root/app.py")

if not dash_script_local_path.exists():
    raise FileNotFoundError(f"Could not find {dash_script_local_path}")

dash_script_mount = modal.Mount.from_local_file(local_path=dash_script_local_path, remote_path=dash_script_remote_path)


@app.function(
    allow_concurrent_inputs=100,
    mounts=[dash_script_mount],
)
@modal.web_server(8000)
def run():
    target = shlex.quote(str(dash_script_remote_path))
    cmd = f"python {target}"
    subprocess.Popen(cmd, shell=True)
