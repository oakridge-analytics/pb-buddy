import os

import modal
from modal import App, Image, Secret, wsgi_app
from app import dash_app

if os.environ.get("API_URL") is None:
    API_URL = "https://dbandrews--bike-buddy-api-autogluonmodelinference-predict.modal.run"
else:
    API_URL = os.environ["API_URL"]

app = App("bike-buddy-ui")
image = (
    Image.debian_slim(python_version="3.9")
    .pip_install(
        [
            "dash==2.9.1",
            "dash-bootstrap-components==1.4.1",
            "yfinance==0.2.43",
        ],
        # force_build=True,
    )
    .pip_install_private_repos(
        "github.com/pb-buddy/pb-buddy@master",
        git_user="dbandrews",
        secrets=[Secret.from_name("pb-buddy-github")],
        # force_build=True,
    )
    .run_commands("playwright install-deps")
    .run_commands("playwright install")
)


@app.function(image=image, secrets=[modal.Secret.from_name("openai-secret")], cpu=2.0, memory=4000, mounts=[modal.Mount.from_local_python_packages("app")])
@wsgi_app()
def flask_app():
    return dash_app.server