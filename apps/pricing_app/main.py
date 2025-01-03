import modal

from .common import app
from .ui import dash_app

# Merge the browser service app
# web_app.merge(browser_app)
image = (
    modal.Image.debian_slim(python_version="3.11")
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
        secrets=[modal.Secret.from_name("pb-buddy-github")],
        # force_build=True,
    )
)


@app.function(
    image=image,
    secrets=[
        modal.Secret.from_name("openai-secret"),
        modal.Secret.from_name("oxy-proxy"),
        modal.Secret.from_name("modal_webhook_tokens"),
    ],
    cpu=2.0,
    memory=4000,
    # mounts=[modal.Mount.from_local_python_packages("app")],
    allow_concurrent_inputs=10,
)
@modal.wsgi_app()
def flask_app():
    return dash_app.server


# if __name__ == "__main__":
#     modal.deploy(web_app)
