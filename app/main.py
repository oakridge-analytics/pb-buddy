import modal
from browser_service import app as browser_app  # noqa

from app import dash_app

# Create the main app
app = modal.App("bike-buddy")

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
    secrets=[modal.Secret.from_name("openai-secret"), modal.Secret.from_name("oxy-proxy")],
    cpu=2.0,
    memory=4000,
    mounts=[modal.Mount.from_local_python_packages("app")],
    keep_warm=1,
    allow_concurrent_inputs=10,
)
@modal.wsgi_app()
def web():
    return dash_app.server


# if __name__ == "__main__":
#     modal.deploy(web_app)
