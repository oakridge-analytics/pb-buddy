import os
from typing import List

import modal
import pandas as pd
from fastapi import FastAPI
from modal import App, Image, enter, web_endpoint
from pydantic import BaseModel, field_validator


class BikeBuddyAd(BaseModel):
    ad_title: str
    description: str
    original_post_date: str
    location: str

    @field_validator("location")
    def validate_location_format(cls, v):
        if "," not in v:
            raise ValueError('Location must be in format "city_name, country_name"')
        city, country = v.split(",", 1)
        if not city.strip() or not country.strip():
            raise ValueError("Both city and country must be non-empty")
        return v


class BikeBuddyAdPredictions(BaseModel):
    predictions: List[float]


# Local paths to download model and pipeline files
BASE_DIR = "/assets"
S3_BUCKET_NAME = "bike-buddy"
UUID = "b9f12bc7cbd34fc39bf300b88f2ab57a"
PIPELINE_FILE = f"models/{UUID}-transformer-auto_mm_bikes.pkl"
MODEL_FILE = f"models/{UUID}-auto_mm_bikes"


# Download model into image to be snapshotted and used in the function
def download_assets():
    import boto3

    s3 = boto3.client("s3")
    for path in [PIPELINE_FILE, MODEL_FILE]:
        desired_dir = os.path.dirname(path)
        if not os.path.exists(desired_dir):
            os.makedirs(desired_dir, exist_ok=True)
    s3.download_file(S3_BUCKET_NAME, PIPELINE_FILE, PIPELINE_FILE)

    s3_resource = boto3.resource("s3")
    bucket = s3_resource.Bucket(S3_BUCKET_NAME)
    for obj in bucket.objects.filter(Prefix=MODEL_FILE):
        if not os.path.exists(os.path.dirname(obj.key)):
            os.makedirs(os.path.dirname(obj.key))
        bucket.download_file(obj.key, obj.key)


web_app = FastAPI(title="Bike Buddy API", description="API bike price prediction", version="0.1")
app = App("bike-buddy-api")
image = (
    Image.debian_slim(python_version="3.11")
    .pip_install("autogluon.multimodal==1.1.1")
    .pip_install("fastapi==0.115.4")
    .pip_install("uvicorn==0.32.0")
    .pip_install("boto3")
    .pip_install_private_repos(
        "github.com/pb-buddy/pb-buddy@master",
        git_user="dbandrews",
        secrets=[modal.Secret.from_name("pb-buddy-github")],
        # force_build=True,
    )
    .run_function(
        download_assets,
        secrets=[modal.Secret.from_name("aws-s3-secrets")],
        # force_build=True,
    )
)


@app.cls(
    image=image,
    gpu="T4",
)
class AutoGluonModelInference:
    @enter()
    def load(self):
        import joblib
        from autogluon.multimodal import MultiModalPredictor

        self.model = MultiModalPredictor.load(MODEL_FILE)
        self.pipeline = joblib.load(PIPELINE_FILE)

    def transform(self, input_data: pd.DataFrame) -> pd.DataFrame:
        return self.pipeline.transform(input_data)

    def get_feature_names_out(self) -> List[str]:
        return self.pipeline.get_feature_names_out()

    # @method()
    @web_endpoint(method="POST", requires_proxy_auth=True)
    def predict(self, ads: List[BikeBuddyAd]) -> BikeBuddyAdPredictions:
        # Due to sklearn pipeline needing our target column price_cpi_adjusted_CAD
        # need to ensure it's present for transform() method
        df_input = pd.DataFrame(data=[dict(single_ad) for single_ad in ads]).assign(price_cpi_adjusted_CAD=0)

        # # TODO: confirm column names needed for pipeline:
        # required_input_cols = set(
        #     itertools.chain(
        #         *[
        #             step[2]
        #             for step in estimator_dict["pipeline"].named_steps["preprocess"].transformers_[
        #                 :-1
        #             ]
        #         ]
        #     )
        # )
        df_predict = pd.DataFrame(
            data=self.pipeline.transform(df_input),
            columns=self.pipeline.get_feature_names_out(),
        )

        predictions = self.model.predict(df_predict, as_pandas=False).tolist()
        if isinstance(predictions, float):
            predictions = [predictions]
        outputs = BikeBuddyAdPredictions(predictions=predictions)

        return outputs
