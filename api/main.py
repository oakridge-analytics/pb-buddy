import os
from typing import List, Optional

import modal
import pandas as pd
from fastapi import FastAPI
from modal import App, Image, enter, web_endpoint
from pydantic import BaseModel


class BikeBuddyAd(BaseModel):
    category: Optional[str] = None
    original_post_date: Optional[str] = None
    last_repost_date: Optional[str] = None
    still_for_sale: Optional[str] = None
    view_count: Optional[float] = None
    watch_count: Optional[float] = None
    price: Optional[float] = None
    currency: Optional[str] = None
    description: Optional[str] = None
    ad_title: Optional[str] = None
    location: Optional[str] = None
    datetime_scraped: Optional[str] = None
    url: Optional[str] = None
    frame_size: Optional[str] = None
    wheel_size: Optional[str] = None
    front_travel: Optional[str] = None
    condition: Optional[str] = None
    material: Optional[str] = None
    rear_travel: Optional[str] = None
    seat_post_diameter: Optional[str] = None
    seat_post_travel: Optional[str] = None
    front_axle: Optional[str] = None
    rear_axle: Optional[str] = None
    shock_eye_to_eye: Optional[str] = None
    shock_stroke: Optional[str] = None
    shock_spring_rate: Optional[str] = None
    restrictions: Optional[str] = None
    price_cpi_adjusted_CAD: Optional[str] = None


class BikeBuddyAdPredictions(BaseModel):
    predictions: List[float]


BASE_DIR = "/assets"
PIPELINE_FILE = "models/7c6aeb0363614859bbb52ba01c177c21-transformer-auto_mm_bikes.pkl"
MODEL_FILE = "models/7c6aeb0363614859bbb52ba01c177c21-auto_mm_bikes"


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
S3_BUCKET_NAME = "bike-buddy"
image = (
    Image.debian_slim(python_version="3.9")
    .pip_install("torch==1.13.1+cpu", extra_index_url="https://download.pytorch.org/whl/cpu/")
    .pip_install("torchvision==0.14.1+cpu", extra_index_url="https://download.pytorch.org/whl/cpu/")
    .pip_install("autogluon==0.7.0")
    .pip_install("fastapi==0.68.0")
    .pip_install("uvicorn==0.14.0")
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
    @web_endpoint(method="POST")
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
