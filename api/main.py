import os
from typing import List, Optional

import modal
import pandas as pd
from fastapi import FastAPI
from joblib import load
from modal import App, Image, asgi_app, enter, method
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


# Reduce the number of times the model is loaded into memory by loading only at launch
# @asynccontextmanager
# async def lifespan(web_app: FastAPI):
#     estimator_dict["pipeline"] = load(PIPELINE_FILE)
#     estimator_dict["model"] = MultiModalPredictor.load(MODEL_FILE)
#     yield
#     estimator_dict.clear()


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
    .run_function(
        download_assets,
        secrets=[modal.Secret.from_name("aws-s3-secrets")],
        # force_build=True,
    )
)


@app.cls(
    image=image,
)
class AutoGluonModel:
    @enter()
    def load(self):
        self.model = MultiModalPredictor.load(MODEL_FILE)
        self.pipeline = load(PIPELINE_FILE)

    @method()
    def transform(self, input_data: pd.DataFrame) -> pd.DataFrame:
        return self.pipeline.transform(input_data)

    @method()
    def get_feature_names_out(self) -> List[str]:
        return self.pipeline.get_feature_names_out()

    @method()
    def predict(self, input_data: pd.DataFrame, **kwargs) -> str:
        return self.model.predict(input_data, **kwargs)


@web_app.post("/text-predict", tags=["text-predictions"])
async def predict(ads: List[BikeBuddyAd]) -> BikeBuddyAdPredictions:
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
    autogluon = AutoGluonModel()
    df_predict = pd.DataFrame(
        data=autogluon.pipeline.transform.remote(df_input),
        columns=autogluon.pipeline.get_feature_names_out.remote(),
    )

    predictions = autogluon.model.predict.remote(df_predict, as_pandas=False).tolist()
    if isinstance(predictions, float):
        predictions = [predictions]
    outputs = BikeBuddyAdPredictions(predictions=predictions)

    return outputs


@app.function(
    image=image,
)
@asgi_app()
def fastapi_app():
    return web_app
