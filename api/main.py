import os
import functools

from fastapi import FastAPI, Body
from autogluon.multimodal import MultiModalPredictor
from joblib import load
import pandas as pd
from model import BikeBuddyAd
import predictor as predictor

app = FastAPI(
    title="Bike Buddy API", description="API bike price prediction", version="0.0"
)


@app.on_event("startup")
async def load_model():
    predictor.pipeline = load(os.environ["PIPELINE_FILE"])
    predictor.model = MultiModalPredictor.load(os.environ["MODEL_FILE"])


@app.post("/text-predict", tags=["text-predictions"])
async def predict(ad: BikeBuddyAd):
    df_predict = pd.DataFrame(
        data=predictor.pipeline.transform(pd.DataFrame(data=[dict(ad)])),
        columns=predictor.pipeline.get_feature_names_out(),
    )

    # print(df_predict.transpose())
    # print(predictor.model.predict(df_predict))

    return {
        "prediction": predictor.model.predict(
            pd.concat([df_predict for i in range(100)])
        ).to_list()
    }
