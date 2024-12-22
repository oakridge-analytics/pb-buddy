import os
import shutil
import tempfile

import boto3
import pandas as pd
from autogluon.multimodal import MultiModalPredictor
from joblib import load
from sklearn.pipeline import Pipeline


def load_model_from_s3(model_uuid: str, bucket_name: str = "bike-buddy") -> tuple:
    """
    Load AutoGluon model and sklearn pipeline from S3 bucket.

    Args:
        model_uuid: UUID of the model to load (e.g. "7c6aeb0363614859bbb52ba01c177c21")
        bucket_name: Name of S3 bucket containing models (default: "bike-buddy")

    Returns:
        tuple: (AutoGluon predictor, sklearn pipeline transformer)
    """
    s3_client = boto3.client("s3")

    # Create temp directory to store downloaded files
    temp_dir = tempfile.mkdtemp()

    try:
        # Download and load transformer pipeline
        transformer_key = f"{model_uuid}-transformer-auto_mm_bikes.pkl"
        transformer_path = os.path.join(temp_dir, transformer_key)

        s3_client.download_file(bucket_name, f"models/{transformer_key}", transformer_path)
        transformer = load(transformer_path)

        # Download AutoGluon model folder
        model_prefix = f"models/{model_uuid}-auto_mm_bikes/"
        model_dir = os.path.join(temp_dir, f"{model_uuid}-auto_mm_bikes")

        # List all objects with model prefix
        paginator = s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket_name, Prefix=model_prefix):
            for obj in page.get("Contents", []):
                # Get relative path by removing prefix
                relative_path = obj["Key"].replace(model_prefix, "")
                if relative_path:
                    # Create subdirectories if needed
                    file_path = os.path.join(model_dir, relative_path)
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)

                    # Download file
                    s3_client.download_file(bucket_name, obj["Key"], file_path)

        # Load AutoGluon predictor
        predictor = MultiModalPredictor.load(model_dir)

        return predictor, transformer

    finally:
        # Cleanup temp directory
        shutil.rmtree(temp_dir)


def predict_price(
    predictor: "MultiModalPredictor", transformer: "Pipeline", ad_data: pd.DataFrame | dict
) -> pd.Series | float:
    """
    Predict price for a single bike ad using loaded model.

    Args:
        predictor: Loaded AutoGluon predictor
        transformer: Loaded sklearn pipeline transformer
        ad_data: DataFrame or dict containing ad data fields

    Returns:
        float: Predicted price in CAD if input is dict
        pd.Series: Series of predictions if input is DataFrame
    """
    # Transform input data
    transformed_data = transformer.transform(ad_data)
    transformed_df = pd.DataFrame(data=transformed_data, columns=transformer.get_feature_names_out())

    # Get prediction
    prediction = predictor.predict(transformed_df)

    return prediction
