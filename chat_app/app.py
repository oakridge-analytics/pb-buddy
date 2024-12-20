import os
from typing import Tuple

import numpy as np
import pandas as pd
from autogluon.multimodal import MultiModalPredictor
from dotenv import load_dotenv
from joblib import dump, load
from swarm import Agent, Swarm

from chat_app.prompts import ASSISTANT_PROMPT
from pb_buddy.modelling.inference import load_model_from_s3
from pb_buddy.utils import mapbox_geocode


def calculate_distance_vectorized(df, ref_lat, ref_lon):
    lat1, lon1, lat2, lon2 = map(np.radians, [ref_lat, ref_lon, df["lat"], df["lon"]])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    r = 6371
    return r * c


def get_relevant_bikes(
    price_limit_cad: float,
    min_front_travel: float,
    max_front_travel: float,
    min_rear_travel: float,
    max_rear_travel: float,
    frame_size: str,
    # frame_material: str,
    category: str,
    lon: float,
    lat: float,
    distance_km_threshold: float,
    max_results: int = 3,
) -> pd.DataFrame:
    """Get bikes matching the specified criteria within a distance threshold.

    Args:
        price_limit_cad (float): Maximum price in CAD to filter bikes by
        min_front_travel (float): Minimum front suspension travel in mm
        max_front_travel (float): Maximum front suspension travel in mm
        min_rear_travel (float): Minimum rear suspension travel in mm
        max_rear_travel (float): Maximum rear suspension travel in mm
        frame_size (str): Normalized frame size (XS, S, M, L, XL)
        category (str): Bike category to filter by
        lon (float): Longitude of reference location
        lat (float): Latitude of reference location
        distance_km_threshold (float): Maximum distance in km from reference location
        max_results (int): Maximum number of results to return
    Returns:
        pd.DataFrame: Filtered dataset containing matching bikes, sorted by predicted price difference.
    """
    df = dataset.query("price_cad < @price_limit_cad")
    df = df.query("front_travel >= @min_front_travel and front_travel <= @max_front_travel")
    df = df.query("rear_travel >= @min_rear_travel and rear_travel <= @max_rear_travel")
    df = df.query("frame_size_normalized == @frame_size")
    # df = df.query("material == @frame_material")
    df = df.query("category == @category")
    df = df.assign(distance_km=lambda _df: calculate_distance_vectorized(_df, lat, lon))
    df = df.query("distance_km <= @distance_km_threshold")
    return df.sort_values(by="predicted_price_diff", ascending=False)[
        [
            "ad_title",
            "url",
            "price",
            "currency",
            "description",
            "last_repost_date",
            "predicted_price_cad",
            "predicted_price_diff",
            "distance_km",
        ]
    ].head(max_results)


def get_user_location(user_city_and_country: str) -> Tuple[float, float]:
    return mapbox_geocode(user_city_and_country)


def embed_new_bike_ad(
    new_ad_data: pd.DataFrame,
) -> np.ndarray:
    """
    New ad data cols required:

    ad_title: str,
    description: str,
    original_post_date: str,
    location: str,
    """
    transformed = transformer.transform(new_ad_data.assign(price_cpi_adjusted_CAD=0))
    return predictor.extract_embedding(pd.DataFrame(data=transformed, columns=transformer.get_feature_names_out()))


def get_similar_sold_bike_ads(
    ad_title: str,
    description: str,
    original_post_date: str,
    location: str,
    num_similar_ads: int = 5,
) -> pd.DataFrame:
    """Find similar sold bike ads based on semantic similarity of ad content.

    Uses embeddings from a trained model to find the most semantically similar sold bike ads
    by computing cosine similarity between the embeddings.

    Args:
        ad_title (str): Title of the bike ad
        description (str): Full description text of the ad
        original_post_date (str): When the ad was originally posted
        location (str): Location string for the ad
        num_similar_ads (int, optional): Number of similar ads to return. Defaults to 5.

    Returns:
        pd.DataFrame: DataFrame containing the most similar sold bike ads, with columns
            - ad_title: Title of the similar bike ad
            - url: URL of the similar bike ad
            - price: Price of the similar bike ad
            - currency: Currency of the price
            - last_repost_date: When the similar ad was last reposted
    """
    new_ad_data = pd.DataFrame(
        {
            "ad_title": [ad_title],
            "description": [description],
            "original_post_date": [original_post_date],
            "location": [location],
        }
    )
    new_ad_embedding = embed_new_bike_ad(new_ad_data)[0, :]

    # calculate cosine similarity between the new ad embedding and the embeddings for the most similar ads, get indices of the most similar ads
    similarities = np.dot(embeddings, new_ad_embedding) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(new_ad_embedding)
    )
    most_similar_indices = np.argsort(similarities)[-num_similar_ads:]
    return sold_dataset.iloc[most_similar_indices][
        [
            "ad_title",
            "url",
            "price",
            "currency",
            "last_repost_date",
        ]
    ]


if __name__ == "__main__":
    load_dotenv(os.path.expanduser("~/.openai/credentials"))
    load_dotenv(".env")

    dataset = pd.read_parquet("/tmp/latest_base_chat_data.parquet").assign(
        front_travel=lambda df: df.front_travel.fillna(0), rear_travel=lambda df: df.rear_travel.fillna(0)
    )

    embeddings = np.load("/home/drumm/projects/pb-buddy/reports/modelling/data/gluon_embeddings.npy")
    sold_dataset = pd.read_csv("/home/drumm/projects/pb-buddy/reports/modelling/data/df_gluon_inspection.csv")
    cache_path = "/tmp/.cache/model_b9f12bc7cbd34fc39bf300b88f2ab57a"
    if os.path.exists(cache_path):
        predictor = MultiModalPredictor.load(cache_path)
        transformer = load(f"{cache_path}_transformer.joblib")
    else:
        os.makedirs(cache_path, exist_ok=True)
        predictor, transformer = load_model_from_s3("b9f12bc7cbd34fc39bf300b88f2ab57a")
        predictor.save(cache_path)
        dump(transformer, f"{cache_path}_transformer.joblib")

    # df = get_relevant_bikes(
    #     price_limit_cad=100000,
    #     front_travel_range=(100, 150),
    #     rear_travel_range=(100, 150),
    #     frame_size="L",
    #     category="Enduro Bikes",
    #     # frame_material="Aluminum",
    #     lon=-113,
    #     lat=52,
    #     distance_km_threshold=200,
    #     dataset=dataset,
    # )
    # print(df)
    client = Swarm()
    agent = Agent(
        name="bike assistant",
        model="gpt-4o-mini",
        instructions=ASSISTANT_PROMPT,
        functions=[get_user_location, get_relevant_bikes, get_similar_sold_bike_ads],
    )
    from swarm.repl import run_demo_loop

    run_demo_loop(agent, stream=True, debug=True)
    # client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    # response = client.chat.completions.create(
    #     model="gpt-4o-mini",
    #     messages=[{"role": "system", "content": "test"}, {"role": "user", "content": "Hello!"}],
    # )
    # print(response)
