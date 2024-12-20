import os
from typing import Tuple

import numpy as np
import openai
import pandas as pd
from dotenv import load_dotenv


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
    front_travel_range: Tuple[float, float],
    rear_travel_range: Tuple[float, float],
    frame_size: str,
    # frame_material: str,
    lon: float,
    lat: float,
    distance_km_threshold: float,
    dataset: pd.DataFrame,
) -> pd.DataFrame:
    df = dataset.query("price_cad < @price_limit_cad")
    df = df.query("front_travel >= @front_travel_range[0] and front_travel <= @front_travel_range[1]")
    df = df.query("rear_travel >= @rear_travel_range[0] and rear_travel <= @rear_travel_range[1]")
    df = df.query("frame_size_normalized == @frame_size")
    # df = df.query("material == @frame_material")
    df = df.assign(distance_km=lambda _df: calculate_distance_vectorized(_df, lat, lon))
    df = df.query("distance_km <= @distance_km_threshold")
    return df.sort_values(by="predicted_price_diff", ascending=False)[
        ["ad_title", "url", "price", "currency", "predicted_price_cad", "predicted_price_diff", "distance_km"]
    ]


if __name__ == "__main__":
    load_dotenv(os.path.expanduser("~/.openai/credentials"))
    load_dotenv()
    dataset = pd.read_parquet("/tmp/latest_base_chat_data.parquet")
    df = get_relevant_bikes(
        price_limit_cad=100000,
        front_travel_range=(100, 150),
        rear_travel_range=(100, 150),
        frame_size="L",
        # frame_material="Aluminum",
        lon=-113,
        lat=52,
        distance_km_threshold=200,
        dataset=dataset,
    )
    print(df)
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "test"}, {"role": "user", "content": "Hello!"}],
    )
    print(response)
