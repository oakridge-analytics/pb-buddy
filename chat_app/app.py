import os
from typing import Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from swarm import Agent, Swarm

from chat_app.prompts import ASSISTANT_PROMPT
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


if __name__ == "__main__":
    load_dotenv(os.path.expanduser("~/.openai/credentials"))
    load_dotenv(".env")

    dataset = pd.read_parquet("/tmp/latest_base_chat_data.parquet").assign(
        front_travel=lambda df: df.front_travel.fillna(0), rear_travel=lambda df: df.rear_travel.fillna(0)
    )

    client = Swarm()
    agent = Agent(
        name="bike assistant",
        model="gpt-4o-mini",
        instructions=ASSISTANT_PROMPT,
        functions=[get_user_location, get_relevant_bikes],
    )
    from swarm.repl import run_demo_loop

    run_demo_loop(agent, stream=True, debug=True)
