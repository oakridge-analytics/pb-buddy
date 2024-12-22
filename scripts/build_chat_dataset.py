# %%
import importlib
import json
import logging
import re
from datetime import datetime
from pathlib import Path

import fire
import pandas as pd
from dotenv import load_dotenv

from pb_buddy import utils as ut
from pb_buddy.data_processors import get_dataset
from pb_buddy.modelling.inference import load_model_from_s3, predict_price
from pb_buddy.utils import get_exchange_rate, mapbox_geocode

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


def get_bike_dataset():
    """Get bike dataset from cache if available, otherwise fetch from database"""
    cache_dir = Path("/tmp/.cache")
    cache_dir.mkdir(exist_ok=True)

    today = datetime.now().strftime("%Y%m%d")
    cache_path = cache_dir / f"bike_dataset_{today}_base.parquet"

    if cache_path.exists():
        logging.info(f"Loading bike dataset from cache: {cache_path}")
        return pd.read_parquet(cache_path)

    # Get fresh data if not in cache
    df = get_dataset(category_num=-1, data_type="base", region_code=None)

    # Convert ObjectId to string before saving
    df["_id"] = df["_id"].astype(str)
    df = df.pipe(ut.convert_to_float, colnames=["price", "watch_count", "view_count"])
    # Cache the data
    df.to_parquet(cache_path)
    return df


def get_bike_categories(categories: list[str]) -> list[str]:
    """Filter categories to only include bike categories."""
    exclude_keywords = ["parts", "frames", "bags", "racks", "stands", "rims", "mx"]

    candidate_categories = []
    for k in set(categories):
        if "bike" in k.lower() and not any(keyword in k.lower() for keyword in exclude_keywords):
            candidate_categories.append(k)

    return sorted(candidate_categories)


def clean_travel_column(value: str) -> float:
    """Clean travel column values to extract decimal numbers."""
    if pd.isna(value):
        return pd.NA

    # Extract numbers (including decimals) from string
    numbers = re.findall(r"\d*\.?\d+", str(value))
    if not numbers:
        return pd.NA

    # Convert to float and handle 0 values
    result = float(numbers[0])
    return pd.NA if result == 0 else result


def clean_travel_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Clean both travel columns in the dataframe."""
    for col in ["front_travel", "rear_travel"]:
        df[col] = df[col].apply(clean_travel_column)
    return df


def convert_prices_from_cad(
    df: pd.DataFrame, to_currency_col: str, cad_price_col: str, price_col_out: str
) -> pd.DataFrame:
    """Convert prices from CAD to other currencies."""
    # Initialize output column with CAD prices and exchange rate of 1
    df[price_col_out] = df[cad_price_col]

    # Convert non-CAD currencies using exchange rates
    for currency in df[df[to_currency_col] != "CAD"][to_currency_col].unique():
        exchange_rate = get_exchange_rate(from_currency="CAD", to_currency=currency)
        df.loc[df[to_currency_col] == currency, price_col_out] = df[cad_price_col] * exchange_rate
    return df


def convert_prices_to_cad(df: pd.DataFrame, currency_col: str, price_col: str, price_col_out: str) -> pd.DataFrame:
    """Convert prices from various currencies to CAD.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    currency_col : str
        Name of column containing currency codes
    price_col : str
        Name of column containing prices in original currencies
    price_col_out : str
        Name of column to store CAD prices

    Returns
    -------
    pd.DataFrame
        Dataframe with new column containing prices in CAD
    """
    df = df.copy()
    df[price_col_out] = df[price_col]  # Copy original prices

    # Convert each non-CAD currency to CAD
    for currency in df[df[currency_col] != "CAD"][currency_col].unique():
        exchange_rate = get_exchange_rate(from_currency=currency, to_currency="CAD")
        df.loc[df[currency_col] == currency, price_col_out] = df[price_col] * exchange_rate

    return df


def get_size_map() -> dict[str, str]:
    resource_list = importlib.resources.files("pb_buddy.resources")

    with importlib.resources.as_file(resource_list) as fh:
        with open(fh / "size_map.json", "r") as f:
            size_map = json.load(f)

    return size_map


def get_location_map() -> list[dict[str, float]]:
    """Get JSON mapping of locations to longitude and latitude.

    Location map is a JSON file with the following structure:
    [{"location": "location_name", "longitude": longitude, "latitude": latitude}]

    Where location_name is city_name, country_name.

    Returns
    -------
    list[dict[str, float]]
        List of dictionaries with location, longitude, and latitude
    """
    resource_list = importlib.resources.files("pb_buddy.resources")

    with importlib.resources.as_file(resource_list) as fh:
        with open(fh / "location_map.json", "r") as f:
            location_map = json.load(f)

    return location_map


def geocode_missing_locations(df_bikes: pd.DataFrame) -> pd.DataFrame:
    """
    Geocode any locations in the dataframe that are missing coordinates.

    Args:
        df_bikes: DataFrame containing bike listings

    Returns:
        DataFrame with geocoded coordinates filled in
    """
    missing_coords = df_bikes["lon"].isna().sum()
    if missing_coords > 0:
        logging.info(f"Geocoding {missing_coords} locations...")
        # Get unique locations that need geocoding
        unique_locations = df_bikes.loc[df_bikes["lon"].isna(), "location"].unique()
        logging.info(f"Found {len(unique_locations)} unique locations to geocode")

        # Create mapping of location to coordinates
        location_coords = {loc: mapbox_geocode(loc) for loc in unique_locations}

        # Update dataframe using the mapping
        for location, (lon, lat) in location_coords.items():
            mask = (df_bikes["lon"].isna()) & (df_bikes["location"] == location)
            df_bikes.loc[mask, ["lon", "lat"]] = lon, lat

    return df_bikes


# %%
# df = get_bike_dataset()
# categories = get_bike_categories(df["category"].unique().tolist())
# df_bikes = df.query("category.isin(@categories)").copy()


# %%
def main(model_uuid: str = "b9f12bc7cbd34fc39bf300b88f2ab57a", subsample_size: int | None = None):
    """
    Main function to process bike dataset and add predictions.

    Args:
        model_uuid: UUID of the model to use for predictions
    """
    logging.info("Loading base dataset...")
    df = get_bike_dataset()

    logging.info("Filtering to bike categories...")
    bike_categories = get_bike_categories(df["category"].unique().tolist())
    df_bikes = df.query("category.isin(@bike_categories)").copy()
    if subsample_size is not None:
        df_bikes = df_bikes.sample(n=subsample_size)

    logging.info("Cleaning travel columns...")
    df_bikes = clean_travel_columns(df_bikes)

    logging.info("Removing zl currency...")
    df_bikes = df_bikes.query("currency != 'zl'")

    logging.info(f"Loading model {model_uuid}...")
    predictor, transformer = load_model_from_s3(model_uuid)

    logging.info("Making predictions...")
    predictions = predict_price(
        predictor,
        transformer,
        df_bikes.assign(price_cpi_adjusted_CAD=0).assign(
            original_post_date=lambda x: pd.to_datetime(x.original_post_date, format="mixed")
        ),
    )

    df_bikes = convert_prices_to_cad(df_bikes, currency_col="currency", price_col="price", price_col_out="price_cad")
    df_bikes["predicted_price_cad"] = predictions.values
    df_bikes = convert_prices_from_cad(
        df_bikes, to_currency_col="currency", cad_price_col="predicted_price_cad", price_col_out="predicted_price"
    )
    df_bikes["predicted_price_diff"] = df_bikes["predicted_price"] - df_bikes["price"]
    df_bikes["model_uuid"] = model_uuid

    # Add location coordinates
    df_lat_lon = pd.DataFrame(get_location_map())
    df_bikes = df_bikes.merge(df_lat_lon, on="location", how="left")
    df_bikes = geocode_missing_locations(df_bikes)

    logging.info("Data cleaning and normalization...")
    df_bikes["material"] = df_bikes["material"].str.strip()
    size_map = get_size_map()
    df_bikes["frame_size_normalized"] = df_bikes["frame_size"].str.strip().map(size_map)

    logging.info("Processing complete!")
    logging.info(f"Final dataset size: {len(df_bikes):,} rows")
    # Save to S3 with timestamp and latest versions
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_key = f"data/chat/{timestamp}_base_chat_data.parquet"
    latest_key = "data/chat/latest_base_chat_data.parquet"

    # logging.info(f"Saving to S3 bucket bike-buddy with key {timestamped_key}...")
    # df_bikes.to_parquet(f"s3://bike-buddy/{timestamped_key}")

    logging.info("Saving latest version...")
    df_bikes.to_parquet(f"s3://bike-buddy/{latest_key}")
    df_bikes.to_parquet("/tmp/latest_base_chat_data.parquet")


if __name__ == "__main__":
    fire.Fire(main)
