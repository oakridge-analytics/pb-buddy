import importlib
import json

import fire
from dotenv import load_dotenv

import pb_buddy.data_processors as dt
from pb_buddy.constants import SPECS_BLOB_NAME, SPECS_CONTAINER_NAME


def get_specs_dataset():
    df_specs = dt.stream_parquet_to_dataframe(blob_container=SPECS_CONTAINER_NAME, blob_name=SPECS_BLOB_NAME)
    df_specs = df_specs.pipe(
        lambda _df: _df.assign(
            **_df.spec_url.str.extract(r"bikes/(.*)/(.*)/(.*)").rename(
                columns={0: "manufacturer", 1: "year", 2: "model"}
            )
        )
    ).assign(
        # combine year manufacturer model, remove all non alphanumeric characters, and lowercase
        year_manufacturer_model=lambda _df: _df.year
        + " "
        + _df.manufacturer
        + " "
        + _df.model.str.replace("-", " ").str.lower()
    )
    return df_specs


def build_year_manufacturer_model_mapping():
    df_specs = get_specs_dataset()

    # Extract all years, then manufacturer for each year as a dictionary, then within each manufacturer,
    # extract all models as a dictionary
    year_manufacturer_model_mapping = {
        year: {
            manufacturer: list(
                df_specs.query(f"year == '{year}' and manufacturer == '{manufacturer}'")
                .model.str.replace(r"-", " ")
                .str.lower()
                .unique()
            )
            for manufacturer in df_specs.query(f"year == '{year}'").manufacturer.unique()
        }
        for year in df_specs.year.unique()
    }

    # Save mapping to "resources/year_make_model_mapping.json"
    resource_list = importlib.resources.files("pb_buddy.resources")

    with importlib.resources.as_file(resource_list) as fh:
        with open(fh / "year_manufacturer_model_mapping.json", "w") as f:
            json.dump(year_manufacturer_model_mapping, f)


def get_year_manufacturer_model_mapping():
    """
    Return current mapping from "resources/year_manufacturer_model_mapping.json"
    """
    resource_list = importlib.resources.files("pb_buddy.resources")

    with importlib.resources.as_file(resource_list) as fh:
        with open(fh / "year_manufacturer_model_mapping.json", "r") as f:
            year_manufacturer_model_mapping = json.load(f)

    return year_manufacturer_model_mapping


if __name__ == "__main__":
    load_dotenv()
    fire.Fire(get_year_manufacturer_model_mapping)
