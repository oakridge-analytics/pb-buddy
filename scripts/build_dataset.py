#%%
import pandas as pd
import os
import json
from tqdm import tqdm
from azure.storage.blob import BlobServiceClient, BlobClient
from dotenv import load_dotenv

load_dotenv("../.env")

#%% Preprocess back filled data
# Read backfill list of dicts
def read_backfill_library(path):
    """
    Read backfill_library.json
    """
    with open(path, "r") as f:
        backfill_library = json.load(f)
    return backfill_library


backfill_list = read_backfill_library("../data/backfill_library.json")
restrictions_list = read_backfill_library("../data/restrictions_backfill_library.json")

# %%
df_backfill = pd.DataFrame(
    backfill_list
)
df_backfill.columns = [
    x.replace(" ", "_").replace(":", "").lower() for x in df_backfill.columns
]

# %%
df_restrictions = pd.DataFrame(restrictions_list)
df_restrictions.columns = [
    x.replace(" ", "_").replace(":", "").lower() for x in df_restrictions.columns
]
# %%
df_historical_data = (
    df_backfill
    .merge(df_restrictions, on="url", how="inner", validate="1:1")
)

# %%
# Need to strip all columns to remove leading and trailing whitespace
# Cast date columns to datetime
datetime_cols = ["original_post_date","last_repost_date","datetime_scraped"]
df_historical_data = df_historical_data.apply(
    lambda x: x.str.strip() if x.dtype == "object" else x,
    axis=0)
df_historical_data[datetime_cols] = (
    df_historical_data[datetime_cols] 
    .apply(
        pd.to_datetime,
        axis=0
    )
)
# %%
df_historical_data.dtypes

#%%
# restrict to just sold ads
df_historical_data = (
    df_historical_data.query("still_for_sale.str.contains('Sold')")
)

#%%
# Save to upload to blob
df_historical_data.to_csv("../data/historical_data.csv", index=False)

#%%
# Upload!
connect_str = os.environ["AZURE_STORAGE_CONN_STR"]
blob_service_client = BlobServiceClient.from_connection_string(connect_str)
container_name = 'pb-buddy-historical'
blob_client = blob_service_client.get_blob_client(container=container_name, blob="historical_sold_data.csv")

with open(os.path.join("../data/", "historical_data.csv"), "rb") as data:
    blob_client.upload_blob(data)   

# %%
