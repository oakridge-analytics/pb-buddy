from matplotlib import blocking_input
import pandas as pd
import os
import pymongo
import certifi
from typing import List
from io import BytesIO
from tqdm import tqdm
from azure.storage.blob import BlobServiceClient


def get_latest_by_scrape_dt(df: pd.DataFrame) -> pd.DataFrame:
    """Get latest rows for each 'url' sorted by 'datetime_scraped'.
    For deduplicating ad data. Converts datetime_scraped to UTC, then Mountain
    time zone for consistent comparisons.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe of ads. Must contain 'url' and 'datetime_scraped' cols

    Returns
    -------
    pd.DataFrame
        Dataframe deduplicated for each 'url' using 'datetime_scraped'
    """
    return (
        df.assign(
            datetime_scraped=lambda x: pd.to_datetime(
                x.datetime_scraped, utc=True
            ).dt.tz_convert("US/Mountain")
        )
        .sort_values("datetime_scraped", ascending=False)
        .drop_duplicates(subset="url")
    )


def get_dataset(category_num: int, data_type: str) -> pd.DataFrame:
    """Get active, sold or changed ad data for a given category number. Normalizes column
    names into nicer forms. Requires env variable "COSMOS_CONN_STR" to
    be set for MongoDB api on CosmosDB access.

    Use `category_num` == -1 to access all data of a certain type.

    Parameters
    ----------
    category_num : int
        Category num to lookup.
    data_type : str
        Either "base","sold" or "changes" for active ads and sold ads respectively.

    Returns
    -------
    pd.DataFrame
        Dataframe of existing data for given category
    """
    if data_type not in ["base", "sold", "changes"]:
        raise ValueError("data_type should be one of 'base','sold','changes'")

    db = get_mongodb()

    if data_type == "base":
        database_mongo = db.base_data
    elif data_type == "sold":
        database_mongo = db.sold_data
    else:
        database_mongo = db.change_data

    if category_num == -1:
        # Blank query returns all data
        query_result = list(database_mongo.find({}))
    else:
        query_result = list(database_mongo.find({"category_num": category_num}))

    if len(query_result) == 0:
        df_out = pd.DataFrame({"url": []})
    else:
        df_out = pd.DataFrame(query_result)

        if "datetime_scraped" in df_out.columns:
            df_out = df_out.assign(
                datetime_scraped=lambda x: pd.to_datetime(
                    x.datetime_scraped, utc=True
                ).dt.tz_convert("US/Mountain")
            )
    return df_out


def write_dataset(category_df: pd.DataFrame, data_type: str):
    """Write active/sold/changed ad data from a given DataFrame. Normalizes column
    names into nicer forms (lower case, "_" separated). Uses unordered insert, ignores index errors.
    Requires env variable "COSMOS_CONN_STR" to
    be set for MongoDB api on CosmosDB access.

    Parameters
    ----------
    category_df : pd.DataFrame
        Data to be written to MongoDB collection.
    data_type : str
        Either "base","sold","changes" for active ads, sold ads, or changes

    """
    if data_type not in ["base", "sold", "changes"]:
        raise ValueError("data")
    db = get_mongodb()

    if data_type == "base":
        database_mongo = db.base_data
    elif data_type == "sold":
        database_mongo = db.sold_data
    else:
        database_mongo = db.change_data

    # normalize column names
    category_df.columns = [
        x.replace(":", "").replace(" ", "_").lower() for x in category_df.columns
    ]

    # Insert as many as possible, return without printing errors for now.
    try:
        database_mongo.insert_many(category_df.to_dict(orient="records"), ordered=False)
    except pymongo.errors.BulkWriteError:
        pass


def update_base_data(
    category_df: pd.DataFrame, index_col: str, cols_to_update: List[str]
):
    """Update MongoDB collection for base ad data documents uniquely identified by `index_col` and update fields
    with values in `cols_to_update` from `category_df`. Requires env variable "COSMOS_CONN_STR" to
    be set for MongoDB api on CosmosDB access.

    Parameters
    ----------
    category_df : pd.DataFrame
        Dataframe of data containing `index_col` and `cols_to_update` data to update MongoDB
        documents with.
    index_col : str
        Column to match each row of `category_df` with a MongoDB document. Must match a field in MongoDB documents.
    cols_to_update : List[str]
        Columns to use for updating data in MongoDB documents. Must all be existing fields in MongoDB documents.
    """
    if len(set(category_df.columns).intersection([index_col] + cols_to_update)) != len(
        [index_col] + cols_to_update
    ):
        raise ValueError(
            "index_col and cols_to_update must all be present in category_df"
        )

    db = get_mongodb()
    database_mongo = db.base_data

    for row in category_df.itertuples():
        updates_dict = {x: getattr(row, x) for x in cols_to_update}
        database_mongo.update_one(
            {index_col: getattr(row, index_col)}, {"$set": updates_dict}
        )


def remove_from_base_data(removal_df: pd.DataFrame, index_col: str):
    """Remove documents from MongoDB `base_data` database based on `index_col`. Requires env variable "COSMOS_CONN_STR" to
    be set for MongoDB api on CosmosDB access.

    Parameters
    ----------
    removal_df : pd.DataFrame
        Dataframe that contains `index_col` of all documents to remove
    index_col : str
        Column to match on in documents fields.
    """
    if index_col not in removal_df.columns:
        raise ValueError("index_col must be in removal_df.columns")

    db = get_mongodb()
    database_mongo = db.base_data

    for val in removal_df[index_col]:
        database_mongo.delete_one({index_col: val})


def get_mongodb():
    """
    Return a `pymongo` database object to work with pb-buddy data
    Requires env variable "COSMOS_CONN_STR" to
    be set for MongoDB api on CosmosDB access.
    """
    # mongo_user = os.environ['MONGO_USER']
    # mongo_pass = os.environ['MONGO_PASS']
    # Replace the uri string with your MongoDB deployment's connection string.
    conn_str = os.environ["COSMOS_CONN_STR"]
    # set a 5-second connection timeout
    client = pymongo.MongoClient(
        conn_str,
        tlsCAFile=certifi.where(),
        serverSelectionTimeoutMS=20000,
        connectTimeoutMS=30000,
        socketTimeoutMS=None,
        socketKeepAlive=True,
        connect=False,
        maxPoolsize=1,
    )
    db = client["pb-buddy"]
    return db


def stream_parquet_to_blob(df: pd.DataFrame, blob_name: str, blob_container: str):
    """Stream pandas DataFrame to parquet.gzip blob in Azure Blob storage.
    Uses `pyarrow` and requires env variable to be set:
    AZURE_STORAGE_CONN_STR=Connection string to authenticate with.

    Parameters
    ----------
    df : pd.DataFrame
        Pandas DataFrame to be uploaded to Azure Blob
    blob_name : str
        File name to upload as in container
    blob_container : str
        Container to upload to
    """    
    # Create a blob client using the local account credentials
    blob_service_client = BlobServiceClient.from_connection_string(
        os.environ["AZURE_STORAGE_CONN_STR"]
    )

    # Create a unique name for the container
    container_name = blob_container

    # Create a blob client using the container client
    blob_client = blob_service_client.get_blob_client(
        container=container_name,
        blob=blob_name
    )

    parquet_file = BytesIO()
    df.to_parquet(parquet_file, engine='pyarrow', compression="gzip")
    parquet_file.seek(0)  # change the stream position back to the beginning after writing

    blob_client.upload_blob(
        data=parquet_file
    )

def stream_parquet_to_dataframe( blob_name: str, blob_container: str) -> pd.DataFrame:
    """Stream pandas DataFrame as parquet.gzip blob from Azure Blob storage.
    Uses `pyarrow` and requires env variable to be set:
    AZURE_STORAGE_CONN_STR=Connection string to authenticate with.

    Parameters
    ----------
    blob_name : str
        File name to upload as in container
    blob_container : str
        Container to upload to

    Returns
    -------
    pd.DataFrame
        DataFrame when blob is downloaded
    """
    # Create a blob client using the local account credentials
    blob_service_client = BlobServiceClient.from_connection_string(
        os.environ["AZURE_STORAGE_CONN_STR"]
    )

    # Create a unique name for the container
    container_name = blob_container

    # Create a blob client using the container client
    blob_client = blob_service_client.get_blob_client(
        container=container_name,
        blob=blob_name
    )

    stream_download = blob_client.download_blob()
    stream = BytesIO()
    stream_download.readinto(stream)

    df = pd.read_parquet(stream, engine="pyarrow")

    return df