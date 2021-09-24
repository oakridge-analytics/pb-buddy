import pandas as pd
import os
import pymongo
import certifi


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
        df
        .assign(datetime_scraped=lambda x: pd.to_datetime(x.datetime_scraped,utc=True).dt.tz_convert("US/Mountain"))
        .sort_values("datetime_scraped", ascending=False)
        .drop_duplicates(subset="url")
    )


def get_dataset(category_num: int, data_type: str) -> pd.DataFrame:
    """Get active, sold or changed ad data for a given category number. Normalizes column
    names into nicer forms. Requires env variables "MONGO_USER", "MONGO_PASS" to
    be set correctly for MongoDB cluster.

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
    if data_type not in ["base","sold","changes"]:
        raise ValueError("data")

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
        query_result = list(database_mongo.find(
            {'category_num': category_num}))

    if len(query_result) == 0:
        df_out = pd.DataFrame({"url":[]})
    else:
        df_out = (
            pd.DataFrame(query_result)
            .assign(datetime_scraped=lambda x: pd.to_datetime(x.datetime_scraped))
        )
    return df_out


def write_dataset(category_df: pd.DataFrame, data_type: str):
    """Write active, sold or changed ad data from a given DataFrame. Normalizes column
    names into nicer forms (lower case, "_" separated). Requires env variables "MONGO_USER", "MONGO_PASS" to
    be set correctly for MongoDB cluster.

    Parameters
    ----------
    category_df : pd.DataFrame
        Data to be written to MongoDB collection. 
    data_type : str
        Either "base","sold" or "changes" for active ads and sold ads respectively.

    """
    if data_type not in ["base","sold","changes"]:
        raise ValueError("data")
    db = get_mongodb()

    if data_type == "base":
        database_mongo = db.base_data
    elif data_type == "sold":
        database_mongo = db.sold_data
    else:
        database_mongo = db.change_data

    database_mongo.insert_many(
        category_df.to_dict(orient="records")
    )


def get_mongodb():
    """Return a `pymongo` object to work with pb-buddy data"""
    mongo_user = os.environ['MONGO_USER']
    mongo_pass = os.environ['MONGO_PASS']
    # Replace the uri string with your MongoDB deployment's connection string.
    conn_str = f"mongodb+srv://{mongo_user}:{mongo_pass}@cluster0.sce8f.mongodb.net/pb-buddy?retryWrites=true&w=majority"
    # set a 5-second connection timeout
    client = pymongo.MongoClient(
        conn_str, tlsCAFile=certifi.where(),serverSelectionTimeoutMS=5000)
    db = client['pb-buddy']
    return pb
