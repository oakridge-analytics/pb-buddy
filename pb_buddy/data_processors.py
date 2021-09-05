import pandas as pd


def get_latest_by_scrape_dt(df: pd.DataFrame) -> pd.DataFrame:
    """Get latest rows for each 'url' sorted by 'datetime_scraped'.
    For deduplicating ad data.

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
        .assign(datetime_scraped=lambda x: pd.to_datetime(x.datetime_scraped))
        .sort_values("datetime_scraped", ascending=False)
        .drop_duplicates(subset="url")
    )


def get_category_base_data(category_num: int) -> pd.DataFrame:
    """Get active ad data for a given category number.

    Parameters
    ----------
    category_num : int
        Category num to lookup.

    Returns
    -------
    pd.DataFrame
        Dataframe of existing data for given category
    """
    return (
        pd.read_parquet(
            f"data/base_data/category_{category_num}_ad_data.parquet.gzip",
        )
    )


def get_category_sold_data(category_num: int) -> pd.DataFrame:
    """Get sold ad data for the category.

    Parameters
    ----------
    category_num : int
        Category number

    Returns
    -------
    pd.DataFrame
        Sold ad data
    """
    return (
        pd.read_parquet(
            f"data/sold_ads/category_{category_num}_sold_ad_data.parquet.gzip",
        )
    )
