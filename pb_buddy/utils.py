import os
from typing import List

import pandas as pd
import requests
import yfinance as yf


def convert_to_cad(price: float, currency: str) -> float:
    """A simple conversion function for USD, GBP, EUR to CAD.
    As of 2021-09-5.

    Parameters
    ----------
    price : float
        Price in `currency`
    currency : str
        Currency - one of "USD", "GBP", "EUR". "CAD" will return same price

    Returns
    -------
    float
        The price in CAD
    """
    if currency == "USD":
        return price * 1.25
    elif currency == "GBP":
        return price * 1.74
    elif currency == "EUR":
        return price * 1.49
    elif currency == "CAD":
        return price
    else:
        print(currency)
        raise ValueError("Currency must be one of USD, GBP, EUR, CAD")


def flatten(nested_list: List[List]) -> List:
    """Flatten nested lists to a list

    Parameters
    ----------
    nested_list : List[List]
        The nested list

    Returns
    -------
    List
        Flattened list
    """
    return [item for sublist in nested_list for item in sublist]


def convert_to_float(df: pd.DataFrame, colnames: List[str]) -> pd.DataFrame:
    """Convert multiple columns to float in a Pandas Dataframe.
    Fills NA with 0 before converting.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to fix columns dtypes in
    colnames : List[str]
        List of columns to change dtype

    Returns
    -------
    pd.DataFrame:
        Dataframe with specified columns cast to float and NA filled with 0.
    """
    df = df.copy()
    for col in colnames:
        df[col] = df.loc[:, col].fillna("0")
        # Replace all non digits or decimal
        df[col] = df[col].astype(str).str.replace("[^\d.]", "", regex=True)
        df = df.loc[df[col] != "", :]
        df[col] = df[col].astype(float)
    return df


def cast_obj_to_string(df: pd.DataFrame) -> pd.DataFrame:
    """Converts object type columns (potentially of mixed types) to str
    for use with pyarrow. Prevents issues of mixed Python types in a column.


    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to fix column types from object -> str

    Returns
    -------
    pd.DataFrame :
        Dataframe with all object type columns cast to str

    """
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].astype(str)
    return df


def generate_changelog(previous_ads: pd.DataFrame, updated_ads: pd.DataFrame, cols_to_check: List[str]) -> pd.DataFrame:
    """Generate a log of column, old value, new value between two
    dataframes of ads. Both dataframes must be sorted by URL and have the
    same membership of ad urls. Uses Last Repost Date of updated version to timestamp
    when change occured.

    Parameters
    ----------
    previous_ads : pd.DataFrame
        Dataframe containing old ad information. Must contain column 'url'.
    updated_ads : pd.DataFrame
        Dataframe containing potentially new ad information. Must contain column 'url'.
    cols_to_check : List[str]
        Columns to log changes for

    Returns
    -------
    pd.DataFrame
        Dataframe with columns:
            - url
            - category
            - field
            - old_value
            - new_value
            - update_date
    """
    if (previous_ads.url == updated_ads.url).sum() != len(previous_ads) or len(previous_ads) != len(updated_ads):
        raise ValueError(
            "previous_ads and updated_ads must have same number of ads, and identical ordering of url column"
        )
    changes = []
    for url in previous_ads.url:
        for col in cols_to_check:
            old_value = previous_ads.loc[previous_ads.url == url, col]
            old_dtype = old_value.dtype
            new_value = updated_ads.loc[updated_ads.url == url, col].astype(old_dtype)
            category = previous_ads.loc[previous_ads.url == url, "category"]
            update_date = pd.to_datetime(updated_ads.loc[updated_ads.url == url, "last_repost_date"]).dt.date

            # if comparing string like, strip first
            if old_dtype in [str, object]:
                if (old_value.str.strip() != new_value.str.strip()).all():
                    changed = {
                        "url": url,
                        "category": category.values[0],
                        "field": col,
                        "old_value": old_value.values[0],
                        "new_value": new_value.values[0],
                        "update_date": str(update_date.values[0]),
                    }
                    changes.append(changed)
            else:
                if (old_value != new_value).all():
                    changed = {
                        "url": url,
                        "category": category.values[0],
                        "field": col,
                        "old_value": old_value.values[0],
                        "new_value": new_value.values[0],
                        "update_date": str(update_date.values[0]),
                    }
                    changes.append(changed)

    return pd.DataFrame(changes)


def convert_currency_symbol(symbol: str) -> str:
    currency_symbols = {
        "$": "USD",
        "€": "EUR",
        "£": "GBP",
        "¥": "JPY",
        "₣": "CHF",
        "C$": "CAD",
        "A$": "AUD",
        "kr": "SEK",
        "₹": "INR",
        "₽": "RUB",
    }
    return currency_symbols.get(symbol, symbol)


def get_exchange_rate(from_currency: str, to_currency: str) -> float:
    """Get the current exchange rate between two currencies using Yahoo Finance."""
    ticker = f"{from_currency}{to_currency}=X"
    exchange_rate = yf.Ticker(ticker).info["regularMarketPreviousClose"]
    return exchange_rate


def convert_currency(amount: float, from_currency: str, to_currency: str) -> float:
    if from_currency == to_currency:
        return amount

    from_currency = convert_currency_symbol(from_currency)
    to_currency = convert_currency_symbol(to_currency)

    exchange_rate = get_exchange_rate(from_currency, to_currency)
    return amount * exchange_rate


def mapbox_geocode(address: str) -> tuple[float, float]:
    """Geocode an address using Mapbox API.

    Requires MAPBOX_API_KEY in environment variables.

    Returns
    -------
    tuple[float, float]
        Longitude and latitude of the address
    """
    url = f"https://api.mapbox.com/search/geocode/v6/forward?q={address}&access_token={os.getenv('MAPBOX_API_KEY')}"
    response = requests.get(url)
    # print(response.json())
    # Get lat,lon from response
    return tuple(response.json()["features"][0]["geometry"]["coordinates"])
