from typing import List
import pandas as pd


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
        df[col] = df.loc[:,col].fillna("0")
        # Replace all non digits or decimal
        df[col] = df[col].astype(str).str.replace(
            "[^\d.]","", regex=True).astype(float)
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
        if df[c].dtype == 'object':
            df[c] = df[c].astype(str)
    return df


def generate_changelog(previous_ads:pd.DataFrame, updated_ads:pd.DataFrame, cols_to_check: List[str]) -> pd.DataFrame:
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
    if (previous_ads.url == updated_ads.url).sum() != len(previous_ads) \
            or len(previous_ads) != len(updated_ads):
        raise ValueError(
            "previous_ads and updated_ads must have same number of ads, and identical ordering of url column")
    changes = []
    for url in previous_ads.url:
        for col in cols_to_check:
            old_value = previous_ads.loc[previous_ads.url == url, col]
            old_dtype = old_value.dtype
            new_value = updated_ads.loc[updated_ads.url == url, col].astype(
                old_dtype)
            category = previous_ads.loc[previous_ads.url == url, "Category:"]
            update_date = pd.to_datetime(
                updated_ads.loc[updated_ads.url == url, "Last Repost Date:"]).dt.date
            if (old_value != new_value).all():
                changed = {
                    "url": url,
                    "category": category.values[0],
                    "field": col,
                    "old_value": old_value.values[0],
                    "new_value": new_value.values[0],
                    "update_date": update_date.values[0]
                }
                changes.append(changed)

    return pd.DataFrame(changes)
