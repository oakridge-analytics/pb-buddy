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


def fix_dtypes(df: pd.DataFrame, colnames: List[str], dtype_desired:int = int):
    """Convert multiple columns to a given dtype in a Pandas Dataframe

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to fix columns dtypes in
    colnames : List[str]
        List of columns to change dtype
    dtype : int, optional
        Datatype to cast to, by default int
    """
    df = df.copy()
    for col in colnames:
        df[col] = df[col].astype(dtype_desired)
    return df
