#%%
from multiprocessing.sharedctypes import Value
import pandas as pd
import numpy as np
import seaborn as sns
import re

#%% Helpers
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


#%%
ad_data = pd.read_csv("20210905_all_mountain_ad_data.csv")
ad_data.columns = [
    x.lower().replace(":", "").replace(" ", "_") for x in ad_data.columns
]
# Add some helper columns
ad_data = ad_data.assign(
    ad_title=lambda x: x.ad_title.astype("string"),
    year=lambda x: x.ad_title.str.extract("^([0-9]{4})"),  # .astype(int),
    price=lambda x: x.price.str.replace("[€£$,]", "", regex=True).astype(float),
    material=lambda x: x.material.str.strip(),
    view_count=lambda x: x.view_count.str.strip().str.replace(",", "").astype(),
)

ad_data = ad_data.dropna(subset=["year"], inplace=False).assign(
    year=lambda x: x.year.astype(int)
)
ad_data["price_cad"] = ad_data.apply(
    lambda x: convert_to_cad(x.price, x.currency), axis=1
)
# %%
sns.displot(
    data=ad_data.query("year > 2015"), x="price_cad", kind="kde", hue="year"
).set(title="List Prices Distribution By Year")

# %%
sns.scatterplot(data=ad_data, x="view_count", y="price_cad", hue="material")
# %%
