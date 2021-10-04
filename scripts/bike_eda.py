#%%
import pandas as pd
import numpy as np
import seaborn as sns
import re
import pb_buddy.utils as ut

#%% Helpers

#%%
ad_data = pd.read_csv("category_75_ad_data.csv")
ad_data.columns = [
    x.lower().replace(":", "").replace(" ", "_") for x in ad_data.columns
]
# Add some helper columns
ad_data = ad_data.assign(
    ad_title=lambda x: x.ad_title.astype("string"),
    year=lambda x: x.ad_title.str.extract("^([0-9]{4})"),  # .astype(int),
    price=lambda x: x.price.str.replace("[€£$,]", "", regex=True).astype(float),
    material=lambda x: x.material.str.strip(),
    view_count=lambda x: x.view_count.str.strip().str.replace(",", "").astype(int),
)

ad_data = ad_data.dropna(subset=["year"], inplace=False).assign(
    year=lambda x: x.year.astype(int)
)
ad_data["price_cad"] = ad_data.apply(
    lambda x: ut.convert_to_cad(x.price, x.currency), axis=1
)
# %%
sns.displot(
    data=ad_data.query("year > 2015 and price_cad < 30000"),
    x="price_cad",
    col="year",
    col_wrap=2,
    kind="hist",
    hue="year",
).set(title="List Prices Distribution By Year")

# %%
sns.scatterplot(data=ad_data, x="view_count", y="price_cad", hue="material")
# %%
