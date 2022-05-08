---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.8
  kernelspec:
    display_name: Python 3.8.12 ('pb-buddy-BzHdUS67-py3.8')
    language: python
    name: python3
---

# Pb-Buddy Scrape Report

```python
%%capture
import pandas as pd
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Custom code
import pb_buddy.scraper as scraper
import pb_buddy.utils as ut
import pb_buddy.data_processors as dt
from pb_buddy.resources import category_dict

load_dotenv("../.env")
warnings.filterwarnings("ignore")
```

```python
plt.rcParams['figure.figsize'] = (12,8)
plt.style.use('seaborn')
```

```python

# %%
# Clean base data -----------------------------------------------------
df_base_data = dt.get_dataset(-1, data_type="base")
df_sold_data = dt.get_dataset(-1, data_type="sold")
```

```python
# Get backfill data over all years of just sold ads in North America
df_historical_sold = (
    dt.stream_parquet_to_dataframe("historical_data.parquet.gzip", "pb-buddy-historical")
    .query("still_for_sale.str.contains('Sold')")
)
```

```python
df_combined = (
    pd.concat([
        df_base_data.assign(type = "base"),
        df_sold_data.assign(type="sold"),
        df_historical_sold.assign(type="sold")
        ],
         axis=0)
.query("currency.isin(['CAD','USD'])")
.assign(
    original_post_date = lambda x: pd.to_datetime(x.original_post_date).dt.date,
    original_post_year = lambda x: pd.to_datetime(x.original_post_date).dt.year,
    original_post_day = lambda x: pd.to_datetime(x.original_post_date).dt.dayofyear,
    original_post_month = lambda x: pd.to_datetime(x.original_post_date).dt.month,
    category = lambda x: x.category.str.strip()
    )
)

```

## Ad Counts in Total Over Time (Sold + Still Active Ads)

```python
(
    df_combined
    .astype({"original_post_year":str})
    .groupby(["original_post_year","currency","original_post_day"])
    .count()
    [["url"]]
    .rolling(window=14)
    .mean()
    .reset_index()
    .rename(columns={"url":"count_ads"})
    .pipe(
        (sns.relplot, "data"), 
        x="original_post_day",
        y="count_ads",
        col="currency",
        hue="original_post_year",
        height=6,
        aspect=1.5,
        kind="line")
)
plt.suptitle("Count of Active+Sold Ads Over Time - 14 Day Rolling Average");
```

## Ad Counts by Type

```python
(
    df_combined
    .query("original_post_date > @pd.to_datetime('01-OCT-2021').date()")
    .groupby(["original_post_date","type"])
    .count()
    [["url"]]
    .reset_index()
    .rename(columns={"url":"count_ads"})
    .plot(x="original_post_date",by="type", xlabel="Original Post Date", ylabel="Cout of Ads Per Day")
)
plt.suptitle("Sold Ads and Active Ads Over Time - Recent Ads");
```

## Ad Counts by Category

```python
(
    df_combined
    .query("original_post_date > @pd.to_datetime('01-OCT-2021').date() and category_num.isin([1,2,6])")
    .groupby(["original_post_date","category"])
    .count()
    [["url"]]
    .reset_index()
    .rename(columns={"url":"count_ads"})
    .plot(x="original_post_date",by="category", xlabel="Original Post Date", ylabel="Cout of Ads Per Day")
)
plt.suptitle("Count of Active+Sold Ads Over Time");
```

