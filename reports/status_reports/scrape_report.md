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
df_combined = (
    pd.concat([
        df_base_data.assign(type = "base"),
        df_sold_data.assign(type="sold")
        ],
         axis=0)
).assign(
    scrape_day = lambda x: pd.to_datetime(x.original_post_date).dt.date,
    category = lambda x: x.category.str.strip()
    )

```

## Ad Counts in Total Over Time (Sold + Still Active Ads)

```python
(
    df_combined
    .query("scrape_day > @pd.to_datetime('01-OCT-2021').date()")
    .groupby(["scrape_day"])
    .count()
    [["url"]]
    .reset_index()
    .rename(columns={"url":"count_ads"})
    .plot(x="scrape_day", xlabel="Original Post Date", ylabel="Cout of Ads Per Day")
)
plt.suptitle("Count of Active+Sold Ads Over Time");
```

## Ad Counts by Type

```python
(
    df_combined
    .query("scrape_day > @pd.to_datetime('01-OCT-2021').date()")
    .groupby(["scrape_day","type"])
    .count()
    [["url"]]
    .reset_index()
    .rename(columns={"url":"count_ads"})
    .plot(x="scrape_day",by="type", xlabel="Original Post Date", ylabel="Cout of Ads Per Day")
)
plt.suptitle("Sold Ads and Active Ads Over Time");
```

## Ad Counts by Category

```python
(
    df_combined
    .query("scrape_day > @pd.to_datetime('01-OCT-2021').date() and category_num.isin([1,2,6])")
    .groupby(["scrape_day","category"])
    .count()
    [["url"]]
    .reset_index()
    .rename(columns={"url":"count_ads"})
    .plot(x="scrape_day",by="category", xlabel="Original Post Date", ylabel="Cout of Ads Per Day")
)
plt.suptitle("Count of Active+Sold Ads Over Time");
```

