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
    display_name: pb-buddy
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

load_dotenv("../.env")
warnings.filterwarnings("ignore")
```

```python
plt.rcParams['figure.figsize'] = (15,15)
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
    datetime_scraped = lambda x: pd.to_datetime(x.original_post_date).dt.date,
    category = lambda x: x.category.str.strip()
    )

```

```python
# Summarize Top 5 largest categories each day, with count over last 10 days, excluding current day
# as report often runs before data is in for the day.
n_largest = 10
ten_days_ago = pd.Timestamp.now() + pd.Timedelta(days=-10)
today = pd.Timestamp.now().date()
(
    df_combined
    .assign(
        datetime_scraped = lambda _df: pd.to_datetime(_df.datetime_scraped)
    )
    .groupby(["datetime_scraped","category"],as_index=False)
    .agg(
        count_ads = ("url", "nunique"),
    )
    .assign(
        daily_rank = lambda _df: _df.groupby("datetime_scraped")["count_ads"].rank(ascending=False, method="dense")
    )
    .query("daily_rank <= @n_largest and datetime_scraped > @ten_days_ago and datetime_scraped < @today") # Top 5 largest category each day
    .sort_values(["datetime_scraped","daily_rank"], ascending=[False,True])
    .pivot(columns="datetime_scraped", index="category", values="count_ads")
    .assign(
        cumulative = lambda _df: _df.sum(axis=1)
    )
    .sort_values("cumulative", ascending=False)
    .drop(columns=["cumulative"])
    .style
    .set_na_rep("")
    .background_gradient(cmap="viridis")
    .set_caption("Top 5 largest categories over last 10 days")

)
```

## Ad Counts in Total Over Time (Sold + Still Active Ads)

```python
(
    df_combined
    .query("datetime_scraped > @pd.to_datetime('01-OCT-2021').date()")
    .groupby(["datetime_scraped"])
    .count()
    [["url"]]
    .reset_index()
    .rename(columns={"url":"count_ads"})
    .plot(x="datetime_scraped", xlabel="Original Post Date", ylabel="Cout of Ads Per Day")
)
plt.suptitle("Count of Active+Sold Ads Over Time");
```

## Ad Counts by Type

```python
(
    df_combined
    .query("datetime_scraped > @pd.to_datetime('01-OCT-2021').date()")
    .groupby(["datetime_scraped","type"])
    .count()
    [["url"]]
    .reset_index()
    .rename(columns={"url":"count_ads"})
    .plot(x="datetime_scraped",by="type", xlabel="Original Post Date", ylabel="Cout of Ads Per Day")
)
plt.suptitle("Sold Ads and Active Ads Over Time");
```

## Ad Counts by Category

```python
(
    df_combined
    .query("datetime_scraped > @pd.to_datetime('01-OCT-2021').date() and (category.str.contains('Enduro') or category.str.contains('Cross Country') or category.str.contains('Trail'))")
    .groupby(["datetime_scraped","category"])
    .count()
    [["url"]]
    .reset_index()
    .rename(columns={"url":"count_ads"})
    .plot(x="datetime_scraped",by="category", xlabel="Original Post Date", ylabel="Count of Ads Per Day")
)
plt.suptitle("Count of Active+Sold Ads Over Time");
```

