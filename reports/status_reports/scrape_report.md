---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.4
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
plt.rcParams["figure.figsize"] = (15, 15)
plt.style.use("ggplot")
```

```python
# %%
# Clean base data -----------------------------------------------------
df_base_data = dt.get_dataset(-1, data_type="base")
df_sold_data = dt.get_dataset(-1, data_type="sold")
```

```python
df_combined = (pd.concat([df_base_data.assign(type="base"), df_sold_data.assign(type="sold")], axis=0)).assign(
    datetime_scraped=lambda x: pd.to_datetime(x.original_post_date, format="mixed").dt.date,
    category=lambda x: x.category.str.strip(),
)
```

```python
# Summarize Top 5 largest categories each day, with count over last 10 days, excluding current day
# as report often runs before data is in for the day.
n_largest = 10
ten_days_ago = pd.Timestamp.now() + pd.Timedelta(days=-10)
today = pd.Timestamp.now()

# Get top 10 categories by total count over last 10 days
top_10_categories = (
    df_combined.query("datetime_scraped > @ten_days_ago.date() and datetime_scraped < @today.date()")
    .groupby("category")["url"]
    .nunique()
    .nlargest(10)
    .index.tolist()
)

# Show daily counts for those categories
(
    df_combined.assign(datetime_scraped=lambda _df: pd.to_datetime(_df.datetime_scraped))
    .query(
        "category in @top_10_categories and datetime_scraped > @ten_days_ago.date() and datetime_scraped < @today.date()"
    )
    .groupby(["datetime_scraped", "category"], as_index=False)
    .agg(count_ads=("url", "nunique"))
    .pivot(columns="datetime_scraped", index="category", values="count_ads")
    .fillna(0)
    .pipe(lambda df: df.reindex(df.sum(axis=1).sort_values(ascending=False).index))
    .style.background_gradient(cmap="viridis", axis=None)
    .format(precision=0)
    .set_caption("Daily ad counts for top 10 categories by total volume over last 10 days")
)
```

```python
# Show daily counts for those categories
(
    df_combined.assign(datetime_scraped=lambda _df: pd.to_datetime(_df.datetime_scraped))
    .query(
        "category in @top_10_categories and datetime_scraped > @ten_days_ago.date() and datetime_scraped < @today.date()"
    )
    .groupby(["datetime_scraped", "category"], as_index=False)
    .agg(count_ads=("url", "nunique"))
    .pivot(columns="category", index="datetime_scraped", values="count_ads")
    .plot(xlabel="Date", ylabel="Count of Ads")
)
plt.suptitle("Daily Ad Counts by Top 10 Categories")
plt.xticks(rotation=45)
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
```

## Ad Counts in Total Over Time (Sold + Still Active Ads)

```python
(
    df_combined.query("datetime_scraped > @pd.to_datetime('01-OCT-2021').date()")
    .groupby(["datetime_scraped"])
    .count()[["url"]]
    .reset_index()
    .rename(columns={"url": "count_ads"})
    .plot(x="datetime_scraped", xlabel="Original Post Date", ylabel="Cout of Ads Per Day")
)
plt.suptitle("Count of Active+Sold Ads Over Time");
```

## Ad Counts by Type

```python
(
    df_combined.query("datetime_scraped > @pd.to_datetime('01-OCT-2021').date()")
    .groupby(["datetime_scraped", "type"])
    .count()[["url"]]
    .reset_index()
    .rename(columns={"url": "count_ads"})
    .plot(x="datetime_scraped", by="type", xlabel="Original Post Date", ylabel="Cout of Ads Per Day")
)
plt.suptitle("Sold Ads and Active Ads Over Time");
```

## Ad Counts by Category

```python
(
    df_combined.query(
        "datetime_scraped > @pd.to_datetime('01-OCT-2021').date() and (category.str.contains('Enduro') or category.str.contains('Cross Country') or category.str.contains('Trail'))"
    )
    .groupby(["datetime_scraped", "category"])
    .count()[["url"]]
    .reset_index()
    .rename(columns={"url": "count_ads"})
    .plot(x="datetime_scraped", by="category", xlabel="Original Post Date", ylabel="Count of Ads Per Day")
)
plt.suptitle("Count of Active+Sold Ads Over Time");
```

