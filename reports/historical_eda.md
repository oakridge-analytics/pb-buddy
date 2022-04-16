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

# EDA of Historical PinkBike Ads


```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pb_buddy.data_processors as dt
from pb_buddy.resources import category_dict

%load_ext autoreload
%autoreload 2
```

```python
df_historical_sold = dt.stream_parquet_to_dataframe("historical_data.parquet.gzip", "pb-buddy-historical")
```

```python
df_current_sold = dt.get_dataset(-1, data_type="sold")
```

```python
# Add in category num for each category, and clean up datatypes.
df_historical_sold_NA = (
    pd.concat([df_historical_sold,df_current_sold], sort=False)
    .query("still_for_sale.str.contains('Sold')")
    .query("currency.isin(['CAD','USD'])")
    .assign(
        category_num = lambda x: x["category"].apply(lambda x: category_dict[x]),
        original_post_date = lambda x: pd.to_datetime(x["original_post_date"]),
        last_repost_date = lambda x: pd.to_datetime(x["last_repost_date"]),
        price = lambda x: x["price"].astype(float),
    )
)

```

## Bikes

We'll first check how many ads we have that can be used for modelling bike prices in general. We'll first check counts by type of bike, and then the metadata available for modelling (i.e frame material etc.).

```python
# Check dataset size to model North American Pricing
(
    df_historical_sold_NA
    .groupby(["category_num","category"],as_index=False)
    .count()
    .query("category.str.contains('Bikes')")
    [["category_num","category","original_post_date"]]
    .rename(columns={"original_post_date":"count_ads"})
    .sort_values("count_ads",ascending=False)
    .style
    .hide(axis="index")
    .set_caption("Bike Ads by Category - North America Scraped Data")
)
```

```python
bike_cats = [2,1,26,75,3,4,77,23,71,74,29,99,63,64]

# for bikes ads - check how often each field is filled. This metadata couldn't have been entered all along
(
    df_historical_sold_NA
    .query("category_num.isin(@bike_cats)")
    .dropna(axis=1, thresh=100) # Drop cols never used for bikes
    .notnull()
    .sum() # Sum TRUES for count
    .sort_values(ascending=False)
    .to_frame("Count Ads with Data For Field")
    )
```

It appears we have good data for the standard ad title, description, frame size and wheel size. Otherwise the other columns aren't fully in use otherwise we could use front travel, rear travel more heavily to model mountain bikes more accurately potentially. Material also isn't fully utilized - otherwise this would be another valuable indicator.

```python
df_sold_bikes = (
    df_historical_sold_NA
    .query("category_num.isin(@bike_cats)")
)
```

Next, we check the quality of key columns for outliers. Here's the findings:

Price - Outliers Removed:
- Numerous ads put at fake high prices ( > 1 Mill.). From looking through ads, $20000 raw price was decided as a threshold. 
- Ads put < $100, asking "Make me an offer" or similar. We don't want to model cheap bikes!
- Ads mention "Stolen" or similar.
- People price ads with $1234 etc. These ads are all removed.

```python
df_sold_bikes_model = (
    df_sold_bikes
    .query("price < 20000 and price > 100")
    .query("description.str.lower().str.contains('stolen') == False")
    .query("description.str.lower().str.contains('looking for') == False")
    .query("price != 1234 and price != 12345 and price !=123")
)
```

```python
top_n = 5
top_bike_categories = (df_sold_bikes_model.category_num.value_counts().index[0:top_n].tolist())
(
    df_sold_bikes_model
    .assign(last_repost_month = lambda x: x["last_repost_date"].dt.to_period('M'))
    .query("category_num.isin(@top_bike_categories)")
    .groupby(["last_repost_month","category"])
    .mean()
    .unstack(1)
    .plot(y="price",figsize=(12,8), title="Average Prices per Month by Category")
);

# Count of Ads 
(
    df_sold_bikes_model
    .assign(last_repost_month = lambda x: x["last_repost_date"].dt.to_period('M'))
    .query("category_num.isin(@top_bike_categories)")
    .groupby(["last_repost_month","category"])
    .count()
    .unstack(1)
    .plot(y="url",figsize=(12,8), title="Counts of Sold Ads per Month by Category")
);
```

```python

```
