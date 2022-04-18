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
        last_repost_year = lambda x: x.last_repost_date.dt.year,
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

It appears we have good data for the standard ad title, description, frame size and wheel size. Otherwise the other columns aren't fully in use otherwise we could use front travel, rear travel more heavily to model mountain bikes more accurately potentially. Material also isn't fully utilized - otherwise this would be another valuable indicator. Finally, condition also isn't fully utilized - this could have potential as another indicator but ~ < 1/3 have this filled out.

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
    .assign(last_repost_month = lambda x: x["last_repost_date"].dt.to_period('M'))
)
```

```python
top_n = 5
top_bike_categories = (df_sold_bikes_model.category_num.value_counts().index[0:top_n].tolist())
(
    df_sold_bikes_model
    .query("category_num.isin(@top_bike_categories)")
    .groupby(["last_repost_month","category"])
    .mean()
    .unstack(1)
    .plot(y="price",figsize=(12,8), title="Average Prices per Month by Category")
);

# Count of Ads 
(
    df_sold_bikes_model
    .query("category_num.isin(@top_bike_categories)")
    .groupby(["last_repost_month","category"])
    .count()
    .unstack(1)
    .plot(y="url",figsize=(12,8), title="Counts of Sold Ads per Month by Category")
);
```

```python
# Count of Ads 
g = (
    df_sold_bikes_model
    .assign(
        last_repost_month = lambda x: x.last_repost_date.dt.month,
        last_repost_year = lambda x: x.last_repost_date.dt.year
    )
    # .query("category_num.isin(@top_bike_categories)")
    .groupby(["last_repost_month","last_repost_year","category"])
    .count()
    [["url"]]
    .rename(columns={"url":"count_ads"})
    .reset_index()
    .pipe(
        (sns.relplot, "data"), 
        x="last_repost_month",
        y="count_ads",
        col="category",
        hue="last_repost_year",
        col_wrap=3,
        facet_kws={'sharey': False, 'sharex': False},
        height=3,
        aspect=1.5,
        # title="Test",
        kind="line")

)
plt.style.use("seaborn")
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle("Count of Sold Ads Over Time")
```



<!-- #region -->
We can see a few interesting trends here:
- All Mountain bikes have a more major bump in volume in the spring, and in the fall.
- All Mountain bikes have increased significantly over the period of the dataset. In 2021, certain months had > 3000 ads that were marked as sold.
- Gravel bikes have increased significantly in the last 2 years going from < 100 ads per month, to over 300 ads per month in 2021.


## Preprocessing

To enable consistent pricing for modelling - we need to adjust for the USD -> CAD conversion, and the inflation over time.

<!-- #endregion -->

### Adjusting for Inflation

As a first attempt - we'll look at "All Items" level inflation to adjust our prices. There is most likely a more relevant index that is available - but this will be good enough for now. We need to find Canadian data and US Data separately.

```python
# Canada data
df_can_cpi_data = pd.read_csv("https://www150.statcan.gc.ca/t1/tbl1/en/dtl!downloadDbLoadingData-nonTraduit.action?pid=1810000501&latestN=0&startDate=20100101&endDate=20220101&csvLocale=en&selectedMembers=%5B%5B2%5D%2C%5B2%2C3%2C79%2C96%2C139%2C176%2C184%2C201%2C219%2C256%2C274%2C282%2C285%2C287%2C288%5D%5D&checkedLevels=")

df_can_cpi_data = (
    df_can_cpi_data
    .query("`Products and product groups`=='All-items'")
    .filter(["REF_DATE","VALUE"])
    .rename(columns={"REF_DATE":"year","VALUE":"cpi"})
    .assign(
        most_recent_cpi = lambda x: x.loc[x.year.idxmax(), "cpi"],
        currency="CAD"
    )
)
```

```python
# US Data
# Based on links found here: https://github.com/palewire/cpi/blob/master/cpi/download.py
df_us_cpi_data = (
    pd.read_csv(
    "https://download.bls.gov/pub/time.series/cu/cu.data.1.AllItems",
    sep='\t',
    header=None,
    skiprows=1,
    names=["series_id","year","period","cpi","footnote"]
    )
    .filter(["year","cpi"])
    .groupby("year", as_index=False)
    .mean()
    .assign(
        most_recent_cpi = lambda x: x.loc[x.year.idxmax(), "cpi"],
        currency="USD"
        )
)
```

```python
# Add records for current year carrying forward prior year CPI values
canadian_current_cpi = df_can_cpi_data.most_recent_cpi.iloc[-1]
us_current_cpi = df_us_cpi_data.most_recent_cpi.iloc[-1]
current_year_cpi = pd.DataFrame(
    data={
        "year" : [df_sold_bikes_model.last_repost_year.max()] * 2,
        "cpi" : [canadian_current_cpi, us_current_cpi],
        "most_recent_cpi" : [canadian_current_cpi, us_current_cpi],
        "currency":["CAD","USD"]
    }
)
```

```python
# Join on all CPI data per currency and adjust historical dollars to most recent year
df_cpi_data_combined = pd.concat([df_us_cpi_data, df_can_cpi_data, current_year_cpi])

df_sold_bikes_model_adjusted = (
    df_sold_bikes_model
    .merge(df_cpi_data_combined,how="left",left_on=["last_repost_year","currency"], right_on=["year", "currency"])
    .assign(
        price_cpi_adjusted = lambda x: (x.price * (x.most_recent_cpi/x.cpi))
    )
)
```

```python
(
    df_sold_bikes_model_adjusted
    .groupby(["last_repost_month"])
    .mean()
    [["price", "price_cpi_adjusted"]]
    .plot(figsize=(12,8), title="Inflation Adjusted Average Prices")
);
```


### Exchange Rates - USD -> CAD

```python
# Use Yahoo Finance API through pandas_datareader
from pandas_datareader.yahoo.fx import YahooFXReader
import numpy as np

fx_reader = YahooFXReader(
    symbols=["CAD"],
    start='01-JAN-2010',
    end=df_sold_bikes_model.last_repost_date.max(),
    interval="mo"
)

# Get data and fix symbols - we want to adjust USD prices -> CAD, leaving
# CAD prices untouched. We'll get the data at a monthly level for now.
# There are data issues with multiple returns for some months, and none for other.
# Average across multiple returns for same month, forward fill for missing.
df_fx = (
    fx_reader.read()
    .reset_index()
    .assign(
        fx_month = lambda x: pd.to_datetime(x.Date).dt.to_period('M'),
        currency = lambda x: np.where(x.PairCode == 'CAD', "USD",'CAD')
    )
    .rename(columns={"Close":"fx_rate_USD_CAD"})
    .filter(["fx_month","currency","fx_rate_USD_CAD"])
    .groupby("fx_month", as_index=False)
    .mean()
    .set_index("fx_month")
    .resample('M')
    .mean()
    .ffill()
    .reset_index()
    .assign(
        currency = "USD"
    ) # Need to extend series to carry forward last fx rate.
    .merge(df_sold_bikes_model['last_repost_month'].drop_duplicates(), how="right", left_on="fx_month", right_on="last_repost_month")
    .sort_values("last_repost_month")
    .assign(
        fx_rate_USD_CAD = lambda x: x.fx_rate_USD_CAD.ffill(),
        currency = lambda x: x.currency.ffill(),
        fx_month = lambda x: x.last_repost_month
    )
    .drop(columns=["last_repost_month"])
)

```

```python
df_sold_bikes_model_adjusted_CAD = (
    df_sold_bikes_model_adjusted
    .merge(df_fx, how="left", left_on=["last_repost_month","currency"], right_on=["fx_month", "currency"])
    .assign(
        fx_month = lambda x: x.fx_month.where(~x.fx_month.isna(), other=x.last_repost_month),
        fx_rate_USD_CAD = lambda x: x.fx_rate_USD_CAD.where(x.currency == 'USD', other=1.00),
        price_cpi_adjusted_CAD = lambda x: x.price_cpi_adjusted * x.fx_rate_USD_CAD
    )
)
```

```python
for 
(
    df_sold_bikes_model_adjusted_CAD
    .groupby(["last_repost_month"])
    .mean()
    [["price", "price_cpi_adjusted", "price_cpi_adjusted_CAD"]]
    # .unstack(1)
    .plot(figsize=(12,8), title="Inflation Adjusted Average Prices")
)
```

```python
# Count of Ads 
g = (
    df_sold_bikes_model_adjusted_CAD
    .assign(
        last_repost_month = lambda x: x.last_repost_date.dt.month,
        last_repost_year = lambda x: x.last_repost_date.dt.year
    )
    .groupby(["last_repost_month","last_repost_year","category","currency"])
    .mean()
    [["price_cpi_adjusted_CAD"]]
    # .rename(columns={"url":"count_ads"})
    .reset_index()
    .pipe(
        (sns.relplot, "data"), 
        x="last_repost_month",
        y="price_cpi_adjusted_CAD",
        col="currency",
        row="category",
        hue="last_repost_year",
        # col_wrap=3,
        facet_kws={'sharey': False, 'sharex': False},
        height=3,
        aspect=1.5,
        # title="Test",
        kind="line")

)
g
# plt.style.use("seaborn")
# g.fig.subplots_adjust(top=0.9)
# g.fig.suptitle("Average Adjusted Price by Category and Original Currency")
```

```python

```
