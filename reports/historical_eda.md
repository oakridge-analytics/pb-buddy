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
# Add in category num for each category
df_historical_sold_NA = (
    df_historical_sold
    .query("still_for_sale.str.contains('Sold')")
    .query("currency.isin(['CAD','USD'])")
    .assign(
        category_num = lambda x: x["category"].apply(lambda x: category_dict[x])
    )
)

```

```python
# Check dataset size to model North American Pricing
(
    df_historical_sold_NA
    .groupby(["category_num","category"],as_index=False)
    .count()
    .head(20)
    [["category_num","category","original_post_date"]]
    .rename(columns={"original_post_date":"count_ads"})
    .sort_values("count_ads",ascending=False)
    .style
    .hide(axis="index")
    .set_caption("Top 20 Largest Categories by Sold Ads in North America")
)
```

```python
# for bikes ads - check how often each field is filled. This metadata couldn't have been entered all along
(
    df_historical_sold_NA
    .query("category_num==2")
    .dropna(axis=1, thresh=100) # Drop cols never used for bikes
    .notna()
    .sum()
    .sort_values(ascending=False)
)
```

```python
(
    df_historical_sold_NA
    .query("category_num==2 and material.notna()")
    .dropna(axis=1, thresh=100) # Drop cols never used for bikes
    .dtypes
)
```

```python

```
