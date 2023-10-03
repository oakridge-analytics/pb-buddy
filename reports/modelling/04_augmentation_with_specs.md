---
jupyter:
  jupytext:
    formats: md,ipynb
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.6
  kernelspec:
    display_name: pb-buddy
    language: python
    name: python3
---

```python
import os
import re
import json

import pandas as pd
import numpy as np
from itertools import product
from joblib import Parallel, delayed

# Visuals ------------------
import matplotlib.pyplot as plt
from IPython.display import Markdown
import seaborn as sns
from rapidfuzz import fuzz


# Custom code ---------------------
import pb_buddy.data_processors as dt
import pb_buddy.utils as ut
from pb_buddy.specs import augment_with_specs, extract_msrp, match_with_default_value
from pb_buddy.data.specs import get_specs_dataset

%load_ext autoreload
%autoreload 2
```

```python
# Set to processed ad data path in Azure Blob storage
input_container_name = 'pb-buddy-historical'
input_blob_name =  '2023-05-13_22_14_08__adjusted_bike_ads.parquet.gzip'


# Set to where bike specs are stored
specs_container_name = 'bike-specs'
specs_blob_name = "20230906_specs.parquet"
```

```python
def make_clickable(val):
    return '<a href="{}">{}</a>'.format(val,val)
```

```python
# Get dataset from blob storage, unless already found in data folder
local_cache_file_path = os.path.join("data",f"{input_blob_name.split('.')[0]}.csv")
if os.path.exists(local_cache_file_path):
    df_modelling = pd.read_csv(local_cache_file_path, index_col=None)
else:
    df_modelling = dt.stream_parquet_to_dataframe(input_blob_name, input_container_name)
```

```python
df_specs = get_specs_dataset()
```

```python
(
    df_specs
    .assign(
        msrp_cleaned=lambda _df: _df.msrp_summary.astype(str).apply(extract_msrp),
    )
    .dropna(subset=['weight_summary',"travel_summary"])
    .query("travel_summary.str.contains(',')")
    .head(20)
    # Extract weights of form xx.x from weights_summary
    .assign(
        weight_summary=lambda _df: _df.weight_summary.astype(str).apply(
            lambda x: float(match_with_default_value(r"([0-9]+\.[0-9]+)", x, 0))
        ).astype(float)
    )
    # Extract from travel_summary xxxmm front, xxxmm rear into front_travel_summary and rear_travel_summary
    .assign(
        front_travel_summary=lambda _df: _df.travel_summary.astype(str).apply(
            lambda x: float(match_with_default_value(r"([0-9]+)mm front", x, 0))
        ).astype(float),
        rear_travel_summary=lambda _df: _df.travel_summary.astype(str).apply(
            lambda x: float(match_with_default_value(r"([0-9]+)mm rear", x, 0))
        ).astype(float),
    )
    [
        [
            "manufacturer",
            "model",
            "year",
            "travel_summary",
            "msrp_cleaned",
            "weight_summary",
            "front_travel_summary",
            "rear_travel_summary",
        ]
    ]
)
```

```python
check_cols = df_modelling.columns[-48:].to_list()

# For each col in check_cols, calculate for each month, in each year
# how many ads by original_post_date have a non-null spec-url value
# and plot the results

df_modelling['year_month'] = pd.to_datetime(df_modelling['original_post_date']).dt.strftime('%Y-%m')

for col in check_cols:
    (
        df_modelling
        .assign(
            **{f"{col}_has_spec": lambda x: x[col].notnull()}
        )
        .groupby(['year_month'])
        .agg({f"{col}_has_spec":'mean'})
        .reset_index()
        .plot(x='year_month', y=f"{col}_has_spec", figsize=(20,3), title=f"{col}_has_spec")
    )
```

```python
# Get unique values of column "col", and the first spec_url that has that value
from IPython.display import display, Markdown

def get_unique_specs(df, col):
    return (
        df
        .loc[df[col].notnull()]
        .assign(
            **{f"{col}": lambda x: x[col].fillna("None")}
        )
        .groupby(col)
        .agg({'spec_url':'first', 'year_manufacturer_model':'count'})
        .sort_values('year_manufacturer_model', ascending=False)
        .reset_index()
    )

for col in [c for c in check_cols if "summary" in c]:
    display(
        get_unique_specs(df_modelling, col).head(10).style.format({'spec_url': make_clickable})
    )
```
