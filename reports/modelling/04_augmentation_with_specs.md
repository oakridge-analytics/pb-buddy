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

import pandas as pd
import numpy as np

# Visuals ------------------
import matplotlib.pyplot as plt
from IPython.display import Markdown
import seaborn as sns
from rapidfuzz import fuzz


# Custom code ---------------------
import pb_buddy.data_processors as dt
import pb_buddy.utils as ut



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
df_specs = dt.stream_parquet_to_dataframe(specs_blob_name, specs_container_name)
```

```python
df_specs = (
    df_specs
    .pipe(lambda _df: _df.assign(**_df.spec_url.str.extract(r"bikes/(.*)/(.*)/(.*)").rename(columns={0: "brand", 1: "year", 2: "model"})))
    .assign(
        # combine year brand model, remove all non alphanumeric characters, and lowercase
        year_brand_model = lambda _df: _df.year + " " +_df.brand + " " + _df.model.str.replace("-"," ")
    )
    .assign(
        year_brand_model = lambda _df: _df.year_brand_model.str.replace(r"[^a-zA-Z0-9 ]", "").str.lower()
    )
)
```

```python
sns.set_theme()
top_n = 30
for col in ["brand", "year", "model"]:
    fig,ax = plt.subplots(figsize=(10, 5))
    print(
        df_specs
        # .head(20)
        .pipe(lambda _df: _df.assign(**_df.spec_url.str.extract(r"bikes/(.*)/(.*)/(.*)").rename(columns={0: "brand", 1: "year", 2: "model"})))
        # plot in horizontal bar chart with seaborn
        .pipe(lambda _df: sns.countplot(y=col, data=_df, order=_df[col].value_counts().iloc[:top_n].index, ax=ax))
        .set_title(f"Top {top_n} {col} in dataset")

        
    )
```

```python
test_str = df_modelling.iloc[4].ad_title.lower()
test_str
```

```python
df_modelling.iloc[4]
```

```python
get_most_similar_spec(df_specs, test_str)
```

```python
# test_str

def get_most_similar_spec(_df, test_str):
    res = (
        _df
        .assign(
            similarity = lambda _df: _df.year_brand_model.apply(lambda _str: fuzz.partial_ratio(_str, test_str))
        )
        .sort_values("similarity", ascending=False)
        .head(1)
        .reset_index()
        [["year_brand_model", "similarity"]]
    )

    return pd.Series([res.year_brand_model.values[0], res.similarity.values[0]])


```

```python

# df_modelling = (
(
    df_modelling
    .sample(30)
    .head(10000)
    .assign(
        ad_title_clean = lambda _df: _df.ad_title.str.lower().str.replace(r"[^a-zA-Z0-9 ]", "", regex=True)
    )
    .pipe(lambda _df: _df.assign(**_df.ad_title_clean.apply(lambda _str: get_most_similar_spec(df_specs, _str.lower())).rename(columns={0: "year_brand_model", 1: "similarity"})))
    [["ad_title_clean", "year_brand_model", "similarity"]]
)

```

```python

```
