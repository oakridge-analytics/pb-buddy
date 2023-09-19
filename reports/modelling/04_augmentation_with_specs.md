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
from pb_buddy.specs import fuzzy_match_bike
from pb_buddy.data.specs import get_specs_dataset, build_year_manufacturer_model_mapping, get_year_manufacturer_model_mapping



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
year_manufacturer_model_mapping = get_year_manufacturer_model_mapping()
# Test function
fuzzy_match_bike("2010 Santa Cruz Heckler".lower(), 2020, year_manufacturer_model_mapping, top_n=1, manufacturer_threshold=10, model_threshold=10, with_similarity=True)
```

```python
# iterate through different model_thresholds and manufacturer_thresholds to see how many ads are matched

# res = []
# def check_matches(manufacturer_threshold, model_threshold):
#     return (
#         manufacturer_threshold,
#         model_threshold,
#         df_modelling
#         .assign(
#             match = lambda _df: _df.ad_title.apply(lambda _str: fuzzy_match_bike(_str, year_make_model_mapping, manufacturer_threshold=manufacturer_threshold, model_threshold=model_threshold)).astype(str)
#         )
#         .query("match.str.contains('None')==False")
#         .shape[0]
#     )

# # Use joblib to parallelize the process
# threshold_range = np.arange(55,100,5)
# n_jobs=20
# res = Parallel(n_jobs=20)(delayed(check_matches)(manufacturer_threshold, model_threshold) for manufacturer_threshold, model_threshold in product(threshold_range, threshold_range))

```

```python
# # convert to res to dataframe and plot by each threshold:
# df_res = pd.DataFrame(res, columns=["manufacturer_threshold", "model_threshold", "num_matches"])
# df_res["num_matches"] = df_res.num_matches.astype(int)
# df_res["manufacturer_threshold"] = df_res.manufacturer_threshold.astype(int)
# df_res["model_threshold"] = df_res.model_threshold.astype(int)
# df_res = df_res.pivot(index="manufacturer_threshold", columns="model_threshold", values="num_matches")
# df_res = df_res.fillna(0)
# df_res = df_res.astype(int)
# df_res.plot(figsize=(10,10))
```

```python
df_modelling = (
    df_modelling
        .assign(
            fuzzy_match = lambda _df: _df[["ad_title","year"]].apply(lambda row: fuzzy_match_bike(row["ad_title"].lower(), row['year'], year_manufacturer_model_mapping, top_n=1, manufacturer_threshold=80, model_threshold=75, with_similarity=False), axis=1)
        )
        .query("fuzzy_match.astype('str').str.contains('None')==False")
)
```

```python


from pb_buddy.specs import augment_with_specs

df_modelling_with_specs = augment_with_specs(df_modelling, year_col="year", ad_title_col="ad_title")
```

```python
df_modelling_with_specs.sample(30)[["msrp_summary","ad_title","fuzzy_match","price", "url", "spec_url"]].style.format({"url": make_clickable, "spec_url": make_clickable})
```

```python
df_modelling_with_specs.shape
```

```python

```
