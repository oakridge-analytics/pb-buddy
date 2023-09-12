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
from pb_buddy.data.specs import get_specs_dataset, build_year_make_model_mapping, get_year_make_model_mapping



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
# Extract all years, then brands for each year as a dictionary, then within each brand, extract all models as a dictionary
year_make_model_mapping = get_year_make_model_mapping()
# with open("data/year_make_model_mapping.json", "w") as f:
#     json.dump(year_make_model_mapping, f)

```

```python
def fuzzy_match_bike(input_string: str, year_make_model_mapping: dict, top_n: int = 3, manufacturer_threshold: float = 0.7, model_threshold: float = 0.7, with_similarity: bool = False):
    """
    Function to fuzzy match a bike ad title to a bike model. 
    Returns a tuple of (year, brand, model) if found, else returns None
    """
    # Clean input string
    input_string = input_string.lower().replace(r"[^a-zA-Z0-9 ]", "")
    # Get all years from input string
    years = re.findall(r"\d{4}", input_string)
    # If no years found, return None
    if len(years) == 0:
        return None
    # If more than one year found, return None
    if len(years) > 1:
        return None
    # Get year
    year = years[0]
    # Get all brands for that year, return None if no brands found
    if year not in year_make_model_mapping.keys():
        return None
    brands = list(year_make_model_mapping[year].keys())
    # Get most similar brand using partial_ratio, and the similarity score
    brand, brand_similarity = max([(brand, fuzz.partial_ratio(brand, input_string)) for brand in brands], key=lambda x: x[1])
    if brand_similarity < manufacturer_threshold:
        brand = None
        model = None
        return (year, brand, model)
    # Get all models for that brand
    models = year_make_model_mapping[year][brand]
    # Get 3 most similar models using partial_ratio, and the similarity score
    models_similar =[(model, fuzz.partial_ratio(model, input_string)) for model in models]
    models_similar = sorted(models_similar, key=lambda x: x[1], reverse=True)[:3]

    # For each of 3 most similar models, check compared to threshold and remove if not above
    for model, model_similarity in models_similar:
        if model_similarity < model_threshold:
            models_similar.remove((model, model_similarity))
    if models_similar is None:
        model = None
    
    if not with_similarity:
        models_similar = [model for model, _ in models_similar]
        
    # Return tuple of (year, brand, model) for each match
    return [(year, brand, model) for model in models_similar]

# Test function
fuzzy_match_bike("2021 Cannondle SuperSix", year_make_model_mapping,manufacturer_threshold=70, model_threshold=70, with_similarity=True)
```

```python
# iterate through different model_thresholds and manufacturer_thresholds to see how many ads are matched

res = []
def check_matches(manufacturer_threshold, model_threshold):
    return (
        manufacturer_threshold,
        model_threshold,
        df_modelling
        .assign(
            match = lambda _df: _df.ad_title.apply(lambda _str: fuzzy_match_bike(_str, year_make_model_mapping, manufacturer_threshold=manufacturer_threshold, model_threshold=model_threshold)).astype(str)
        )
        .query("match.str.contains('None')==False")
        .shape[0]
    )

# Use joblib to parallelize the process
threshold_range = np.arange(55,100,5)
n_jobs=20
res = Parallel(n_jobs=20)(delayed(check_matches)(manufacturer_threshold, model_threshold) for manufacturer_threshold, model_threshold in product(threshold_range, threshold_range))

```

```python
# convert to res to dataframe and plot by each threshold:
df_res = pd.DataFrame(res, columns=["manufacturer_threshold", "model_threshold", "num_matches"])
df_res["num_matches"] = df_res.num_matches.astype(int)
df_res["manufacturer_threshold"] = df_res.manufacturer_threshold.astype(int)
df_res["model_threshold"] = df_res.model_threshold.astype(int)
df_res = df_res.pivot(index="manufacturer_threshold", columns="model_threshold", values="num_matches")
df_res = df_res.fillna(0)
df_res = df_res.astype(int)
# df_res.style.background_gradient(cmap='viridis_r', axis=None)
df_res.plot(figsize=(10,10))
```

```python
# from IPython.display import display
# for manufacturer_threshold, model_threshold in product(threshold_range, threshold_range):
        
#         display(
#         df_modelling
#         .sample(20, random_state=42)
#         .assign(
#             match = lambda _df: _df.ad_title.apply(lambda _str: fuzzy_match_bike(_str, year_make_model_mapping,top_n=3, manufacturer_threshold=manufacturer_threshold, model_threshold=model_threshold)).apply(lambda row: "<br>".join([str(item) for item in row]) if row else "None")
#         )
#         .query("match.str.contains('None')==False")
#         [["ad_title", "match"]]
#         .style
#         .set_caption(f"manufacturer_threshold: {manufacturer_threshold}, model_threshold: {model_threshold}")
#         )
```

```python
for model in [model for model in year_make_model_mapping["2017"]["trek"] if "820" in model]:
    print(f"Similarity with {model}:  {fuzz.partial_ratio('2017 Trek Fuel EX 8'.lower(), model)}")
```

```python
[model for model in year_make_model_mapping["2017"]["trek"] if "fuel" in model]
```
