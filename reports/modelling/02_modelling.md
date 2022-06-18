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

# Modelling Prices - Bike Ads


```python
import pandas as pd
import numpy as np

# Sklearn helpers ------------------------------
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    PredefinedSplit
)

from sklearn.feature_extraction.text import (
    CountVectorizer
)
from sklearn.base import (
    BaseEstimator,
    RegressorMixin,
    TransformerMixin
)

from sklearn.pipeline import (
    Pipeline,
)

from sklearn.compose import (
    ColumnTransformer
)

# Visuals ------------------
import matplotlib.pyplot as plt
from IPython.display import Markdown
import seaborn as sns

# Custom code ---------------------
import pb_buddy.data_processors as dt
import pb_buddy.utils as ut
from pb_buddy.resources import category_dict

%load_ext autoreload
%autoreload 2
```

```python
# Where set processed data path in Azure Blob storage
input_container_name = 'pb-buddy-historical'
input_blob_name =  '2022-05-20_05_00_04__adjusted_bike_ads.parquet.gzip'
```

```python
df_modelling = dt.stream_parquet_to_dataframe(input_blob_name, input_container_name)
```

```python
df_modelling.columns
```

```python
df_modelling = (
    df_modelling
    .assign(
        original_post_date = lambda _df: pd.to_datetime(_df.original_post_date),
        last_repost_date = lambda _df: pd.to_datetime(_df.last_repost_date),
    )
)
```

Load the preprocessed dataset with inflation adjusted, and currency adjusted price as target.

Target column = `price_cpi_adjusted_cad`

Likely predictor columns:
- `description`: Ad description, free form text
- `original_post_date`: Original date ad was posted
- `ad_title`: Title of the ad 


# Train/Valid/Test Split


We likely want to mix Covid era ads into our train, valid, test sets - we don't want to hold out just a period of time. It's unlikely there is any other temporal trend in the data but we'll check residuals in our fitting process.

```python
# Use set validation set for use later with deep learning approaches that are too hard to do CV on
df_train, df_valid_test = train_test_split(df_modelling, test_size=0.3, random_state=42)
df_valid, df_test = train_test_split(df_valid_test, test_size=0.5, random_state=42)

X_train = df_train[["ad_title","description","original_post_date"]]
y_train = df_train[["price_cpi_adjusted_CAD"]]

X_valid = df_valid[["ad_title","description","original_post_date"]]
y_valid = df_valid[["price_cpi_adjusted_CAD"]]

```

```python
sns.set_theme()
```

```python
fig,ax = plt.subplots()
for _df in [df_train,df_valid,df_test]:
    sns.kdeplot(data=_df, x="price_cpi_adjusted_CAD", ax=ax)
ax.set_title("Train/Valid/Test Price Distributions")
```

```python
# Helper code for sklearn pipelines --------------------------

class add_age_of_equipment(BaseEstimator, TransformerMixin):
    """
    Calculate age of equipment being sold based on:
    - Year extracted from `ad_title`
    - Year extracted from `original_post_date`, requires datetime type

    Ads column `age_at_post` that is number of years old.
    Rows where year of equipment couldn't be extracted from ad title,
    are returned as -1000
    """

    def __init__(self):
        return None

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        return (
            X
            .assign(
                age_at_post = lambda _df: _df.original_post_date.dt.year - _df.ad_title.str.extract("(19|20[0-9]{2})", expand=False).fillna(10000).astype(int),
            )
            .assign(    
                age_at_post = lambda _df: np.where(_df.age_at_post < 0, -1000, _df.age_at_post)
            )
            .drop(
                columns=["ad_title","original_post_date"]
            )
        )

```

## Baseline

We'll first do some baselines - first a simple bag of words, with logistic regression

```python
transformer = ColumnTransformer(
    transformers = [
        ("add_age", add_age_of_equipment(), ["original_post_date","ad_title"]),
        ("title_bow", CountVectorizer(token_pattern = '[a-zA-Z0-9$&+,:;=?@#|<>.^*()%!-]+'), "ad_title"),
        ("description_bow", CountVectorizer(token_pattern = '[a-zA-Z0-9$&+,:;=?@#|<>.^*()%!-]+'), "description")
    ],
    remainder="drop"
)

```

```python

pipe = Pipeline(
    steps = [
        ("transform", transformer),
        ("model", LinearRegression())
    ]
)

hparams ={
    "transform__title_bow__max_features" : [100,500,1000],
    "transform__description_bow__max_features" : [100,500,1000]
}

# Assign static validation split
# -1 indicates train set, 0 is valid set
split_mask = np.vstack((np.ones((len(X_train),1))*-1, np.zeros((len(X_valid),1))))
cv_strat = PredefinedSplit(test_fold=split_mask)

grid_search = GridSearchCV(pipe, param_grid=hparams, cv=cv_strat,scoring="neg_root_mean_squared_error", n_jobs=4, error_score="raise")

grid_search.fit(pd.concat([X_train,X_valid]), pd.concat([y_train,y_valid]))

```

```python
pd.DataFrame(grid_search.cv_results_)
```

```python
(
    pd.concat(
        [
            X_valid,
            y_valid
        ], axis=1)
    .sample(20, random_state=42)
    .assign(
        pred = lambda _df : grid_search.predict(_df),
    )
)
```

```python

```
