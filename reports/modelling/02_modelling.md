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
from tempfile import mkdtemp
from shutil import rmtree

import pandas as pd
import numpy as np
from joblib import dump, load

# Other models -----------
from catboost import CatBoostRegressor, Pool


# Sklearn helpers ------------------------------
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.dummy import DummyRegressor
from sklearn.svm import LinearSVR

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    PredefinedSplit
)

from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer
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

from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    MaxAbsScaler
)

# Visuals ------------------
import matplotlib.pyplot as plt
from IPython.display import Markdown
import seaborn as sns
import shap

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

## Sklearn Utils

```python
# -----------------------------Helper code for sklearn transformers --------------------------
# `get_feature_names_out()` on Pipeline instances requires us to have callable functions
# for returning calculated feature names and be able to pickle the resulting estimator. 
# We create small functions to do this despite how silly they look.

# For encoding country info:----------------------
def add_country(df):
    """
    Extract country from location string, of form "city, country"
    """
    return (
        df
        .assign(
            country = lambda _df:_df.location.str.extract(r', ([A-Za-z\s]+)')
        )
        [["country"]]
    )

def add_country_names(func_transformer, feature_names_in):
    return ["country"]
    
add_country_transformer = FunctionTransformer(add_country, feature_names_out=add_country_names)

# For calculating how old bike was at time of posting:----------------------------------
def add_age(df):
    """
    Calculate the age of bike, using year identified in `ad_title` and the 
    `original_post_date`. Assings Jan,1 of year identified in `ad_title` and
    calculates difference in days. Ads where year can't be identified in title,
    assigns `age_at_post`=-1000.

    Requires:
    - Columns: `ad_title`, `original_post_date`
    """
    return (
        df
        .assign(
            bike_model_date = lambda _df: pd.to_datetime(_df.ad_title.str.extract("((?:19|20)\d{2})", expand=False)),
            age_at_post = lambda _df: (_df.original_post_date - _df.bike_model_date).dt.days,
        )
        .fillna(-1000)
        .astype({"age_at_post":int})
        [["age_at_post"]]
    )

def add_age_names(func_transformer, feature_names_in):
    return ["age_at_post"]

add_age_transformer = FunctionTransformer(add_age, feature_names_out=add_age_names)

# Add a flag for Covid affected market or not. End of Covid market not totally done yet.....
def add_covid_flag(df):
    """
    Use `original_post_date` and check if inbetween March 2020 - July 2022 as
    a flag for COVID affected market...may need to tune these dates once impact
    is more sorted out

    Requires:
    - Column `original_post_date`

    Returns:
    - 1 column dataframe with `covid_flag` column
    """
    return (
        df
        .assign(
            covid_flag = lambda _df: np.where((_df.original_post_date > '20-MARCH-2020') & (_df.original_post_date < '01-AUG-2022'), 1, 0)
        )
        [["covid_flag"]]
    )

def add_covid_name(func_transformer, feature_names_in):
    return ["covid_flag"]

add_covid_transformer = FunctionTransformer(add_covid_flag, feature_names_out=add_covid_name)

```

```python
# Do general pre processing of dataset
# Notes:
# - Cast date columns to correct datetime type
# - Remove ads where the year of the bike can't be identified in ad_title
# - Remove ads that appear to be more than 1 model year ahead of current year (age_at_post<-365)
# - Remove ads posted from countries outside Canada, United States
# - Remove calculated columns so Sklearn transformer pipeline can add back in later for ease of downstream use
df_modelling = (
    df_modelling
    .assign(
        original_post_date = lambda _df: pd.to_datetime(_df.original_post_date),
        last_repost_date = lambda _df: pd.to_datetime(_df.last_repost_date),
        age_at_post = lambda _df: add_age(_df).fillna(-1000),
        country  = lambda _df: add_country(_df)
    )
    .query("age_at_post>-365 and (country=='Canada' or country=='United States')")
    .drop(columns=["age_at_post","country"])
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

X_train = df_train.drop(columns=["price_cpi_adjusted_CAD"])
y_train = df_train["price_cpi_adjusted_CAD"]

X_valid = df_valid.drop(columns=["price_cpi_adjusted_CAD"])
y_valid = df_valid["price_cpi_adjusted_CAD"]

X_test = df_test.drop(columns=["price_cpi_adjusted_CAD"])
y_test = df_test['price_cpi_adjusted_CAD']

# For deterministic validation splits, need train and valid in one array and can pass
# a mask to indicate beginning is train, end of data is valid. See PredefinedSplit docs for more:
X_train_valid = pd.concat([X_train,X_valid],axis=0)
y_train_valid = pd.concat([y_train,y_valid],axis=0)

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

## Baseline

We'll first do some baselines - first a simple bag of words, with dummy regressor (predict the mean) and with a linear regression

```python
transformer = Pipeline(steps=[
    ('preprocess',
        ColumnTransformer(
            transformers = [
                ("add_age", add_age_transformer, ["ad_title","original_post_date"]),
                (
                    "add_country",
                    Pipeline(
                        steps=[
                            ("get_country",add_country_transformer),
                            ("one_hot", OneHotEncoder(handle_unknown="infrequent_if_exist"))
                        ]
                    ),
                    ['location']
                ),
                ("add_covid_flag", add_covid_transformer,["original_post_date"]),
                ("title_bow", TfidfVectorizer(token_pattern = '[a-zA-Z0-9$&+,:;=?@#|<>.^*()%!-]+'), "ad_title"),
                ("description_bow", TfidfVectorizer(token_pattern = '[a-zA-Z0-9$&+,:;=?@#|<>.^*()%!-]+'), "description")
            ],
        remainder="drop"
    )),
    ('scale', MaxAbsScaler()) # Maintains sparsity!
]
)

```

```python
# Assign static validation split
# -1 indicates train set, 0 is valid set
split_mask = np.vstack((np.ones((len(X_train),1))*-1, np.zeros((len(X_valid),1))))
cv_strat = PredefinedSplit(test_fold=split_mask)
```

```python
# Store modelling results over all experiments:
results = {}
```

```python
# cachedir = mkdtemp()
base_pipe = Pipeline(
    steps = [
        ("model", DummyRegressor())
    ],
)

hparams ={
}

base_grid_search = GridSearchCV(base_pipe, param_grid=hparams, cv=cv_strat,scoring="neg_root_mean_squared_error", n_jobs=4, error_score="raise")
base_grid_search.fit(X_train_valid, y_train_valid)

```

```python

results["baseline"] = pd.DataFrame(base_grid_search.cv_results_).drop(columns=[c for c in base_grid_search.cv_results_.keys() if "param_" in c])
pd.DataFrame(base_grid_search.cv_results_)
```

# Stronger Baseline

```python
# Setup caching of pipeline
# cachedir = mkdtemp()

svr_pipe = Pipeline(
    steps = [
        ("transform", transformer),
        ("model", LinearSVR(max_iter=1000, random_state=42))
    ],
)

hparams ={
    "transform__preprocess__title_bow__max_features" : [10000,100000,200000],
    "transform__preprocess__title_bow__ngram_range": [(1,3),],
    # "transform__title_bow__max_df":[0.6,],
    "transform__preprocess__description_bow__max_features" : [10000,100000,200000],
    "transform__preprocess__description_bow__ngram_range": [(1,3),],
    # "transform__preprocess__description_bow__max_df":[0.6,],
    "model__C": [30]
}

svr_grid_search = GridSearchCV(svr_pipe, param_grid=hparams, cv=cv_strat,scoring="neg_root_mean_squared_error", error_score="raise")
svr_grid_search.fit(X_train_valid, y_train_valid)

```

```python
dump(svr_grid_search.best_estimator_, "../../data/svr_pipe.joblib")

results["linear_svr"] = pd.DataFrame(svr_grid_search.cv_results_).drop(columns=[c for c in svr_grid_search.cv_results_.keys() if "param_" in c])
pd.DataFrame(svr_grid_search.cv_results_)
```

```python
# gbm_pipe = Pipeline(
#     steps = [
#         ("transform", transformer),
#         ("model", CatBoostRegressor())
#     ]
# )

# hparams ={
#     "transform__title_bow__max_features" : [1000],
#     "transform__description_bow__max_features" : [1000],
#     "model__max_depth": [30],
#     # "model__num_leaves": [100],
#     "model__n_estimators": [10]
# }

# gbm_grid_search = GridSearchCV(gbm_pipe, param_grid=hparams, cv=cv_strat,scoring="neg_root_mean_squared_error", n_jobs=8, error_score="raise")
# gbm_grid_search.fit(X_train_valid, y_train_valid)

# results["gbm"] = pd.DataFrame(gbm_grid_search.cv_results_).drop(columns=[c for c in gbm_grid_search.cv_results_.keys() if "param_" in c])
# pd.DataFrame(gbm_grid_search.cv_results_)
```

## Model Interpretation

```python
sample_data = pd.DataFrame(
        data={
            "ad_title":["2013 Rocky Mountain Altitude - Custom"],
        #     "age_at_post":[9*365],
            "location":["Calgary, Canada"],
            "original_post_date":[pd.to_datetime("2022-MAR-31")],
    "description":[
        """Custom built Rocky Mountain Altitude - one of the best bikes for big climbs and big descents. 28.5 pounds as built so it can handle the climbs but still has no problem getting down the gnar.

Highlights:
- Full carbon frame - front triangle warrantied and replaced brand new in 2017.
- Pike RCT3 Fork (2015) - full damper rebuild in 2021. Lowers serviced consistently.
- RockShox Monarch Debonair shock - full rebuild in 2019 at Fluid Function.
- Race Face Next R Carbon rear wheel, Aeffect R 30 aluminum front wheel.
- Easton Haven dropper - full rebuild in 2019. Very quick cable actuated post, without relying on hydraulic systems.
- Deore XTR groupset - 1x11 - new chain, chain ring (30T), and cassette (11-46) in July 2021.
- Enve Carbon bar + Thompson Elite stem
- Minion DHF + DHR tires

Ride wrapped from day one - have kept up service of all the pivots as well.

*Doesn't include bottle cages pictured."""
]
})

sample_data.assign(
        pred = lambda _df : svr_grid_search.predict(_df),
    )
```

```python
sample_data = pd.DataFrame(
        data={
            "ad_title":["2016 Norco Sight C7.4"],
            "original_post_date":[pd.to_datetime("2022-MAR-31")],
                "location":["Calgary, Canada"],
    "description":[
        """
        Well loved bike, needs $600 dollars work

        Seat Post	KS E-Ten Integra 30.9mm adj post w / remote stealth
Seat Post Clamp	Norco design alloy nutted clamp (no cable guide)
Saddle	WTB Volt Sport
Shifter Casing	Jagwire LEX housing
Headset	Cane Creek 10 series tapered (integrated)
Headset Spacer	1x10mm - Black / 1x5mm - Black
Top Cap	Black alloy with Cane Creek logo
Stem	Alloy MTB stem with 4 bolt front clamp
Handlebar	Norco 1" rise bar 760 mm width, 6061 DB alloy
Grips	Norco single side lock
Front Brake	Shimano Acera hydraulic disc brake M396
Chain Tensioner	N/A
Rear Brake	Shimano Acera hydraulic disc brake M396
Brake Levers	Shimano Acera
Brake Rotors	Shimano RT-54  180 mm rotors ( center lock)
Brake Cable Casing	N/A
Rims	Alex DP23 650B 28.5 mm wide w/TRS
Tires	Schwalbe 27.5 x 2.25 Nobby Nic Performance
Tubes	Schwalbe presta 27.5 inner tube
Front Hub	Shimano Deore 618 15mm
Rear Hub	Shimano Deore 618 142 x 12
Spokes/Nipples	Stainless steel black 2.0 spokes
Shifter Front	Shimano Deore M610 rapid fire 2/3 sp
Shifter Rear	Shimano Deore M610 rapid fire 10 sp
Front Derailleur	Sram X5 high direct mount FD
Rear Derailleur	Shimano Deore M615 Shadow Plus SGS
Cassette	Shimano HG-50 11-36T 10 sp
Crankset	Shimano Deore M615 38/24T 2x10 w/bb
Bottom Bracket	N/A
Pedals	N/A
Chain	KMC X-10

"""]
})

sample_data.assign(
        pred = lambda _df : svr_grid_search.predict(_df),
    )
```

```python
# --------------------- Attempt to use SHAP----------------------------
# sample_expanded_data = svr_grid_search.best_estimator_.named_steps["transform"].transform(sample_data).toarray()
# # df_sample_expanded_data = pd.DataFrame(sample_expanded_data, columns = svr_grid_search.best_estimator_.get_feature_names_out())
# ex = shap.KernelExplainer(svr_grid_search.best_estimator_.named_steps["model"].predict, sample_expanded_data)
# shap_values = ex.shap_values(sample_expanded_data, silent=True)
```

```python
year_sweep = pd.DataFrame()
for year in range(2010,2023):
    new_result = (
        sample_data
        .assign(
            # original_post_date = pd.to_datetime(f"{year}-01-01"),
            ad_title = lambda _df: _df.ad_title.str.replace("\d{4}",str(year), regex=True),
            pred = lambda _df : svr_grid_search.predict(_df),
        )
        [["original_post_date","ad_title","pred"]]
    )
    year_sweep = pd.concat([
        year_sweep,
        new_result

    ]
    )

year_sweep
```

```python
pd.set_option("display.max_colwidth", None)
(
    pd.concat([
        X_test,
        y_test
    ],
    axis=1
    )
    .sample(20, random_state=42)
    .assign(
        pred = lambda _df : svr_grid_search.predict(_df),
        diff = lambda _df: _df.price_cpi_adjusted_CAD - _df.pred
    )
    [["ad_title","description","original_post_date","price_cpi_adjusted_CAD","pred","diff"]]
    .sort_values("diff")
    .style
    .hide_index()
    .background_gradient(subset=["diff"])
)
```

```python
# text_features = ["ad_title","description"]

# def fit_catboost(X_train, X_valid, y_train, y_valid, catboost_params={}, verbose=100):
#     learn_pool = Pool(
#         X_train[text_features], 
#         y_train, 
#         text_features=text_features,
#         feature_names=["ad_title","description"]
#     )
#     eval_pool = Pool(
#         X_valid[text_features], 
#         y_valid, 
#         text_features=text_features,
#         feature_names=["ad_title","description"]
#     )
    
#     catboost_default_params = {
#         'iterations': 1000,
#         'learning_rate': 0.03,
#         # 'eval_metric': 'Accuracy',
#         'task_type': 'GPU'
#     }
    
#     catboost_default_params.update(catboost_params)
    
#     model = CatBoostRegressor(**catboost_default_params)
#     model.fit(learn_pool, eval_set=eval_pool, verbose=verbose)

#     return model


```

```python
# fit_catboost(X_train, X_valid, y_train, y_valid)
```

```python

```
