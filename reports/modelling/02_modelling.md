---
jupyter:
  jupytext:
    formats: ipynb,md
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

# Modelling Prices - Bike Ads

These datasets have been pre calculated in `01_historical_data_preprocessing.md`. This has taken care of adjusting for inflation, and currency conversion on ads.

Ads have been pre filtered to complete bike categories.

```python
from tempfile import mkdtemp
from shutil import rmtree
import os

import pandas as pd
import numpy as np
from joblib import dump, load

# Other models -----------
from catboost import CatBoostRegressor, Pool

# Sklearn helpers ------------------------------
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import IsolationForest
from sklearn.svm import LinearSVR
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    PredefinedSplit
)

from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer
)

from sklearn.impute import SimpleImputer

from sklearn.pipeline import (
    Pipeline,
)

from sklearn.compose import (
    ColumnTransformer
)

from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    MaxAbsScaler,
    TargetEncoder,
)

# Transformers modelling ---------------
# from datasets import Dataset
# from transformers import TrainingArguments, Trainer
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# from transformers import RobertaModel

from sklearn.base import clone

from autogluon.multimodal import MultiModalPredictor
from ray import tune
import uuid


# Visuals ------------------
import matplotlib.pyplot as plt
from IPython.display import Markdown
import seaborn as sns

# Custom code ---------------------
import pb_buddy.data_processors as dt
import pb_buddy.utils as ut

from pb_buddy.modelling.skhelpers import (
    add_age,
    add_country,
    add_country_transformer,
    add_age_transformer,
    add_covid_transformer,
    remove_year_transformer,
    get_post_month_transformer,
    AugmentSpecFeatures
)
from pb_buddy.specs import augment_with_specs

%load_ext autoreload
%autoreload 2
```

# Setup Cells

```python
# Where set processed data path in Azure Blob storage
input_container_name = 'pb-buddy-historical'
input_blob_name =  '2023-05-13_22_14_08__adjusted_bike_ads.parquet.gzip'
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
df_modelling.to_csv(local_cache_file_path, index=None)
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
        country  = lambda _df: add_country(_df),
        ad_title_description = lambda _df: _df.ad_title + " " + _df.description
    )
    .pipe(
        augment_with_specs, year_col="year",
          manufacturer_threshold=80,
            model_threshold=80
            )
    # .dropna(subset=["spec_url"])
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
- `location`: What country, and potentially what city/region was the bike listed in?
- `Covid`: Was COVID-19 impacting the market when this bike was listed? We know retail sale of bikes was overwhelmed by demand, likely this occured in used market as well and saw increased prices.


## Train/Valid/Test Split


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
print(f"Train set size: {X_train.shape[0]}, valid set size: {X_valid.shape[0]}, test set size: {X_test.shape[0]}")
```

```python
fig,ax = plt.subplots()
for _df in [df_train,df_valid,df_test]:
    sns.kdeplot(data=_df, x="price_cpi_adjusted_CAD", bw_adjust=0.5, ax=ax)
ax.set_title("Train/Valid/Test Price Distributions")
```

# End Setup

```python
import torch
torch.cuda.is_available()
```

```python

```

# Pipeline Definition

```python
# Allow periods only inside numbers for model names,
# otherwise, we don't want to include periods as part of tokens
# as end of sentence words will get treated as a different token
# with a period on the end.
token_pattern = "[a-zA-Z0-9$&+,:;=?@#|<>^%!]+|[0-9$&+,.:;=?@#|<>^%!]+"

transformer = Pipeline(steps=[
    ('preprocess',
        ColumnTransformer(
            transformers = [
                ("add_age", add_age_transformer, ["ad_title","original_post_date"]),
                (
                    "add_post_month",
                    Pipeline(
                        steps=[
                            ("get_post_month",get_post_month_transformer),
                            ("one_hot", OneHotEncoder(handle_unknown="infrequent_if_exist"))
                        ]
                    ),
                    ['original_post_date']
                ),
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
                (
                    "title_text", 
                    Pipeline(
                        steps=[
                            # Remove mentions of year so model doesn't learn to predict based on that year's prices
                            ('remove_year', remove_year_transformer),
                            ('tfidf',
                                TfidfVectorizer(
                                        token_pattern = token_pattern 
                                )
                            ),
                        ]
                    ), 
                    "ad_title"
                ),
                (
                    "description_text", 
                    Pipeline(
                        steps=[
                            # Remove mentions of year so model doesn't learn to predict based on that year's prices
                            ('remove_year', remove_year_transformer), 
                            ('tfidf',
                                TfidfVectorizer(
                                        # Allow periods only inside numbers for model names etc. 
                                        token_pattern = token_pattern
                                )
                            ),
                        ]
                    ), 
                    "description"
                ),
                (
                    "specs_data_one_hot", 
                    OneHotEncoder(handle_unknown="infrequent_if_exist"),
                    [
                    "frame_summary",
                    "wheels_summary",
                    "drivetrain_summary",
                    "brakes_summary",
                    "suspension_summary",
                    ]
                ),
                (
                    "specs_data_high_cardinality",
                    TargetEncoder(target_type="continuous"),
                    [
                    "groupset_summary",
                    "fork_summary",
                    ]
                ),
                (
                    "specs_data_numerical",
                    SimpleImputer(strategy="median"),
                    ["msrp_cleaned", "weight_summary", "front_travel_summary", "rear_travel_summary"]
                )
            ],
        remainder="drop"
    )),
    ('scale', MaxAbsScaler()) # Maintains sparsity!
]
)

transformer
```

```python
# transform Check ---------------------------------
n=10
pd.DataFrame(
    data=transformer.fit_transform(X_train.head(n), y_train.head(n)).toarray(),
    columns=transformer.fit(X_train.head(n), y_train.head(n)).get_feature_names_out()
)
```

# Outlier Detection

To help remove ads that are likely erroneous - we'll first check for data points that are suspicious before modelling.

```python
# Add in price to detect outliers on pricing as well as inputs
token_pattern = "[a-zA-Z0-9$&+,:;=?@#|<>^%!]+|[0-9$&+,.:;=?@#|<>^%!]+"

outlier_transformer = Pipeline(steps=[
    ('preprocess',
        ColumnTransformer(
            transformers = [
                (
                "price",
                "passthrough",
                ["price_cpi_adjusted_CAD"]
                ),
                ("add_age", add_age_transformer, ["ad_title","original_post_date"]),
                (
                    "add_post_month",
                    Pipeline(
                        steps=[
                            ("get_post_month",get_post_month_transformer),
                            ("one_hot", OneHotEncoder(handle_unknown="infrequent_if_exist"))
                        ]
                    ),
                    ['original_post_date']
                ),
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
                (
                    "title_text", 
                    Pipeline(
                        steps=[
                            # Remove mentions of year so model doesn't learn to predict based on that year's prices
                            ('remove_year', remove_year_transformer),
                            ('tfidf',
                                TfidfVectorizer(
                                        token_pattern = token_pattern 
                                )
                            ),
                        ]
                    ), 
                    "ad_title"
                ),
                (
                    "description_text", 
                    Pipeline(
                        steps=[
                            # Remove mentions of year so model doesn't learn to predict based on that year's prices
                            ('remove_year', remove_year_transformer), 
                            ('tfidf',
                                TfidfVectorizer(
                                        # Allow periods only inside numbers for model names etc. 
                                        token_pattern = token_pattern
                                )
                            ),
                        ]
                    ), 
                    "description"
                ),
                (
                    "specs_data", 
                    Pipeline(
                        steps=[
                            ("get_specs",augment_spec_features_transformer),
                            ("one_hot", OneHotEncoder(handle_unknown="infrequent_if_exist"))
                        ]
                    ),
                    ["year","ad_title"]
                ),
            ],
        remainder="drop"
    )),
    ('scale', MaxAbsScaler()) # Maintains sparsity!
]
)


isolation_pipe = Pipeline(steps=[
    ("preprocess", outlier_transformer),
    ("isoforest", IsolationForest(n_estimators=1000, max_samples=10000, n_jobs=30, contamination=0.01))
])

isolation_pipe.fit(df_modelling)
```

```python
X_train_outlier_scored = (
    df_modelling
    .assign(
        outlier_flag = lambda _df : isolation_pipe.predict(_df),
        outlier_score = lambda _df : isolation_pipe.score_samples(_df)
    )
)
```

```python
(
    X_train_outlier_scored
    .outlier_flag
    .value_counts()
)
```

```python
X_train_outlier_scored.query("outlier_flag==-1").outlier_score.hist(bins=100)
```

```python
(
    X_train_outlier_scored
    .assign(
        price_cpi_adjusted_CAD = df_modelling.price_cpi_adjusted_CAD
    )
    .query("outlier_flag == -1")
    .sort_values("outlier_score", ascending=True)
    .head(20)
    [["price", "url","price_cpi_adjusted_CAD","ad_title", "original_post_date","description", "outlier_score"]]
    .assign(
        url = lambda _df : _df.url.apply(make_clickable)
    )
    .style
)
    
```

# Baseline

We'll first do some baselines - first a simple bag of words, with dummy regressor (predict the mean) and with a linear regression

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
def predefined_grid_search(X_train_valid: pd.DataFrame, y_train_valid: pd.DataFrame, pipeline: Pipeline, hyperparams: dict, cv: PredefinedSplit, **kwargs):
    """
    Helper func for modelling with predefined split, getting valid results, and refitting just on train set:
    - Do normal grid search to check hyperparams with valid set, refit=False to prevent refitting on all data.
    - Fit on just train set
    - Return grid search validation set results, fitted best estimator
    """
    pipeline = clone(pipeline)
    grid = GridSearchCV(estimator=pipeline, param_grid=hyperparams, refit=False, cv=cv, **kwargs)
    grid.fit(X_train_valid, y_train_valid)
    
    # Refit to just train set:
    for train_set_ix,_ in cv.split():
        X_train, y_train = X_train_valid.iloc[train_set_ix], y_train_valid.iloc[train_set_ix]
    best_estimator = pipeline.set_params(**grid.best_params_).fit(X_train,y_train)

    return grid.cv_results_, best_estimator
```

```python
# Setup caching of pipeline
# cachedir = mkdtemp()

svr_pipe = Pipeline(
    steps = [
        ("transform", transformer),
        ("model", LinearSVR(max_iter=100000, random_state=42))
    ],
)

hparams ={
    "transform__preprocess__title_text__tfidf__max_features" : [100000,200000],
    "transform__preprocess__title_text__tfidf__ngram_range": [(1,3),(1,5),(2,5)],
    # "transform__preprocess__title_text__tfidf__max_df":[0.4,0.6,0.7,1],
    "transform__preprocess__description_text__tfidf__max_features" : [100000,200000],
    "transform__preprocess__description_text__tfidf__ngram_range": [(1,3),(1,5),(2,5)],
    # "transform__preprocess__description _text__tfidf__max_df":[0.4,0.6,0.7,1],
    
    "model__C": [30],
    "model__tol": [1e-4]
}

svr_results, svr_estimator = predefined_grid_search(
    X_train_valid,
    y_train_valid,
    svr_pipe, 
    hparams,
    cv=cv_strat,
    scoring="neg_root_mean_squared_error",
    n_jobs=4, # Keep as 1 if using large tfidf matrices due to RAM limits
    verbose=3
)


```

```python
results["linear_svr"] = pd.DataFrame(svr_results).drop(columns=[c for c in svr_results.keys() if "param_" in c])
pd.DataFrame(svr_results).sort_values("split0_test_score", ascending=False)
```

# Boosting Models

```python
catboost_pipe = Pipeline(
    steps = [
        ("transform", transformer),
        ("model", CatBoostRegressor(verbose=False, task_type="GPU"))
    ]
)

hparams ={
    "transform__preprocess__title_text__tfidf__max_features" : [10000],
    "transform__preprocess__title_text__tfidf__ngram_range": [(1,3),],
    # "transform__preprocess__title_text__tfidf__max_df": [0.6,0.7],
    "transform__preprocess__description_text__tfidf__max_features" : [10000],
    "transform__preprocess__description_text__tfidf__ngram_range": [(1,3),],
    # "transform__preprocess__specs_data__get_specs__spec_cols": [
    #     [
    #         "groupset_summary",
    #         "wheels_summary",
    #         "suspension_summary",
    #     ],
    #     [
    #         "groupset_summary",
    #         "wheels_summary",
    #         "suspension_summary",
    #         "manufacturer",
    #         "model"
    #     ]
    # ],
    # "transform__preprocess__description_text__tfidf__max_df": [0.6,0.7],
    # "model__C": [30]
}

catboost_results, catboost_estimator = predefined_grid_search(
    X_train_valid,
    y_train_valid,
    catboost_pipe, 
    hparams,
    cv=cv_strat,
    scoring="neg_root_mean_squared_error",
    n_jobs=1,
    verbose=3
)

results["catboost"] = pd.DataFrame(catboost_results).drop(columns=[c for c in catboost_results.keys() if "param_" in c])
pd.DataFrame(catboost_results)
```

# Model Interpretation

```python
feature_imp = pd.DataFrame(
    data=svr_estimator.named_steps['model'].coef_,
    index=svr_estimator.named_steps["transform"].get_feature_names_out()
)

(
    feature_imp
    .reset_index()
    .rename(columns={"index":"feature",0:"importance"})
    .assign(abs_importance = lambda _df: np.abs(_df.importance))
    .sort_values("abs_importance", ascending=False)
    .query("feature.str.contains('add_post_month')")
    .head(50)
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
        pred = lambda _df : catboost_estimator.predict(_df),
    )
```

```python
sample_data = pd.DataFrame(
        data={
            "ad_title":["2018 Yeti SB5.5 Medium TURQ Mountain Bike"],
            "year": [2018],
            "location":["Calgary, United States"],
            "original_post_date":[pd.to_datetime("2022-MAR-31")],
    "description":[
        """
        2018 Yeti SB5.5 Medium TURQ Mountain Bike. Will clean up like new. everything works and is well lubrcated. it looks alittle dirty in the pics but this bike will clean right up. rims ae perfect and the shocks still ride like new. the derailer is scratched up. that is the only mrks on the bike tha wont assh off.
"""
]
})

sample_data.assign(
        pred = lambda _df : catboost_estimator.predict(_df),
    )
```

```python
sample_data = pd.DataFrame(
        data={
            "ad_title":["2011 Norco CRR SL H&R Block Team Edition - Dura Ace"],
            "location":["Calgary, Canada"],
            "original_post_date":[pd.to_datetime("2022-MAR-31")],
    "description":[
        """Excellent condition H&R Block Team Bike. Based on the Norco CRR SL Frame from 2011: https://www.norco.com/bike-archives/2011/crr-sl/

Hasn't been ridden since 2015 - chain has  20 rides on it. 

Highlights:
- Full carbon frame
- Full Dura Ace shifters, derailleurs, and cassette (2x10)
- FSA K-Force full carbon cranks (length=172.5)
- Brand new Fulcrum Racing 6 wheelset
- FSA Bar & Stem (110 mm stem, 42 cm wide bars)
- Fully integrated seat mast (+/- 1 cm adjustment). Currently set at 73 cm saddle height (center of BB to saddle surface). Most likely will fit people  5'6 to 5'10kept up service of all the pivots as well.
"""
]
})

sample_data.assign(
        pred = lambda _df : catboost_estimator.predict(_df),
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
sample_data = pd.DataFrame(
        data={
            "ad_title":["2021 Trek Fuel EX 7"],
        #     "age_at_post":[9*365],
            "location":["Calgary, Canada"],
            "original_post_date":[pd.to_datetime("2022-AUG-31")],
    "description":[
        """2021 Trek Fuel EX 7
        Full SRAM NX Eagle 1x12, Bontrager Kovee Pro 30 TLR Boost 29" MTB Wheelset."""
]
})

sample_data.assign(
        pred = lambda _df : catboost_estimator.predict(_df),
)
```

```python
sample_data = df_valid.sample(1)
year_sweep = pd.DataFrame()
for year in range(2010,2030):
    new_result = (
        sample_data
        .assign(
            original_post_date = pd.to_datetime(f"{year}-01-01"),
            # bike_model_year = lambda _df: pd.to_datetime(f"{year}-01-01"),
            # ad_title = lambda _df: _df.ad_title.str.replace("\d{4}",str(year), regex=True),
            pred = lambda _df : catboost_estimator.predict(_df),
        )
        [["original_post_date","ad_title","pred"]]
    )
    year_sweep = pd.concat([
        year_sweep,
        new_result
        ]
    )

year_sweep.plot(x="original_post_date", y="pred", marker='o', title=sample_data.ad_title.iloc[0]);
```

```python
sample_data = df_valid.sample(1)
month_sweep = pd.DataFrame()
for month in range(1,13):
    new_result = (
        sample_data
        .assign(
            original_post_date = pd.to_datetime(f"2022-{month}-01"),
            month_of_sale = month,
            pred = lambda _df : catboost_estimator.predict(_df),
        )
        [["original_post_date","month_of_sale","ad_title","pred"]]
    )
    month_sweep = pd.concat([
        month_sweep,
        new_result
        ]
    )

month_sweep.plot(x="month_of_sale", y="pred", marker='o', title=sample_data.ad_title.iloc[0]);
```

```python
# Rough exploration of validation set with SVR model
# - Appears we overpredict more heavily on 2022 ads, appears 2022 prices might have gotten weaker?
# - We overpredict on certain E-bike categories (E-Bike MTB)
# - Doesn't appear to cause issues if price is mentioned in Ad text (residuals distribution ~ the same with and without)
# - By month residuals don't show clear trends
# - Otherwise no clear trends to improve modelling with

pd.set_option("display.max_colwidth", None)
df_valid_inspection = (
    pd.concat([
        X_valid,
        y_valid
    ],
    axis=1
    )
    .assign(
        pred = lambda _df : catboost_estimator.predict(_df),
        residual = lambda _df: _df.price_cpi_adjusted_CAD - _df.pred,
        price_in_description = lambda _df: _df.description.str.contains('[$][0-9]+[.]', regex=True),
        month = lambda _df : _df.original_post_date.dt.month,
        condition = lambda _df: _df.condition.str.strip(),
        shipping_restrictions = lambda _df: np.select(
            [
                _df.restrictions.str.contains("Local pickup"),
                _df.restrictions.str.contains("within country"),
                _df.restrictions.str.contains("within continent"),
                _df.restrictions.str.contains("globally"),
                _df.restrictions.str.contains("ship locally"),
            ],
            [
                "local_pickup_only",
                "within_country",
                "within_continent",
                "globally",
                "ship_locally"
            ],
            default="None"
    )
    )
)
```

```python
for col in ["year","month", "price_in_description", "condition", "shipping_restrictions"]:
    (
        df_valid_inspection
            .pipe((sns.displot, "data"), x="residual", hue=col, kind="kde", common_norm=False, bw_adjust=0.5, height=5, aspect=2, rug=True)
    )
```

# Transformers Approach

```python
os.environ['MLFLOW_TRACKING_URI'] = "http://192.168.0.26:5000"
```

```python
model_name = "roberta-large"
```

```python
# Need columns of "text" and "label" for typical datasets usage
hf_train_dataset = Dataset.from_pandas(df_train.rename(columns={"ad_title_description":"text", "price_cpi_adjusted_CAD":"label"}))
hf_valid_dataset = Dataset.from_pandas(df_valid.rename(columns={"ad_title_description":"text", "price_cpi_adjusted_CAD":"label"}))

tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

hf_train_encoded_dataset = hf_train_dataset.map(tokenize_function, batched=True)
hf_valid_encoded_dataset = hf_valid_dataset.map(tokenize_function, batched=True)

```

```python
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    rmse = mean_squared_error(labels, predictions, squared=False)
    return {"rmse": rmse}

batch_size = 10

args = TrainingArguments(
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=3,
    fp16=True,
    # report_to=MLFlow,
    weight_decay=0.01,
     output_dir='/mnt/h/',
    metric_for_best_model='rmse'
    )

trainer = Trainer(
    model,
    args,
    train_dataset=hf_train_encoded_dataset,
    eval_dataset=hf_valid_encoded_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
```

# AutoGluon MultiModal


## Data Setup

```python
# Setup transformations prior to fit using Sklearn helpers
gluon_transformer = Pipeline(steps=[
    ('preprocess',
        ColumnTransformer(
            transformers = [
                ("price", "passthrough", ["price_cpi_adjusted_CAD"] ),
                ("add_age", add_age_transformer, ["ad_title","original_post_date"]),
                (
                    "add_post_month",
                    Pipeline(
                        steps=[
                            ("get_post_month",get_post_month_transformer),
                        ]
                    ),
                    ['original_post_date']
                ),
                (
                    "add_country",
                    Pipeline(
                        steps=[
                            ("get_country",add_country_transformer),
                        ]
                    ),
                    ['location']
                ),
                # ("image", "passthrough",["image"]),
                ("add_covid_flag", add_covid_transformer,["original_post_date"]),
                 (
                    "title_text", 
                    # "passthrough",
                    Pipeline(
                        steps=[
                            # Remove mentions of year so model doesn't learn to predict based on that year's prices
                            ('remove_year', remove_year_transformer),
                        ]
                    ), 
                    ["ad_title"]
                ),
                (
                    "description_text", 
                    # "passthrough",
                    Pipeline(
                        steps=[
                            # Remove mentions of year so model doesn't learn to predict based on that year's prices
                            ('remove_year', remove_year_transformer), 
                        ]
                    ), 
                    ["description"]
                ),
                (
                    "specs_data_categories", 
                    "passthrough",
                    [
                    "frame_summary",
                    "wheels_summary",
                    "drivetrain_summary",
                    "brakes_summary",
                    "suspension_summary",
                    "groupset_summary",
                    "fork_summary",
                    ]
                ),
                (
                    "specs_data_numerical",
                    SimpleImputer(strategy="median"),
                    ["msrp_cleaned", "weight_summary", "front_travel_summary", "rear_travel_summary"]
                )
            ],
        remainder="drop"
    )
    ),
    ]
)
gluon_transformer

```

```python
gluon_transformer.fit(df_train)
df_train_gluon = pd.DataFrame(data=gluon_transformer.transform(df_train), columns=gluon_transformer.get_feature_names_out())#.astype({"add_age__age_at_post":int})
df_valid_gluon = pd.DataFrame(data=gluon_transformer.transform(df_valid), columns=gluon_transformer.get_feature_names_out())#.astype({"add_age__age_at_post":int})
```

```python
# Save transformation pipeline from raw ad data
run_uuid = uuid.uuid4().hex
os.makedirs(os.path.join(os.getcwd(),"tmp"), exist_ok=True)
dump(gluon_transformer, f"./tmp/{run_uuid}-transformer-auto_mm_bikes")
```

## Model Fitting

```python

# time_limit = 12*60*60  # set to larger value in your applications
model_path = f"./tmp/{run_uuid}-auto_mm_bikes"
predictor = MultiModalPredictor(
        label='price__price_cpi_adjusted_CAD',
        problem_type="regression",
        path=model_path,
        # eval_metric="mean",
        verbosity=4
        ) 
predictor.fit(
        train_data=df_train_gluon,
        tuning_data=df_valid_gluon,
        time_limit=None,
        hyperparameters={
                # "optimization.learning_rate": tune.uniform(0.00001, 0.00006),
                # "optimization.optim_type": tune.choice(["adamw"]),
                # "model.names": ["hf_text", "timm_image", "clip", "categorical_mlp", "numerical_mlp", "fusion_mlp"],
                "optimization.max_epochs": 10,
                "optimization.patience": 6, # Num checks without valid improvement, every 0.5 epoch by default
                "env.per_gpu_batch_size": 14,
                "env.num_workers": 20,
                "env.num_workers_evaluation": 20,
                # "model.hf_text.checkpoint_name": tune.choice(["google/electra-base-discriminator", 'roberta-base','roberta-large']),
                # "model.hf_text.checkpoint_name": 'roberta-large',
                # "model.hf_text.text_trivial_aug_maxscale": tune.choice(["0.0","0.10","0.15","0.2"])
        },
        # hyperparameter_tune_kwargs = {
        #         "searcher": "bayes",
        #         "scheduler": "ASHA",
        #         "num_trials": 5,
        # }
)
```

```python
predictor.fit_summary()
```

## Model Load and Results Inspection

```python
predictor = MultiModalPredictor.load("tmp/be901a11a3234fef85729799a07a4965-auto_mm_bikes")
```

```python
df_gluon_inspection = pd.DataFrame(
    data=gluon_transformer.transform(
        pd.concat([df_valid, df_train])
    ),
    columns= gluon_transformer.fit(df_valid).get_feature_names_out().tolist(),
).assign(pred=lambda _df: predictor.predict(_df))

```

```python
# Augment with all initial columns in dataset
df_gluon_inspection = pd.concat([
        df_gluon_inspection.assign(
        resid = lambda _df: _df.price__price_cpi_adjusted_CAD - _df.pred
        ),
         pd.concat([df_valid.assign(split="valid"), df_train.assign(split="train")]).reset_index()
    ],
axis=1
)

```

```python
print(f"""Mean absolute percentage error on selected model: {mean_absolute_percentage_error(df_gluon_inspection.query("split=='valid'").price__price_cpi_adjusted_CAD, df_gluon_inspection.query("split=='valid'").pred):.2%}""")
```

## Sweep Over Features


### Depreciation Curves

```python
initial_df = df_valid_gluon.sample(1)
sample_df = pd.concat(
    [
        initial_df.assign(
            add_covid_flag__covid_flag=0,
            add_age__age_at_post = str(x)
        )
        for x in range(100,3000,365)
    ]
).assign(pred=lambda _df: predictor.predict(_df))
sample_df.plot(x="add_age__age_at_post", y="pred",
            #    title=sample_df["title_text__ad_title"].iloc[0],
               marker='o');
# sample_df
initial_df.head(1)
```

### PDP Plots

```python
df_valid_gluon
```

```python
from sklearn.inspection import PartialDependenceDisplay

# Patch to allow partial dependence to work with Gluon models
predictor.fitted_ = True
predictor._estimator_type = "regressor"
predictor.set_verbosity(0)

fig, ax = plt.subplots(figsize=(30, 10))
PartialDependenceDisplay.from_estimator(
    predictor, df_valid_gluon.sample(10000),
    ["add_age__age_at_post",("add_age__age_at_post","specs_data_numerical__msrp_cleaned")],
    grid_resolution=10,
    verbose=4,
    ax=ax
    # n_jobs=3
    )
```

```python
initial_df = df_valid_gluon.sample(1)
sample_df = pd.concat(
    [
        initial_df.assign(
            add_covid_flag__covid_flag=0,
            title_text__ad_title = lambda _df: str(x) + _df.title_text__ad_title.str.replace(r'((?:19|20)\d{2})', '', regex=True).iloc[0],
            model_year = x
        )
        for x in range(2010,2030)
    ]
).assign(pred=lambda _df: predictor.predict(_df))
sample_df.plot(x="model_year", y="pred",
               title=sample_df["title_text__ad_title"].iloc[0],
               marker='o');
initial_df
```

```python
initial_df = df_valid_gluon.sample(1)
sample_df = pd.concat(
    [
        initial_df.assign(
            add_covid_flag__covid_flag=0,
            add_post_month__original_post_date=x,
        )
        for x in pd.date_range("01-JAN-2022", "31-DEC-2022", freq="M").strftime("%B")
    ]
).assign(pred=lambda _df: predictor.predict(_df))
sample_df
sample_df.plot(x="add_post_month__original_post_date", y="pred",
               title=sample_df["title_text__ad_title"].iloc[0],
               marker='o');
# initial_df.head(1)
```

```python
# initial_df = df_valid_gluon.sample(1)
# sample_df = pd.concat(
#     [
#         initial_df.assign(
#             add_covid_flag__covid_flag=0,
#             image__image=x,
#         )
#         for x in df_valid_gluon["image__image"].sample(10)
#     ]
# ).assign(pred=lambda _df: predictor.predict(_df))
# sample_df
# sample_df.plot(x="image__image", y="pred",
#                title=f"{sample_df['title_text__ad_title'].iloc[0]} - with random images",
#                marker='o');
# plt.xticks(rotation=90);
```

```python
pd.set_option('display.max_colwidth', 0)

def make_clickable(val):
    return '<a href="{}">{}</a>'.format(val,val)

(
    df_gluon_inspection
    .assign(
        abs_resid = lambda _df: np.abs(_df.resid)
    )
    [["url","add_age__age_at_post","original_post_date", "resid", "abs_resid","pred","price_cpi_adjusted_CAD", "ad_title", "description"]]
    .sort_values("abs_resid", ascending=False)
    .head(20)
    .assign(
        url = lambda _df : _df.url.apply(make_clickable)
    )
    .style
    .background_gradient(subset="resid", cmap="RdBu")
    .set_caption("Largest Absolute Residuals")
)
```

### Residuals by Spec Feature

```python
# Show the residuals distribution, facet by if certain columns are NA or not
# - Appears that we overpredict on ads with no description, and underpredict on ads with no image
check_cols = [
    'wheels_summary',
    'drivetrain_summary',
    'brakes_summary',
    'suspension_summary',
    'groupset_summary',
    'fork_summary',
    'msrp_cleaned',
    'weight_summary',
    'front_travel_summary',
    'rear_travel_summary'
]
for col in check_cols:
    g= (
        df_gluon_inspection
        .query("split == 'valid'")
        .query("resid.abs() < 2500")
        .assign(
            **{f"{col}__isna": lambda _df: (_df[col].isna()) | (_df[col] == 0.0)}
        )
        .astype({"price__price_cpi_adjusted_CAD":float})
        .pipe((sns.displot, "data"), x="resid", hue=f"{col}__isna", kind="kde", common_norm=False, bw_adjust=0.5, height=5, aspect=2, rug=True)
        # .pipe((sns.jointplot, "data"),y="pred", x="price__price_cpi_adjusted_CAD", hue=f"{col}__isna", height=10, alpha=0.1)
    )
    g.fig.suptitle(f"Residuals by Specs Data Availability, Col: {col}");


```

```python
g= (
    df_gluon_inspection.astype({"price__price_cpi_adjusted_CAD":float})
    .pipe((sns.jointplot, "data"),kind="reg", y="pred", x="price__price_cpi_adjusted_CAD", joint_kws={"scatter_kws": {"alpha":0.1}}, height=10)
)
g.fig.suptitle("Predicted vs. Actual")
```

```python
g= (
    df_valid_gluon.astype({"price__price_cpi_adjusted_CAD":float})
    .assign(
        years_old = lambda _df: _df.add_age__age_at_post/365
    )
    .sample(10000)
    .pipe((sns.jointplot, "data"),x="years_old", y="price__price_cpi_adjusted_CAD", height=10)
)
g.fig.suptitle("Price vs. Age of Ad")
```

## Embeddings Exploration

```python
# gluon_embeddings = predictor.extract_embedding(pd.concat([df_valid_gluon, df_train_gluon]))
```

### Save Files for External Exploration

```python
# gluon_embeddings.shape
```

```python
# # Save data for exploring in separate app
# df_gluon_inspection.to_csv("data/df_gluon_inspection.csv", index=False)
# np.save("data/gluon_embeddings", gluon_embeddings)

# df_valid_gluon_inspection = pd.read_csv("data/df_valid_gluon_inspection.csv", index_col=None)
# valid_gluon_embeddings = np.load("data/valid_gluon_embeddings.npy")
```

```python

```
