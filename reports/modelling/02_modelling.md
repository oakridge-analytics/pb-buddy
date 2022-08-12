---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.0
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

from sklearn.base import clone

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

# Remove year information from ad title and ad description - so model learns based on age of ad instead of encoding specific knowledge about each
# year.
def remove_year(df):
    return df.replace("((?:19|20)\d{2})","", regex=True)

remove_year_transformer = FunctionTransformer(remove_year, feature_names_out="one-to-one")


# Get month ad is posted, encode as one hot for modelling
def get_post_month(df):
    """ 
    Extracts month from `original_post_date`, returns as string for encoding in later
    transforms.
    """
    return df.original_post_date.dt.month_name().to_frame()

get_post_month_transformer = FunctionTransformer(get_post_month, feature_names_out="one-to-one")
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
- `location`: What country, and potentially what city/region was the bike listed in?
- `Covid`: Was COVID-19 impacting the market when this bike was listed? We know retail sale of bikes was overwhelmed by demand, likely this occured in used market as well and saw increased prices.


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
                            ('remove_year', remove_year_transformer),
                            ('tfidf',
                                TfidfVectorizer(
                                        # Allow periods only inside numbers for model names etc. 
                                        token_pattern = "[a-zA-Z0-9$&+,:;=?@#|<>^%!]+|[0-9$&+,.:;=?@#|<>^%!]+"
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
                            ('remove_year', remove_year_transformer),
                            ('tfidf',
                                TfidfVectorizer(
                                        # Allow periods only inside numbers for model names etc. 
                                        token_pattern = "[a-zA-Z0-9$&+,:;=?@#|<>^%!]+|[0-9$&+,.:;=?@#|<>^%!]+"
                                )
                            ),
                        ]
                    ), 
                    "description"
                ),
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
# pd.DataFrame(
#     data=transformer.fit_transform(X_train.head(10)).toarray(),
#     columns=transformer.fit(X_train.head(10)).get_feature_names_out()
# ).iloc[0:5,0:20]
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
        ("model", LinearSVR(max_iter=1000, random_state=42))
    ],
)

hparams ={
    "transform__preprocess__title_text__tfidf__max_features" : [200000],
    "transform__preprocess__title_text__tfidf__ngram_range": [(1,3),],
    # "transform__title_bow__max_df":[0.6,],
    "transform__preprocess__description_text__tfidf__max_features" : [200000],
    "transform__preprocess__description_text__tfidf__ngram_range": [(1,3),],
    # "transform__preprocess__description_bow__max_df":[0.6,],
    "model__C": [30]
}

svr_results, svr_estimator = predefined_grid_search(
    X_train_valid,
    y_train_valid,
    svr_pipe, 
    hparams,
    cv=cv_strat,
    scoring="neg_root_mean_squared_error",
    n_jobs=4
)


```

```python
# dump(svr_estimator, "../../data/svr_pipe.joblib")
```

```python

results["linear_svr"] = pd.DataFrame(svr_results).drop(columns=[c for c in svr_results.keys() if "param_" in c])
pd.DataFrame(svr_results).sort_values("split0_test_score", ascending=False)
```

```python
catboost_pipe = Pipeline(
    steps = [
        ("transform", transformer),
        ("model", CatBoostRegressor(verbose=False, task_type="GPU"))
    ]
)

hparams ={
    "transform__preprocess__title_text__tfidf__max_features" : [10000],
    "transform__preprocess__title_text__tfidf__ngram_range": [(1,1),],
    # "transform__preprocess__title_text__tfidf__max_df": [0.6,0.7],
    "transform__preprocess__description_text__tfidf__max_features" : [10000],
    "transform__preprocess__description_text__tfidf__ngram_range": [(1,1),],
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
    n_jobs=-1
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
year_sweep = pd.DataFrame()
for year in range(2010,2023):
    new_result = (
        sample_data
        .assign(
            # original_post_date = pd.to_datetime(f"{year}-01-01"),
            bike_model_year = lambda _df: pd.to_datetime(f"{year}-01-01"),
            ad_title = lambda _df: _df.ad_title.str.replace("\d{4}",str(year), regex=True),
            pred = lambda _df : catboost_estimator.predict(_df),
        )
        [["original_post_date","bike_model_year","ad_title","pred"]]
    )
    year_sweep = pd.concat([
        year_sweep,
        new_result
        ]
    )

year_sweep.plot(x="bike_model_year", y="pred")
```

```python
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

month_sweep.plot(x="month_of_sale", y="pred")
```

```python
pd.set_option("display.max_colwidth", None)
(
    pd.concat([
        X_valid,
        y_valid
    ],
    axis=1
    )
    .query("description.str.contains('[$][0-9]+[.]', regex=True)") #Check ads where prices are mentioned!
    .sample(20, random_state=42)
    .assign(
        pred = lambda _df : svr_estimator.predict(_df),
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

```
