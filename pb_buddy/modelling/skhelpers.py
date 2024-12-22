import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer

from pb_buddy.specs import augment_with_specs

# -----------------------------Helper code for sklearn transformers --------------------------
# `get_feature_names_out()` on Pipeline instances requires us to have callable functions
# for returning calculated feature names and be able to pickle the resulting estimator.
# We create small functions to do this despite how silly they look.


# For encoding country info:----------------------
def add_country(df):
    """
    Extract country from location string, of form "city, country"
    """
    return df.assign(country=lambda _df: _df.location.str.extract(r", ([A-Za-z\s]+)"))[["country"]].astype("category")


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
        df.assign(original_post_date=lambda _df: pd.to_datetime(_df.original_post_date, format="mixed"))
        .assign(
            bike_model_date=lambda _df: pd.to_datetime(_df.ad_title.str.extract(r"((?:19|20)\d{2})", expand=False)),
            age_at_post=lambda _df: (_df.original_post_date - _df.bike_model_date).dt.days,
        )
        .fillna(-1000)
        .astype({"age_at_post": int})[["age_at_post"]]
    )


def add_age_names(func_transformer, feature_names_in):
    return ["age_at_post"]


add_age_transformer = FunctionTransformer(add_age, feature_names_out=add_age_names)


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
    return df.assign(
        covid_flag=lambda _df: np.where(
            (_df.original_post_date > "20-MARCH-2020") & (_df.original_post_date < "01-AUG-2022"),
            1,
            0,
        )
    )[["covid_flag"]]


def add_covid_name(func_transformer, feature_names_in):
    return ["covid_flag"]


add_covid_transformer = FunctionTransformer(add_covid_flag, feature_names_out=add_covid_name)


# Remove year information from ad title and ad description - so model learns based on
# age of ad instead of encoding specific knowledge about each
# year.
def remove_year(df):
    return df.replace(r"((?:19|20)\d{2})", "", regex=True)


remove_year_transformer = FunctionTransformer(remove_year, feature_names_out="one-to-one")


# Get month ad is posted, encode as one hot for modelling
def get_post_month(df):
    """
    Extracts month from `original_post_date`, returns as string for encoding in later
    transforms.
    """
    return (
        df.assign(original_post_date=lambda _df: pd.to_datetime(_df.original_post_date))
        .original_post_date.dt.month_name()
        .astype("category")
        .to_frame()
    )


get_post_month_transformer = FunctionTransformer(get_post_month, feature_names_out="one-to-one")


def augment_spec_features(
    df,
    spec_cols: list[str] = ["groupset_summary", "wheels_summary", "suspension_summary"],
    year_col: str = "year",
    ad_title_col: str = "ad_title",
    manufacturer_threshold: int = 80,
    model_threshold: int = 80,
):
    "Add specs data to input dataframe"
    df_out = df

    # print(df_out)
    return df_out[spec_cols]


def spec_feature_names(func_transformer, feature_names_in):
    return ["covid_flag"]


augment_spec_features_transformer = FunctionTransformer(augment_spec_features, feature_names_out="one-to-one")


class AugmentSpecFeatures(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        spec_cols: list[str] = [
            "groupset_summary",
            "wheels_summary",
            "suspension_summary",
        ],
        year_col: str = "year",
        ad_title_col: str = "ad_title",
        manufacturer_threshold: int = 80,
        model_threshold: int = 80,
    ) -> None:
        super().__init__()
        self.spec_cols = spec_cols
        self.year_col = year_col
        self.ad_title_col = ad_title_col
        self.manufacturer_threshold = manufacturer_threshold
        self.model_threshold = model_threshold

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df_out = X.pipe(
            augment_with_specs,
            year_col=self.year_col,
            ad_title_col=self.ad_title_col,
            manufacturer_threshold=self.manufacturer_threshold,
            model_threshold=self.model_threshold,
        )
        return df_out[self.spec_cols].values

    def get_feature_names_out(self, input_features=None):
        return self.spec_cols
