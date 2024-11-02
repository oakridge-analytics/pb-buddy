# # Modelling Prices - Bike Ads
#
# These datasets have been pre calculated in `01_historical_data_preprocessing.md`. This has taken care of adjusting for inflation, and currency conversion on ads.
#
# Ads have been pre filtered to complete bike categories.

import os
import uuid

# Visuals ------------------
import pandas as pd
import seaborn as sns
import torch.multiprocessing as mp
from autogluon.multimodal import MultiModalPredictor

# Other models -----------
from joblib import dump

# Transformers modelling ---------------
# from datasets import Dataset
# from transformers import TrainingArguments, Trainer
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# from transformers import RobertaModel
from sklearn.compose import ColumnTransformer

# Sklearn helpers ------------------------------
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import (
    Pipeline,
)

# Custom code ---------------------
from pb_buddy.modelling.skhelpers import (
    add_age,
    add_age_transformer,
    add_country,
    add_country_transformer,
    add_covid_transformer,
    get_post_month_transformer,
    # AugmentSpecFeatures,
    remove_year_transformer,
)

mp.set_start_method("spawn", force=True)

# # Setup Cells

# In[2]:

if __name__ == "__main__":
    # Where set processed data path in Azure Blob storage
    input_path_name = "s3://bike-buddy/data/historical/adjusted/"
    input_file_name = "2024-10-05_09_13_40__adjusted_bike_ads.parquet.gzip"

    # In[3]:

    # Get dataset from blob storage, unless already found in data folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_cache_file_path = os.path.join(script_dir, "data", f"{input_file_name.split('.')[0]}.csv")
    if os.path.exists(local_cache_file_path):
        df_modelling = pd.read_csv(local_cache_file_path, index_col=None)
    else:
        df_modelling = pd.read_parquet(input_path_name + input_file_name)

    # In[4]:

    df_modelling.to_csv(local_cache_file_path, index=None)

    # In[5]:

    # Do general pre processing of dataset
    # Notes:
    # - Cast date columns to correct datetime type
    # - Remove ads where the year of the bike can't be identified in ad_title
    # - Remove ads that appear to be more than 1 model year ahead of current year (age_at_post<-365)
    # - Remove ads posted from countries outside Canada, United States
    # - Remove calculated columns so Sklearn transformer pipeline can add back in later for ease of downstream use
    df_modelling = (
        df_modelling.assign(
            original_post_date=lambda _df: pd.to_datetime(_df.original_post_date),
            last_repost_date=lambda _df: pd.to_datetime(_df.last_repost_date),
            age_at_post=lambda _df: add_age(_df).fillna(-1000),
            country=lambda _df: add_country(_df),
            ad_title_description=lambda _df: _df.ad_title + " " + _df.description,
            days_between_posts=lambda _df: (_df.last_repost_date - _df.original_post_date).dt.days,
        )
        ####### Augment with specs ########
        # .pipe(
        #     augment_with_specs,
        #     year_col="year",
        #     manufacturer_threshold=80,
        #     model_threshold=80,
        # )
        # .dropna(subset=["spec_url"])
        .query("age_at_post>-365 and days_between_posts<365")
        .query("~price_cpi_adjusted_CAD.isna()")
        .drop(columns=["age_at_post", "country"])
    )

    # Load the preprocessed dataset with inflation adjusted, and currency adjusted price as target.
    #
    # Target column = `price_cpi_adjusted_cad`
    #
    # Likely predictor columns:
    # - `description`: Ad description, free form text
    # - `original_post_date`: Original date ad was posted
    # - `ad_title`: Title of the ad
    # - `location`: What country, and potentially what city/region was the bike listed in?
    # - `Covid`: Was COVID-19 impacting the market when this bike was listed? We know retail sale of bikes was overwhelmed by demand, likely this occured in used market as well and saw increased prices.

    # ## Train/Valid/Test Split

    # We likely want to mix Covid era ads into our train, valid, test sets - we don't want to hold out just a period of time. It's unlikely there is any other temporal trend in the data but we'll check residuals in our fitting process.

    # In[6]:

    # Use set validation set for use later with deep learning approaches that are too hard to do CV on
    df_train, df_valid_test = train_test_split(df_modelling, test_size=0.3, random_state=42)
    df_valid, df_test = train_test_split(df_valid_test, test_size=0.5, random_state=42)

    X_train = df_train.drop(columns=["price_cpi_adjusted_CAD"])
    y_train = df_train["price_cpi_adjusted_CAD"]

    X_valid = df_valid.drop(columns=["price_cpi_adjusted_CAD"])
    y_valid = df_valid["price_cpi_adjusted_CAD"]

    X_test = df_test.drop(columns=["price_cpi_adjusted_CAD"])
    y_test = df_test["price_cpi_adjusted_CAD"]

    # For deterministic validation splits, need train and valid in one array and can pass
    # a mask to indicate beginning is train, end of data is valid. See PredefinedSplit docs for more:
    X_train_valid = pd.concat([X_train, X_valid], axis=0)
    y_train_valid = pd.concat([y_train, y_valid], axis=0)

    # In[7]:

    # %%

    # In[8]:

    print(f"Train set size: {X_train.shape[0]}, valid set size: {X_valid.shape[0]}, test set size: {X_test.shape[0]}")
    gluon_transformer = Pipeline(
        steps=[
            (
                "preprocess",
                ColumnTransformer(
                    transformers=[
                        ("price", "passthrough", ["price_cpi_adjusted_CAD"]),
                        (
                            "add_age",
                            add_age_transformer,
                            ["ad_title", "original_post_date"],
                        ),
                        (
                            "add_post_month",
                            Pipeline(
                                steps=[
                                    ("get_post_month", get_post_month_transformer),
                                ]
                            ),
                            ["original_post_date"],
                        ),
                        (
                            "add_country",
                            Pipeline(
                                steps=[
                                    ("get_country", add_country_transformer),
                                ]
                            ),
                            ["location"],
                        ),
                        # ("image", "passthrough",["image"]),
                        ("add_covid_flag", add_covid_transformer, ["original_post_date"]),
                        (
                            "title_text",
                            # "passthrough",
                            Pipeline(
                                steps=[
                                    # Remove mentions of year so model doesn't learn to predict based on that year's prices
                                    ("remove_year", remove_year_transformer),
                                ]
                            ),
                            ["ad_title"],
                        ),
                        (
                            "description_text",
                            # "passthrough",
                            Pipeline(
                                steps=[
                                    # Remove mentions of year so model doesn't learn to predict based on that year's prices
                                    ("remove_year", remove_year_transformer),
                                ]
                            ),
                            ["description"],
                        ),
                        ###### Add in specs data ##########
                        # (
                        #     "specs_data_categories",
                        #     "passthrough",
                        #     [
                        #         "frame_summary",
                        #         "wheels_summary",
                        #         "drivetrain_summary",
                        #         "brakes_summary",
                        #         "suspension_summary",
                        #         "groupset_summary",
                        #         "fork_summary",
                        #     ],
                        # ),
                        # (
                        #     "specs_data_numerical",
                        #     SimpleImputer(strategy="median"),
                        #     [
                        #         "msrp_cleaned",
                        #         "weight_summary",
                        #         "front_travel_summary",
                        #         "rear_travel_summary",
                        #     ],
                        # ),
                    ],
                    remainder="drop",
                ),
            ),
        ]
    )

    gluon_transformer.fit(df_train)
    df_train_gluon = pd.DataFrame(
        data=gluon_transformer.transform(df_train),
        columns=gluon_transformer.get_feature_names_out(),
    )  # .astype({"add_age__age_at_post":int})
    df_valid_gluon = pd.DataFrame(
        data=gluon_transformer.transform(df_valid),
        columns=gluon_transformer.get_feature_names_out(),
    )  # .astype({"add_age__age_at_post":int})

    # Save transformation pipeline from raw ad data
    run_uuid = uuid.uuid4().hex
    tmp_dir = os.path.join(os.path.dirname(__file__), "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    dump(gluon_transformer, os.path.join(tmp_dir, f"{run_uuid}-transformer-auto_mm_bikes"))
    # ## Model Fitting

    # time_limit = 12*60*60  # set to larger value in your applications
    model_path = os.path.join(tmp_dir, f"{run_uuid}-auto_mm_bikes")
    predictor = MultiModalPredictor(
        label="price__price_cpi_adjusted_CAD",
        problem_type="regression",
        path=model_path,
        # eval_metric="mean",
        verbosity=1,
        presets="best_quality",
    )

    predictor.fit(
        train_data=df_train_gluon,
        tuning_data=df_valid_gluon,
        time_limit=None,
        presets="best_quality",
        # hyperparameters={
        #     "optimization.learning_rate": tune.loguniform(1e-5, 1e-2),
        #     "model.hf_text.checkpoint_name": tune.choice(["microsoft/deberta-v3-base"]),
        #     # "optimization.optim_type": tune.choice(["adamw", "sgd"]),
        #     # "optimization.max_epochs": tune.choice(list(range(5, 31))),
        #     # "env.batch_size": tune.choice([16, 32, 64, 128, 256]),
        #     #     # "optimization.learning_rate": tune.uniform(0.00001, 0.00006),
            #     #     "optimization.optim_type": tune.choice(["adamw"]),
            #     #     # "model.names": ["hf_text", "timm_image", "clip", "categorical_mlp", "numerical_mlp", "fusion_mlp"],
            # "optimization.max_epochs": 1,
            #     #     # "optimization.patience": 6,  # Num checks without valid improvement, every 0.5 epoch by default
            # "env.batch_size": tune.choice([32, 64]),
            # "env.per_gpu_batch_size": 16,
            #     #     "env.num_workers": 1,
            #     #     "env.num_workers_evaluation": 1,
            #     #     # "model.hf_text.checkpoint_name": tune.choice(["google/electra-base-discriminator", 'roberta-base','roberta-large']),
            #     #     # "model.hf_text.checkpoint_name": 'roberta-large',
            # "model.hf_text.text_trivial_aug_maxscale": 0,
        # },
        # hyperparameter_tune_kwargs={
        #     "searcher": "bayes",
        #     "scheduler": "ASHA",
        #     "num_trials": 2,
        # },
    )

    print(predictor.fit_summary())

    # torch.set_float32_matmul_precision(precision='medium')
    # predictor._config.env.per_gpu_batch_size = 70

    df_gluon_inspection = pd.DataFrame(
        data=gluon_transformer.transform(pd.concat([df_valid, df_train])),
        columns=gluon_transformer.fit(df_valid).get_feature_names_out().tolist(),
    ).assign(pred=lambda _df: predictor.predict(_df))

    # In[16]:

    # Augment with all initial columns in dataset
    df_gluon_inspection = pd.concat(
        [
            df_gluon_inspection.assign(resid=lambda _df: _df.price__price_cpi_adjusted_CAD - _df.pred),
            pd.concat([df_valid.assign(split="valid"), df_train.assign(split="train")]).reset_index(),
        ],
        axis=1,
    )

    # In[17]:

    print(
        f"""Mean absolute percentage error on selected model: {mean_absolute_percentage_error(df_gluon_inspection.query("split=='valid'").price__price_cpi_adjusted_CAD, df_gluon_inspection.query("split=='valid'").pred):.2%}"""
    )
