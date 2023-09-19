import re
import pandas as pd

from rapidfuzz import fuzz

from pb_buddy.data.specs import get_specs_dataset, get_year_manufacturer_model_mapping


def fuzzy_match_bike(
    input_string: str,
    year: int,
    year_manufacturer_model_mapping: dict,
    top_n: int = 3,
    manufacturer_threshold: float = 70,
    model_threshold: float = 70,
    with_similarity: bool = False,
) -> list:
    """Fuzzy match a bike string to a year, manufacturer, and model,
    using hierarchical mapping of year, manufacturer, and model.

    Parameters
    ----------
    input_string : str
        Ad title that contains manufacturer and model of bike
    year : int
        Year of bike
    year_manufacturer_model_mapping : dict
        Mapping of form: {year: {manufacturer: [model1, model2, ...]}}
    top_n : int, optional
        Top relevant matches to return, respecting thresholds,
        by default 3
    manufacturer_threshold : float, optional
        Minimum similarity for a match to be considered, by default 70
    model_threshold : float, optional
        Minimum similarity for a match to be considered, by default 70
    with_similarity : bool, optional
        Return model, and similarity score for each match, by default False

    Returns
    -------
    list
        A list of tuples of form (year, manufacturer, model) or
        (year, manufacturer, (model, similarity_score)) if with_similarity=True
    """
    year = str(year)
    manufacturers = list(year_manufacturer_model_mapping[year].keys())
    # Get most similar manufacturer using partial_ratio, and the similarity score
    manufacturer, manufacturer_similarity = max(
        [
            (manufacturer, fuzz.partial_ratio(manufacturer, input_string))
            for manufacturer in manufacturers
        ],
        key=lambda x: x[1],
    )
    if manufacturer_similarity < manufacturer_threshold:
        manufacturer = None
        model = None
        return (year, manufacturer, model)

    # Get all models for that manufacturer
    models = year_manufacturer_model_mapping[year][manufacturer]
    # Get 3 most similar models using partial_ratio, and the similarity score
    models_similar = [
        (model, fuzz.partial_ratio(model, input_string)) for model in models
    ]
    models_similar = sorted(models_similar, key=lambda x: x[1], reverse=True)[:top_n]

    # For each of 3 most similar models, check compared to threshold and remove if not above
    for model, model_similarity in models_similar:
        if model_similarity < model_threshold:
            models_similar.remove((model, model_similarity))
    if models_similar is None or models_similar == []:
        return [None]

    if not with_similarity:
        models_similar = [model for model, _ in models_similar]

    return [(year, manufacturer, model) for model in models_similar]


def augment_with_specs(
    df_modelling: pd.DataFrame,
    year_col: str = "year",
    ad_title_col: str = "ad_title",
    manufacturer_threshold: int = 80,
    model_threshold: int = 75,
) -> pd.DataFrame:
    """Load specs dataset and fuzzy match each ad title to a year, manufacturer, and model.

    Parameters
    ----------
    df_modelling : pd.DataFrame
        DataFrame to merge bike spec data to
    year_col : str, optional
        Column to find 4 digit years in, by default 'year'
    ad_title_col : str, optional
        Column for ad titles to match manufacturer and model, by default "ad_title"

    Returns
    -------
    pd.DataFrame
        df_modelling with additional columns from spec dataset.
    """
    df_specs = get_specs_dataset()
    year_manufacturer_model_mapping = get_year_manufacturer_model_mapping()
    df_modelling = (
        df_modelling.assign(
            fuzzy_match=lambda _df: _df[["ad_title", "year"]].apply(
                lambda row: fuzzy_match_bike(
                    row["ad_title"].lower(),
                    row["year"],
                    year_manufacturer_model_mapping,
                    top_n=1,
                    manufacturer_threshold=manufacturer_threshold,
                    model_threshold=model_threshold,
                    with_similarity=False,
                ),
                axis=1,
            )
        )
        .assign(
            fuzzy_match=lambda _df: _df.fuzzy_match.apply(
                lambda _tuple: None if _tuple == [None] else " ".join(list(_tuple[0]))
            )
        )
        .merge(
            df_specs,
            left_on="fuzzy_match",
            right_on="year_manufacturer_model",
            how="left",
        )
    )

    return df_modelling
