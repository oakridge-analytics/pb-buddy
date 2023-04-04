import re
import os

import dash
import pandas as pd
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import plotly.express as px
import requests

from parser import parse_buysell_ad

# constants
current_year = pd.Timestamp.now().year
if os.environ.get("API_URL") is None:
    API_URL = "https://bikebuddy-api.azurewebsites.net/text-predict"
else:
    API_URL = os.environ["API_URL"]

dash_app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SLATE])
app = dash_app.server

dash_app.layout = dbc.Container(
    [
        html.H1("Bike Buddy", className="text-center mb-4"),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H2("Enter Pinkbike URL:"),
                        html.Br(),
                        dcc.Input(id="ad-url", placeholder="Enter Pinkbike Buy Sell URL"),
                        html.Br(),
                        dbc.Button(
                            "Parse Ad",
                            id="parse-ad",
                            color="primary",
                            className="mt-3",
                        ),
                        html.Br(),
                        html.Hr(),
                        html.H2("Or manually enter:"),
                        html.Br(),
                        dbc.Label("Model Year", html_for="model-year"),
                        dcc.Dropdown(
                            id="model-year",
                            options=list(range(current_year + 1, 2000, -1)),
                            placeholder="Select Model Year",
                        ),
                        html.Br(),
                        dbc.Label("Ad Title", html_for="ad-title"),
                        html.Br(),
                        dcc.Input(
                            id="ad-title", type="text", placeholder="Enter Ad Title"
                        ),
                        html.Br(),
                        dbc.Label("Ad Description", html_for="ad-description"),
                        html.Br(),
                        dcc.Textarea(
                            id="ad-description",
                            placeholder="Enter Ad Description",
                            style={"height": 200, "width": "100%"},
                        ),
                        html.Br(),
                        dbc.Label("Country", html_for="country"),
                        dcc.Dropdown(
                            id="country",
                            options=[
                                {"label": "Canada", "value": "Canada"},
                                {"label": "United States", "value": "United States"},
                            ],
                            placeholder="Select Country",
                        ),
                        html.Br(),
                        dbc.Label("Post Date: ", html_for="post-date"),
                        html.Br(),
                        dcc.DatePickerSingle(
                            id="post-date",
                            date=str(pd.Timestamp.now()),
                            placeholder="Select Post Date",
                        ),
                        html.Br(),
                        dbc.Button(
                            "Submit for Prediction",
                            id="submit-prediction",
                            color="primary",
                            className="mt-3",
                        ),
                    ],
                    md=4,
                ),
                dbc.Col(
                    [
                        dcc.Loading(
                            [html.Div(id="kpi")],
                            type="cube",
                        )
                    ],
                    md=8,
                    align="center",
                ),
            ],
        ),
        # dbc.Row(
        #     [
        #         dbc.Col(
        #             [
        #                 dcc.Loading(
        #                     [
        #                         dcc.Graph(id="depreciation-graph"),
        #                     ],
        #                     type="cube",
        #                 )
        #             ],
        #             width=12,
        #         ),
        #     ]
        # ),
    ]
)


# Callback function
@dash_app.callback(
    [
        Output("model-year", "value"),
        Output("ad-title", "value"),
        Output("ad-description", "value"),
        Output("post-date", "date"),
        Output("country", "value"),
    ],
    [Input("parse-ad", "n_clicks")],
    [State("ad-url", "value")],
)
def update_ad_fields(n_clicks, ad_url):
    if n_clicks is None or ad_url is None:
        return [None, None, None, str(pd.Timestamp.now()), None]

    parsed_ad = parse_buysell_ad(buysell_url=ad_url, delay_s=0)

    # Post process for dash_app inputs:
    year_match = re.search(r"((?:19|20)\d{2})", parsed_ad["ad_title"])
    if year_match:
        parsed_ad["model_year"] = year_match.group(1)
    else:
        parsed_ad["model_year"] = None

    country_match = re.search(r"((Canada)|(United States))", parsed_ad["location"])
    if country_match is None:
        parsed_ad["country"] = None
    parsed_ad["country"] = country_match.group(1)

    return (
        parsed_ad["model_year"],
        parsed_ad["ad_title"],
        parsed_ad["description"],
        pd.to_datetime(parsed_ad["original_post_date"]).date(),
        parsed_ad["country"],
    )


@dash_app.callback(
    Output("kpi", "children"),
    [Input("submit-prediction", "n_clicks")],
    [
        State("model-year", "value"),
        State("ad-title", "value"),
        State("ad-description", "value"),
        State("country", "value"),
        State("post-date", "date"),
    ],
)
def update_output(n_clicks, model_year, ad_title, ad_description, country, post_date):
    if n_clicks is None:
        return html.H3(
            "Please Enter Data For Price Prediction and Click Submit"
        )  # , go.Figure()

    # Single request
    data = [
        {
            "ad_title": f"{model_year} {ad_title}",
            "description": ad_description,
            "location": f"Unknown, {country}",
            "original_post_date": post_date,
        }
    ]

    response = requests.post(API_URL, json=data)
    response_data = response.json()
    kpi_value = response_data["predictions"][0]
    kpi_element = html.H3(
        f"Predicted Price: ${round(kpi_value, 2)} CAD", className="text-center"
    )

    # # Depreciation curve over next 5 years
    # depreciaton_data = [
    #     {
    #         "ad_title": f"{model_year} {ad_title}",
    #         "description": ad_description,
    #         "location": f"Unknown, {country}",
    #         "original_post_date": str(date),
    #     }
    #     for date in pd.date_range(
    #         start=f"{current_year-1}-01-01", periods=6, freq="Y"
    #     ).tolist()
    # ]
    # response = requests.post(api_url, json=depreciaton_data)
    # response_data = response.json()["predictions"]
    # df_depreciation = pd.DataFrame(
    #     data=[dict(single_ad) for single_ad in depreciaton_data]
    # ).assign(prediction=response_data)

    # depreciation_fig = px.line(
    #     data_frame=df_depreciation, x="original_post_date", y="prediction", markers=True
    # )

    return kpi_element  # , depreciation_fig


if __name__ == "__main__":
    dash_app.run_server(host="0.0.0.0", debug=True)
