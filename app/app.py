import dash
import pandas as pd
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import plotly.express as px
import requests

# constants
current_year = pd.Timestamp.now().year

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container(
    [
        html.H1("Bike Buddy", className="text-center mb-4"),
        dbc.Row(
            [
                dbc.Col(
                    [
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
                        dcc.Input(
                            id="ad-description",
                            type="text",
                            placeholder="Enter Ad Description",
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
                            id="post-date", placeholder="Select Post Date"
                        ),
                        html.Br(),
                        dbc.Button(
                            "Submit for Prediction",
                            id="submit-prediction",
                            color="primary",
                            className="mt-3",
                        ),
                    ],
                    width=4,
                ),
                dbc.Col(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dcc.Loading(
                                            [html.Div(id="kpi", className="text-center")],
                                            type="cube",
                                        )
                                    ],
                                    width=12,
                                ),
                            ]
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dcc.Loading(
                                            [
                                                dcc.Graph(id="depreciation-graph"),
                                            ],
                                            type="cube",
                                        )
                                    ],
                                    width=12,
                                ),
                            ]
                        ),
                    ],
                    width=8,
                ),
            ]
        ),
    ]
)


@app.callback(
    [Output("kpi", "children"), Output("depreciation-graph", "figure")],
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
        return html.H3("Please Enter Data For Price Prediction"), go.Figure()

    api_url = "https://bikebuddy-api.azurewebsites.net/text-predict"

    # Single request
    data = [
        {
            "ad_title": f"{model_year} {ad_title}",
            "description": ad_description,
            "location": f"Unknown, {country}",
            "original_post_date": post_date,
        }
    ]

    response = requests.post(api_url, json=data)
    response_data = response.json()
    kpi_value = response_data["predictions"][0]
    kpi_element = html.H3(
        f"Predicted Price: ${round(kpi_value, 2)} CAD", className="text-center"
    )

    # Depreciation curve over next 5 years
    depreciaton_data = [
        {
            "ad_title": f"{model_year} {ad_title}",
            "description": ad_description,
            "location": f"Unknown, {country}",
            "original_post_date": str(date),
        }
        for date in pd.date_range(
            start=f"{current_year-1}-01-01", periods=6, freq="Y"
        ).tolist()
    ]
    response = requests.post(api_url, json=depreciaton_data)
    response_data = response.json()["predictions"]
    df_depreciation = pd.DataFrame(
        data=[dict(single_ad) for single_ad in depreciaton_data]
    ).assign(prediction=response_data)

    depreciation_fig = px.line(
        data_frame=df_depreciation, x="original_post_date", y="prediction", markers=True
    )

    return kpi_element, depreciation_fig


if __name__ == "__main__":
    app.run_server(debug=True)
