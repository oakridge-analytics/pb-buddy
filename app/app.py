import os
import re

from modal import App, Image, Secret, wsgi_app

if os.environ.get("API_URL") is None:
    API_URL = "https://dbandrews--bike-buddy-api-autogluonmodelinference-predict.modal.run"
else:
    API_URL = os.environ["API_URL"]

app = App("bike-buddy-ui")
image = (
    Image.debian_slim(python_version="3.9")
    .pip_install(
        [
            "dash==2.9.1",
            "dash-bootstrap-components==1.4.1",
        ],
        # force_build=True,
    )
    .pip_install_private_repos(
        "github.com/pb-buddy/pb-buddy@feat/add_other_ad_parsing",
        git_user="dbandrews",
        secrets=[Secret.from_name("pb-buddy-github")],
        # force_build=True,
    )
    .run_commands("playwright install-deps")
    .run_commands("playwright install chromium")
)


@app.function(image=image)
@wsgi_app()
def flask_app():
    import logging

    import dash
    import dash_bootstrap_components as dbc
    import pandas as pd
    import requests
    from bs4 import BeautifulSoup
    from dash import dcc, html
    from dash.dependencies import Input, Output, State
    from playwright.sync_api import sync_playwright

    from pb_buddy.scraper import (AdType, parse_buysell_buycycle_ad,
                                  parse_buysell_pinkbike_ad)

    dash_app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SLATE])

    # Configure logging
    logging.basicConfig(level=logging.DEBUG)

    def get_page_from_playwright_scraper(url: str) -> str:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto(url)
            page_content = page.content()
            browser.close()
        return page_content

    current_year = pd.Timestamp.now().year
    dash_app.layout = dbc.Container(
        [
            html.H1("Bike Buddy", className="text-center mb-4"),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H3("Enter used bike ad URL to predict price:", id="h3-heading"),
                            dbc.Tooltip("Supported platforms: Pinkbike, Buycycle.", target="h3-heading"),
                            html.Br(),
                            dcc.Input(id="ad-url", placeholder="Enter URL", type="text"),
                            html.Br(),
                            dbc.Button(
                                "Parse Ad",
                                id="parse-ad",
                                color="primary",
                                className="mt-3",
                            ),
                            html.Br(),
                            html.Hr(),
                            dbc.Button(
                                "Or Manually Enter ▼",
                                id="collapse-button",
                                className="mb-3",
                                color="primary",
                            ),
                            html.Br(),
                            dbc.Collapse(
                                id="collapse-section",
                                is_open=False,
                                children=[
                                    dbc.Label("Model Year", html_for="model-year"),
                                    dcc.Dropdown(
                                        id="model-year",
                                        options=[
                                            {"label": str(year), "value": year}
                                            for year in range(current_year + 1, 2000, -1)
                                        ],
                                        placeholder="Select Model Year",
                                    ),
                                    html.Br(),
                                    dbc.Label("Ad Title", html_for="ad-title"),
                                    html.Br(),
                                    dcc.Input(id="ad-title", type="text", placeholder="Enter Ad Title"),
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
                                ],
                            ),
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
        ]
    )

    @dash_app.callback(
        Output("collapse-section", "is_open"),
        [Input("collapse-button", "n_clicks")],
        [State("collapse-section", "is_open")],
    )
    def toggle_collapse(n_clicks, is_open):
        if n_clicks:
            return not is_open
        return is_open

    def determine_ad_type(url: str) -> AdType:
        if "pinkbike" in url:
            return AdType.PINKBIKE
        elif "buycycle" in url:
            return AdType.BUYCYCLE
        else:
            return AdType.OTHER

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

        ad_type = determine_ad_type(ad_url)
        page_content = get_page_from_playwright_scraper(ad_url)
        soup = BeautifulSoup(page_content, features="html.parser")
        if ad_type == AdType.PINKBIKE:
            parsed_ad = parse_buysell_pinkbike_ad(soup)
            logging.info(f"Parsed ad: {ad_type}")
        elif ad_type == AdType.BUYCYCLE:
            parsed_ad = parse_buysell_buycycle_ad(soup)
        else:
            parsed_ad = {"ad_title": "Unknown", "description": "Unknown", "original_post_date": pd.Timestamp("now")}

        # Post process for dash_app inputs:
        try:
            year_match = re.search(r"((?:19|20)\d{2})", parsed_ad["ad_title"])
        except KeyError as e:
            logging.info(f"KeyError: {e} in parsed_ad: {parsed_ad}")

        if year_match:
            parsed_ad["model_year"] = year_match.group(1)
        else:
            parsed_ad["model_year"] = None

        country_match = re.search(r"((Canada)|(United States))", parsed_ad["location"])
        if country_match is None:
            parsed_ad["country"] = None
        else:
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
            return html.H3("Please Enter Data For Price Prediction and Click Submit")  # , go.Figure()

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
        kpi_element = html.H3(f"Predicted Price: ${round(kpi_value, 2)} CAD", className="text-center")

        return kpi_element

    return dash_app.server
        kpi_element = html.H3(f"Predicted Price: ${round(kpi_value, 2)} CAD", className="text-center")

        return kpi_element

    return dash_app.server
