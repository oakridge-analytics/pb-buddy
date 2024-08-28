import os

import modal
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


@app.function(image=image, secrets=[modal.Secret.from_name("openai-secret")])
@wsgi_app()
def flask_app():
    import base64
    import json
    import random
    import re

    import dash
    import dash_bootstrap_components as dbc
    import openai
    import pandas as pd
    import requests
    from bs4 import BeautifulSoup
    from dash import dcc, html
    from dash.dependencies import Input, Output, State
    from playwright.sync_api import sync_playwright

    from pb_buddy.scraper import (
        AdType,
        parse_buysell_buycycle_ad,
        parse_buysell_pinkbike_ad,
    )

    dash_app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SLATE])

    user_agents_opts = [
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36",
        "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36",
        "Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.96 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.81 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36 Edge/16.16299",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.96 Safari/537.36 Edge/16.16299",
    ]

    # Need to use playwright to evade Cloudflare's bot detection
    # in certain cases.
    def get_page_from_playwright_scraper(url: str) -> str:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.set_extra_http_headers({"User-Agent": random.choice(user_agents_opts)})
            page.goto(url)
            page_content = page.content()
            browser.close()
        return page_content

    def get_page_screenshot(url: str) -> str:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.set_extra_http_headers({"User-Agent": random.choice(user_agents_opts)})
            page.goto(url)
            screenshot = page.screenshot(full_page=True)
            browser.close()

        return base64.b64encode(screenshot).decode("utf-8")

    def parse_other_buysell_ad(url: str) -> tuple[dict, str]:
        """
        Use a screenshot, and gpt-4o w/ function calling to parse the ad.
        """
        tool_name = "parse_buysell_ad"
        tools = [
            {
                "type": "function",
                "function": {
                    "name": f"{tool_name}",
                    "description": "Parse information about a bicycle ad",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "model_year": {
                                "description": "The year the bike was manufactured",
                                "title": "Model Year",
                                "type": "integer",
                            },
                            "ad_title": {"title": "Ad Title", "type": "string"},
                            "description": {
                                "description": "Detailed information about the bike for sale. Should include all possible details.",
                                "title": "Ad Description",
                                "type": "string",
                            },
                            "location": {
                                "description": 'The city and country where the bike is being sold. If city is unknown, just use "Unknown, Country_Name". The country where the bike is located, either "Canada" or "United States"',
                                "title": "Country",
                                "type": "string",
                            },
                            "original_post_date": {
                                "description": "The original date the ad was posted",
                                "title": "Original Post Date",
                                "type": "string",
                            },
                        },
                        "required": ["model_year", "ad_title", "description", "location", "original_post_date"],
                    },
                },
            }
        ]
        b64_img = get_page_screenshot(url)
        client = openai.Client()
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant, for extracting information from a bike ad for calling a downstream function.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract the information about the main ad on this page."},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}},
                    ],
                },
            ],
            tools=tools,
            tool_choice={"type": "function", "function": {"name": f"{tool_name}"}},
        )
        return json.loads(response.choices[0].message.tool_calls[0].function.arguments), b64_img

    current_year = pd.Timestamp.now().year
    dash_app.layout = dbc.Container(
        [
            html.H1("Bike Buddy", className="text-center mb-4"),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H3("Enter used bike ad URL to parse and predict price:", id="h3-heading"),
                            dbc.Tooltip("Supported platforms: Pinkbike, Buycycle.", target="h3-heading"),
                            html.Br(),
                            dcc.Input(id="ad-url", placeholder="Enter URL", type="text"),
                            html.Br(),
                            dbc.Button(
                                "Parse Ad",
                                id="parse-ad",
                                color="info",
                                className="mt-3",
                            ),
                            html.Br(),
                            html.Div(
                                [
                                    dbc.Alert(
                                        "Failed to parse URL, please try again",
                                        id="url-alert",
                                        color="danger",
                                        dismissable=True,
                                        is_open=False,
                                    ),
                                ],
                                id="alert-container",
                            ),
                            html.Hr(),
                            dbc.Button(
                                "Or Manually Enter/Edit ▼",
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
                            html.Br(),
                            html.Br(),
                            html.Br(),
                        ],
                        md=4,
                    ),
                    dbc.Col(
                        [
                            dcc.Loading(
                                [html.Div(id="kpi"), html.Img(id="screenshot", src="")],
                                type="dot",
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
        [
            Output("submit-prediction", "disabled"),
            Output("submit-prediction", "color"),
        ],
        [
            Input("ad-title", "value"),
            Input("ad-description", "value"),
            Input("country", "value"),
            Input("post-date", "date"),
        ],
    )
    def update_button_state(title, description, country, date):
        if not title or not description or not country or not date:
            return True, "seconday"
        return False, "success"

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
            Output("url-alert", "is_open"),
            Output("screenshot", "src"),
        ],
        [Input("parse-ad", "n_clicks")],
        [State("ad-url", "value")],
    )
    def update_ad_fields(n_clicks, ad_url):
        if n_clicks is None or ad_url is None:
            return [None, None, None, str(pd.Timestamp.now()), None, False, ""]

        ad_type = determine_ad_type(ad_url)
        page_content = get_page_from_playwright_scraper(ad_url)
        soup = BeautifulSoup(page_content, features="html.parser")
        if ad_type == AdType.PINKBIKE:
            parsed_ad = parse_buysell_pinkbike_ad(soup)
        elif ad_type == AdType.BUYCYCLE:
            parsed_ad = parse_buysell_buycycle_ad(soup)
        else:
            parsed_ad, b64_img = parse_other_buysell_ad(ad_url)

        if not parsed_ad:
            return [None, None, None, str(pd.Timestamp.now()), None, True, ""]

        year_match = re.search(r"((?:19|20)\d{2})", parsed_ad["ad_title"])
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
            False,
            f"data:image/png;base64,{b64_img}" if ad_type == AdType.OTHER else "",
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
            return html.H3("Please enter a URL to parse an ad, or enter manually - then click submit.")  # , go.Figure()

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
