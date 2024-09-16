import base64
import json
import logging
import os
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
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright
import yfinance as yf

from pb_buddy.scraper import (
    AdType,
    parse_buysell_buycycle_ad,
    parse_buysell_pinkbike_ad,
)

if os.environ.get("API_URL") is None:
    API_URL = "https://dbandrews--bike-buddy-api-autogluonmodelinference-predict.modal.run"
else:
    API_URL = os.environ["API_URL"]

dash_app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SLATE])
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

user_agents_opts = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
]

# Need to use playwright to evade Cloudflare's bot detection
# in certain cases.
def get_page_from_playwright_scraper(url: str) -> str:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        user_agent = random.choice(user_agents_opts)
        logger.info(f"Using user agent: {user_agent}")
        page.set_extra_http_headers({"User-Agent": user_agent})
        response = page.goto(url)
        if response.status == 403:
            logger.warning(f"Received 403 Forbidden status for URL: {url}")
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
                        "price": {
                            "description": "The price of the bike",
                            "title": "Price",
                            "type": "number",
                        },
                        "currency": {
                            "description": "The currency of the price",
                            "title": "Currency",
                            "type": "string",
                        },
                    },
                    "required": ["model_year", "ad_title", "description", "location", "original_post_date", "price", "currency"],
                },
            },
        }
    ]
    try:
        b64_img = get_page_screenshot(url)
    except PlaywrightTimeoutError:
        return {}, ""

    current_date = pd.Timestamp.now().date().strftime("%Y-%m-%d")
    client = openai.Client()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant, for extracting information from a bike ad"
                    " for calling a downstream function. If you can't determine a value, just leave it blank."
                    f" The current date is {current_date} if it's a relative date."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Extract the information about the main ad in this image. If the image isn't about a bicycle ad, return blanks for all fields.",
                    },
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}},
                ],
            },
        ],
        tools=tools,
        tool_choice={"type": "function", "function": {"name": f"{tool_name}"}},
    )
    parsed_ad = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
    for key, value in parsed_ad.items():
        if value == "Unknown":
            parsed_ad[key] = None
    return parsed_ad, b64_img

def convert_currency_symbol(symbol: str) -> str:
    currency_symbols = {
        "$": "USD",
        "€": "EUR",
        "£": "GBP",
        "¥": "JPY",
        "₣": "CHF",
        "C$": "CAD",
        "A$": "AUD",
        "kr": "SEK",
        "₹": "INR",
        "₽": "RUB",
    }
    return currency_symbols.get(symbol, symbol)

def convert_currency(amount: float, from_currency: str, to_currency: str) -> float:
    if from_currency == to_currency:
        return amount
    
    from_currency = convert_currency_symbol(from_currency)
    to_currency = convert_currency_symbol(to_currency)
    
    ticker = f"{from_currency}{to_currency}=X"
    exchange_rate = yf.Ticker(ticker).info['regularMarketPreviousClose']
    return amount * exchange_rate


current_year = pd.Timestamp.now().year
SAMPLE_URLS = [
    "https://buycycle.com/en-ca/bike/megatower-s-carbon-c-29-2019-60199",
    "https://www.pinkbike.com/buysell/3924063/",
    "https://www.pinkbike.com/buysell/3896775/",
    "https://www.pinkbike.com/buysell/3905160/",
]

dash_app.layout = dbc.Container(
    [
        html.H1("Bike Buddy", className="text-center mb-4"),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Div(
                            children=[
                                html.H3(
                                    [
                                        "Enter used bike ad URL* to parse and predict price :",
                                    ],
                                    id="h3-heading",
                                ),
                            ],
                        ),
                        dbc.Tooltip(
                            "Supported platforms: Pinkbike, Buycycle. Other URL types will use slower screenshot+LLM parsing.",
                            target="h3-heading",
                        ),
                        html.Br(),
                        dcc.Dropdown(
                            id="sample-urls",
                            options=[{"label": url, "value": url} for url in SAMPLE_URLS],
                            placeholder="Select a sample URL or enter your own below",
                            style={'width': '100%'}
                        ),
                        html.Br(),
                        dcc.Input(
                            id="ad-url",
                            placeholder="Enter URL",
                            type="text",
                            style={'width': '100%'}
                        ),
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
                        dcc.Loading(
                            [
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
                                                {"label": "United Kingdom", "value": "United Kingdom"},
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
                                        dbc.Row([
                                            dbc.Col([
                                                dbc.Label("Listed Price", html_for="ad-price"),
                                                dcc.Input(
                                                    id="ad-price",
                                                    type="number",
                                                    placeholder="Enter Price",
                                                    style={"width": "100%"}
                                                ),
                                            dbc.Tooltip(
                                                "Enter the listed price for comparison with the predicted price.",
                                                target="ad-price",
                                                placement="top",
                                            ),
                                            ], width=8),
                                            dbc.Col([
                                                dbc.Label("Currency", html_for="price-currency"),
                                                dcc.Dropdown(
                                                    id="price-currency",
                                                    options=[
                                                        {"label": "CAD", "value": "CAD"},
                                                        {"label": "USD", "value": "USD"},
                                                        {"label": "GBP", "value": "GBP"},
                                                        {"label": "EUR", "value": "EUR"},
                                                    ],
                                                    value="CAD",
                                                    style={"width": "100%"}
                                                ),
                                            ], width=4),
                                        ]),
                                        html.Br(),
                                    ],
                                ),
                            ],
                            type="dot",
                        ),
                        dbc.Button(
                            "Submit for Prediction",
                            id="submit-prediction",
                            color="secondary",
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
                        html.Div(id="screenshot-container", children=""),
                        dcc.Loading(
                            [
                                html.Div(id="kpi"),
                            ],
                            type="dot",
                        ),
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
        Output("submit-prediction", "children"),
    ],
    [
        Input("ad-title", "value"),
        Input("ad-description", "value"),
        Input("country", "value"),
        Input("post-date", "date"),
        Input("model-year", "value"),
        Input("ad-price", "value"),
        Input("price-currency", "value"),
    ],
)
def update_button_state(title, description, country, date, model_year, price, currency):
    all_inputs_filled = all([title, description, country, date, model_year, price is not None, currency])
    
    if all_inputs_filled:
        return False, "success", "Submit for Prediction"
    else:
        return True, "secondary", "Fill all input fields for prediction"

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

@dash_app.callback(
    Output("ad-url", "value"),
    Input("sample-urls", "value")
)
def update_url_input(sample_url):
    return sample_url

@dash_app.callback(
    [
        Output("model-year", "value"),
        Output("ad-title", "value"),
        Output("ad-description", "value"),
        Output("post-date", "date"),
        Output("country", "value"),
        Output("url-alert", "is_open"),
        Output("screenshot-container", "children"),
        Output("collapse-button", "children"),
        Output("ad-price", "value"),
        Output("price-currency", "value"),
    ],
    [Input("parse-ad", "n_clicks")],
    [State("ad-url", "value"), State("sample-urls", "value")],
)
def update_ad_fields(n_clicks, custom_url, sample_url):
    if n_clicks is None:
        return [None, None, None, str(pd.Timestamp.now()), None, False, "", "Or Manually Enter/Edit ▼", None, "CAD"]

    ad_url = custom_url or sample_url
    if ad_url is None:
        return [None, None, None, str(pd.Timestamp.now()), None, True, "", "Or Manually Enter/Edit ▼", None, "CAD"]

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
        return [None, None, None, str(pd.Timestamp.now()), None, True, "", "Or Manually Enter/Edit ▼", None, "CAD"]

    if parsed_ad.get("model_year") is None:
        year_match = re.search(r"((?:19|20)\d{2})", f'{parsed_ad["ad_title"]}: {parsed_ad["description"]}')
        if year_match:
            parsed_ad["model_year"] = year_match.group(1)
        else:
            parsed_ad["model_year"] = None  

    country_match = re.search(r"((Canada)|(United States)|(United Kingdom))", parsed_ad["location"])
    if country_match is None:
        parsed_ad["country"] = None
    else:
        parsed_ad["country"] = country_match.group(1)

    if parsed_ad["original_post_date"] is None:
        parsed_ad["original_post_date"] = pd.Timestamp.now()

    return (
        parsed_ad["model_year"],
        parsed_ad["ad_title"],
        parsed_ad["description"],
        pd.to_datetime(parsed_ad["original_post_date"]).date(),
        parsed_ad["country"],
        False,
        html.Details([html.Summary("Screenshot used"), html.Img(src=f"data:image/png;base64,{b64_img}")])
        if ad_type == AdType.OTHER
        else "",
        "Edit ▼",
        parsed_ad["price"],
        convert_currency_symbol(parsed_ad["currency"]),
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
        State("ad-price", "value"),
        State("price-currency", "value"),
        State("ad-url", "value"),
    ],
)
def update_output(n_clicks, model_year, ad_title, ad_description, country, post_date, ad_price, price_currency, ad_url):
    if n_clicks is None:
        return html.H3("")

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
    kpi_value = response_data["predictions"][0]  # Assuming this is in CAD

    if ad_price is not None:
        ad_price = float(ad_price)
        ad_price_cad = convert_currency(ad_price, price_currency, "CAD")
        price_difference_cad = ad_price_cad - kpi_value
        price_difference_original = convert_currency(price_difference_cad, "CAD", price_currency)
        percentage_difference = (price_difference_cad / kpi_value) * 100
        predicted_price_original = convert_currency(kpi_value, "CAD", price_currency)
        if price_difference_cad > 0:
            price_color = "danger"
            price_text = f"Overpriced by {price_currency} ${abs(round(price_difference_original, 2))} ({round(percentage_difference, 2)}%)"
        elif price_difference_cad < 0:
            price_color = "success"
            price_text = f"Underpriced by {price_currency} ${abs(round(price_difference_original, 2))} ({round(abs(percentage_difference), 2)}%)"
        else:
            price_color = "secondary"
            price_text = "Priced perfectly"

        kpi_element = dbc.Card(
            dbc.CardBody([
                html.H2(html.A(f"{ad_title}", href=ad_url, target="_blank"), className="card-title text-center"),
                html.H3(f"Predicted Price: ${round(predicted_price_original, 2)} {price_currency} / ${round(kpi_value, 2)} CAD ", className="card-title text-center"),
                html.H3(f"Listed Price: ${round(ad_price, 2)} {price_currency}", className="card-title text-center"),
                html.H3(price_text, className="card-title text-center text-" + price_color)
            ]),
            className="mt-3"
        )
    else:
        kpi_element = dbc.Card(
            dbc.CardBody([
                html.H2(html.A(f"{ad_title}", href=ad_url, target="_blank"), className="card-title text-center"),
                html.H3(f"Predicted Price: ${round(kpi_value, 2)} CAD", className="text-center")
            ]),
            className="mt-3"
        )

    return kpi_element


if __name__ == "__main__":
    dash_app.run_server(debug=True)