import base64
import json
import logging
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import openai
import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from flask import Flask, Response, render_template, request, stream_with_context
from prompts import ASSISTANT_PROMPT
from pydantic import BaseModel
from swarm import Agent, Swarm

from pb_buddy.scraper import (
    AdType,
    parse_buysell_ad,
    parse_buysell_buycycle_ad,
    parse_buysell_pinkbike_ad,
)
from pb_buddy.utils import convert_currency, mapbox_geocode

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# Get the logger for this module
logger = logging.getLogger(__name__)

app = Flask(__name__)


def calculate_distance_vectorized(df, ref_lat, ref_lon):
    lat1, lon1, lat2, lon2 = map(np.radians, [ref_lat, ref_lon, df["lat"], df["lon"]])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    r = 6371
    return r * c


def get_relevant_bikes(
    price_limit_cad: float,
    min_front_travel: float,
    max_front_travel: float,
    min_rear_travel: float,
    max_rear_travel: float,
    frame_size: str,
    category: str,
    lon: float,
    lat: float,
    distance_km_threshold: float,
    max_results: int = 3,
) -> pd.DataFrame:
    df = dataset.query("price_cad < @price_limit_cad")
    df = df.query("front_travel >= @min_front_travel and front_travel <= @max_front_travel")
    df = df.query("rear_travel >= @min_rear_travel and rear_travel <= @max_rear_travel")
    df = df.query("frame_size_normalized == @frame_size")
    df = df.query("category == @category")
    df = df.assign(distance_km=lambda _df: calculate_distance_vectorized(_df, lat, lon))
    df = df.query("distance_km <= @distance_km_threshold")
    return (
        df.sort_values(by="predicted_price_diff", ascending=False)[
            [
                "ad_title",
                "url",
                "price",
                "currency",
                "last_repost_date",
                "predicted_price",
                "predicted_price_diff",
                "distance_km",
            ]
        ]
        .head(max_results)
        .to_markdown(index=False)
    )


def get_user_location(user_city_and_country: str) -> Tuple[float, float]:
    return mapbox_geocode(user_city_and_country)


class BikeBuddyAd(BaseModel):
    ad_title: str
    description: str
    original_post_date: str
    location: str


class BikeBuddyAdPredictions(BaseModel):
    predictions: List[float]


def get_auth_headers() -> dict:
    """Create authentication headers for Modal webhook endpoints.

    Returns
    -------
    dict
        Headers with Basic authentication for Modal endpoints
    """
    if os.environ.get("MODAL_TOKEN_ID") is None or os.environ.get("MODAL_TOKEN_SECRET") is None:
        raise Exception("MODAL_TOKEN_ID and MODAL_TOKEN_SECRET environment variables must be set")

    credentials = f"{os.environ['MODAL_TOKEN_ID']}:{os.environ['MODAL_TOKEN_SECRET']}"
    encoded_credentials = base64.b64encode(credentials.encode()).decode()
    return {"Proxy-Authorization": f"Basic {encoded_credentials}"}


def get_price_prediction(
    ad_title: str,
    description: str,
    location: str,
    original_post_date: str,
) -> tuple[float, str]:
    """Get a price prediction for a bike ad from the pricing API.

    Parameters
    ----------
    ad_title : str
        The title of the bike ad
    description : str
        The description of the bike ad
    location : str
        Location in format "city, country"
    original_post_date : str
        Original post date in format "YYYY-MM-DD"

    Returns
    -------
    tuple[float, str]
        Predicted price in CAD and currency
    """
    try:
        api_url = os.environ.get(
            "PRICING_API_URL", "https://dbandrews--bike-buddy-api-autogluonmodelinference-predict.modal.run"
        )
        ad = BikeBuddyAd(
            ad_title=ad_title,
            description=description,
            original_post_date=original_post_date,
            location=location,
        )

        headers = {"Content-Type": "application/json", **get_auth_headers()}
        logger.info(f"Sending ad to pricing API: {ad.model_dump()}")

        # Add timeout to prevent hanging
        response = requests.post(
            api_url,
            json=[ad.model_dump()],
            headers=headers,
            timeout=30,  # 30 second timeout
        )

        response.raise_for_status()  # Raise error for bad status codes

        logger.info(f"Pricing API response: {response.json()}")
        predictions = BikeBuddyAdPredictions(**response.json())
        return predictions.predictions[0], "CAD"

    except requests.Timeout:
        logger.error("Timeout while connecting to pricing API", exc_info=True)
        raise Exception("Timeout while getting price prediction. Please try again.")
    except requests.RequestException as e:
        logger.error(f"Error connecting to pricing API: {str(e)}", exc_info=True)
        raise Exception(f"Failed to get price prediction: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in price prediction: {str(e)}", exc_info=True)
        raise Exception(f"Unexpected error in price prediction: {str(e)}")


def determine_ad_url_parsing_type(url: str) -> AdType:
    """Determine the type of ad from the URL.

    Parameters
    ----------
    url : str
        URL of the ad

    Returns
    -------
    AdType
        Type of ad (PINKBIKE, BUYCYCLE, or OTHER)
    """
    if "pinkbike" in url:
        return "pinkbike"
    elif "buycycle" in url:
        return "buycycle"
    else:
        return "other ad requires screenshot"


def get_page_from_browser_service(url: str) -> str:
    """Get page content from browser service.

    Parameters
    ----------
    url : str
        URL to get content from

    Returns
    -------
    str
        Page content
    """
    browser_service_url = os.environ.get(
        "BROWSER_SERVICE_URL", "https://dbandrews--browser-service-browserservice-web.modal.run"
    )
    headers = get_auth_headers()
    logger.info(f"Requesting page content from browser service for URL: {url}")
    response = requests.get(f"{browser_service_url}/page/{url}", headers=headers)

    if response.status_code != 200:
        logger.error(f"Failed to get page content: {response.text}")
        raise Exception(f"Failed to get page content: {response.text}")

    logger.info("Successfully retrieved page content")
    return response.json()["content"]


def normalize_parsed_ad(parsed_ad: Dict[str, Any], url: str) -> Dict[str, Any]:
    """Normalize a parsed bike ad by adding standard fields and extracting additional information.

    Parameters
    ----------
    parsed_ad : Dict[str, Any]
        The raw parsed ad data
    url : str
        The URL of the ad

    Returns
    -------
    Dict[str, Any]
        Normalized ad data with additional fields
    """
    # Add standard fields
    parsed_ad["url"] = url
    parsed_ad["datetime_scraped"] = str(pd.Timestamp.today(tz="US/Mountain"))

    # Extract year from title/description if not present
    if parsed_ad.get("model_year") is None:
        year_match = re.search(r"((?:19|20)\d{2})", f'{parsed_ad["ad_title"]}: {parsed_ad["description"]}')
        if year_match:
            parsed_ad["model_year"] = year_match.group(1)

    # Extract country from location
    country_match = re.search(r"((Canada)|(United States)|(United Kingdom))", parsed_ad["location"])
    if country_match is None:
        parsed_ad["country"] = None
    else:
        parsed_ad["country"] = country_match.group(1)

    return parsed_ad


def get_page_screenshot(url: str) -> Path:
    """Get page screenshot from browser service API and save to temp file.

    Parameters
    ----------
    url : str
        URL to get screenshot from

    Returns
    -------
    Path
        Path to temporary screenshot file
    """
    headers = get_auth_headers()
    browser_service_url = os.environ.get(
        "BROWSER_SERVICE_URL", "https://dbandrews--browser-service-browserservice-web.modal.run"
    )
    response = requests.get(f"{browser_service_url}/screenshot/{url}", headers=headers)
    if response.status_code == 407:
        raise Exception("Authentication failed - check MODAL_TOKEN_ID and MODAL_TOKEN_SECRET")
    if response.status_code != 200:
        raise Exception(f"Failed to get screenshot: {response.status_code}")

    # Create a temporary file that will be automatically cleaned up
    temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    try:
        # Decode base64 and write to temp file
        image_data = base64.b64decode(response.json()["image_base64"])
        temp_file.write(image_data)
        temp_file.close()
        return Path(temp_file.name)
    except Exception as e:
        # Clean up the temp file if there's an error
        os.unlink(temp_file.name)
        raise e


def parse_other_adtype_from_screenshot(screenshot_path: Path) -> Dict[str, Any]:
    """Parse a bike ad from a screenshot file using GPT-4o.

    Parameters
    ----------
    screenshot_path : Path
        Path to the screenshot file

    Returns
    -------
    Dict[str, Any]
        Parsed ad data
    """
    tool_name = "parse_buysell_ad"
    current_date = pd.Timestamp.now().date().strftime("%Y-%m-%d")
    client = openai.Client()

    # Read the image file and encode to base64
    with open(screenshot_path, "rb") as f:
        b64_screenshot = base64.b64encode(f.read()).decode()

    # Clean up the temporary file
    try:
        os.unlink(screenshot_path)
    except Exception as e:
        logger.warning(f"Failed to clean up temporary file {screenshot_path}: {e}")

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
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_screenshot}"}},
                ],
            },
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": tool_name,
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
                        "required": [
                            "model_year",
                            "ad_title",
                            "description",
                            "location",
                            "original_post_date",
                            "price",
                            "currency",
                        ],
                    },
                },
            }
        ],
        tool_choice={"type": "function", "function": {"name": tool_name}},
    )

    if not response.choices or not response.choices[0].message.tool_calls:
        return {}

    parsed_ad = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
    for key, value in parsed_ad.items():
        if value == "Unknown":
            parsed_ad[key] = None
    return parsed_ad


def parse_pinkbike_or_buycycle_ad(url: str) -> Optional[Dict[str, Any]]:
    """Parse a bike ad from a URL.

    Parameters
    ----------
    url : str
        URL of the ad

    Returns
    -------
    Optional[Dict[str, Any]]
        Parsed ad data if successful, None if failed
    """
    try:
        logger.info(f"Starting to parse ad from URL: {url}")
        if "pinkbike" in url:
            ad_type = AdType.PINKBIKE
        elif "buycycle" in url:
            ad_type = AdType.BUYCYCLE
        else:
            ad_type = AdType.OTHER

        logger.info(f"Determined ad type: {ad_type}")

        page_content = get_page_from_browser_service(url)
        parsed_ad = parse_buysell_ad(page_content, url, region_code=1, ad_type=ad_type)
        if parsed_ad:
            logger.info("Successfully parsed ad, normalizing data")
            return normalize_parsed_ad(parsed_ad, url)

        logger.warning("Failed to parse ad - no data returned from parser")
        return None
    except Exception as e:
        logger.error(f"Error parsing ad {url}: {str(e)}", exc_info=True)
        return None


def convert_currency_logging(price: float, from_currency: str, to_currency: str) -> float:
    """Convert a price to the specified currency and log the conversion.

    Parameters
    ----------
    price : float
        Price to convert
    from_currency : str
        Currency to convert from
    to_currency : str
        Currency to convert to

    Returns
    -------
    float
        Converted price amount
    """
    converted_price = convert_currency(price, from_currency, to_currency)
    logger.info(f"Converted price from {from_currency} to {to_currency}: {converted_price}")
    return converted_price


def process_streaming_response(response):
    content = ""
    last_sender = ""
    has_started = False
    needs_space = False

    for chunk in response:
        if "sender" in chunk:
            last_sender = chunk["sender"]

        if "content" in chunk and chunk["content"] is not None:
            # Add a space if we're coming from a tool call
            if needs_space:
                yield " "
                needs_space = False

            if not has_started:
                has_started = True
            yield chunk["content"]
            content += chunk["content"]

        if "tool_calls" in chunk and chunk["tool_calls"] is not None:
            for tool_call in chunk["tool_calls"]:
                f = tool_call["function"]
                name = f["name"]
                if not name:
                    continue
                # Format tool call as markdown code block
                tool_call_msg = f"\n\n```\nCalling function: {tool_map[name]}\n```\n\n"
                yield tool_call_msg
                needs_space = True

        if "delim" in chunk and chunk["delim"] == "end" and content:
            yield "\n"
            content = ""
            has_started = False


tool_map = {
    "get_user_location": "Get User Location",
    "get_relevant_bikes": "Get Relevant Bikes",
    "get_price_prediction": "Get Price Prediction",
    "parse_pinkbike_or_buycycle_ad": "Parse Ad",
    "determine_ad_url_parsing_type": "Determine Ad URL Parsing Type",
    "normalize_parsed_ad": "Normalize Parsed Ad",
    "get_page_screenshot": "Get Page Screenshot",
    "parse_other_adtype_from_screenshot": "Parse Ad from Screenshot",
    "convert_currency_logging": "Convert Currency",
}


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    message = data.get("message")
    conversation_history = data.get("history", [])

    if not message:
        return {"error": "No message provided"}, 400

    def generate():
        client = Swarm()
        agent = Agent(
            name="Bike Broker",
            model="gpt-4o",
            instructions=ASSISTANT_PROMPT,
            functions=[globals()[name] for name in tool_map.keys()],
        )

        messages = conversation_history + [{"role": "user", "content": message}]

        response = client.run(
            agent=agent,
            messages=messages,
            stream=True,
        )

        for chunk in process_streaming_response(response):
            yield chunk

    return Response(stream_with_context(generate()), mimetype="text/event-stream")


@app.route("/")
def home():
    return render_template("index.html")


load_dotenv(os.path.expanduser("~/.openai/credentials"))
load_dotenv("../../.env")

dataset = pd.read_parquet("s3://bike-buddy/data/chat/latest_base_chat_data.parquet").assign(
    front_travel=lambda df: df.front_travel.fillna(0), rear_travel=lambda df: df.rear_travel.fillna(0)
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
