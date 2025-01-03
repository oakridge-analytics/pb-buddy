import base64
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from flask import Flask, Response, render_template, request, stream_with_context
from prompts import ASSISTANT_PROMPT
from pydantic import BaseModel
from swarm import Agent, Swarm

app = Flask(__name__)


def mapbox_geocode(address: str) -> tuple[float, float]:
    """Geocode an address using Mapbox API.

    Requires MAPBOX_API_KEY in environment variables.

    Returns
    -------
    tuple[float, float]
        Longitude and latitude of the address
    """
    url = f"https://api.mapbox.com/search/geocode/v6/forward?q={address}&access_token={os.getenv('MAPBOX_API_KEY')}"
    response = requests.get(url)
    # Get lat,lon from response
    return tuple(response.json()["features"][0]["geometry"]["coordinates"])


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
                "predicted_price_cad",
                "predicted_price_diff",
                "distance_km",
            ]
        ]
        .head(max_results)
        .to_markdown(index=False)
    )


def get_user_location(user_city_and_country: str) -> Tuple[float, float]:
    return mapbox_geocode(user_city_and_country)


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
                tool_call_msg = f"\n\n```\nCalling function: {name}\n```\n\n"
                yield tool_call_msg
                needs_space = True

        if "delim" in chunk and chunk["delim"] == "end" and content:
            yield "\n"
            content = ""
            has_started = False


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
) -> float:
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
    float
        Predicted price in CAD
    """
    api_url = os.environ["PRICING_API_URL"]
    ad = BikeBuddyAd(
        ad_title=ad_title,
        description=description,
        original_post_date=original_post_date,
        location=location,
    )

    headers = {"Content-Type": "application/json", **get_auth_headers()}

    response = requests.post(api_url, json=[ad.model_dump()], headers=headers)

    if response.status_code != 200:
        raise Exception(f"API request failed: {response.text}")

    predictions = BikeBuddyAdPredictions(**response.json())
    return predictions.predictions[0]


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
            model="gpt-4o-mini",
            instructions=ASSISTANT_PROMPT,
            functions=[get_user_location, get_relevant_bikes, get_price_prediction],
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
load_dotenv(".env")

dataset = pd.read_parquet("s3://bike-buddy/data/chat/latest_base_chat_data.parquet").assign(
    front_travel=lambda df: df.front_travel.fillna(0), rear_travel=lambda df: df.rear_travel.fillna(0)
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
