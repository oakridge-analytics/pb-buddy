from pydantic import BaseModel
from typing import List


class BikeBuddyAd(BaseModel):
    category: str = None
    original_post_date: str = None
    last_repost_date: str = None
    still_for_sale: str = None
    view_count: float = None
    watch_count: float = None
    price: float = None
    currency: str = None
    description: str = None
    ad_title: str = None
    location: str = None
    datetime_scraped: str = None
    url: str = None
    frame_size: str = None
    wheel_size: str = None
    front_travel: str = None
    condition: str = None
    material: str = None
    rear_travel: str = None
    seat_post_diameter: float = None
    seat_post_travel: str = None
    front_axle: str = None
    rear_axle: str = None
    shock_eye_to_eye: str = None
    shock_stroke: str = None
    shock_spring_rate: float = None
    restrictions: str = None
    price_cpi_adjusted_CAD: float = None


class BikeBuddyAdPredictions(BaseModel):
    predictions: List[float]
