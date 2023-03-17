from pydantic import BaseModel


class BikeBuddyAd(BaseModel):
    category: str
    original_post_date: str
    last_repost_date: str
    still_for_sale: str
    view_count: float
    watch_count: float
    price: float
    currency: str
    description: str
    ad_title: str
    location: str
    datetime_scraped: str
    url: str
    frame_size: str
    wheel_size: str
    front_travel: str
    condition: str
    material: str
    rear_travel: str
    seat_post_diameter: float = None
    seat_post_travel: str
    front_axle: str
    rear_axle: str
    shock_eye_to_eye: str
    shock_stroke: str
    shock_spring_rate: float = None
    restrictions: str
    _id: int = None
    category_num: int
    scrape_rank: float
    last_repost_year: int
    last_repost_month: str
    year: int
    cpi: float
    most_recent_cpi: float
    price_cpi_adjusted: float
    fx_month: str
    fx_rate_USD_CAD: float
    price_cpi_adjusted_CAD: float = None
    url_rank: float
    ad_title_description: str
