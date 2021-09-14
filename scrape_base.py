# %%
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed
import os
import json
import time
from piapy import PiaVpn
import seaborn as sns

# Custom code
import pb_buddy.scraper as scraper
import pb_buddy.utils as ut

# %%
# Settings -----------------------------------------------------------------
categories_to_scrape = range(100, 101)
num_jobs = 8  # Curently only for initial link grab
delay_s = 0.1

# Setup PIA vpn object for manipulation. Make sure to connect
vpn = PiaVpn()
vpn.connect(verbose=False, timeout=20)

# Setup output folder
folder_name = str(pd.Timestamp("today").date())
output_dir = Path("data") / folder_name
os.makedirs(output_dir, exist_ok=True)

# Get category lookup
with open("category_dict.json", "r") as fh:
    cat_dict = json.load(fh)

vpn.set_region("random")
time.sleep(10)

# Iterate through all categories in random order, prevent noticeable patterns?
num_categories_scraped = 0
for category_to_scrape in np.random.choice(
    categories_to_scrape, size=len(categories_to_scrape), replace=False
):
    print(
        f"*************Starting Category {[x for x,v in cat_dict.items() if v== category_to_scrape]} - Number: {category_to_scrape}. {num_categories_scraped} Categories Scraped So Far"
    )

    # Get existing open ads and stats of last scrape
    base_data = pd.read_parquet(os.path.join(
        "data",
        "base_data",
        f"category_{category_to_scrape}_ad_data.parquet.gzip"))
    last_scrape_dt = pd.to_datetime(base_data.datetime_scraped.max())

    # Get total number of pages-----------------------------------------------
    total_page_count = scraper.get_total_pages(category_to_scrape)

    # Build List of Ad URLS----------------------------------------------------
    ad_urls = []
    print(f"Scraping {total_page_count} pages of ads......")
    pages_to_check = list(range(1, total_page_count + 1))
    page_urls = [
        f"https://www.pinkbike.com/buysell/list/?region=3&page={x}&category={category_to_scrape}"
        for x in pages_to_check
    ]
    ad_urls = Parallel(n_jobs=1)(
        delayed(scraper.get_buysell_ads)(x, delay_s=delay_s)
        for x in tqdm(page_urls)
    )
    ad_urls = ut.flatten(ad_urls)
    # Change region and wait for reconnect before continuing
    vpn.set_region("random")
    time.sleep(10)

    # Get new ad data ---------------------------------------------------------
    ad_data = pd.DataFrame()
    intermediate_ad_data = []
    print(f"Extracting ad data from {len(ad_urls)} ads")

    # Do sequentially and check datetime of last scrape
    # Only add new ads
    for url in ad_urls:
        single_ad_data = scraper.parse_buysell_ad(url, delay_s=0)
        if pd.to_datetime(single_ad_data["Last Repost Date:"]) > \
                last_scrape_dt:
            intermediate_ad_data.append(single_ad_data)
        else:
            break

    print(f" Adding {len(intermediate_ad_data)} new ads to base data")

    ad_data = pd.concat(
        [base_data, pd.DataFrame(intermediate_ad_data)], axis=0)
    ad_data.to_csv(
        output_dir.joinpath(f"category_{category_to_scrape}_ad_data.csv"),
        index=False,
    )

    # Get sold ad data ------------------------------------------------
    sold_urls = base_data.loc[~ base_data.url.isin(ad_urls), "url"]

    # Rescrape sold ads to make sure, check for "SOLD"
    sold_ad_data = pd.DataFrame()
    intermediate_ad_data = []
    print(f"Extracting ad data from {len(sold_urls)} potentially sold ads")

    # Do sequentially and check datetime of last scrape
    # Only add new ads
    for url in sold_urls:
        single_ad_data = scraper.parse_buysell_ad(url, delay_s=0)
        if single_ad_data and "sold" in single_ad_data["Still For Sale:"].lower():
            intermediate_ad_data.append(single_ad_data)

    print(f"Found {len(intermediate_ad_data)} sold ads")

    sold_ad_data = pd.concat(
        [sold_ad_data, pd.DataFrame(intermediate_ad_data)], axis=0)
    sold_ad_data.to_csv(
        output_dir.joinpath(f"category_{category_to_scrape}_sold_ad_data.csv"),
        index=False,
    )
    # check for all ads that dropped out

    f"*************Finished Category {[x for x,v in cat_dict.items() if v== category_to_scrape]}"

    num_categories_scraped += 1
    # Change region and wait for reconnect before continuing
    vpn.set_region("random")
    time.sleep(10)

# %%
