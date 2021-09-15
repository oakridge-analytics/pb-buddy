# %%
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm, utils
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
categories_to_scrape = range(50, 51)
num_jobs = os.cpu_count()  # Curently only for initial link grab
delay_s = 0.0
use_vpn = False


# Setup PIA vpn object for manipulation. Make sure to connect
vpn = PiaVpn()
vpn.connect(verbose=False, timeout=20)
vpn.set_region("random")
time.sleep(15)

# Get category lookup
with open("category_dict.json", "r") as fh:
    cat_dict = json.load(fh)

# Iterate through all categories in random order, prevent noticeable patterns?
num_categories_scraped = 0
for category_to_scrape in np.random.choice(
    categories_to_scrape, size=len(categories_to_scrape), replace=False
):
    print(
        f"** ***********Starting Category {[x for x,v in cat_dict.items() if v == category_to_scrape]} \
        - Number: {category_to_scrape}. {num_categories_scraped} Categories Scraped So Far"
    )

    # Get existing open ads and stats of last scrape
    base_data_path = os.path.join(
        "data",
        "base_data",
        f"category_{category_to_scrape}_ad_data.parquet.gzip")
    if os.path.isfile(base_data_path):
        base_data = pd.read_parquet(os.path.join(
            "data",
            "base_data",
            f"category_{category_to_scrape}_ad_data.parquet.gzip"))
        last_scrape_dt = pd.to_datetime(base_data.datetime_scraped).max()
    else:
        base_data = pd.DataFrame()
        last_scrape_dt = pd.to_datetime("01-JAN-1900")

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
    ad_urls = Parallel(n_jobs=num_jobs)(
        delayed(scraper.get_buysell_ads)(x, delay_s=delay_s)
        for x in tqdm(page_urls)
    )
    ad_urls = ut.flatten(ad_urls)

    # Get new ad data ---------------------------------------------------------
    intermediate_ad_data = []
    print(f"Extracting ad data from {len(ad_urls)} ads")

    # Do sequentially and check datetime of last scrape
    # Only add new ads
    for url in ad_urls:
        single_ad_data = scraper.parse_buysell_ad(url, delay_s=0)
        display_string = f"\r Checking ad from {single_ad_data['Last Repost Date:'].strip()} vs. last scrape date {last_scrape_dt}"
        print(display_string, sep=' ', end='', flush=True)

        if single_ad_data != {}:
            if pd.to_datetime(single_ad_data["Last Repost Date:"]).date() >= \
                    last_scrape_dt.date():
                intermediate_ad_data.append(single_ad_data)
            else:
                print("")
                break

    # New ads where not in existing urls
    if len(intermediate_ad_data) == 0:
        continue
    else:
        recently_added_ads = pd.DataFrame(intermediate_ad_data)
        new_ads = recently_added_ads.loc[~recently_added_ads.url.isin(
            base_data.url),:]

        # Get ads that might have an update. check price, description and title
        # for change.
        cols_to_check = ["price", "description", "ad_title"]
        updated_ads = (recently_added_ads
                       .loc[recently_added_ads.url.isin(base_data.url),:]
                       .sort_values("url")
                       .reset_index(drop=True)
                       )
        previous_versions = (base_data
                             .loc[base_data.url.isin(updated_ads.url)]
                             .sort_values("url")
                             .reset_index(drop=True)
                             )
        unchanged_mask = updated_ads[cols_to_check].eq(
            previous_versions[cols_to_check]).all(1)
        updated_ads = updated_ads[~unchanged_mask]

        ad_data = pd.concat(
            [base_data.loc[~base_data.url.isin(updated_ads)],
             updated_ads,
             new_ads], axis=0)

    print(f" Adding {len(new_ads)} new ads to base data")
    print(f" Updating {len(updated_ads)} ads")

    # Get sold ad data ------------------------------------------------
    potentially_sold_urls = base_data.loc[~ base_data.url.isin(ad_urls), "url"]

    # Rescrape sold ads to make sure, check for "SOLD"
    sold_data_path = os.path.join(
        "data",
        "sold_ads",
        f"category_{category_to_scrape}_sold_ad_data.parquet.gzip"
    )
    if os.path.isfile(sold_data_path):
        sold_data = pd.read_parquet(sold_data_path)
    else:
        sold_ad_data = pd.DataFrame()

    intermediate_ad_data = []
    print(
        f"Extracting ad data from {len(potentially_sold_urls)} potentially sold ads")

    # Do sequentially and check datetime of last scrape
    # Only add new ads
    urls_to_remove = []
    for url in tqdm(potentially_sold_urls):
        single_ad_data = scraper.parse_buysell_ad(url, delay_s=0)
        if single_ad_data and "sold" in single_ad_data["Still For Sale:"].lower():
            intermediate_ad_data.append(single_ad_data)
        else:
            urls_to_remove.append(url)

    print(
        f"Found {len(intermediate_ad_data)} sold ads, {len(urls_to_remove)} removed ads")

    if len(intermediate_ad_data) > 0:
        sold_ad_data = pd.concat(
            [sold_ad_data, pd.DataFrame(intermediate_ad_data)], axis=0)
        sold_ad_data = sold_ad_data.dropna()
        sold_ad_data = ut.fix_dtypes(sold_ad_data, colnames=[
            "price", "Watch Count:"], dtype_desired=int)
        sold_ad_data.to_parquet(
            sold_data_path,
            compression="gzip",
            index=False,
        )

    # remove ads that didn't sell and shouldn't be in anymore ---------------
    ad_data = ut.fix_dtypes(ad_data, colnames=[
        "price", "Watch Count:"], dtype_desired=int)
    ad_data = ad_data.dropna()
    ad_data.loc[~ ad_data.url.isin(urls_to_remove), :].to_parquet(
        base_data_path,
        index=False,
        compression="gzip"
    )
    print(
        f"*************Finished Category {[x for x,v in cat_dict.items() if v== category_to_scrape]}")

    num_categories_scraped += 1


# %%
