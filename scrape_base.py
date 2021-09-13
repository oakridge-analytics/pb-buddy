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

# Custom code
import pb_buddy.scraper as scraper
import pb_buddy.utils as ut


if __name__ == "__main__":
    #%%
    # Settings -----------------------------------------------------------------
    categories_to_scrape = range(1, 101 + 1)
    num_jobs = 8  # Curently only for initial link grab
    delay_s = 0.1

    # Setup PIA vpn object for manipulation:
    vpn = PiaVpn()

    # Setup output folder
    folder_name = str(pd.Timestamp("today").date())
    output_dir = Path("data") / folder_name
    os.makedirs(output_dir, exist_ok=True)

    # Get category lookup
    with open("category_dict.json", "r") as fh:
        cat_dict = json.load(fh)

    # Iterate through all categories in random order, prevent noticeable patterns?
    num_categories_scraped = 0
    for category_to_scrape in np.random.choice(
        categories_to_scrape, size=len(categories_to_scrape), replace=False
    ):
        print(
            f"*************Starting Category {[x for x,v in cat_dict.items() if v== category_to_scrape]} - Number: {category_to_scrape}. {num_categories_scraped} Categories Scraped So Far"
        )

        # Get total number of pages-----------------------------------------------
        base_url = f"https://www.pinkbike.com/buysell/list/?region=3&page=1&category={category_to_scrape}"
        search_results = requests.get(base_url).content
        soup = BeautifulSoup(search_results, features="html.parser")
        total_page_count = scraper.get_total_pages(soup)

        if total_page_count == 0:
            print(search_results)
            break

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
        # Change region and wait for reconnect before continuing
        vpn.set_region("random")
        time.sleep(10)

        # Get ad data -----------------------------------------------------------
        ad_data = pd.DataFrame()
        print(f"Extracting ad data from {len(ad_urls)} ads")

        intermediate_ad_data = Parallel(n_jobs=num_jobs, backend="threading")(
            delayed(scraper.parse_buysell_ad)(ad, delay_s=delay_s)
            for ad in tqdm(ad_urls)
        )

        ad_data = pd.concat([ad_data, pd.DataFrame(intermediate_ad_data)], axis=0)
        ad_data.to_csv(
            output_dir.joinpath(f"category_{category_to_scrape}_ad_data.csv"),
            index=False,
        )
        f"*************Finished Category {[x for x,v in cat_dict.items() if v== category_to_scrape]}"

        num_categories_scraped += 1
        # Change region and wait for reconnect before continuing
        vpn.set_region("random")
        time.sleep(10)
