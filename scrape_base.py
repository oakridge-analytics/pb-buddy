# %%
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
from joblib import Parallel, delayed
import sys

# Custom code
import pb_buddy.scraper as scraper
import pb_buddy.utils as ut


#%%
# Settings -----------------------------------------------------------------
category_to_scrape = 6
num_jobs = 7

# %%
# Logic:
# - Initial check of how many pages
# - Loop through each page with a delay -> append to list of add urls
# - Loop through ad urls with a delay -> append to DF, with progressive save out

base_url = f"https://www.pinkbike.com/buysell/list/?region=3&page=1&category={category_to_scrape}"

search_results = requests.get(base_url).content
soup = BeautifulSoup(search_results, features="html.parser")
total_page_count = scraper.get_total_pages(soup)

# %%
# Build List of Ad URLS............................................
ad_urls = []
print(f"Scraping {total_page_count} pages of ads......")
pages_to_check = list(range(1, total_page_count + 1))
page_urls = [
    f"https://www.pinkbike.com/buysell/list/?region=3&page={x}&category={category_to_scrape}"
    for x in pages_to_check
]
ad_urls = Parallel(n_jobs=num_jobs)(
    delayed(scraper.get_buysell_ads)(x) for x in tqdm(page_urls)
)
ad_urls = ut.flatten(ad_urls)

# %%
# Ad data columns
# ad_data = pd.read_csv("all_mountain_ad_data.csv", index_col=)
ad_data = pd.DataFrame()
print(f"Extracting ad data from {len(ad_urls)} ads")

intermediate_ad_data = Parallel(n_jobs=1)(
    delayed(scraper.parse_buysell_ad)(ad) for ad in tqdm(ad_urls)
)
# for i, url in tqdm(list(enumerate(ad_urls))):
#     try:
#         single_ad_data = scraper.parse_buysell_ad(url)
#     except requests.exceptions.RequestException:
#         continue

#     intermediate_ad_data.append(single_ad_data)

#     # Progress capture every so often
#     if i % 100 == 0:
#         # print(f"Logging ad data at ad {i}")
#         ad_data = pd.concat([ad_data, pd.DataFrame(intermediate_ad_data)], axis=0)
#         ad_data.to_csv(f"category_{category_to_scrape}_ad_data.csv", index=False)
#     # time.sleep(0.1)

# Catch remainder
ad_data = pd.concat([ad_data, pd.DataFrame(intermediate_ad_data)], axis=0)
ad_data.to_csv(f"category_{category_to_scrape}_ad_data.csv", index=False)
# %%
