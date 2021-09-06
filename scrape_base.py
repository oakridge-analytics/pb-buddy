# %%
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from tqdm import tqdm
import pb_buddy.scraper as scraper

#%%
# Settings -----------------------------------------------------------------
category_to_scrape = 75

# %%
# Logic:
# - Initial check of how many pages
# - Loop through each page with a delay -> append to list of add urls
# - Loop through ad urls with a delay -> append to DF, with progressive save out

page_num = 1
base_url = f"https://www.pinkbike.com/buysell/list/?region=3&page={page_num}&category={category_to_scrape}"

search_results = requests.get(base_url).content
soup = BeautifulSoup(search_results, features="html.parser")
total_page_count = scraper.get_total_pages(soup)

# %%
# Build List of Ad URLS............................................
ad_urls = []
print(f"Scraping {total_page_count} pages of ads......")
for page in tqdm(range(1, total_page_count)):
    # print(f"Getting ad urls on page {page}")
    base_url_for_pages = f"https://www.pinkbike.com/buysell/list/?region=3&page={page}&category={category_to_scrape}"
    search_results = requests.get(base_url_for_pages).content
    soup = BeautifulSoup(search_results, features="html.parser")
    ad_urls.extend(scraper.get_buysell_ads(soup))

    # time.sleep(0.1)

# %%
# Ad data columns
# ad_data = pd.read_csv("all_mountain_ad_data.csv", index_col=)
ad_data = pd.DataFrame()

print(f"Extracting ad data from {len(ad_urls)} ads")
for i, url in tqdm(list(enumerate(ad_urls))):

    if url in ad_data.index:
        continue

    try:
        ad_intermediate_data = scraper.parse_buysell_ad(url)
    except ValueError:
        continue

    ad_data = pd.concat([ad_data, ad_intermediate_data], axis=0)

    if i % 100 == 0:
        # print(f"Logging ad data at ad {i}")
        ad_data.to_csv(f"category_{category_to_scrape}_ad_data.csv", index=False)
    time.sleep(0.1)

ad_data.to_csv(f"category_{category_to_scrape}_ad_data.csv", index=False)
# %%
