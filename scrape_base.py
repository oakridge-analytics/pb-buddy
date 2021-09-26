# %%
import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import os
import json
import logging

# Custom code
import pb_buddy.scraper as scraper
import pb_buddy.utils as ut
import pb_buddy.data_processors as dt
from pb_buddy.resources import category_dict

# %%
# Settings -----------------------------------------------------------------
categories_to_scrape = range(1, 101 + 1)
num_jobs = os.cpu_count()  # Curently only for initial link grab
delay_s = 0.0
log_level = "INFO"


# Config main settings -----------------------------------------------
logging.basicConfig(
    filename=os.path.join(
        "logs", "scrape.log"),
    filemode="w",
    level=getattr(logging, log_level.upper(), None),
    format='%(asctime)s %(message)s')

# Setup ------------------------------------------------------------
all_base_data = dt.get_dataset(category_num=-1, data_type="base")
all_sold_data = dt.get_dataset(category_num=-1, data_type="sold")

# Main loop --------------------------------------------------
# Iterate through all categories in random order, prevent noticeable patterns?
logging.info("######## Starting new scrape session #########")
num_categories_scraped = 0
for category_to_scrape in np.random.choice(
    categories_to_scrape, size=len(categories_to_scrape), replace=False
):
    # Get category, some don't have entries so skip to next.(category 10 etc.)
    category_name = [x for x,v in category_dict.items()
                     if v == category_to_scrape]
    if not category_name:
        continue

    logging.info(
        f"**************Starting Category {category_name} - Number: {category_to_scrape}. {num_categories_scraped} Categories Scraped So Far"
    )

    # Get existing open ads and stats of last scrape
    base_data = all_base_data.query("category_num == @category_to_scrape")
    if len(base_data) == 0:
        last_scrape_dt = pd.to_datetime("01-JAN-1900")
    else:
        last_scrape_dt = pd.to_datetime(base_data.datetime_scraped).max()

    # Setup for sold ad data
    sold_ad_data = all_sold_data.query("category_num == @category_to_scrape")
    intermediate_sold_ad_data = []

    # Get total number of pages of ads-----------------------------------------------
    total_page_count = scraper.get_total_pages(category_to_scrape)
    if total_page_count == 0:
        continue

    # Build List of Ad URLS----------------------------------------------------
    ad_urls = []
    logging.info(f"Scraping {total_page_count} pages of ads......")
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
    logging.info(f"Extracting ad data from {len(ad_urls)} ads")

    # Do sequentially and check datetime of last scrape
    # Only add new ads. Note Pinkbike doesn't note AM/PM so can't do by time.
    # Have to round to date and check anything >= last scrape date.
    for url in tqdm(ad_urls):
        single_ad_data = scraper.parse_buysell_ad(url, delay_s=0)
        if single_ad_data != {}:
            if pd.to_datetime(single_ad_data["Last Repost Date:"]).date() >= \
                    last_scrape_dt.date():
                # Sometimes sold ads kept in main results, ad for later
                if "sold" in single_ad_data["Still For Sale:"].lower():
                    intermediate_sold_ad_data.append(single_ad_data)
                else:
                    intermediate_ad_data.append(single_ad_data)
            else:
                break

    # Split out brand new ads, versus updates ------------------
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
            previous_versions[cols_to_check]).all(axis=1)
        updated_ads = updated_ads.loc[~unchanged_mask.values,:]

        # Log the changes in each field
        if len(updated_ads) != 0:
            changes = ut.generate_changelog(
                previous_versions.loc[previous_versions.url.isin(
                    updated_ads.url),:],
                updated_ads=updated_ads,
                cols_to_check=cols_to_check
            )
            logging.info(
                f"{len(changes)} changes across {len(updated_ads)} ads")
            dt.write_dataset(changes, data_type="changes")

        # Update ads that had changes
        dt.update_base_data(updated_ads, index_col="url",
                            cols_to_update=cols_to_check)

        # Write new ones !
        dt.write_dataset(new_ads.assign(
            category_num=category_to_scrape),
            data_type="base")

    logging.info(f"Adding {len(new_ads)} new ads to base data")

    # Get sold ad data ------------------------------------------------
    potentially_sold_urls = (
        base_data
        .loc[~ base_data.url.isin(ad_urls), "url"]
        .dropna()
    )
    # Rescrape sold ads to make sure, check for "SOLD"
    logging.info(
        f"Extracting ad data from {len(potentially_sold_urls)} potentially sold ads")

    # Do sequentially and check datetime of last scrape
    # Only add new ads. Removed ads won't return a usable page.
    urls_to_remove = []
    for url in tqdm(potentially_sold_urls):
        single_ad_data = scraper.parse_buysell_ad(url, delay_s=0)
        if single_ad_data and "sold" in single_ad_data["Still For Sale:"].lower() \
                and single_ad_data["url"] not in sold_ad_data.url:
            intermediate_sold_ad_data.append(single_ad_data)
        else:
            urls_to_remove.append(url)

    logging.info(
        f"Found {len(intermediate_sold_ad_data)} sold ads, {len(urls_to_remove)} removed ads")

    # If any sold ads found in normal listings, or missing from new scrape.
    # Write out to sold ad file. Deduplicate just in case
    if len(intermediate_sold_ad_data) > 0:
        sold_ad_data = pd.DataFrame(intermediate_sold_ad_data)
        sold_ad_data = sold_ad_data.dropna()
        sold_ad_data = dt.get_latest_by_scrape_dt(sold_ad_data)
        sold_ad_data = ut.convert_to_float(
            sold_ad_data,
            colnames=["price", "Watch Count:", "View Count:"]
        )
        dt.write_dataset(
            sold_ad_data.assign(category_num=category_to_scrape),
            data_type="sold"
        )

    # remove ads that didn't sell or sold and shouldn't be in anymore ---------------
    dt.remove_from_base_data(
        removal_df=base_data.loc[(base_data.url.isin(urls_to_remove)) |
                                 (base_data.url.isin(sold_ad_data.url)),:],
        index_col="url"
    )
    logging.info(
        f"*************Finished Category {category_name}")

    num_categories_scraped += 1


logging.info("######## Finished scrape session #########")
