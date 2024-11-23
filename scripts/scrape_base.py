# %%
import logging
import os
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import fire
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

import pb_buddy.data_processors as dt
# Custom code
import pb_buddy.scraper as scraper
import pb_buddy.utils as ut


class PlaywrightThread(threading.local):
    def __init__(self) -> None:
        self.scraper = scraper.PlaywrightScraper()
        logging.debug(f"Created playwright instance in Thread {threading.current_thread().name}")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10), reraise=True)
def process_chunk(urls):
    """Process a chunk of URLs using thread-local Playwright instance with retries"""
    thread_local = PlaywrightThread()
    try:
        results = thread_local.scraper.process_urls(urls, scraper.get_buysell_ads)
        return results
    except Exception as e:
        logging.error(f"Error processing URLs: {e}")
        raise


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10), reraise=True)
def parse_with_retry(url, delay_s, region_code, playwright_scraper):
    try:
        return scraper.parse_buysell_ad(
            url, delay_s=delay_s, region_code=region_code, playwright_scraper=playwright_scraper
        )
    except Exception as e:
        logging.error(f"Error parsing ad {url}: {e}")
        raise


def main(full_refresh=False, delay_s=1, num_jobs=8, categories_to_scrape: Optional[List[int]] = [2], region=3, headless=True):
    # TODO: Fix how we handle poor formatted inputs when using
    # workflow_dispatch vs. cron scheduled runs
    num_jobs = int(num_jobs) if num_jobs else 8
    playwright_scraper = scraper.PlaywrightScraper(headless=headless)
    category_dict = scraper.get_category_list(playwright=playwright_scraper)
    # Settings -----------------------------------------------------------------
    log_level = "INFO"
    show_progress = False if os.environ.get("PROGRESS", "0") == "0" else True

    # If categories to scrape not specified, scrape all
    if categories_to_scrape is None:
        start_category = min(category_dict.values())
        end_category = max(category_dict.values())
        categories_to_scrape = range(start_category, end_category + 1)

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), None),
        format="%(asctime)s %(message)s",
    )

    logging.info("######## Starting new scrape session #########")
    all_base_data = dt.get_dataset(category_num=-1, data_type="base", region_code=int(region))
    all_sold_data = dt.get_dataset(category_num=-1, data_type="sold", region_code=int(region))

    logging.info("All previous data loaded from MongoDB")
    warnings.filterwarnings(action="ignore")

    # Main loop --------------------------------------------------
    # Iterate through all categories in random order, prevent noticeable patterns?

    num_categories_scraped = 0
    for category_to_scrape in np.random.choice(
        np.asarray(categories_to_scrape), size=len(categories_to_scrape), replace=False
    ):
        # Get category, some don't have entries so skip to next.(category 10 etc.)
        category_name = [x for x, v in category_dict.items() if v == category_to_scrape]
        if not category_name:
            continue

        logging.info(
            f"**************Starting Category {category_name} - Number: {category_to_scrape}. {num_categories_scraped} Categories Scraped So Far"
        )

        # Get existing open ads and stats of last scrape
        base_data = all_base_data.query("category_num == @category_to_scrape")

        # Setup date the compare ads to for refresh. If we want to check all current ads
        # for existence in our system, don't restrict based on date.
        if len(base_data) == 0:
            last_scrape_dt = pd.to_datetime("01-JAN-1900", utc=True)
        else:
            if full_refresh:
                last_scrape_dt = pd.to_datetime("01-JAN-1900", utc=True)
            else:
                last_scrape_dt = pd.to_datetime(base_data.datetime_scraped).max()

        # Setup for sold ad data
        sold_ad_data = all_sold_data.query("category_num == @category_to_scrape")

        # Get total number of pages of ads-----------------------------------------------
        total_page_count = scraper.get_total_pages(
            category_to_scrape, region=region, playwright_scraper=playwright_scraper
        )
        if total_page_count == 0:
            continue

        # Build List of Ad URLS----------------------------------------------------
        ad_urls = []
        logging.info(f"Scraping {total_page_count} pages of ads......")
        pages_to_check = list(range(1, total_page_count + 1))
        page_urls = [
            f"https://www.pinkbike.com/buysell/list/?region={region}&page={x}&category={category_to_scrape}"
            for x in pages_to_check
        ]
        url_chunks = np.array_split(page_urls, num_jobs)

        with ThreadPoolExecutor(max_workers=num_jobs) as executor:
            futures = [executor.submit(process_chunk, chunk) for chunk in url_chunks]
            ad_urls = []
            for future in tqdm(futures, disable=(not show_progress)):
                ad_urls.extend(future.result())

        ad_urls = {key: value for d in ad_urls for key, value in d.items() if d is not None}

        # Get new ad data ---------------------------------------------------------
        intermediate_ad_data = []
        intermediate_sold_ad_data = []
        logging.info(f"Extracting ad data from {len(ad_urls)} ads")

        # Do sequentially and check datetime of last scrape
        # Only add new ads. Note Pinkbike doesn't note AM/PM so can't do by time.
        # Have to round to date and check anything > 36 hours since last scrape date
        # Boosted ads are always at top of results, and may have older dates
        for url, boosted_status in tqdm(ad_urls.items(), disable=(not show_progress)):
            try:
                single_ad_data = parse_with_retry(
                    url, delay_s=0, region_code=region, playwright_scraper=playwright_scraper
                )
                if single_ad_data != {}:
                    if (
                        pd.to_datetime(single_ad_data["last_repost_date"], utc=False).tz_localize("MST")
                        - last_scrape_dt
                    ).total_seconds() / (60 * 60) > -48 or boosted_status == "boosted":
                        # Sometimes sold ads kept in main results, ad for later
                        if "sold" in single_ad_data["still_for_sale"].lower() and url not in sold_ad_data.url.values:
                            intermediate_sold_ad_data.append(single_ad_data)
                        else:
                            intermediate_ad_data.append(single_ad_data)
                    else:
                        logging.info(f"Checked up until ~ page: { (len(intermediate_ad_data)//20)+1} of ads")
                        break
            except Exception as e:
                logging.error(f"Error parsing ad {url}: {e}")
                continue

        # If any ads missed from previous scrape, add back in
        ads_missed = list(
            set(ad_urls).difference(set(all_base_data.url)).difference(set([ad["url"] for ad in intermediate_ad_data]))
        )
        logging.info(f"Extracting ad data from {len(ads_missed)} missed ads")
        ads_missed_data = [
            parse_with_retry(url, delay_s=0, region_code=region, playwright_scraper=playwright_scraper)
            for url in ads_missed
        ]
        ads_missed_data = [ad for ad in ads_missed_data if ad != {} and "sold" not in ad["still_for_sale"].lower()]

        intermediate_ad_data.extend(ads_missed_data)

        logging.info(f"Found: {len(intermediate_sold_ad_data)} sold ads in buysell normal pages")
        # Split out brand new ads, versus updates ------------------
        if len(intermediate_ad_data) == 0:
            continue
        else:
            recently_added_ads = pd.DataFrame(intermediate_ad_data)
            logging.info(
                f"Found {len(recently_added_ads)} recently modified ads, {len(intermediate_sold_ad_data)} sold ads"
            )
            # Check membership across categories in case of changes!
            new_ads = recently_added_ads.loc[~recently_added_ads.url.isin(all_base_data.url), :]
            logging.info(f"Found {len(new_ads)} new ads")

            # Get ads that might have an update. check price, description,title and category
            # for change. If category has changed, capture change here and update old entry.
            cols_to_check = ["price", "description", "ad_title", "category", "currency"]
            updated_ads = (
                recently_added_ads.loc[recently_added_ads.url.isin(all_base_data.url), :]
                .sort_values("url")
                .reset_index(drop=True)
            )
            previous_versions = (
                all_base_data.loc[all_base_data.url.isin(updated_ads.url)]
                .pipe(dt.get_latest_by_scrape_dt)
                .sort_values("url")
                .reset_index(drop=True)
            )
            unchanged_mask = updated_ads[cols_to_check].eq(previous_versions[cols_to_check]).all(axis=1)

            # Filter to adds that actually had changes in columns of interest
            updated_ads = updated_ads.loc[~unchanged_mask.values, :]
            previous_versions = previous_versions.loc[~unchanged_mask.values, :]

            # Log the changes in each field
            if len(updated_ads) != 0:
                changes = ut.generate_changelog(
                    previous_ads=previous_versions,
                    updated_ads=updated_ads,
                    cols_to_check=cols_to_check,
                )
                if len(changes) > 0:
                    logging.info(f"{len(changes)} changes across {len(updated_ads)} ads")
                    dt.write_dataset(
                        changes.assign(category_num=category_to_scrape),
                        data_type="changes",
                    )

            # Update ads that had changes in key columns
            dt.update_base_data(
                updated_ads,
                index_col="url",
                cols_to_update=cols_to_check + ["datetime_scraped", "last_repost_date"],
            )

            # Write new ones !
            if len(new_ads) > 0:
                dt.write_dataset(new_ads.assign(category_num=category_to_scrape), data_type="base")

        logging.info(f"Adding {len(new_ads)} new ads to base data")

        # Get sold ad data ------------------------------------------------
        potentially_sold_urls = base_data.loc[~base_data.url.isin(ad_urls.keys()), "url"].dropna()
        # Rescrape sold ads to make sure, check for "SOLD"
        logging.info(f"Extracting ad data from {len(potentially_sold_urls)} potentially sold ads")

        # Do sequentially and check datetime of last scrape
        # Only add new ads. Removed ads won't return a usable page.
        urls_to_remove = []
        for url in tqdm(potentially_sold_urls, disable=(not show_progress)):
            try:
                single_ad_data = parse_with_retry(
                    url, delay_s=0, region_code=region, playwright_scraper=playwright_scraper
                )
                if (
                    single_ad_data
                    and "sold" in single_ad_data["still_for_sale"].lower()
                    and url not in sold_ad_data.url.values
                ):
                    intermediate_sold_ad_data.append(single_ad_data)
                else:
                    urls_to_remove.append(url)
            except Exception as e:
                logging.error(f"Error parsing ad {url}: {e}")
                continue

        logging.info(f"Found {len(intermediate_sold_ad_data)} sold ads, {len(urls_to_remove)} removed ads")

        # If any sold ads found in normal listings, or missing from new scrape.
        # Write out to sold ad file. Deduplicate just in case
        if len(intermediate_sold_ad_data) > 0:
            new_sold_ad_data = (
                pd.DataFrame(intermediate_sold_ad_data)
                .dropna()
                .pipe(ut.convert_to_float, colnames=["price", "watch_count", "view_count"])
                .pipe(dt.get_latest_by_scrape_dt)
            )
            if len(new_sold_ad_data) > 0:
                dt.write_dataset(
                    new_sold_ad_data.assign(category_num=category_to_scrape),
                    data_type="sold",
                )

        # remove ads that didn't sell or sold and shouldn't be in anymore ---------------
        dt.remove_from_base_data(
            removal_df=base_data.loc[
                (base_data.url.isin(potentially_sold_urls)) | (base_data.url.isin(sold_ad_data.url)),
                :,
            ],
            index_col="url",
        )
        logging.info(f"*************Finished Category {category_name}")

        num_categories_scraped += 1
        logging.info(
            f"Done Category {category_name} - Number: {category_to_scrape}. {num_categories_scraped} Categories Scraped So Far"
        )

    logging.info("######## Finished scrape session #########")


if __name__ == "__main__":
    load_dotenv("../.env")
    fire.Fire(main)
