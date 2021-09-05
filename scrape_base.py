# %%
import requests
from typing import List
from bs4 import BeautifulSoup
import re
import pandas as pd
import time
from tqdm import tqdm

# %%


def get_buysell_ads(soup: BeautifulSoup) -> List[str]:
    """Grab all buysell URL's from a response from Pinkbike's buysell results

    Parameters
    ----------
    soup : BeautifulSoup
        A Beautiful Soup object to use for parsing

    Returns
    -------
    List[str]
        List of URL's to the buysell ads on that page
    """
    buysell_ads = []
    for link in soup.find_all("a"):
        if re.match("https://www.pinkbike.com/buysell/[0-9]{7}", link.get("href")):
            buysell_ads.append(link.get("href"))
    return list(set(buysell_ads))


def get_total_pages(soup: BeautifulSoup) -> int:
    """Get the total number of pages in the Pinkbike buysell results

    Parameters
    ----------
    soup : BeautifulSoup
        A Beautiful Soup object sorted by relevance for any category

    Returns
    -------
    int
        Total number of pages possible to click through to.
    """
    largest_page_num = 0
    for link in soup.find_all("a"):
        page_num = re.match(
            ".*region=3&page=([0-9]{1,20})&category=2", link.get("href")
        )
        # If page number is found, convert to int from only match and compare
        if page_num is not None and int(page_num.group(1)) > largest_page_num:
            largest_page_num = int(page_num.group(1))
    return largest_page_num


def parse_buysell_ad(buysell_url: str) -> pd.DataFrame:
    """Takes a Pinkbike buysell URL and extracts all attributes listed for product.

    Parameters
    ----------
    buysell_url : str
        URL to the buysell ad

    Returns
    -------
    pd.DataFrame
        Dataframe with all ad data, plus URL and date scraped.
    """
    page_request = requests.get(buysell_url)
    if page_request.status_code > 200:
        raise ValueError("Error requesting Ad")
    soup = BeautifulSoup(page_request.content)

    # Get data about product
    for details in soup.find_all("div", class_="buysell-container details"):
        data_dict = {
            tag.text: tag.next_sibling for tag in details.find_all("b")}

    # Get price
    for pricing in soup.find_all("div", class_="buysell-container buysell-price"):

        # Grab price and currency. In case of issue, store None and handle downstream
        price_search = re.search("[^\s\$].*([0-9.,]+)", pricing.text)
        if price_search is not None:
            data_dict["price"] = price_search.group(0)
        else:
            data_dict["price"] = None

        currency_search = re.search("([A-Za-z]{1,3})", pricing.text)
        if currency_search is not None:
            data_dict["currency"] = currency_search.group(0)
        else:
            data_dict["currency"] = None

    # Get description
    for description in soup.find_all("div", class_="buysell-container description"):
        data_dict["description"] = description.text

    # Get ad title
    for title in soup.find_all("h1", class_="buysell-title"):
        data_dict["ad_title"] = title.text

    # Add scrape datetime
    data_dict["datetime_scraped"] = pd.Timestamp.today()
    data_dict["url"] = buysell_url

    return pd.DataFrame(data=data_dict, index=list([buysell_url]))


# %%
# Logic:
# - Initial check of how many pages
# - Loop through each page with a delay -> append to list of add urls
# - Loop through ad urls with a delay -> append to DF, with progressive save out

page_num = 1
all_mountain_base_url = (
    f"https://www.pinkbike.com/buysell/list/?region=3&page={page_num}&category=2"
)

search_results = requests.get(all_mountain_base_url).content
soup = BeautifulSoup(search_results)
total_page_count = get_total_pages(soup)

# %%
# Build List of Ad URLS............................................
ad_urls = []
print(f"Scraping {total_page_count} pages of ads......")
for page in tqdm(range(1, total_page_count)):
    # print(f"Getting ad urls on page {page}")
    all_mountain_base_url_for_page = (
        f"https://www.pinkbike.com/buysell/list/?region=3&page={page}&category=2"
    )
    search_results = requests.get(all_mountain_base_url_for_page).content
    soup = BeautifulSoup(search_results)
    ad_urls.extend(get_buysell_ads(soup))

    # time.sleep(0.1)

# %%
# Ad data columns
# ad_data = pd.read_csv("all_mountain_ad_data.csv", index_col=)
ad_data = pd.DataFrame(
    columns=[
        "Category:",
        "Condition:",
        "Frame Size:",
        "Wheel Size:",
        "Material:",
        "Front Travel:",
        "Rear Travel:",
        "Original Post Date:",
        "Last Repost Date:",
        "Still For Sale:",
        "View Count:",
        "Watch Count:",
        "price",
        "currency",
        "description",
        "ad_title",
        "datetime_scraped",
        "year",
        "url",
    ]
)
ad_data.to_csv("all_mountain_ad_data.csv")

print(f"Extracting ad data from {len(ad_urls)} ads")
for i, url in tqdm(list(enumerate(ad_urls))):

    if url in ad_data.index:
        continue

    try:
        ad_intermediate_data = parse_buysell_ad(url)
    except ValueError:
        continue

    ad_data = pd.concat([ad_data, ad_intermediate_data], axis=0)

    if i % 100 == 0:
        # print(f"Logging ad data at ad {i}")
        ad_data.to_csv("all_mountain_ad_data.csv",
                       index=False, mode="a", header=False)
    time.sleep(0.1)

ad_data.to_csv("all_mountain_ad_data.csv", index=False, mode="a", header=False)
# %%
