import requests
from typing import List
from bs4 import BeautifulSoup
import re
import pandas as pd


def get_buysell_ads(url: str) -> List[str]:
    """Grab all buysell URL's from a page of Pinkbike's buysell results

    Parameters
    ----------
    url : str
        URL to a buysell results page to extract all ad links from

    Returns
    -------
    List[str]
        List of URL's to the buysell ads on that page
    """
    search_results = requests.get(url)
    if search_results.status_code > 200:
        raise requests.exceptions.RequestException("Buysell url status error")
    soup = BeautifulSoup(search_results.content, features="html.parser")
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
        page_num = re.match(".*page=([0-9]{1,20})", link.get("href"))
        # If page number is found, convert to int from only match and compare
        if page_num is not None and int(page_num.group(1)) > largest_page_num:
            largest_page_num = int(page_num.group(1))
    return largest_page_num


def parse_buysell_ad(buysell_url: str) -> dict:
    """Takes a Pinkbike buysell URL and extracts all attributes listed for product.

    Parameters
    ----------
    buysell_url : str
        URL to the buysell ad

    Returns
    -------
    dict
        dictionary with all ad data, plus URL and date scraped.
    """
    # print("processing ad: ", buysell_url)
    page_request = requests.get(buysell_url)
    if page_request.status_code > 200:
        # raise requests.exceptions.RequestException("Error requesting Ad")
        return {}
    soup = BeautifulSoup(page_request.content, features="html.parser")

    data_dict = {}
    # Get data about product
    for details in soup.find_all("div", class_="buysell-container details"):
        for tag in details.find_all("b"):
            data_dict[tag.text] = tag.next_sibling

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

    # Get user city, country
    for sp in soup.find_all("span", class_="f11"):
        if "Member" in sp.text:
            location = sp.text.split("\n")[2].strip()
            data_dict["location"] = location

    # Add scrape datetime
    data_dict["datetime_scraped"] = str(pd.Timestamp.today())
    data_dict["url"] = buysell_url

    # Clean whitespace a little:
    pattern = re.compile(r"\s+")
    data_dict = {
        k: re.sub(pattern, " ", v) if type(v) == str else v
        for k, v in data_dict.items()
    }

    return data_dict
