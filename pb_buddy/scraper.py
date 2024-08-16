import re
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_fixed


def get_category_list() -> dict:
    """
    Get the mapping of category name to category number
    for all categories from https://www.pinkbike.com/buysell/
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36"
    }

    try:
        page_request = requests.get("https://www.pinkbike.com/buysell/", headers=headers, timeout=20)
    except TimeoutError as e:
        print(e)
        return {}
    except requests.exceptions.Timeout as e:
        print(e)
        return {}
    except requests.exceptions.ConnectionError as e:
        print(e)
        return {}

    if page_request.status_code > 200:
        print("Error requesting Categories")
        if page_request.status_code == 404:
            print("404 - Ad missing")
        return {}

    soup = BeautifulSoup(page_request.content, features="html.parser")

    category_dict = {}
    for link in soup.find_all("a"):
        category_num_match = re.match(".*category=([0-9]{1,20})", link.get("href"))

        if category_num_match is not None:
            category_text = link.text
            category_dict[category_text] = int(category_num_match.group(1))

    return category_dict


@retry(stop=stop_after_attempt(5), wait=wait_fixed(10))
def get_buysell_ads(url: str, delay_s: int = 1) -> dict:
    """Grab all buysell URL's from a page of Pinkbike's buysell results

    Parameters
    ----------
    url : str
        URL to a buysell results page to extract all ad links from
    delay_s : int
        Time in seconds to delay before returning.

    Returns
    -------
    dict
        Mapping of ad URL: "boosted" or "not boosted"
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36"
    }
    search_results = requests.get(url, headers=headers, timeout=200)
    if search_results.status_code > 200:
        print(search_results.content)
        print(search_results.status_code)
        raise requests.exceptions.RequestException("Buysell url status error")
    soup = BeautifulSoup(search_results.content, features="html.parser")
    buysell_ad_links = []

    # Get all buysell ad links
    for link in soup.find_all("a"):
        if re.match("https://www.pinkbike.com/buysell/[0-9]{6,7}", link.get("href")):
            buysell_ad_links.append(link.get("href"))

    # De-duplicate, order maintained
    buysell_ad_links = list(dict.fromkeys(buysell_ad_links))

    # Get if each ad is boosted or not
    buysell_ad_status = {}
    divs = soup.find_all("div", class_=["bsitem boosted", "bsitem"])
    for i, div in enumerate(divs):
        if "boosted" in div.get("class"):
            buysell_ad_status[buysell_ad_links[i]] = "boosted"
        else:
            buysell_ad_status[buysell_ad_links[i]] = "not boosted"

    time.sleep(delay_s)
    # Return de duplicated, order maintained
    return buysell_ad_status


def get_total_pages(category_num: str, region: int = 3) -> int:
    """Get the total number of pages in the Pinkbike buysell results
    for a given category number

    Parameters
    ----------
    category_num : int
        Category number to grab total number of paes of ads for
    region : int
        Region to get results for. Default of 3 is North America.

    Returns
    -------
    int
        Total number of pages possible to click through to.
    """
    # Get total number of pages-----------------------------------------------
    base_url = f"https://www.pinkbike.com/buysell/list/?region={region}&page=1&category={category_num}"
    search_results = requests.get(base_url, timeout=200).content
    soup = BeautifulSoup(search_results, features="html.parser")

    largest_page_num = 0
    for link in soup.find_all("a"):
        page_num = re.match(".*page=([0-9]{1,20})", link.get("href"))
        # If page number is found, convert to int from only match and compare
        if page_num is not None and int(page_num.group(1)) > largest_page_num:
            largest_page_num = int(page_num.group(1))
    return largest_page_num


@retry(stop=stop_after_attempt(5), wait=wait_fixed(10))
def parse_buysell_ad(buysell_url: str, delay_s: int, region_code: int) -> dict:
    """Takes a Pinkbike buysell URL and extracts all attributes listed for product.

    Parameters
    ----------
    buysell_url : str
        URL to the buysell ad
    delay_s : int
        Number of seconds to sleep before returning

    Returns
    -------
    dict
        dictionary with all ad data, plus URL and date scraped.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36"
    }

    page_request = requests.get(buysell_url, headers=headers, timeout=200)

    if page_request.status_code > 200:
        print("Error requesting Ad")
        if page_request.status_code == 404:
            print("404 - Ad missing")
        return {}

    soup = BeautifulSoup(page_request.content, features="html.parser")

    data_dict = {}
    # Get data about product
    for details in soup.find_all("div", class_="buysell-container details"):
        for tag in details.find_all("b"):
            # Still For Sale has a filler element we need to skip to get text.
            # Sometimes using text attribute of next_sibling fails, cast to str and strip HTML instead.
            if "Still For Sale" in tag.text:
                data_dict[tag.text] = re.sub("<[^<]+?>", "", str(tag.next_sibling.next_sibling))
            else:
                data_dict[tag.text] = re.sub("<[^<]+?>", "", str(tag.next_sibling))

    # Get price
    for pricing in soup.find_all("div", class_="buysell-container buysell-price"):
        # Grab price and currency. In case of issue, store None and handle downstream
        price_search = re.search("([\d,]+)", pricing.text)
        if price_search is not None:
            data_dict["price"] = float(price_search.group(0).replace(",", ""))
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

    # Get restrictions on shipping
    for sp in soup.find_all("div", class_="buysell-container"):
        if "Restrictions" in sp.text:
            data_dict["restrictions"] = sp.text.split(":")[-1]

    # Add scrape datetime
    data_dict["datetime_scraped"] = str(pd.Timestamp.today(tz="US/Mountain"))
    data_dict["url"] = buysell_url

    # Clean non standard whitespace, fix key names:
    pattern = re.compile(r"\s+")
    data_dict = {
        k.replace(":", "").replace(" ", "_").lower(): re.sub(pattern, " ", v).strip() if type(v) == str else v
        for k, v in data_dict.items()
    }
    data_dict["region_code"] = region_code
    time.sleep(delay_s)

    return data_dict


def retry(times, exceptions, time_delay=60):
    """
    Retries the wrapped function/method `times` times if the exceptions listed
    in `exceptions` are thrown

    Params
    ------
    times : int
        Number of times to retry with `time_delay` between attempts
    exceptions : list
        List of exceptions to capture and retry for. Otherwise, won't
        retry.
    time_delay : int
        Number of seconds to sleep in between attempts
    """

    def decorator(func):
        def newfn(*args, **kwargs):
            attempt = 0
            while attempt < times:
                try:
                    return func(*args, **kwargs)
                except exceptions:
                    print("Exception thrown when attempting to run %s, attempt " "%d of %d" % (func, attempt, times))
                    time.sleep(time_delay)
                    attempt += 1
            return func(*args, **kwargs)

        return newfn

    return decorator

    return decorator
