import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import time


def parse_buysell_ad(buysell_url: str, delay_s: int) -> dict:
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

    try:
        page_request = requests.get(buysell_url, headers=headers, timeout=20)
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
                data_dict[tag.text] = re.sub(
                    "<[^<]+?>", "", str(tag.next_sibling.next_sibling)
                )
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
        k.replace(":", "").replace(" ", "_").lower(): re.sub(pattern, " ", v).strip()
        if type(v) == str
        else v
        for k, v in data_dict.items()
    }

    time.sleep(delay_s)

    return data_dict
