import re
import time
from enum import Enum

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


@retry(stop=stop_after_attempt(5), wait=wait_fixed(2))
def request_ad(url: str, delay_s: int = 1) -> requests.models.Response:
    """Request a URL and return the response object

    Parameters
    ----------
    url : str
        URL to request
    delay_s : int
        Time in seconds to delay before returning.

    Returns
    -------
    requests.models.Response
        Response object from the request
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36"
    }
    page_request = requests.get(url, headers=headers, timeout=200)
    # Add error handling if ad not found return a empty dict, otherwise raise exception
    if page_request.status_code > 200:
        print("Error requesting Ad")
        if page_request.status_code == 404:
            print("404 - Ad missing")
            return {}
        else:
            raise requests.exceptions.RequestException("Ad request error")

    time.sleep(delay_s)
    return page_request


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
        Category number to grab total number of pages of ads for
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


class AdType(Enum):
    PINKBIKE = "pinkbike"
    OTHER = "other"


def parse_buysell_ad(buysell_url: str, delay_s: int, region_code: int, ad_type: AdType = AdType.PINKBIKE) -> dict:
    """Takes a buysell URL and extracts all attributes listed for product.

    Parameters
    ----------
    buysell_url : str
        URL to the buysell ad
    delay_s : int
        Number of seconds to sleep before returning
    region_code : int
        Region code for the ad
    ad_type : AdType
        Type of the ad, default is AdType.PINKBIKE

    Returns
    -------
    dict
        Dictionary with all ad data, plus URL and datetime scraped.
    """
    page_request = request_ad(buysell_url, delay_s=delay_s)

    soup = BeautifulSoup(page_request.content, features="html.parser")

    if ad_type == AdType.PINKBIKE:
        results = parse_buysell_pinkbike_ad(soup)
    else:
        results = parse_buysell_other_ad(soup)

    # Add region code
    results["region_code"] = region_code
    results["url"] = buysell_url
    results["datetime_scraped"] = str(pd.Timestamp.today(tz="US/Mountain"))
    return results


def parse_buysell_pinkbike_ad(page_content: BeautifulSoup) -> dict:
    """Takes a Pinkbike buysell URL and extracts all attributes listed for a product.

    Parameters
    ----------
    page_content : BeautifulSoup
        BeautifulSoup object representing the HTML content of the page

    Returns
    -------
    dict
        Dictionary with all ad data
    """
    soup = page_content

    data_dict = {}

    # Combine all relevant divs into a single find_all call
    details_divs = soup.select("div.buysell-container.details")
    price_divs = soup.select("div.buysell-container.buysell-price")
    description_divs = soup.select("div.buysell-container.description")
    title_divs = soup.select("h1.buysell-title")
    location_spans = soup.select("span.f11")
    restriction_divs = soup.select("div.buysell-container")

    # Get data about product
    for details in details_divs:
        for tag in details.find_all("b"):
            if "Still For Sale" in tag.text:
                data_dict[tag.text] = tag.next_sibling.next_sibling.get_text().strip()
            else:
                data_dict[tag.text] = str(tag.next_sibling).strip()

    # Precompile the regular expression for efficiency
    price_currency_re = re.compile(r"([\d,]+)\s*([A-Za-z]{1,3})")

    # Get price and currency
    for pricing in price_divs:
        match = price_currency_re.search(pricing.text)
        if match:
            price_str, currency = match.groups()
            data_dict["price"] = float(price_str.replace(",", ""))
            data_dict["currency"] = currency
        else:
            data_dict["price"] = None
            data_dict["currency"] = None

    # Get description
    for description in description_divs:
        data_dict["description"] = description.text

    # Get ad title
    for title in title_divs:
        data_dict["ad_title"] = title.text

    # Get user city, country
    for sp in location_spans:
        if "Member" in sp.text:
            location = sp.text.split("\n")[2].strip()
            data_dict["location"] = location

    # Get restrictions on shipping
    for sp in restriction_divs:
        if "Restrictions" in sp.text:
            data_dict["restrictions"] = sp.text.split(":")[-1]

    # Clean non standard whitespace, fix key names:
    data_dict = {
        k.replace(":", "").replace(" ", "_").lower(): " ".join(v.split()).strip() if isinstance(v, str) else v
        for k, v in data_dict.items()
    }

    return data_dict


def parse_buysell_buycycle_ad(page_content: BeautifulSoup) -> dict:
    """Takes a Buycycle buysell URL and extracts all attributes listed for a product.

    Parameters
    ----------
    page_content : BeautifulSoup
        BeautifulSoup object representing the HTML content of the page

    Returns
    -------
    dict
        Dictionary with all ad data
    """
    soup = page_content

    data_dict = {}

    ad_title = soup.find("h1", class_="text-3xl font-500 content-primary mb-1").text
    ad_title = " ".join(ad_title.split())
    price_text = soup.find("p", class_="text-3xl font-500 mb-0 content-sale").text
    price = price_text.split()[0].replace(".", "")
    currency = price_text.split()[1]
    # No post date found on site, so using current date
    original_post_date = str(pd.Timestamp.today(tz="US/Mountain"))
    country = soup.find("p", class_="text-sm content-tertiary mb-0").text.strip()
    details_divs = soup.find_all("div", class_="pdp-modal-content")
    for div in details_divs:
        if "Condition & details" in div.text:
            # # Remove occurences of multiple whitespace in a row
            # # with a single occurence of respective whitespace
            details = re.sub(r" +", " ", div.text)
            details = re.sub(r"\n +", r"\n", details)
            details = re.sub(r"\n+", r"\n", details)
            # Remove the newline before each colon, and then add a newline after each colon
            details = re.sub(r"\n:", ":", details)
            details = re.sub(r":", ":\n", details)
    data_dict["ad_title"] = ad_title
    data_dict["price"] = price
    data_dict["currency"] = currency
    data_dict["original_post_date"] = original_post_date
    data_dict["location"] = country
    data_dict["description"] = details

    return data_dict
