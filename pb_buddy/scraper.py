import re
import time
from enum import Enum

import pandas as pd
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
from tenacity import retry, stop_after_attempt, wait_fixed


class PlaywrightScraper:
    cookies = []
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36"
    }

    def __init__(self, headless: bool = True):
        self.page = None
        self.browser = None
        self.headless = headless
        self.initialized = True

    def start_browser(self):
        if self.browser is None:
            self._playwright = sync_playwright().start()
            self.browser = self._playwright.chromium.launch(headless=self.headless)
            self.context = self.browser.new_context()
            self.context.set_extra_http_headers(self.headers)
            self.page = self.context.new_page()

    def close_browser(self):
        if self.browser:
            self.browser.close()
            self.browser = None
        if hasattr(self, "_playwright"):
            self._playwright.stop()

    def set_cookies(self, cookies: list):
        if self.context is None:
            raise Exception("Browser context is not started. Call start_browser() first.")
        self.context.add_cookies(cookies)

    def get_page_content(self, url: str):
        if self.page is None:
            self.start_browser()
        self.page.goto(url)
        return self.page.content()

    # def get_soup(self):
    #     return BeautifulSoup(self.get_page_content(), features="html.parser")


def get_category_list(playwright_scraper: PlaywrightScraper) -> dict:
    """
    Get the mapping of category name to category number
    for all categories from https://www.pinkbike.com/buysell/
    """
    page_content = playwright_scraper.get_page_content("https://www.pinkbike.com/buysell/")

    soup = BeautifulSoup(page_content, features="html.parser")

    category_dict = {}
    for link in soup.find_all("a"):
        if link.get("href") is None:
            continue
        category_num_match = re.match(".*category=([0-9]{1,20})", link.get("href"))

        if category_num_match is not None:
            category_text = link.text
            category_dict[category_text] = int(category_num_match.group(1))

    return category_dict


@retry(stop=stop_after_attempt(5), wait=wait_fixed(2))
def request_ad(url: str, playwright_scraper: PlaywrightScraper, delay_s: int = 1) -> str:
    """Request a URL and return the page HTML content using Playwright.

    Parameters
    ----------
    url : str
        URL to request
    delay_s : int
        Time in seconds to delay before returning.

    Returns
    -------
    str
        Page content as a string
    """
    page_content = playwright_scraper.get_page_content(url)

    time.sleep(delay_s)
    return page_content


def get_buysell_ads(url: str, playwright_scraper: PlaywrightScraper, delay_s: int = 1) -> dict:
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
    search_results = request_ad(url, playwright_scraper, delay_s=delay_s)
    soup = BeautifulSoup(search_results, features="html.parser")
    buysell_ad_links = []

    # Get all buysell ad links
    for link in soup.find_all("a"):
        if link.get("href") is None:
            continue
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
    return buysell_ad_status


def get_total_pages(category_num: int, playwright_scraper: PlaywrightScraper, region: int = 3) -> int:
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
    search_results = request_ad(
        f"https://www.pinkbike.com/buysell/list/?region={region}&page=1&category={category_num}",
        playwright_scraper=playwright_scraper,
        delay_s=0,
    )
    soup = BeautifulSoup(search_results, features="html.parser")

    largest_page_num = 0
    for link in soup.find_all("a"):
        if link.get("href") is None:
            continue
        page_num = re.match(".*page=([0-9]{1,20})", link.get("href"))
        # If page number is found, convert to int from only match and compare
        if page_num is not None and int(page_num.group(1)) > largest_page_num:
            largest_page_num = int(page_num.group(1))
    return largest_page_num


class AdType(Enum):
    PINKBIKE = "pinkbike"
    BUYCYCLE = "buycycle"
    OTHER = "other"


def parse_buysell_ad(
    buysell_url: str,
    delay_s: int,
    region_code: int,
    playwright_scraper: PlaywrightScraper,
    ad_type: AdType = AdType.PINKBIKE,
) -> dict:
    """Takes a buysell URL and extracts all attributes listed for product.

    Parameters
    ----------
    buysell_url : str
        URL to the buysell ad
    delay_s : int
        Number of seconds to sleep before returning
    region_code : int
        Region code for the ad
    playwright_scraper : PlaywrightScraper
        PlaywrightScraper object to store playwright browser context for requests
    ad_type : AdType
        Type of the ad, default is AdType.PINKBIKE

    Returns
    -------
    dict
        Dictionary with all ad data, plus URL and datetime scraped.
    """
    page_request = request_ad(buysell_url, delay_s=delay_s, playwright_scraper=playwright_scraper)

    soup = BeautifulSoup(page_request, features="html.parser")
    if ad_type == AdType.PINKBIKE:
        results = parse_buysell_pinkbike_ad(soup)
    elif ad_type == AdType.BUYCYCLE:
        results = parse_buysell_buycycle_ad(soup)
    else:
        # results = parse_buysell_other_ad(soup)
        raise NotImplementedError(f"Ad type {ad_type} not implemented")

    if results:
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
