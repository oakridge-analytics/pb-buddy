import random
import re
import time
from enum import Enum
from typing import Callable

import pandas as pd
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
from tenacity import retry, stop_after_attempt, wait_fixed


class PlaywrightScraper:
    # List of common user agents
    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
    ]

    VIEWPORT_SIZES = [
        {"width": 1920, "height": 1080},
        {"width": 1366, "height": 768},
        {"width": 1536, "height": 864},
        {"width": 1440, "height": 900},
        {"width": 1280, "height": 720},
    ]

    def __init__(self, headless: bool = True):
        self.page = None
        self.browser = None
        self.headless = headless
        self.initialized = True
        self.user_agent = random.choice(self.USER_AGENTS)
        self.viewport = random.choice(self.VIEWPORT_SIZES)
        self.start_browser()

    def start_browser(self):
        if self.browser is None:
            self._playwright = sync_playwright().start()
            self.browser = self._playwright.chromium.launch(
                headless=self.headless,
                args=[
                    "--disable-gpu",
                    "--disable-dev-shm-usage",
                    "--disable-setuid-sandbox",
                    "--no-sandbox",
                ],
            )
            self.context = self.browser.new_context(
                viewport=self.viewport,
                user_agent=self.user_agent,
                java_script_enabled=True,
                bypass_csp=True,
                color_scheme="dark" if random.random() > 0.5 else "light",
                locale=random.choice(["en-US", "en-GB", "en-CA"]),
                timezone_id=random.choice(["America/New_York", "America/Los_Angeles", "America/Chicago"]),
            )
            self.page = self.context.new_page()

    def random_delay(self, min_seconds: float = 1.0, max_seconds: float = 4.0):
        """Add a random delay between actions"""
        time.sleep(random.uniform(min_seconds, max_seconds))

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
        """Use playwright to avoid detection for getting a page's content"""
        self.page.goto(url, wait_until="domcontentloaded", timeout=30000)
        return self.page.content()

    def process_urls(self, urls: list, callable_func: Callable):
        """Process a list of URLs with a callable function.

        Parameters
        ----------
        urls : list
            URLs to visit and get page contents for
        callable_func : Callable
            callable function to process the page content.
        """
        results = []
        for url in urls:
            try:
                page_content = self.get_page_content(url)
                results.append(callable_func(page_content))
            except Exception as e:
                print(f"Error processing URL {url}: {str(e)}")
                results.append(None)
                continue

        return results


@retry(stop=stop_after_attempt(5), wait=wait_fixed(2))
def get_category_list(playwright: PlaywrightScraper) -> dict:
    """
    Get the mapping of category name to category number
    for all categories from the given page content of the buysell listing page.
    """
    page_content = playwright.get_page_content("https://www.pinkbike.com/buysell/")
    soup = BeautifulSoup(page_content, features="html.parser")

    category_dict = {}

    if len(soup.find_all("a")) == 0:
        raise Exception("No links found on the buysell page")

    for link in soup.find_all("a"):
        if link.get("href") is None:
            continue
        category_num_match = re.match(".*category=([0-9]{1,20})", link.get("href"))

        if category_num_match is not None:
            category_text = link.text
            category_dict[category_text] = int(category_num_match.group(1))

    if not category_dict:
        raise Exception("No category URLs found matching expected pattern")

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


def get_buysell_ads(search_results: str) -> dict:
    """Grab all buysell URL's from a page of Pinkbike's buysell results

    Parameters
    ----------
    search_results : str
        HTML content of the search results page

    Returns
    -------
    dict
        Mapping of ad URL: "boosted" or "not boosted"
    """
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
    price_element = soup.find("p", class_="text-3xl font-500 mb-0 content-sale")
    if price_element:
        price_text = price_element.text.strip()
        price_match = re.search(r"([\D]*)(\d[\d.,]*)([\D]*)", price_text)
        if price_match:
            pre_text, price, post_text = price_match.groups()
            price = price.replace(".", "").replace(",", "")
            currency = pre_text.strip() if pre_text else post_text.strip()
        else:
            price = None
            currency = None
    else:
        price = None
        currency = None
    # No post date found on site, so using current date
    original_post_date = str(pd.Timestamp.today(tz="US/Mountain"))
    country = soup.find("p", class_="text-sm content-tertiary mb-0").text.strip()
    details = []

    # Extract brand and model
    brand_model = soup.find("div", class_="py-3")
    if brand_model:
        brand = brand_model.find("span", class_="text-sm primary d-block")
        model = brand_model.find("strong", class_="text-lg primary d-block font-500")
        if brand:
            details.append(f"Brand: {brand.text.strip()}")
        if model:
            details.append(f"Model: {model.text.strip()}")

    # Extract summary information
    summary_info = soup.find("p", class_="text-sm content-secondary")
    if summary_info:
        summary_text = summary_info.text.strip()
        details.append(f"Summary: {summary_text}")

    # Extract condition and details
    condition_details = soup.find("ul", class_="pdp-modal-bike-info-list")
    if condition_details:
        for li in condition_details.find_all("li"):
            strong = li.find("strong")
            span = li.find("span", class_="secondary-3")
            if strong and span:
                details.append(f"{strong.text.strip().rstrip(':').rstrip()}: {span.text.strip()}")

    # Extract components
    components = soup.find("div", class_="pdp-modal-bike-components-list")
    if components:
        for div in components.find_all("div", class_="pb-2 text-sm"):
            component = div.find("p", class_="d-flex align-items-center primary gap-1 mb-1")
            value = div.find("div", class_="content-secondary")
            if component and value:
                component_text = component.text.strip().rstrip(":").rstrip().rstrip("\n")
                value_text = value.text.strip()
                if value_text and value_text != "-":
                    details.append(f"{component_text}: {value_text}")

    # Extract additional details
    additional_details = soup.find("div", class_="pdp-modal-bike-other")
    if additional_details:
        info_div = additional_details.find("div", class_="text-sm content-secondary mb-0 info-div-content")
        if info_div:
            details.append(f"Additional Details: {info_div.text.strip()}")

    details_text = " ".join(details)
    details_text = re.sub(r"\s+", " ", details_text).strip()

    data_dict["ad_title"] = ad_title
    data_dict["price"] = price
    currency_mapping = {
        "\u20ac": "EUR",  # Euro
        "\u00a3": "GBP",  # British Pound
        "$": "USD",  # US Dollar (assuming $ is always USD)
    }
    data_dict["currency"] = currency_mapping.get(currency, currency)
    data_dict["original_post_date"] = original_post_date
    data_dict["location"] = country
    data_dict["description"] = details_text

    return data_dict
