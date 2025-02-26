import logging
import random
import re
import time
from enum import Enum
from typing import Callable, List, Optional

import pandas as pd
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
from tenacity import retry, stop_after_attempt, wait_exponential, wait_fixed

logger = logging.getLogger(__name__)


class PlaywrightScraper:
    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    ]

    def __init__(self, headless: bool = True):
        self.page = None
        self.browser = None
        self.context = None
        self.headless = headless
        self.initialized = True
        self.rotate_identity()
        self.start_browser()

    def rotate_identity(self):
        """Rotate browser fingerprint settings"""
        self.user_agent = random.choice(self.USER_AGENTS)
        # Use standardized viewport sizes
        self.viewport = {"width": 1920, "height": 1080}
        self.timezone = random.choice(
            ["America/Los_Angeles", "America/New_York", "America/Chicago", "America/Denver", "America/Vancouver"]
        )
        self.locale = random.choice(["en-US", "en-CA", "en-GB"])

    def start_browser(self):
        """Start or restart the browser with new identity"""
        # Clean up existing resources first
        if self.page:
            self.page.close()
            self.page = None
        if self.context:
            self.context.close()
            self.context = None
        if self.browser:
            self.browser.close()
            self.browser = None

        if hasattr(self, "_playwright"):
            self._playwright.stop()

        self._playwright = sync_playwright().start()
        self.browser = self._playwright.chromium.launch(
            headless=self.headless,
            args=[
                "--disable-gpu",
                "--disable-dev-shm-usage",
                "--disable-setuid-sandbox",
                "--no-sandbox",
                "--disable-blink-features=AutomationControlled",
            ],
        )
        self.context = self.browser.new_context(
            viewport=self.viewport,
            user_agent=self.user_agent,
            java_script_enabled=True,
            bypass_csp=True,
            color_scheme="dark" if random.random() > 0.5 else "light",
            locale=self.locale,
            timezone_id=self.timezone,
            permissions=["geolocation"],
            extra_http_headers={
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate, br",
                "DNT": "1",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "none",
                "Sec-Fetch-User": "?1",
            },
        )
        self.page = self.context.new_page()
        # Emulate human-like behavior
        self.page.set_default_timeout(60000)
        self.page.set_default_navigation_timeout(60000)

    def handle_cloudflare(self, url: str, max_attempts: int = 3) -> bool:
        """Handle Cloudflare challenge if encountered"""

        for attempt in range(max_attempts):
            try:
                # Navigate to the page
                self.page.goto(url, wait_until="domcontentloaded")

                # Check for Cloudflare challenge
                if self._check_cloudflare_challenge():
                    logging.info(f"Cloudflare challenge detected (attempt {attempt + 1}/{max_attempts})")

                    # Wait longer for challenge to complete
                    self.page.wait_for_load_state("networkidle", timeout=30000)
                    time.sleep(random.uniform(5, 10))

                    # If still blocked, rotate identity and retry
                    if self._check_cloudflare_challenge():
                        self.rotate_identity()
                        self.start_browser()
                        continue

                return True

            except Exception as e:
                logging.warning(f"Error handling Cloudflare (attempt {attempt + 1}/{max_attempts}): {str(e)}")
                if attempt < max_attempts - 1:
                    self.rotate_identity()
                    self.start_browser()
                    continue
                return False

        return False

    def _check_cloudflare_challenge(self) -> bool:
        """Check if the page is a Cloudflare challenge"""
        return (
            "challenge-platform" in self.page.content()
            and "/cdn-cgi/challenge-platform/h/b/orchestrate/" in self.page.content()
        )

    def get_page_content(self, url: str) -> str:
        """Get page content with Cloudflare handling"""
        if self.handle_cloudflare(url):
            content = self.page.content()
            if "you have been blocked" in content.lower():
                raise Exception("Blocked by Cloudflare")
            return content
        else:
            raise Exception("Failed to bypass Cloudflare protection")

    def random_delay(self, min_seconds: float = 1.0, max_seconds: float = 4.0):
        """Add a random delay between actions"""
        time.sleep(random.uniform(min_seconds, max_seconds))

    def close_browser(self):
        """Clean up all browser resources"""
        if self.page:
            self.page.close()
            self.page = None
        if self.context:
            self.context.close()
            self.context = None
        if self.browser:
            self.browser.close()
            self.browser = None
        if hasattr(self, "_playwright"):
            self._playwright.stop()
            delattr(self, "_playwright")

    def set_cookies(self, cookies: list):
        if self.context is None:
            raise Exception("Browser context is not started. Call start_browser() first.")
        self.context.add_cookies(cookies)

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
    def process_urls(self, urls: list, callable_func: Callable) -> List[Optional[dict]]:
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
                # Add retry logic for each individual URL
                result = self._process_single_url_with_retry(url, callable_func)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process URL {url} after all retries: {str(e)}")
                results.append(None)
        return results

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _process_single_url_with_retry(self, url: str, callable_func: Callable) -> Optional[dict]:
        """Process a single URL with retries

        Parameters
        ----------
        url : str
            URL to process
        callable_func : Callable
            Function to process the URL

        Returns
        -------
        Optional[dict]
            Processed result or None if failed
        """
        try:
            page_content = self.get_page_content(url)
            if "you have been blocked" in page_content.lower():
                raise Exception("Blocked by Cloudflare")
            result = callable_func(page_content)
            return result
        except Exception as e:
            logger.warning(f"Attempt failed for URL {url}: {str(e)}")
            raise  # This will trigger the retry


@retry(stop=stop_after_attempt(5), wait=wait_fixed(10))
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


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=2, max=20))
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


def request_buysell_ad(
    buysell_url: str,
    delay_s: int,
    playwright_scraper: PlaywrightScraper,
) -> str:
    """Request the HTML content of a buysell ad and return raw HTML.

    Parameters
    ----------
    buysell_url : str
        URL to the buysell ad
    delay_s : int
        Number of seconds to sleep before returning
    playwright_scraper : PlaywrightScraper
        PlaywrightScraper object to store playwright browser context for requests

    Returns
    -------
    str
        Raw HTML content of the ad page
    """
    return request_ad(buysell_url, delay_s=delay_s, playwright_scraper=playwright_scraper)


def parse_buysell_ad(
    page_content: str,
    buysell_url: str,
    region_code: int,
    ad_type: AdType = AdType.PINKBIKE,
) -> dict:
    """Takes raw HTML buysell page content and extracts all attributes listed for product.

    Parameters
    ----------
    page_content : str
        Raw HTML content of the ad page
    buysell_url : str
        URL to the buysell ad
    region_code : int
        Region code for the ad
    ad_type : AdType
        Type of the ad, default is AdType.PINKBIKE

    Returns
    -------
    dict
        Dictionary with all ad data, plus URL and datetime scraped.
    """
    soup = BeautifulSoup(page_content, features="html.parser")

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
    """Takes a Buycycle buysell URL and extracts all attributes listed for a product."""
    data_dict = {}

    # Cache commonly used selectors
    header_div = page_content.find("div", class_="px-3 py-4 bike-info-header")
    data_dict["ad_title"] = " ".join(header_div.find("div").text.split()) if header_div else None

    # Optimize price extraction
    if price_element := page_content.find("p", class_="text-2xl font-500 mb-0 content-primary"):
        price_text = price_element.text.strip()
        if price_match := re.search(r"([\D]*)(\d[\d.,]*)([\D]*)", price_text):
            pre_text, price, post_text = price_match.groups()
            data_dict["price"] = price.replace(".", "").replace(",", "")
            data_dict["currency"] = pre_text.strip() or post_text.strip()
        else:
            data_dict["price"] = data_dict["currency"] = None
    else:
        data_dict["price"] = data_dict["currency"] = None

    # Set fixed values
    data_dict["original_post_date"] = str(pd.Timestamp.today(tz="US/Mountain"))
    data_dict["location"] = page_content.find("p", class_="text-sm content-tertiary wrap-1 mb-0").text.strip()

    # Build description more efficiently
    text_parts = []

    # Extract header info - single query
    if header := page_content.select_one("div.modal-header"):
        if (brand := header.select_one("p.content-secondary")) and (model := header.select_one("h2.content-primary")):
            text_parts.extend([f"Brand: {brand.text.strip()}", f"Model: {model.text.strip()}"])

    # Extract general information - minimize queries
    if gen_info := page_content.select_one("div.general-information"):
        text_parts.append("General Information:")
        rows = gen_info.select("div.d-flex.align-items-center.flex-nowrap")
        for row in rows:
            if (label := row.select_one("span.content-tertiary")) and (value := row.select_one("span.content-primary")):
                value_text = value.text.strip()
                if value_text != "-":
                    text_parts.append(f"{label.text.strip().rstrip(':')}: {value_text}")

    # Extract seller information - single query
    if seller_info := page_content.select_one("div.seller-info .info-text"):
        if seller_text := seller_info.text.strip():
            text_parts.extend(["Seller Description:", seller_text])

    # Extract parts information - batch process
    for section in page_content.select("div.parts.parts-modal"):
        if section_title := section.select_one("h3"):
            text_parts.append(f"\n{section_title.text.strip()}:")

            rows = section.select("div.d-flex.align-items-center")
            for row in rows:
                if (label := row.select_one("span.content-tertiary")) and (
                    value := row.select_one("span.content-primary")
                ):
                    value_text = value.text.strip()
                    if value_text != "-":
                        text_parts.append(f"{label.text.strip().rstrip(':')}: {value_text}")

    data_dict["description"] = "\n\n".join(text_parts)

    # Map currency symbols more efficiently
    currency_mapping = {"\u20ac": "EUR", "\u00a3": "GBP", "$": "USD"}
    data_dict["currency"] = currency_mapping.get(data_dict["currency"], data_dict["currency"])

    return data_dict
