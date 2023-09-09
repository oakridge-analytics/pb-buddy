import time
import re
from pathlib import Path

from playwright.sync_api import Playwright, sync_playwright
import fire
import pandas as pd
import logging
import unicodedata
from tqdm import tqdm

BASE_URL = "https://99spokes.com"
LANG_REGION = "en-CA"
NO_BIKES_TEXT = "No bikes found :-("
URL_MODIFIERS = "region=earth"
START_YEAR = 2000

logging.basicConfig(level=logging.INFO)


def normalize_manufacturer_name(manufacturer: str) -> str:
    return remove_accents(manufacturer.lower().replace(" ", "").replace("-", ""))


def remove_accents(input_str: str) -> str:
    return (
        unicodedata.normalize("NFKD", input_str).encode("ASCII", "ignore").decode("utf-8")
    )


def get_manufacturer_names():
    """
    Navigate to main browsing page, access all manufacturer names
    from filter dialog box
    """
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()
        page.goto(f"{BASE_URL}/{LANG_REGION}/bikes")
        page.get_by_role("button", name="Brand").click()
        manufacturers = []
        for manufacturer in (
            page.get_by_role("dialog")
            .locator("li")
            .filter(has_text=re.compile(r"\([\d]+\)"))
            .all()
        ):
            manufacturers.append(manufacturer.inner_text().split("\n")[0])

        return manufacturers


def scrape_manufacturer_links(
    links_path_out: str,
    base_url: str = BASE_URL,
    delay_s: int = 1,
    headless: bool = True,
) -> None:
    manufacturer_list = get_manufacturer_names()
    for manufacturer in manufacturer_list:
        with sync_playwright() as playwright:
            get_manufacturer_links(
                manufacturer,
                links_path_out=links_path_out,
                playwright_context=playwright,
                base_url=base_url,
                delay_s=delay_s,
                headless=headless,
            )


def get_manufacturer_links(
    manufacturer: str,
    links_path_out: str,
    playwright_context: Playwright,
    base_url: str = BASE_URL,
    delay_s: int = 1,
    headless: bool = True,
) -> None:
    browser = playwright_context.chromium.launch(headless=headless)
    context = browser.new_context()
    page = context.new_page()
    manufacturer_normalized = normalize_manufacturer_name(manufacturer)

    # Logic:
    # First get model families per year
    # Then get models per model family
    # Then get specs per model
    year_delay_s = 2
    model_links_out = []
    run_timestamp = pd.Timestamp.now().strftime("%Y%m%d%H%M%S")
    for year in range(START_YEAR, pd.Timestamp.now().year + 2):
        page.goto(
            f"{base_url}/{LANG_REGION}/bikes?{URL_MODIFIERS}&makerId={manufacturer_normalized}&year={year}"
        )
        time.sleep(year_delay_s)

        # Check if there are any bikes for this year, if pattern present anywhere on page
        if page.get_by_text("No bikes found :-(").is_visible():
            logging.info(f"No bikes found for {manufacturer} {year}...")
            continue

        logging.info(f"Scraping Manufactuer: {manufacturer} Year: {year}...")
        model_family_pattern = re.compile(rf"{manufacturer}.*")
        model_family_links = get_pattern_links(page, model_family_pattern)
        for model_family_link in model_family_links:
            logging.info(f"Scraping Model Family: {model_family_link}...")
            page.goto(f"{base_url}/{model_family_link}")
            time.sleep(delay_s)
            model_pattern = re.compile(rf"{year} {manufacturer}.*")
            model_links = get_pattern_links(page, model_pattern)

            # In certain cases, only one model in family, so no subpage
            # TODO: This logic lets a few family links through
            if model_links:
                model_links_out.extend(model_links)
            elif "family" in model_family_link:
                raise ValueError(f"No models found for family link {model_family_link}")

    model_links_out = [f"{base_url}{model_link}" for model_link in model_links_out]

    logging.info(f"Writing links to {links_path_out}...")
    pd.DataFrame({"URL": model_links_out}).to_csv(
        Path(links_path_out) / f"{run_timestamp}_{manufacturer}.csv", index=False
    )


def get_pattern_links(page, pattern: re.Pattern):
    "Get links matching re.Pattern for further scraping"
    matching_links = []
    for matching_link in page.get_by_role("link", name=pattern).all():
        matching_links.append(matching_link.get_attribute("href"))
    return matching_links


if __name__ == "__main__":
    fire.Fire(scrape_manufacturer_links)
