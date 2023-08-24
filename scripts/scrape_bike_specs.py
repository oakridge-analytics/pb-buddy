import time
import re
import uuid
from pathlib import Path

from playwright.sync_api import Playwright, sync_playwright, expect
import fire
import pandas as pd
import logging
import unicodedata

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


def run(playwright: Playwright) -> None:
    browser = playwright.chromium.launch(headless=False)
    context = browser.new_context()
    page = context.new_page()
    page.goto("https://99spokes.com/en-CA/bikes?makerId=3t")
    page.get_by_role("button", name="Year").click()
    page.get_by_label("2023 (58)").check()
    page.get_by_role("button", name="Apply").click()
    page.get_by_role("link", name="3T Strada $4,999—$12,299").click()
    page.get_by_role(
        "link", name="2023 3T STRADA RED AXS 1X12/CLASSIFIED R50 $9,799"
    ).click()
    page.locator(
        ".sc-5cd5ae29-0 > div:nth-child(2) > div > div > div > div"
    ).first.click()
    page.get_by_role("img", name="2023 3T STRADA RED AXS 1X12/CLASSIFIED R50").click()
    page.get_by_role(
        "link", name="2023 3T Strada X Automobili Lamborghini $12,299"
    ).click()
    page.goto("https://99spokes.com/en-CA/bikes?makerId=3t&year=2023&family=3t-strada")
    page.get_by_role("link", name="2023 3T STRADA RIVAL AXS 2X12 $7,599").click()

    # ---------------------
    context.close()
    browser.close()


def scrape_manufacturer_specs(
    manufacturers_file: str,
    links_path_out: str,
    base_url: str = BASE_URL,
    delay_s: int = 1,
) -> None:
    manufacturer_list = pd.read_csv(manufacturers_file, header=None)[0].tolist()
    # Update on progress
    logging.info(f"Scraping manufacturers: {manufacturer_list}...")
    for manufacturer in manufacturer_list:
        with sync_playwright() as playwright:
            get_manufacturer_data(
                manufacturer,
                links_path_out=links_path_out,
                playwright_context=playwright,
                base_url=base_url,
                delay_s=delay_s,
            )


def get_manufacturer_data(
    manufacturer: str,
    links_path_out: str,
    playwright_context: Playwright,
    base_url: str = BASE_URL,
    delay_s: int = 1,
) -> None:
    browser = playwright_context.chromium.launch(headless=False)
    context = browser.new_context()
    page = context.new_page()
    manufacturer_normalized = normalize_manufacturer_name(manufacturer)

    # Logic:
    # First get model families per year
    # Then get models per model family
    # Then get specs per model
    year_delay_s = 2
    model_links_out = []
    for year in range(START_YEAR, pd.Timestamp.now().year + 1):
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
            if model_links:
                model_links_out.extend(model_links)
            else:
                model_links_out.append(model_family_link)

    model_links_out = [f"{base_url}{model_link}" for model_link in model_links_out]

    logging.info(f"Writing links to {links_path_out}...")
    pd.DataFrame({"URL": model_links_out}).to_csv(
        Path(links_path_out) / f"{uuid.uuid4()}_{manufacturer}.csv", index=False
    )


def get_pattern_links(page, pattern: re.Pattern):
    "Get links matching re.Pattern for further scraping"
    matching_links = []
    for matching_link in page.get_by_role("link", name=pattern).all():
        matching_links.append(matching_link.get_attribute("href"))
    return matching_links


if __name__ == "__main__":
    fire.Fire(scrape_manufacturer_specs)
