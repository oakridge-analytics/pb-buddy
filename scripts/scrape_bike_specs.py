import time
import re

from playwright.sync_api import Playwright, sync_playwright, expect
import fire
import pandas as pd
import logging

BASE_URL = "https://99spokes.com/en-CA"
NO_BIKES_TEXT = "No bikes found :-("
START_YEAR = 2020

logging.basicConfig(level=logging.INFO)


def normalize_manufacturer_name(manufacturer: str) -> str:
    return manufacturer.lower().replace(" ", "-")


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


def scrape_specs(manufacturer: str, base_url: str = BASE_URL) -> None:
    with sync_playwright() as playwright:
        get_manufacturer_data("3T", playwright)


def get_manufacturer_data(
    manufacturer: str, playwright_context: Playwright, base_url: str = BASE_URL
) -> None:
    browser = playwright_context.chromium.launch(headless=False)
    context = browser.new_context()
    page = context.new_page()
    manufacturer_normalized = normalize_manufacturer_name(manufacturer)

    # Logic:
    # First get model families per year
    # Then get models per model family
    # Then get specs per model
    delay_s = 1
    for year in range(START_YEAR, pd.datetime.now().year + 1):
        page.goto(f"{base_url}/bikes?makerId={manufacturer_normalized}&year={year}")
        logging.info(f"Scraping {manufacturer} {year}...")
        model_family_pattern = re.compile(rf"{manufacturer}.*")
        model_family_links = get_pattern_links(page, model_family_pattern)
        specs = []
        print(model_family_links)
        for model_family_link in model_family_links:
            logging.info(f"Scraping {model_family_link}...")
            page.goto(f"{base_url}/{model_family_link}")
            time.sleep(delay_s)
            model_pattern = re.compile(rf"{year} {manufacturer}.*")
            model_links = get_pattern_links(page, model_pattern)
            # all_links = get_pattern_links(page, re.compile(r".*"))
            # print(all_links)
            for model_link in model_links:
                page.goto(f"{base_url}/{model_link}")
                # model_specs = get_specs(page)
                logging.info(f"Getting specs for model {model_link}...")
                # specs.append(model_specs)
                # logging.info(specs)


def get_pattern_links(page, pattern: re.Pattern):
    "Get links matching re.Pattern for further scraping"
    matching_links = []
    for matching_link in page.get_by_role("link", name=pattern).all():
        matching_links.append(matching_link.get_attribute("href"))
    return matching_links


if __name__ == "__main__":
    fire.Fire(scrape_specs)
