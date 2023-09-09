import re
import os
import uuid
from pathlib import Path

from playwright.sync_api import Playwright, sync_playwright, Page
from playwright._impl._api_types import TimeoutError
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from urllib.parse import unquote
import fire
from tqdm import tqdm

# Print all pandas columns
pd.set_option("display.max_columns", None)


def extract_specs_from_links(
    url_list: str,
    headless: bool = True,
    save_frequency: int = 100,
    specs_folder_path: str = "resources/specs",
    job_number: int = 0,
) -> None:
    
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=headless)
        context = browser.new_context()
        context.set_default_timeout(3000)
        page = context.new_page()

        df_specs = pd.DataFrame()
        for url in tqdm(url_list, position=job_number, leave=True):
            df_specs = pd.concat([df_specs, extract_specs(url, page)])
            if df_specs.shape[0] % save_frequency == 0:
                df_specs.to_csv(
                    os.path.join(specs_folder_path, f"specs_{df_specs.shape[0]}_{uuid.uuid4()}.csv")
                )

        df_specs.to_csv(f"specs_{df_specs.shape[0]}.csv")
        context.close()
        browser.close()


def process_html_table(html_table: str) -> pd.DataFrame:
    """Read a html table transpose the table
    and promote the first row to headers
    """
    df = pd.read_html(html_table)[0].transpose()
    df.columns = df.iloc[0]
    df = df.drop(df.index[0])
    df = normalize_column_names(df)
    return df


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to remove spaces and special characters"""
    df.columns = df.columns.str.replace(r"[\s-]", "_", regex=True)
    df.columns = df.columns.str.lower()
    return df


def extract_specs(url: str, page: Page):
    """Given a page with a bike specs table, extract the specs and return a dataframe"""
    page.goto(url, timeout=30000)

    try:
        # Due to an injected div for top rated bikes, need to check for this and grab
        # next div
        initial_summary_table_html = page.locator("div:nth-child(4) > div").first.inner_html()
        if "99 Spokes" in initial_summary_table_html:
            initial_summary_table_html = page.locator("div:nth-child(5) > div").first.inner_html()
        df_summary_specs = process_html_table(
            initial_summary_table_html
        )
        df_summary_specs.columns = [f"{col}_summary" for col in df_summary_specs.columns]
    except TimeoutError:
        df_summary_specs = pd.DataFrame()

    try:
        manufacturer_link = unquote(
            page.get_by_role("link", name=re.compile(r"View on.*"))
            .get_attribute("href")
            .split("target=")[1]
        ).split("?99spokes")[0]
    except TimeoutError:
        manufacturer_link = ""

    # Some manufacturers are missing all specs, including frame specs
    try:
        df_frame_specs = process_html_table(
            page.get_by_text(re.compile(r"^Build.+"))
            .filter(has=page.get_by_role("table"))
            .inner_html()
        )
    except TimeoutError:
        df_frame_specs = pd.DataFrame()

    # For framesets, no wheels or specs listed
    # TODO: refactor all this to iterate over all tables
    try:
        df_drivetrain_specs = process_html_table(
            page.get_by_text(re.compile(r"^Groupset.+"))
            .filter(has=page.get_by_role("table"))
            .inner_html()
        )
    except TimeoutError:
        df_drivetrain_specs = pd.DataFrame()

    try:
        df_wheel_specs = process_html_table(
            page.get_by_text(re.compile(r"^Wheels.+"))
            .filter(has=page.get_by_role("table"))
            .inner_html()
        )
    except TimeoutError:
        df_wheel_specs = pd.DataFrame()

    df_complete = pd.concat(
        [df_summary_specs, df_frame_specs, df_drivetrain_specs, df_wheel_specs], axis=1
    ).assign(manufacturer_url=manufacturer_link, spec_url=url)

    return df_complete


def extract_all_specs(
    csv_folder_path: str = "resources/links",
    specs_folder_path: str = "resources/specs",
    existing_specs_path: str = "resources/specs",
    n_jobs: int = 10,
    save_frequency: int = 1000,
    headless: bool = True,
):
    """Given a path to a folder of CSV files,
    extract all specs from the links in the CSV files

    Parameters
    ----------
    csv_folder_path : str
        Folder with CSV's containing bike links
        in a column named 'URL'
    """
    csv_files = Path(csv_folder_path).glob("*.csv")
    url_list = []
    for file in csv_files:
        url_list.extend(pd.read_csv(file)["URL"].tolist())

    # Remove any URL found in existing_specs_path files:
    existing_specs_files = Path(existing_specs_path).glob("*.csv")
    existing_specs_urls = []
    for file in existing_specs_files:
        existing_specs_urls.extend(pd.read_csv(file)["spec_url"].tolist())
    url_list = [url for url in url_list if url not in existing_specs_urls]

    url_chunks = np.array_split(url_list, n_jobs)
    jobs = list(range(n_jobs))
    Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(extract_specs_from_links)(
            url_list=url_chunk,
            save_frequency=save_frequency,
            specs_folder_path=specs_folder_path,
            headless=headless,
            job_number=job_number,
        )
        for url_chunk, job_number in zip(url_chunks,jobs)
    )


if __name__ == "__main__":
    fire.Fire(extract_all_specs)
