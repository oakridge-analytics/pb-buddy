# %%
import json
import os
import time

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from joblib import Parallel, delayed
from tqdm import tqdm

# Custom code
import pb_buddy.scraper as scraper
from pb_buddy.data_processors import get_dataset, stream_parquet_to_dataframe

load_dotenv()

# %%
# Setings --------------------------------------------------------------
delay_s = 0.1
chunk_delay_s = 10
n_jobs = 8


proxy_servers = [
    "amsterdam.nl.socks.nordhold.net",
    "atlanta.us.socks.nordhold.net",
    "dallas.us.socks.nordhold.net",
    "los-angeles.us.socks.nordhold.net",
    "nl.socks.nordhold.net",
    "se.socks.nordhold.net",
    "stockholm.se.socks.nordhold.net",
    "us.socks.nordhold.net",
    # "new-york.us.socks.nordhold.net",
    # "san-francisco.us.socks.nordhold.net",
    # "detroit.us.socks.nordhold.net",
]
# ----------------------------------------------------------------------

# %%
# Check if datasets already exist in cache folder, up one level from current file,
# then in "data/cache"
data_cache_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "cache")
os.makedirs(data_cache_folder, exist_ok=True)
sold_file = os.path.join(data_cache_folder, "df_sold.csv")
historical_sold_file = os.path.join(data_cache_folder, "df_historical_sold.csv")
active_file = os.path.join(data_cache_folder, "df_active.csv")


# %%
if not os.path.exists(sold_file):
    # Get and save df_sold dataset
    print("Getting sold dataset")
    df_sold = get_dataset(category_num=-1, data_type="sold")
    df_sold.to_csv(sold_file)
else:
    # Load df_sold dataset from cache
    df_sold = pd.read_csv(sold_file)

if not os.path.exists(active_file):
    # Get and save df_active dataset
    print("Getting active dataset")
    df_active = get_dataset(category_num=-1, data_type="base")
    df_active.to_csv(active_file)
else:
    # Load df_active dataset from cache
    df_active = pd.read_csv(active_file)

if not os.path.exists(historical_sold_file):
    # Get and save df_historical_sold dataset
    print("Getting historical sold dataset")
    df_historical_sold = stream_parquet_to_dataframe("historical_data.parquet.gzip", "pb-buddy-historical")
    df_historical_sold.to_csv(historical_sold_file)
else:
    # Load df_historical_sold dataset from cache
    df_historical_sold = pd.read_csv(historical_sold_file)


# %%
existing_url_ints = pd.concat([df_active["url"], df_sold["url"], df_historical_sold["url"]]).unique().tolist()
existing_ad_integers = [int(url_int.split("/")[-2]) for url_int in existing_url_ints]

# %%
print("Making chunks")
# %%
# __file__ = "/home/drumm/projects/pb-buddy/scripts/scrape_backfill.py"

# Create chunks, add factor to max integer
# %%
# Load previous scrape files, get URL out of all JSON in cache dir after reading them
url_cache_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "backfill")
previous_files = os.listdir(url_cache_folder)
previous_integers = []
for file in previous_files:
    with open(os.path.join(url_cache_folder, file), "r") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print(f"Error reading {file}")
            # Remove
            os.remove(os.path.join(url_cache_folder, file))
            continue
        previous_integers.extend([int(ad["url"].split("/")[-2]) for ad in data])

print(f"Number of previous integers: {len(previous_integers)}")
all_potential_ad_integers = list(range(2500000, int(round(max(existing_ad_integers) * 1.2))))
existing_ad_integers_set = set(existing_ad_integers)
previous_integers_set = set(previous_integers)
remaining_integers = list(set(all_potential_ad_integers) - existing_ad_integers_set - previous_integers_set)
remaining_integers.sort()

print(f"Number of remaining integers: {len(remaining_integers)}")

# Divide URLS over n_jobs in equal chunks
chunks = np.array_split(remaining_integers, n_jobs)

# Use a specific proxy server for each job
request_sessions = [requests.Session() for _ in range(n_jobs)]


# %%
def process_chunk(chunk, job_number, request_session, proxy_server):
    start = chunk[0]
    end = chunk[-1]
    results = []
    chunk_file = f"data/backfill/{start}_{end}.json"
    print(f"Starting job {job_number} with {len(chunk)} URLs")
    if os.path.exists(chunk_file):
        return
    # Get the data
    with request_session as s:
        for url_int in tqdm(chunk, desc=f"Job {job_number}, Proxy: {proxy_server}", position=job_number, leave=False):
            data = scraper.parse_buysell_ad(
                f"https://www.pinkbike.com/buysell/{url_int}/",
                delay_s=delay_s,
                session=s,
                proxy_server=proxy_server,
            )
            if data == {}:
                continue
            results.append(data)

            if len(results) % 100 == 0:
                min_int = min([int(ad["url"].split("/")[-2]) for ad in results])
                max_int = max(int(ad["url"].split("/")[-2]) for ad in results)
                chunk_file = f"data/backfill/{min_int}_{max_int}.json"
                with open(chunk_file, "w") as f:
                    json.dump(results, f)
                results = []

    time.sleep(chunk_delay_s)


# %%
# Process each chunk with multiprocessing with pool
Parallel(n_jobs=8)(
    delayed(process_chunk)(chunk, i, request_sessions[i], proxy_servers[i]) for i, chunk in enumerate(chunks)
)
# %%
