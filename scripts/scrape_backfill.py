# %%
import json
import os
import time

import pandas as pd
from dotenv import load_dotenv
from joblib import Parallel, delayed
from tqdm import tqdm

# Custom code
import pb_buddy.scraper as scraper
from pb_buddy.data_processors import get_dataset, stream_parquet_to_dataframe

load_dotenv()

#%%
# Setings --------------------------------------------------------------
delay_s = 0.1
chunk_delay_s = 10
chunk_size = 50

# ----------------------------------------------------------------------

# %%
# Check if datasets already exist in cache folder, up one level from current file,
# then in "data/cache"
cache_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'cache')
os.makedirs(cache_folder, exist_ok=True)
sold_file = os.path.join(cache_folder, 'df_sold.csv')
historical_sold_file = os.path.join(cache_folder, 'df_historical_sold.csv')
active_file = os.path.join(cache_folder, 'df_active.csv')


#%%
if not os.path.exists(sold_file):
    # Get and save df_sold dataset
    print('Getting sold dataset')
    df_sold = get_dataset(category_num=-1, data_type='sold')
    df_sold.to_csv(sold_file)
else:
    # Load df_sold dataset from cache
    df_sold = pd.read_csv(sold_file)

if not os.path.exists(active_file):
    # Get and save df_active dataset
    print('Getting active dataset')
    df_active = get_dataset(category_num=-1, data_type='base')
    df_active.to_csv(active_file)
else:
    # Load df_active dataset from cache
    df_active = pd.read_csv(active_file)

if not os.path.exists(historical_sold_file):
    # Get and save df_historical_sold dataset
    print('Getting historical sold dataset')
    df_historical_sold = stream_parquet_to_dataframe(
    "historical_data.parquet.gzip", "pb-buddy-historical"
    )
    df_historical_sold.to_csv(historical_sold_file)
else:
    # Load df_historical_sold dataset from cache
    df_historical_sold = pd.read_csv(historical_sold_file)


# %%
existing_url_ints = pd.concat([df_active['url'], df_sold['url'], df_historical_sold["url"]]).unique().tolist()
existing_ad_integers = [int(url_int.split('/')[-2]) for url_int in existing_url_ints]
# From inspection, ads start around 1e6
all_potential_ad_integers = list(range(1000000, int(round(max(existing_ad_integers)*1.2))))

# %%    
print("Making chunks")
# Create chunks, add factor to max integer
# %%
# Load previous scrape files, get URL out of all JSON in cache dir after reading them
previous_files = os.listdir(cache_folder)
previous_integers = []
for file in previous_files:
    with open(os.path.join(cache_folder, file), 'r') as f:
        data = json.load(f)
        previous_integers.extend([int(ad['url'].split('/')[-2]) for ad in data])

print(f"Number of previous integers: {len(previous_integers)}")
print(previous_integers)
chunks = [all_potential_ad_integers[i:i + chunk_size] for i in range(0, int(round(len(existing_ad_integers)*1.3)), chunk_size)]
# Remove all integers that are from previous, or existing
existing_ad_integers_set = set(existing_ad_integers)
previous_integers_set = set(previous_integers)
chunks = [list(set(chunk) - existing_ad_integers_set - previous_integers_set) for chunk in chunks]
chunks = [chunk for chunk in chunks if chunk]

print(f"Number of chunks to process: {len(chunks)} with total number URLS to check of {sum([len(chunk) for chunk in chunks])}")
#%%
def process_chunk(chunk, job_number):
    start = chunk[0]
    end = chunk[-1]
    results = []
    chunk_file = f'data/backfill/{start}_{end}.json'
    if os.path.exists(chunk_file):
        return
    # Get the data
    for url_int in tqdm(chunk, desc=f'Job {job_number}', position=job_number, leave=False):
        data = scraper.parse_buysell_ad(f"https://www.pinkbike.com/buysell/{url_int}/", delay_s=delay_s)
        if data == {}:
            continue
        results.append(data)

    with open(chunk_file, 'w') as f:
        json.dump(results, f)

    time.sleep(chunk_delay_s)

# %%
#Process each chunk with multiprocessing with pool
Parallel(n_jobs=8)(delayed(process_chunk)(chunk, i) for i,chunk in enumerate(chunks))
# %%
