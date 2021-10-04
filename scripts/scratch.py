# %%
import pandas as pd
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv

# Custom code
import pb_buddy.scraper as scraper
import pb_buddy.utils as ut
import pb_buddy.data_processors as dt
from pb_buddy.resources import category_dict

load_dotenv("/workspaces/pb-buddy/.env")

# %%
# Clean base data -----------------------------------------------------
all_data = dt.get_dataset(-1, data_type="base")
print(all_data.shape)

# %%
all_data["scrape_rank"] = all_data.groupby("url", as_index=False)[
    "datetime_scraped"
].rank("dense", ascending=False)

# %%
data_remove = all_data.query("scrape_rank >= 2")
print(data_remove.shape)


#%%
all_data.loc[
    all_data.url.isin(data_remove.url), ["url", "datetime_scraped", "category"]
].sort_values("url")

# %%
dt.remove_from_base_data(data_remove, index_col="_id")

# %%
changes = dt.get_dataset(-1, data_type="changes")
# %%
# %%
# Clean base data -----------------------------------------------------
all_sold_data = dt.get_dataset(-1, data_type="sold")
print(all_sold_data.shape)

# %%
all_sold_data["scrape_rank"] = all_sold_data.groupby("url", as_index=False)[
    "datetime_scraped"
].rank("dense", ascending=False)

# %%
all_sold_data_remove = all_sold_data.query("scrape_rank >= 2")
print(all_sold_data_remove.shape)

# %%
index_col = "_id"
sold_db_driver = dt.get_mongodb()
sold_db = sold_db_driver.sold_data

for val in all_sold_data_remove[index_col]:
    sold_db.delete_one({index_col: val})

# %%
