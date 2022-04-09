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

#%%
df_testing = all_data.drop(columns=["_id"])

#%% 
# Plot count over time
(
    all_data
    .assign(scrape_day = lambda x: pd.to_datetime(x.original_post_date).dt.date)
    .query("scrape_day > @pd.to_datetime('01-OCT-2021')")
    .groupby("scrape_day")
    .count()
    [["url"]]
    .plot()
)

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
# Remove fake changes ----------------------------------------
changes = dt.get_dataset(-1, data_type="changes")

# %%
fake_changes = changes.query("old_value.str.strip() == new_value.str.strip()")

# %%
db = dt.get_mongodb()
changes_collection = db.change_data

for val in fake_changes["_id"]:
    changes_collection.delete_one({"_id": val})

# %%
# Clean sold data -----------------------------------------------------
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
