# %%
import pymongo
import certifi
import pandas as pd
from tqdm import tqdm
import os
import json
import pb_buddy.scraper as scraper
import pb_buddy.utils as ut
import pb_buddy.old_data_processors as dt
from pb_buddy.resources import category_dict

# %%
from dotenv import load_dotenv
load_dotenv()

# %%
mongo_user = os.environ['MONGO_USER']
mongo_pass = os.environ['MONGO_PASS']
# Replace the uri string with your MongoDB deployment's connection string.
conn_str = f"mongodb+srv://{mongo_user}:{mongo_pass}@cluster0.sce8f.mongodb.net/pb-buddy?retryWrites=true&w=majority"
# set a 5-second connection timeout
client = pymongo.MongoClient(
    conn_str, tlsCAFile=certifi.where(),serverSelectionTimeoutMS=10000)

# %%
db = client['pb-buddy']

# %%
# LOAD Base DATA!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
base_data_mongo = db.base_data


for i in tqdm(list(range(1,101 + 1))):
    try:
        data = dt.get_category_base_data(i)
    except:
        continue

    if len(data) == 0:
        continue

    data = (
        data
        .assign(
            category_num=lambda x: category_dict[data.category.str.strip().unique()[
                0]],
            still_for_sale=lambda x:x.still_for_sale.str.replace('<[^<]+?>', "", regex=True))
    )

    base_data_mongo.insert_many(data.to_dict(orient="records"))
    base_data_mongo.create_index([('url', pymongo.ASCENDING)],unique=True)
# %%
# LOAD Sold DATA!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
sold_data_mongo = db.sold_data


for i in tqdm(list(range(1,101 + 1))):
    try:
        data = dt.get_category_sold_data(i)
    except:
        continue

    if len(data) == 0:
        continue

    data = (
        data
        .assign(
            category_num=lambda x: category_dict[data.category.str.strip().unique()[
                0]],
            still_for_sale=lambda x:x.still_for_sale.str.replace('<[^<]+?>', "", regex=True))
    )

    sold_data_mongo.insert_many(data.to_dict(orient="records"))
    sold_data_mongo.create_index([('url', pymongo.ASCENDING)],unique=True)
# %%
# LOAD CHANGE RECORDS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
change_data_mongo = db.change_data
change_data_in = (
    pd.read_parquet("data/changed_data/changed_ad_data.parquet.gzip")
    .assign(
        category_num=lambda x: category_dict[x.category.str.strip().unique()[
            0]]
    )
)
change_data_mongo.insert_many(
    change_data_in.to_dict(orient="records"))

# %%
# CHECKS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# base_data_mongo = db.base_data
# total_output = base_data_mongo.find(
#     {'category': {"$regex":'.*'}})

# # %%
# total_output = pd.DataFrame(list(total_output))
# total_output.shape
# %%
