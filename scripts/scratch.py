# %%

# Custom code
import pb_buddy.data_processors as dt

# Clean base data -----------------------------------------------------
all_data = dt.get_dataset(-1, data_type="base")
print(all_data.shape)

# %%
all_data.astype({col: "str" for col in all_data.columns}).to_parquet(
    "base_data.parquet.gzip", engine="pyarrow", compression="gzip"
)

# %%
# Write each row of "all_data" as a JSON to "data/base_data" folder
for index, row in all_data.astype({col: "str" for col in all_data.columns}).iterrows():
    json_data = row.to_json()
    file_name = f"data/base_data/{index}.json"
    with open(file_name, "w") as file:
        file.write(json_data)


# %%


# %%
import time

import duckdb

# Read parquet file from S3 bucket bike-buddy/data/base_data.parquet.gzip
# get start time
start_time = time.time()
con = duckdb.connect(database=":memory:")
con.execute("""
    CREATE SECRET secret2 (
    TYPE S3,
    PROVIDER CREDENTIAL_CHAIN
);
""")
con.execute("CREATE TABLE base_data AS SELECT * FROM parquet_scan('s3://bike-buddy/data/base_data.parquet.gzip')")
# con.execute("CREATE TABLE base_data AS SELECT * FROM 's3://bike-buddy/data/base_data_json/*.json'")
df = con.execute("SELECT * FROM base_data").fetch_df()
print(df.head())
print(df.shape)
print(f"Time taken to read parquet file: {time.time() - start_time}")

# %%
