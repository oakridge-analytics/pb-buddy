# %%
import pandas as pd
from pathlib import Path


# %%
csvs = Path("resources/links").glob("*.csv")
urls = []
total_rows = 0
for csv in csvs:
    total_rows += pd.read_csv(csv).shape[0]

print(total_rows)

# %%
