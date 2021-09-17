# %%
import pandas as pd
import json
import os

# Custom code
import pb_buddy.emailer as et
import pb_buddy.data_processors as dt

# %%
# Get rules to check and email for each
alerts = json.load(open(os.path.join("alerts","alerts.json")))

# %%
for alert in alerts["alerts"]:
