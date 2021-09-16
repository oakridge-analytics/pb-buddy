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
    df = dt.get_category_base_data(alert["category_num"])
    last_check_dt = pd.to_datetime(alert['last_checked'])
    # Check for matches in recent data
    filtered_df = df.query(alert["search_string"])
    filtered_df = (
        filtered_df
        .assign(
            datetime_scraped=lambda x: pd.to_datetime(x.datetime_scraped)
        )
        .query("datetime_scraped > @last_check_dt")
        [["url", "ad_title", "price","currency", "last_repost_date"]]
    )

    timestamp = pd.Timestamp.now(
        tz='US/Mountain').strftime('%Y-%m-%d %H:%M')

    if len(filtered_df) != 0:
        target_email = alert["email_stem"] + alert["email_domain"]
        email_subject = f"{alert['alert_name']} alert updates at {timestamp} US/Mtn time"
        et.email_df(filtered_df,
                    email_to=target_email,
                    email_subject=email_subject)

    # Update last_checked so we only send new ads
    alert["last_checked"] = timestamp

# Save out just in case last_checked updated:
with open(os.path.join("alerts","alerts.json"), "w") as out:
    json.dump(alerts, out)
