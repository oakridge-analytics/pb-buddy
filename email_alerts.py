# %%
import pandas as pd
import json
import os

# Custom code
import pb_buddy.emailer as et
import pb_buddy.data_processors as dt

# %%
# Get rules to check and email for each
alerts = json.load(open(os.path.join("alerts", "alerts.json")))
changes = dt.get_dataset(-1, data_type="changes")

# %%
for alert in alerts["alerts"]:
    df = dt.get_dataset(alert["category_num"], data_type="base")
    last_check_dt = pd.to_datetime(alert["last_checked"], utc=True).tz_convert(
        "US/Mountain"
    )

    # Get price drop data

    price_changes = (
        changes.assign(update_date=lambda x: pd.to_datetime(x.update_date))
        .query(
            "field == 'price' and new_value < old_value and update_date >= @last_check_dt.date()"
        )
        .assign(
            price_change=lambda x: x.old_value - x.new_value,
            url=lambda x: x.url.astype(object),
        )[["url", "price_change"]]
    )

    # Check for matches in recent data.
    # Handle query errors and provide some sort of feedback in email.
    try:
        filtered_df = df.query(alert["search_string"])
        filtered_df = pd.merge(
            filtered_df, price_changes, left_on="url", right_on="url", how="left"
        )
        filtered_df = (
            filtered_df.assign(
                datetime_scraped=lambda x: pd.to_datetime(
                    x.datetime_scraped, utc=True
                ).dt.tz_convert("US/Mountain")
            )
            .query("datetime_scraped > @last_check_dt")[
                [
                    "url",
                    "ad_title",
                    "price",
                    "price_change",
                    "currency",
                    "location",
                    "last_repost_date",
                ]
            ]
            .sort_values("price", ascending=True)
            .fillna("0")
        )
        error_message = None
    except pd.core.computation.ops.UndefinedVariableError:
        error_message = "Check all columns in alert are spelled correctly!"
        filtered_df = pd.DataFrame({"error": "Check alert search string definition"})
    except AttributeError:
        error_message = (
            "Ensure you are using Pandas query syntax"
            " correctly and '.str' accesor only on string columns!"
        )
        filtered_df = pd.DataFrame({"error": "Check alert search string definition"})

    timestamp = str(pd.Timestamp.now(tz="US/Mountain"))

    if len(filtered_df) != 0:
        target_email = alert["email_stem"] + alert["email_domain"]

        if error_message is not None:
            email_subject = error_message
        else:
            email_subject = (
                f"{alert['alert_name']} alert updates at {timestamp} US/Mtn time"
            )
        et.email_df(
            filtered_df,
            email_to=target_email,
            email_subject=email_subject,
        )

    # Update last_checked so we only send new ads
    alert["last_checked"] = timestamp

# Save out just in case last_checked updated:
with open(os.path.join("alerts", "alerts.json"), "w") as out:
    json.dump(alerts, out, indent=4)
