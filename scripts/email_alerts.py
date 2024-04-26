# %%
import json
import os

import pandas as pd
import requests

import pb_buddy.data_processors as dt

# Custom code
import pb_buddy.emailer as et
import pb_buddy.utils as ut

# %%
# Get rules to check and email for each
alerts = json.load(open(os.path.join("alerts", "alerts.json")))
changes = dt.get_dataset(-1, data_type="changes")
api_url = "https://dbandrews--bike-buddy-api-autogluonmodelinference-predict.modal.run"
email_cols = [
    "url",
    "ad_title",
    "price",
    "price_change",
    "pred_price",
    "pred_price_diff",
    "original_post_date",
    "location",
    "last_repost_date",
    "datetime_scraped",
]
api_cols = [
    "ad_title",
    "description",
    "original_post_date",
    "location",
]

# %%
for alert in alerts["alerts"]:
    df = dt.get_dataset(alert["category_num"], data_type="base")
    last_check_dt = pd.to_datetime(alert["last_checked"], utc=True).tz_convert("US/Mountain")

    # Get price drop data

    price_changes = (
        changes.assign(update_date=lambda x: pd.to_datetime(x.update_date))
        .query("field == 'price' and new_value < old_value and update_date >= @last_check_dt.date()")
        .pipe(ut.convert_to_float, colnames=["old_value", "new_value"])
        .assign(
            price_change=lambda x: x.old_value - x.new_value,
            url=lambda x: x.url.astype(object),
        )[["url", "price_change"]]
    )

    # Check for matches in recent data.
    # Handle query errors and provide some sort of feedback in email.
    try:
        filtered_df = df.query(alert["search_string"])
        filtered_df = (
            pd.merge(filtered_df, price_changes, left_on="url", right_on="url", how="left")
            .assign(
                datetime_scraped=lambda x: pd.to_datetime(x.datetime_scraped, utc=True).dt.tz_convert("US/Mountain")
            )
            .query("datetime_scraped > @last_check_dt")
        )

        if len(filtered_df) > 0:
            filtered_df = (
                filtered_df.assign(
                    # TODO: implement batched requests. For now, do single ad at time
                    pred_price=lambda _df: [
                        requests.post(api_url, json=[row], timeout=(None, None)).json()["predictions"][0]
                        for row in _df[api_cols].to_dict(orient="records")
                    ],
                    price=lambda _df: _df.apply(lambda x: ut.convert_to_cad(x.price, x.currency), axis=1),
                    pred_price_diff=lambda _df: _df.pred_price - _df.price,
                )
                .sort_values(["price_change", "price"], ascending=[False, True])[email_cols]
                .query(alert["price_search_string"] if alert.get("price_search_string") is not None else "price > 0")
                .fillna("0")
            )
        error_message = None
    except pd.core.computation.ops.UndefinedVariableError:
        error_message = "Check all columns in alert are spelled correctly!"
        filtered_df = pd.DataFrame({"error": "Check alert search string definition"})
    except AttributeError:
        error_message = (
            "Ensure you are using Pandas query syntax" " correctly and '.str' accesor only on string columns!"
        )
        filtered_df = pd.DataFrame({"error": "Check alert search string definition"})

    timestamp = str(pd.Timestamp.now(tz="US/Mountain"))

    if len(filtered_df) != 0:
        target_email = alert["email_stem"] + alert["email_domain"]

        if error_message is not None:
            email_subject = error_message
        else:
            email_subject = f"{alert['alert_name']} alert updates at {timestamp} US/Mtn time"
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
