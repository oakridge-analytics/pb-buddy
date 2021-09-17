# %%
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import pandas as pd
import os


def email_df(df: pd.DataFrame, email_to: str, email_subject: str):
    """Email a pandas Dataframe as an HTML table to a specific
    recipient. Relies on environment variables being set to handle
    sender's email and password.

    Required env variables:
    - `HOTMAIL_USER` - sender's email address
    - `HOTMAIL_PASS` - sender's password

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to convert to html and insert
    email_to : str
        Recipient's email
    email_subject : str
        Subject for email
    """
    # Get sender's user info
    env_vars = ["HOTMAIL_USER","HOTMAIL_PASS"]
    for v in env_vars:
        try:
            os.environ[v]
        except KeyError:
            raise KeyError(
                "Ensure HOTMAIL_USER,HOTMAIL_PASS environ variables are set prior to running!")

    msg = MIMEMultipart()
    msg['From'] = os.environ["HOTMAIL_USER"]
    msg['To'] = email_to
    msg['Subject'] = email_subject

    # Build, attach table
    html = f"""
    <html>
    <head></head>
    <body>
        {df.to_html()}
    </body>
    </html>
    """

    part1 = MIMEText(html, 'html')
    msg.attach(part1)

    s = smtplib.SMTP("smtp.live.com",587)
    # Hostname to send for this command defaults to the fully qualified domain name of the local host.
    s.ehlo()
    s.starttls()  # Puts connection to SMTP server in TLS mode
    s.ehlo()
    s.login(msg["From"],os.environ["HOTMAIL_PASS"])
    s.sendmail(msg["From"], msg["To"], msg.as_string())

    s.quit()

    # %%
