# %%
import smtplib
from dotenv import load_dotenv
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import pandas as pd
import os

load_dotenv()


def email_df(df: pd.DataFrame, email_to: str, email_subject: str):
    """Email a pandas Dataframe as an HTML table to a specific
    recipient using Hotmail(Outlook) SMTP server through Twilio SendGrid relay. Relies on environment variables being set to handle
    sender's email and username/password for Twilio API.

    Required env variables:
    - `TWILIO_USER` - sender's user name
    - `TWILIO_PASS` - sender's password
    - `HOTMAIL_ADDRESS` - sender's email address

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
    env_vars = ["TWILIO_USER", "TWILIO_PASS", "HOTMAIL_ADDRESS"]
    for v in env_vars:
        try:
            os.environ[v]
        except KeyError:
            raise KeyError(
                "Ensure TWILIO_USER,TWILIO_PASS,HOTMAIL_ADDRESS environ variables are set prior to running!"
            )

    msg = MIMEMultipart()
    msg["From"] = os.environ["HOTMAIL_ADDRESS"]
    msg["To"] = email_to
    msg["Subject"] = email_subject

    # Build, attach table
    # html = f"""
    # <html>
    # <head></head>
    # <body>
    #     {df.to_html()}
    # </body>
    # </html>
    # """

    html = f"""
    <table border='2' cellpadding='0' cellspacing='0' width='100%' bgcolor='#e37b46' style='background: rgb(227,123,70); background: linear-gradient(315deg, rgba(227,123,70,1) 3%, rgba(198,51,92,1) 44%, rgba(86,50,139,1) 85%);'>
        {df.to_html()}
    </table>
    """

    part1 = MIMEText(html, "html")
    msg.attach(part1)

    s = smtplib.SMTP("smtp.sendgrid.net", 587)
    # Hostname to send for this command defaults to the fully qualified domain name of the local host.
    s.ehlo()
    s.starttls()  # Puts connection to SMTP server in TLS mode
    s.ehlo()
    s.login(os.environ["TWILIO_USER"], os.environ["TWILIO_PASS"])
    s.sendmail(msg["From"], msg["To"], msg.as_string())

    s.quit()

    # %%
