# Quick script to email status report to myself.
# Ensure report is rendered from Markdown to html before calling
from dotenv import load_dotenv
from pb_buddy.emailer import email_html_report
import pandas as pd

# For local development, load .env file
load_dotenv("../.env")
subject_str = pd.Timestamp.now(tz="US/Mountain").strftime("%Y-%m-%d %H:%m") + \
    " PB-Buddy Status Report"
email_html_report(report_path="reports/scrape_report.html", email_subject=subject_str)
