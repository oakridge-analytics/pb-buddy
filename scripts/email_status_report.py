# Quick script to email status report to myself.
# Ensure report is rendered from Markdown to html before calling
from pb_buddy.emailer import email_html_status_report

email_html_status_report(report_path="reports/scrape_report.html", email_subject="Pb-Buddy Status Report")