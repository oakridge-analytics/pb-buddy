import os

import duckdb


class AlertHandler:
    BUCKET = "bike-buddy"
    PREFIX = "alerts"

    def __init__(self):
        self.s3 = boto3.client("s3")
        self.conn = duckdb.connect(":memory:")
        self.conn.execute("INSTALL aws; LOAD aws")
        self.conn.execute(f"""
            CREATE SECRET secret1 (
            TYPE S3,
            KEY_ID '{os.environ['AWS_ACCESS_KEY_ID']}',
            SECRET '{os.environ['AWS_SECRET_ACCESS_KEY']}',
            REGION 'us-west-1'
        );""")

    def load_alerts(self):
        self.conn.execute(
            f"CREATE TABLE alerts as SELECT * FROM 's3://{self.BUCKET}/{self.PREFIX}/alerts.json'"
        ).fetchdf()

    def get_alerts(self):
        return self.conn.execute("SELECT * FROM alerts").fetchdf()

    def update_alert_datetime(self, uuid):
        self.conn.execute(f"UPDATE alerts SET last_checked = CURRENT_TIMESTAMP WHERE uuid = '{uuid}'")

    def write_alerts(self):
        self.conn.execute(f"COPY alerts TO 's3://{self.BUCKET}/{self.PREFIX}/alerts.json'")
