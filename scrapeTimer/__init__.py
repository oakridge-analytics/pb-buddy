import datetime
import logging

import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import os
import logging
import json

# Custom code
import pb_buddy.scraper as scraper
import pb_buddy.utils as ut
import pb_buddy.data_processors as dt
import pb_buddy.emailer as et
from pb_buddy.resources import category_dict

import azure.functions as func


def main(mytimer: func.TimerRequest) -> None:
    utc_timestamp = (
        datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc).isoformat()
    )

    if mytimer.past_due:
        logging.info("The timer is past due!")

    logging.info("Python timer trigger function ran at %s", utc_timestamp)
