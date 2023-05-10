import os
import time
import requests
from random import random
from flask import Flask, render_template, jsonify
from multiprocessing import Pool, Value
from functools import partial
from bs4 import BeautifulSoup
import pandas as pd
import logging

log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)

app = Flask(__name__)
counter = Value("i", 0)
total = 0


def download_image(url, save_dir, delay, max_retries, backoff_factor, error_folder):
    global counter

    retries = 0
    file_stem = url.split("/")[-2] + ".jpg"
    filename = os.path.join(save_dir, file_stem)
    while retries <= max_retries:
        try:
            time.sleep(delay)
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, "html.parser")
                image = soup.find("div", {"id": "buysell-image", "class": "selectable"})
                if image:
                    image_url = image["data-fullimageurl"]
                    with open(filename, "wb") as file:
                        for chunk in requests.get(image_url).iter_content(
                            chunk_size=8192
                        ):
                            file.write(chunk)
                    print(f"Downloaded image for: {url}")
                    with counter.get_lock():
                        counter.value += 1
                        # if counter.value % 1000 < 10:
                        #     time.sleep(60)
                        #     print("60 second sleep to reset")
                    return
                else:
                    print(f"No image found to download for: {url}")
                    with open(os.path.join(error_folder, file_stem), "w") as file:
                        file.write("error")
                    return
            else:
                print(f"Failed to download: {url}")
                with open(os.path.join(error_folder, file_stem), "w") as file:
                    file.write("error")
                return

        except Exception as e:
            print(f"Error downloading: {url} - {e}")

        retries += 1
        delay *= backoff_factor
        print(f"Retrying {url} in {delay} seconds...")

    print(f"Failed to download (max retries): {url}")


def download_images(
    urls, save_dir, num_processes, delay, max_retries, backoff_factor, error_folder
):
    global counter
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Remove URLs of already downloaded images, update progress bar
    existing_files = os.listdir(save_dir)
    error_files = os.listdir(error_folder)
    urls = [
        url
        for url in urls
        if f"{url.split('/')[-2]}.jpg" not in existing_files
        and f"{url.split('/')[-2]}.jpg" not in error_files
    ]

    with Pool(num_processes) as pool:
        pool.map(
            partial(
                download_image,
                save_dir=save_dir,
                delay=delay,
                max_retries=max_retries,
                backoff_factor=backoff_factor,
                error_folder=error_folder,
            ),
            urls,
        )


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/progress")
def progress():
    global counter

    progress = round((counter.value / total) * 100, 2)
    return jsonify({"progress": progress})


def run_app():
    app.run(host="0.0.0.0", port=8000)


if __name__ == "__main__":
    from threading import Thread

    # ----------------------settings----------------------------------
    urls = pd.read_csv(
        "../data/2022-05-20_05_00_04__adjusted_bike_ads.csv",
        index_col=False,
    ).url.to_list()
    total = len(urls)

    save_dir = "../data/images/"
    error_dir = "../data/error/"
    num_processes = 4
    delay = 1
    max_retries = 12
    backoff_factor = 2

    # ------------------start processes------------------------------
    web_app_thread = Thread(target=run_app)
    web_app_thread.start()

    download_images_thread = Thread(
        target=download_images,
        args=(
            urls,
            save_dir,
            num_processes,
            delay,
            max_retries,
            backoff_factor,
            error_dir,
        ),
    )
    download_images_thread.start()
