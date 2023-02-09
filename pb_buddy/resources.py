import requests
from bs4 import BeautifulSoup
import re


def get_category_list():
    """
    Get the mapping of category name to category number
    for all categories from https://www.pinkbike.com/buysell/
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36"
    }

    try:
        page_request = requests.get(
            "https://www.pinkbike.com/buysell/", headers=headers, timeout=20
        )
    except TimeoutError as e:
        print(e)
        return {}
    except requests.exceptions.Timeout as e:
        print(e)
        return {}
    except requests.exceptions.ConnectionError as e:
        print(e)
        return {}

    if page_request.status_code > 200:
        print("Error requesting Categories")
        if page_request.status_code == 404:
            print("404 - Ad missing")
        return {}

    soup = BeautifulSoup(page_request.content, features="html.parser")

    category_dict = {}
    for link in soup.find_all("a"):
        category_num_match = re.match(".*category=([0-9]{1,20})", link.get("href"))

        if category_num_match is not None:
            category_text = link.text
            category_dict[category_text] = int(category_num_match.group(1))

    return category_dict
