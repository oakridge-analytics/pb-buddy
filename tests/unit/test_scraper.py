import pytest

from pb_buddy.scraper import (
    AdType,
    PlaywrightScraper,
    get_category_list,
    get_total_pages,
    parse_buysell_ad,
)


@pytest.fixture
def category_list():
    return {"Downhill Bikes": 1, "Enduro Bikes": 2, "Dirt Jump Bikes": 3}


@pytest.fixture
def playwright_scraper():
    playwright_scraper = PlaywrightScraper()
    yield playwright_scraper
    playwright_scraper.close_browser()


@pytest.fixture
def sample_pinkbike_url():
    return "https://www.pinkbike.com/buysell/3144637/"


@pytest.fixture
def sample_buycycle_url():
    return "https://buycycle.com/en-ca/bike/gs1-custom-wedu-edition-20588"


def test_get_category_list(category_list, playwright_scraper):
    res = get_category_list(playwright=playwright_scraper)
    # sort by values
    res = dict(sorted(res.items(), key=lambda x: x[1]))
    assert all(res[key] == value for key, value in category_list.items() if key in res)


def test_get_total_pages(playwright_scraper):
    category_num = 1
    region = 3
    total_pages = get_total_pages(category_num, region=region, playwright_scraper=playwright_scraper)
    assert isinstance(total_pages, int)
    assert total_pages >= 0


def test_parse_buysell_ad_fields(playwright_scraper):
    buysell_url = "https://www.pinkbike.com/buysell/1234567"
    page_content = playwright_scraper.get_page_content(buysell_url)
    region_code = 3
    ad_data = parse_buysell_ad(
        page_content,
        buysell_url,
        region_code,
        AdType.PINKBIKE,
    )
    assert isinstance(ad_data, dict)
    assert "url" in ad_data
    assert "datetime_scraped" in ad_data
    assert "ad_title" in ad_data
    assert "description" in ad_data
    assert "price" in ad_data
    assert "currency" in ad_data
    assert "location" in ad_data
    assert "restrictions" in ad_data
    assert "region_code" in ad_data


def test_parse_pinkbike_buysell_content(sample_pinkbike_url, playwright_scraper):
    page_content = playwright_scraper.get_page_content(sample_pinkbike_url)
    ad_data = parse_buysell_ad(
        page_content,
        sample_pinkbike_url,
        region_code=3,
        ad_type=AdType.PINKBIKE,
    )
    expected = {
        "category": "Trail Bikes",
        "condition": "Excellent - Lightly Ridden",
        "frame_size": "L",
        "wheel_size": '29"',
        "material": "Aluminium",
        "front_travel": "150 mm",
        "rear_travel": "140 mm",
        "original_post_date": "Aug-29-2021 1:06:56",
        "last_repost_date": "Sep-21-2021 11:46:21",
        "still_for_sale": "Sold",
        "view_count": "962",
        "watch_count": "3",
        "price": 2950.0,
        "currency": "USD",
        "description": "2020 mostly-custom Commencal Meta TR This bike was ridden lightly by my partner, a beginner-intermediate rider for 1 season, and most of the original components have been exchanged with upgrades from the parts bin, or New Take Off's (NTO) from a bike purchased in spring of this year. COMPONENTS: FRAME: 2020 Commencal Meta TR, size large FORKS: 2020 Rockshox 35 Gold - could use a 50h service. I can take care of this for you for an extra $100, just lmk your preference! SHOCK: 2020 Cane Creek DB Coil IL, 500-610lb progressive spring. less than 20 hours ridden since purchase. great for riders in the 200-250lb range depending on rider style. STEM: Acros 50mm DROPPER LEVER: Wolf Tooth light action DROPPER POST: 2021 Bike Yoke 185mm, very low hours BAR: Ride Alpha R27 GRIPS: Chromag - they're comfy! and red! BRAKES: Sram Level T OR Magura MT 7 Pro's for an extra $350 WHEEL (R): Light Bicycles carbon AM935 rim, DT brass nips + double butted spokes, Hope Pro 4 hub, blue. hub has been ridden 1 season, but it's just been rebuilt and greased. rim and spokes are brand new WHEEL (F): Stans Arch Mk III, NTO TIRES: Vee Tire Co, plenty of tread on both SHIFTER: sram GX Eagle, brand new DERAILLEUR Sram NX Eagle CASSETTE: sram GX Eagle CHAINRING: absolute black 28T Oval CRANKS: sram SX eagle PEDALS: I've got Crank Brothers Mallets and a couple options for flats - lmk what you're interested in, if any and we'll work it out. Let me know if you have any questions! feel free to shoot me a text (708-fiveTHREEnine-eightEIGHTzeroFOUR) or a PB message.",
        "ad_title": "2020 Commencal Meta TR - PRICE DROP!",
        "location": "Evergreen, United States",
        "restrictions": "Reasonable offers only, No Trades, Local pickup only",
        "datetime_scraped": "2024-08-16 10:34:21.591959-06:00",
        "url": "https://www.pinkbike.com/buysell/3144637/",
        "region_code": 3,
    }
    keys_to_check = [
        "category",
        "condition",
        "frame_size",
        "wheel_size",
        "material",
        "front_travel",
        "rear_travel",
        "original_post_date",
        "last_repost_date",
        "still_for_sale",
        "price",
        "currency",
        "description",
        "ad_title",
        "location",
        "restrictions",
        "url",
        "region_code",
    ]

    assert all(ad_data[key] == expected[key] for key in keys_to_check), "Mismatch found in dictionary values"


def test_parse_buycycle_ad_fields(sample_buycycle_url, playwright_scraper):
    page_content = playwright_scraper.get_page_content(sample_buycycle_url)
    ad_data = parse_buysell_ad(
        page_content,
        sample_buycycle_url,
        region_code=3,
        ad_type=AdType.BUYCYCLE,
    )
    assert isinstance(ad_data, dict)
    expected = {
        "ad_title": "Ventum GS1 Custom WED\u016a edition",
        "currency": "CAD",
        "original_post_date": "2024-12-20 11:17:00.087125-07:00",
        "location": "Clermont, United States",
        "description": "Brand: Ventum\n\nModel: GS1 Custom WEDŪ edition\n\nGeneral Information:\n\nCondition: New\n\nYear: 2022\n\nFrame sizes: 54 cm\n\nFits to height: 168 - 178cm\n\nSeller Description:\n\nSuper Rare Ventum GS1 Team WEDŪ edition. Maybe like 1 of 3. IYKYK about the founder of WEDU and here is your chance to own one!\n\n\nOriginal Parts:\n\n\nParts replaced by the seller:\n\nFork material: Carbon\n\nFront derailleur: -sram rival axs front derailleur\n\nRear derailleur: -sram rival axs rear derailleur\n\nCrank: -sram rival crank arms 172.5\n\nBrakes: -hydraulic disc\n\nWheels: -new Bontrager Aeolus 35mm carbon rims\n\nTires: -new continental gp5000’s 700x28cc\n\nSaddle: -new Bontrager saddle\n\nSeatpost: Carbon\n\nStem: -xxx Bontrager 90mm stem\n\nHandlebars: -44mm bars\n\nCassette: -sram rival 11/28\n\nChainrings: -sram rival 48x35",
    }
    keys_to_check = [
        "ad_title",
        "currency",
        "location",
        "description",
    ]
    assert all(ad_data[key] == expected[key] for key in keys_to_check), "Mismatch found in dictionary values"
