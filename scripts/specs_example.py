from playwright.sync_api import Playwright, sync_playwright, expect


def run(playwright: Playwright) -> None:
    browser = playwright.chromium.launch(headless=False)
    context = browser.new_context()
    page = context.new_page()
    page.goto("https://99spokes.com/bikes?makerId=3t")
    page.get_by_role("button", name="Year").click()
    page.get_by_role("dialog").locator("div").filter(
        has_text="ReleaseNewest Model (62)Year2023 (62)2022 (26)2021 (19)2020 (38)2019 (23)"
    ).first.click()
    page.get_by_text("2023 (62)").click()
    page.locator("div").filter(
        has_text="ClearCloseReleaseNewest Model (62)Year2023 (62)2022 (26)2021 (19)2020 (38)2019 ("
    ).nth(2).click()
    page.get_by_label("2022 (26)").click()
    page.get_by_role("button", name="Apply").click()
    page.get_by_role("link", name="3T EXPLORO TEAM $4,699").click()
    page.get_by_role(
        "link", name="2023 3T EXPLORO TEAM RIVAL AXS XPLR 1X12 $4,699"
    ).click()
    page.get_by_role("navigation").get_by_role("link", name="Bikes").click()
    page.get_by_role("button", name="Brand").click()
    page.get_by_placeholder("Filter…").fill("tre")
    page.get_by_text("Trek (5,590)").click()
    page.get_by_text("See resultsApplyClear").click()
    page.get_by_role("button", name="Apply").click()
    page.get_by_role("link", name="Trek Marlin $349—$1,329").click()
    page.get_by_role("link", name="2023 Trek Marlin 6 Gen 3 $899").click()
    page.locator("#overview").get_by_text("Marlin 6 Gen 3").click()
    page.locator(".sc-762dedaa-0 > div > div > div:nth-child(4)").first.click()
    page.get_by_text(
        "BuildFrameSize: XXS, XS, Alpha Silver Aluminum, curved top tube, internal derail"
    ).click()
    page.get_by_text(
        "GroupsetRear DerailleurShimano Deore M5120, long cageCrankSize: XXS, Prowheel C1"
    ).click()
    page.get_by_text(
        "WheelsRimsBontrager Kovee, double-wall, Tubeless Ready, 28-hole, 23mm width, Pre"
    ).click()

    # ---------------------
    context.close()
    browser.close()


with sync_playwright() as playwright:
    run(playwright)
