import base64
import logging
import os
import random
from typing import Optional

import modal

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = modal.App("browser-service")

# Create image with Playwright dependencies
playwright_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "playwright==1.47.0",
    )
    .run_commands(
        "playwright install firefox",
        "playwright install-deps firefox",
    )
)

user_agents_opts = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36"
]


@app.cls(
    image=playwright_image,
    keep_warm=1,  # Keep one instance always running
    allow_concurrent_inputs=10,  # Allow multiple requests to be processed concurrently
    container_idle_timeout=120,  # Keep container alive for 2 minutes after last request
    secrets=[modal.Secret.from_name("oxy-proxy")],
    cpu=2.0,
    memory=4000,
)
class BrowserService:
    def __init__(self):
        self.browser = None
        self.proxy_config = self.get_proxy_config()

    def get_proxy_config(self) -> Optional[dict]:
        """Get proxy configuration from environment variables."""
        proxy_server = os.environ.get("PROXY_SERVER")
        proxy_username = os.environ.get("PROXY_USERNAME")
        proxy_password = os.environ.get("PROXY_PASSWORD")

        if not all([proxy_server, proxy_username, proxy_password]):
            return None

        return {"server": proxy_server, "username": proxy_username, "password": proxy_password}

    @modal.enter()
    def start_browser(self):
        """Start browser when container starts."""
        import playwright.sync_api as pw

        launch_options = {
            "slow_mo": 300,
        }
        if self.proxy_config:
            launch_options["proxy"] = self.proxy_config
            logger.info(f"Using proxy server: {self.proxy_config['server']}")

        playwright = pw.sync_playwright().start()
        self.browser = playwright.firefox.launch(**launch_options)
        logger.info("Browser started")

    @modal.exit()
    def stop_browser(self):
        """Stop browser when container stops."""
        if self.browser:
            self.browser.close()
            logger.info("Browser stopped")

    @modal.method()
    def get_page_content(self, url: str) -> str:
        """Get page content using existing browser instance."""
        page = self.browser.new_page()
        try:
            user_agent = random.choice(user_agents_opts)
            logger.info(f"Using user agent: {user_agent}")
            page.set_extra_http_headers({"User-Agent": user_agent})
            response = page.goto(url)
            if response and response.status == 403:
                logger.warning(f"Received 403 Forbidden status for URL: {url}")
            return page.content()
        finally:
            page.close()

    @modal.method()
    def get_page_screenshot(self, url: str) -> str:
        """Get page screenshot using existing browser instance."""
        page = self.browser.new_page()
        try:
            page.set_extra_http_headers({"User-Agent": random.choice(user_agents_opts)})
            page.goto(url)
            screenshot = page.screenshot(full_page=True)
            return base64.b64encode(screenshot).decode("utf-8")
        finally:
            page.close()
