import base64
import logging
import os
import random
import time
from typing import Optional

import modal
from playwright.async_api import Browser, Playwright

from .common import app

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        self.browser: Optional[Browser] = None
        self.playwright: Optional[Playwright] = None
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
    async def start_browser(self):
        """Start browser when container starts."""
        from playwright.async_api import async_playwright

        launch_options = {
            "headless": True,
            "args": [
                "--disable-gpu",
                "--disable-dev-shm-usage",
                "--disable-setuid-sandbox",
                "--no-sandbox",
                "--disable-web-security",
                "--disable-features=IsolateOrigins,site-per-process",
                "--disable-site-isolation-trials",
                "--dns-prefetch-disable",  # Disable DNS prefetching
                "--no-zygote",  # Disable zygote process
                "--no-xshm",  # Disable X Shared Memory
                "--disable-dev-tools",
                "--disable-logging",
                "--disable-permissions-api",
                "--disable-audio-output",
                "--disable-background-networking",
                "--disable-background-timer-throttling",
                "--disable-backgrounding-occluded-windows",
                "--disable-breakpad",
                "--disable-component-extensions-with-background-pages",
                "--disable-features=TranslateUI",
                "--disable-ipc-flooding-protection",
                "--disable-renderer-backgrounding",
                "--enable-features=NetworkService,NetworkServiceInProcess",
                "--force-color-profile=srgb",
                "--metrics-recording-only",
                "--mute-audio",
            ],
            "firefox_user_prefs": {
                # Disable unnecessary features
                "media.autoplay.default": 0,
                "media.autoplay.blocking_policy": 0,
                "media.autoplay.block-webaudio": False,
                "media.autoplay.blocked": False,
                # Network optimizations
                "network.http.max-connections": 1000,
                "network.http.max-persistent-connections-per-server": 10,
                "network.dns.disablePrefetch": True,
                "network.prefetch-next": False,
                # Cache settings
                "browser.cache.disk.enable": False,
                "browser.cache.memory.enable": False,
                "browser.cache.offline.enable": False,
            },
        }
        if self.proxy_config:
            launch_options["proxy"] = self.proxy_config
            logger.info(f"Using proxy server: {self.proxy_config['server']}")

        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.firefox.launch(**launch_options)
        logger.info("Browser started with optimized settings")

    @modal.exit()
    async def stop_browser(self):
        """Stop browser when container stops."""
        if self.browser:
            await self.browser.close()
            logger.info("Browser stopped")

    @modal.method()
    async def get_page_content(self, url: str) -> str:
        """Get page content using existing browser instance."""
        if not self.browser:
            raise RuntimeError("Browser not initialized")

        start_time = time.time()
        logger.info(f"Starting page content fetch for URL: {url}")

        context = await self.browser.new_context(
            viewport={"width": 800, "height": 600},
            java_script_enabled=False,
            bypass_csp=True,
            proxy=self.proxy_config if self.proxy_config else None,
            # Add aggressive timeouts
            service_workers="block",
            strict_selectors=True,
            has_touch=False,
            is_mobile=False,
            reduced_motion="reduce",
        )

        try:
            page = await context.new_page()
            await page.set_extra_http_headers(
                {
                    "User-Agent": random.choice(user_agents_opts),
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.5",
                    "Accept-Encoding": "gzip, deflate, br",
                    "Connection": "keep-alive",
                    "Upgrade-Insecure-Requests": "1",
                    "Sec-Fetch-Dest": "document",
                    "Sec-Fetch-Mode": "navigate",
                    "Sec-Fetch-Site": "none",
                    "Sec-Fetch-User": "?1",
                    "Cache-Control": "no-cache",
                    "Pragma": "no-cache",
                }
            )

            # Block resources we don't need
            await page.route(
                "**/*",
                lambda route: route.abort()
                if route.request.resource_type in ["image", "stylesheet", "font", "media", "websocket", "other"]
                else route.continue_(),
            )

            goto_start = time.time()
            response = await page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=10000,
            )
            logger.info(f"Page navigation took {time.time() - goto_start:.2f} seconds")

            if response and response.status == 403:
                logger.warning(f"Received 403 Forbidden status for URL: {url}")

            content = await page.content()
            total_time = time.time() - start_time
            logger.info(f"Total page content fetch took {total_time:.2f} seconds")
            return content
        finally:
            await context.close()

    @modal.method()
    async def get_page_screenshot(self, url: str) -> str:
        """Get page screenshot using existing browser instance."""
        start_time = time.time()
        logger.info(f"Starting page screenshot capture for URL: {url}")

        page = await self.browser.new_page()
        try:
            await page.set_extra_http_headers({"User-Agent": random.choice(user_agents_opts)})

            goto_start = time.time()
            await page.goto(url)
            logger.info(f"Page navigation took {time.time() - goto_start:.2f} seconds")

            screenshot = await page.screenshot(full_page=True)
            total_time = time.time() - start_time
            logger.info(f"Total screenshot capture took {total_time:.2f} seconds")
            return base64.b64encode(screenshot).decode("utf-8")
        finally:
            await page.close()
