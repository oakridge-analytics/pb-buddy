import base64
import logging
import os
import random
import time
from typing import Optional
from urllib.parse import unquote

import modal
from fastapi import FastAPI, HTTPException, Response
from fastapi.security import HTTPBearer
from modal import App
from playwright.async_api import Browser, BrowserContext, Page, Playwright
from pydantic import BaseModel

app = App(name="browser-service")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create image with Playwright dependencies
playwright_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "playwright==1.47.0",
        "fastapi==0.104.1",
        "uvicorn==0.24.0",
    )
    .run_commands(
        "playwright install firefox",
        "playwright install-deps firefox",
    )
)

# Set up authentication scheme
auth_scheme = HTTPBearer()

user_agents_opts = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36"
]


class PageResponse(BaseModel):
    content: str
    status: int
    elapsed_time: float


class ScreenshotResponse(BaseModel):
    image_base64: str
    elapsed_time: float


@app.cls(
    image=playwright_image,
    allow_concurrent_inputs=10,
    container_idle_timeout=60,
    secrets=[modal.Secret.from_name("oxy-proxy"), modal.Secret.from_name("browser-service-token")],
    cpu=2.0,
    memory=2000,
)
class BrowserService:
    def __init__(self):
        self.browser: Optional[Browser] = None
        self.playwright: Optional[Playwright] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        self.proxy_config = None
        self.device = None

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

        # Initialize proxy config
        self.proxy_config = self.get_proxy_config()

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
                "--dns-prefetch-disable",
                "--no-zygote",
                "--no-xshm",
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
                "media.autoplay.default": 0,
                "media.autoplay.blocking_policy": 0,
                "media.autoplay.block-webaudio": False,
                "media.autoplay.blocked": False,
                "network.http.max-connections": 1000,
                "network.http.max-persistent-connections-per-server": 10,
                "network.dns.disablePrefetch": True,
                "network.prefetch-next": False,
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

        # Create a reusable context
        self.context = await self.browser.new_context(
            viewport={"width": 800, "height": 600},
            java_script_enabled=False,
            bypass_csp=True,
            service_workers="block",
            strict_selectors=True,
            has_touch=False,
            is_mobile=False,
            reduced_motion="reduce",
        )

        # Create a reusable page
        self.page = await self.context.new_page()
        await self.page.set_extra_http_headers(
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
        await self.page.route(
            "**/*",
            lambda route: route.abort()
            if route.request.resource_type in ["image", "stylesheet", "font", "media", "websocket", "other"]
            else route.continue_(),
        )

        # Warmup with a test page
        await self.page.goto("about:blank")
        logger.info("Browser warmup completed")

    @modal.exit()
    async def stop_browser(self):
        """Stop browser when container stops."""
        if self.page:
            await self.page.close()
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
            logger.info("Browser stopped")

    def ensure_url_protocol(self, url: str) -> str:
        """Ensure URL has a protocol prefix."""
        if not url.startswith(("http://", "https://")):
            return f"https://{url}"
        return url

    async def get_page_content(self, url: str) -> PageResponse:
        """Get page content using existing browser instance."""
        if not self.page:
            raise HTTPException(status_code=500, detail="Browser not initialized")

        url = self.ensure_url_protocol(url)
        start_time = time.time()
        logger.info(f"Starting page content fetch for URL: {url}")

        goto_start = time.time()
        response = await self.page.goto(
            url,
            wait_until="domcontentloaded",
            timeout=10000,
        )
        logger.info(f"Page navigation took {time.time() - goto_start:.2f} seconds")

        status = response.status if response else 500
        if status == 403:
            logger.warning(f"Received 403 Forbidden status for URL: {url}")

        content = await self.page.content()
        total_time = time.time() - start_time
        logger.info(f"Total page content fetch took {total_time:.2f} seconds")

        return PageResponse(content=content, status=status, elapsed_time=total_time)

    async def get_page_screenshot(self, url: str) -> ScreenshotResponse:
        """Get page screenshot using existing browser instance."""
        if not self.page:
            raise HTTPException(status_code=500, detail="Browser not initialized")

        url = self.ensure_url_protocol(url)
        start_time = time.time()
        logger.info(f"Starting page screenshot capture for URL: {url}")

        goto_start = time.time()
        await self.page.goto(url)
        logger.info(f"Page navigation took {time.time() - goto_start:.2f} seconds")

        screenshot = await self.page.screenshot(full_page=True)
        total_time = time.time() - start_time
        logger.info(f"Total screenshot capture took {total_time:.2f} seconds")

        return ScreenshotResponse(image_base64=base64.b64encode(screenshot).decode("utf-8"), elapsed_time=total_time)

    @modal.asgi_app(requires_proxy_auth=True)
    def web(self):
        web_app = FastAPI()

        @web_app.get("/status")
        async def status():
            return Response(status_code=200)

        @web_app.get("/page/{url:path}", response_model=PageResponse)
        async def page(url: str):
            decoded_url = unquote(url)  # Decode the URL-encoded string
            return await self.get_page_content(decoded_url)

        @web_app.get("/screenshot/{url:path}", response_model=ScreenshotResponse)
        async def screenshot(url: str):
            decoded_url = unquote(url)  # Decode the URL-encoded string
            return await self.get_page_screenshot(decoded_url)

        return web_app
