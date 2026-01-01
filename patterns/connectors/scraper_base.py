"""
Scraper Base - Universal Business Solution Framework

Robust web scraping foundation with session management, retry logic,
anti-detection measures, and proxy support.

Example:
```python
from patterns.connectors import ScraperBase, ScraperConfig

# Create scraper with configuration
scraper = ScraperBase(ScraperConfig(
    rate_limit_seconds=2.0,
    max_retries=3,
    random_delay=True,
    rotate_user_agents=True
))

# Fetch pages
html = scraper.get("https://example.com/page1")
soup = scraper.get_soup("https://example.com/page2")

# Session-based scraping
with scraper.session() as session:
    session.get("https://example.com/login")
    session.post("https://example.com/login", data={"user": "x"})
    data = session.get("https://example.com/protected")
```
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Iterator, Union, Tuple
import time
import random
import threading
import hashlib
import json
import urllib.request
import urllib.error
import urllib.parse
import http.cookiejar
from pathlib import Path
from contextlib import contextmanager


# Common user agents for rotation
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/120.0.0.0",
]


class ProxyType(Enum):
    """Proxy protocol types."""
    HTTP = "http"
    HTTPS = "https"
    SOCKS5 = "socks5"


@dataclass
class ProxyConfig:
    """Proxy configuration."""
    host: str
    port: int
    proxy_type: ProxyType = ProxyType.HTTP
    username: Optional[str] = None
    password: Optional[str] = None

    @property
    def url(self) -> str:
        """Get proxy URL."""
        auth = ""
        if self.username and self.password:
            auth = f"{self.username}:{self.password}@"
        return f"{self.proxy_type.value}://{auth}{self.host}:{self.port}"


@dataclass
class ScraperConfig:
    """Scraper configuration."""
    rate_limit_seconds: float = 1.0        # Delay between requests
    max_retries: int = 3                    # Max retry attempts
    retry_delay: float = 2.0                # Initial retry delay
    timeout_seconds: int = 30               # Request timeout
    random_delay: bool = True               # Add random delay variance
    random_delay_range: Tuple[float, float] = (0.5, 1.5)  # Delay multiplier range
    rotate_user_agents: bool = True         # Rotate user agents
    user_agents: List[str] = field(default_factory=lambda: USER_AGENTS.copy())
    default_headers: Dict[str, str] = field(default_factory=dict)
    follow_redirects: bool = True
    max_redirects: int = 10
    verify_ssl: bool = True
    proxies: List[ProxyConfig] = field(default_factory=list)
    rotate_proxies: bool = False


@dataclass
class ScrapeResult:
    """Scrape result wrapper."""
    url: str
    status_code: int
    content: str
    headers: Dict[str, str]
    elapsed_ms: float
    final_url: str = ""           # After redirects
    from_cache: bool = False
    retries_used: int = 0

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300

    def json(self) -> Any:
        """Parse content as JSON."""
        return json.loads(self.content)


class ScraperSession:
    """
    Session for stateful scraping with cookies.

    Maintains cookies and session state across requests.
    """

    def __init__(self, scraper: 'ScraperBase'):
        self.scraper = scraper
        self.cookie_jar = http.cookiejar.CookieJar()
        self.opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(self.cookie_jar)
        )
        self._closed = False

    def get(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> ScrapeResult:
        """Make GET request with session cookies."""
        if self._closed:
            raise RuntimeError("Session is closed")
        return self._request("GET", url, params=params, headers=headers)

    def post(
        self,
        url: str,
        data: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> ScrapeResult:
        """Make POST request with session cookies."""
        if self._closed:
            raise RuntimeError("Session is closed")
        return self._request("POST", url, data=data, json_data=json_data, headers=headers)

    def _request(
        self,
        method: str,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> ScrapeResult:
        """Make request using session."""
        # Apply rate limiting
        self.scraper._apply_rate_limit()

        # Build URL with params
        if params:
            query = urllib.parse.urlencode(params)
            url = f"{url}?{query}"

        # Build headers
        request_headers = self.scraper._get_headers()
        if headers:
            request_headers.update(headers)

        # Build body
        body = None
        if json_data:
            body = json.dumps(json_data).encode('utf-8')
            request_headers['Content-Type'] = 'application/json'
        elif data:
            body = urllib.parse.urlencode(data).encode('utf-8')
            request_headers['Content-Type'] = 'application/x-www-form-urlencoded'

        # Create request
        req = urllib.request.Request(url, data=body, headers=request_headers, method=method)

        # Make request with retry
        return self.scraper._request_with_retry(url, req, opener=self.opener)

    def get_cookies(self) -> Dict[str, str]:
        """Get all session cookies."""
        return {cookie.name: cookie.value for cookie in self.cookie_jar}

    def set_cookie(self, name: str, value: str, domain: str):
        """Set a session cookie."""
        cookie = http.cookiejar.Cookie(
            version=0, name=name, value=value,
            port=None, port_specified=False,
            domain=domain, domain_specified=True, domain_initial_dot=False,
            path="/", path_specified=True,
            secure=False, expires=None, discard=True,
            comment=None, comment_url=None, rest={}, rfc2109=False
        )
        self.cookie_jar.set_cookie(cookie)

    def clear_cookies(self):
        """Clear all session cookies."""
        self.cookie_jar.clear()

    def close(self):
        """Close the session."""
        self._closed = True


class ScraperBase:
    """
    Robust web scraper with anti-detection and session support.

    Features:
    - Automatic rate limiting
    - User agent rotation
    - Proxy support and rotation
    - Session management with cookies
    - Retry with exponential backoff
    - Random delay variance
    - Response caching

    Example:
    ```python
    scraper = ScraperBase(ScraperConfig(
        rate_limit_seconds=2.0,
        rotate_user_agents=True
    ))

    # Simple fetch
    result = scraper.get("https://example.com")
    print(result.content)

    # With BeautifulSoup
    soup = scraper.get_soup("https://example.com")
    links = soup.find_all("a")

    # Session-based
    with scraper.session() as s:
        s.get("https://example.com/login")
        s.post("https://example.com/login", data={"user": "x"})
    ```
    """

    def __init__(self, config: Optional[ScraperConfig] = None):
        """
        Initialize scraper.

        Args:
            config: Scraper configuration
        """
        self.config = config or ScraperConfig()
        self._last_request_time = 0.0
        self._request_lock = threading.Lock()
        self._ua_index = 0
        self._proxy_index = 0

        # Response cache
        self._cache: Dict[str, Tuple[ScrapeResult, datetime]] = {}
        self._cache_lock = threading.Lock()
        self._cache_ttl = 0  # Set to enable caching

    def get(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        cache_ttl: int = 0
    ) -> ScrapeResult:
        """
        Make GET request.

        Args:
            url: URL to fetch
            params: Query parameters
            headers: Additional headers
            cache_ttl: Cache TTL in seconds (0 = no cache)

        Returns:
            ScrapeResult with content
        """
        # Build URL with params
        if params:
            query = urllib.parse.urlencode(params)
            url = f"{url}?{query}"

        # Check cache
        if cache_ttl > 0:
            cached = self._get_cached(url)
            if cached:
                return cached

        # Apply rate limiting
        self._apply_rate_limit()

        # Build request
        request_headers = self._get_headers()
        if headers:
            request_headers.update(headers)

        req = urllib.request.Request(url, headers=request_headers, method="GET")

        # Make request
        result = self._request_with_retry(url, req)

        # Cache if successful
        if cache_ttl > 0 and result.ok:
            self._set_cached(url, result, cache_ttl)

        return result

    def post(
        self,
        url: str,
        data: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> ScrapeResult:
        """
        Make POST request.

        Args:
            url: URL to post to
            data: Form data
            json_data: JSON data
            headers: Additional headers

        Returns:
            ScrapeResult with response
        """
        # Apply rate limiting
        self._apply_rate_limit()

        # Build headers
        request_headers = self._get_headers()
        if headers:
            request_headers.update(headers)

        # Build body
        body = None
        if json_data:
            body = json.dumps(json_data).encode('utf-8')
            request_headers['Content-Type'] = 'application/json'
        elif data:
            body = urllib.parse.urlencode(data).encode('utf-8')
            request_headers['Content-Type'] = 'application/x-www-form-urlencoded'

        req = urllib.request.Request(url, data=body, headers=request_headers, method="POST")

        return self._request_with_retry(url, req)

    def get_soup(self, url: str, **kwargs) -> Any:
        """
        Fetch URL and return BeautifulSoup object.

        Requires beautifulsoup4 to be installed.

        Args:
            url: URL to fetch
            **kwargs: Additional args for get()

        Returns:
            BeautifulSoup object
        """
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError("beautifulsoup4 is required for get_soup(). Install with: pip install beautifulsoup4")

        result = self.get(url, **kwargs)
        return BeautifulSoup(result.content, 'html.parser')

    def get_json(self, url: str, **kwargs) -> Any:
        """
        Fetch URL and parse as JSON.

        Args:
            url: URL to fetch
            **kwargs: Additional args for get()

        Returns:
            Parsed JSON data
        """
        result = self.get(url, **kwargs)
        return result.json()

    @contextmanager
    def session(self):
        """
        Create a session context for stateful scraping.

        Yields:
            ScraperSession with cookie support
        """
        s = ScraperSession(self)
        try:
            yield s
        finally:
            s.close()

    def download(
        self,
        url: str,
        path: str,
        chunk_size: int = 8192,
        headers: Optional[Dict[str, str]] = None
    ) -> Path:
        """
        Download file to disk.

        Args:
            url: URL to download
            path: Local file path
            chunk_size: Download chunk size
            headers: Additional headers

        Returns:
            Path to downloaded file
        """
        self._apply_rate_limit()

        request_headers = self._get_headers()
        if headers:
            request_headers.update(headers)

        req = urllib.request.Request(url, headers=request_headers)
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with urllib.request.urlopen(req, timeout=self.config.timeout_seconds) as response:
            with open(file_path, 'wb') as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)

        return file_path

    def fetch_all(
        self,
        urls: List[str],
        delay_between: Optional[float] = None,
        on_error: str = "skip"  # "skip", "raise", "none"
    ) -> List[Optional[ScrapeResult]]:
        """
        Fetch multiple URLs sequentially.

        Args:
            urls: List of URLs to fetch
            delay_between: Override delay between requests
            on_error: Error handling ("skip", "raise", "none")

        Returns:
            List of results (None for failed if on_error="none")
        """
        results = []

        for url in urls:
            try:
                result = self.get(url)
                results.append(result)
            except Exception as e:
                if on_error == "raise":
                    raise
                elif on_error == "none":
                    results.append(None)
                # "skip" - just don't add to results

            if delay_between:
                time.sleep(delay_between)

        return results

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with user agent rotation."""
        headers = {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }

        # Add user agent
        if self.config.rotate_user_agents and self.config.user_agents:
            headers["User-Agent"] = self._get_next_user_agent()
        elif self.config.user_agents:
            headers["User-Agent"] = self.config.user_agents[0]

        # Add custom headers
        headers.update(self.config.default_headers)

        return headers

    def _get_next_user_agent(self) -> str:
        """Get next user agent in rotation."""
        ua = self.config.user_agents[self._ua_index % len(self.config.user_agents)]
        self._ua_index += 1
        return ua

    def _get_next_proxy(self) -> Optional[ProxyConfig]:
        """Get next proxy in rotation."""
        if not self.config.proxies:
            return None
        proxy = self.config.proxies[self._proxy_index % len(self.config.proxies)]
        if self.config.rotate_proxies:
            self._proxy_index += 1
        return proxy

    def _apply_rate_limit(self):
        """Apply rate limiting delay."""
        with self._request_lock:
            now = time.time()
            elapsed = now - self._last_request_time

            delay = self.config.rate_limit_seconds
            if self.config.random_delay:
                min_mult, max_mult = self.config.random_delay_range
                delay *= random.uniform(min_mult, max_mult)

            if elapsed < delay:
                time.sleep(delay - elapsed)

            self._last_request_time = time.time()

    def _request_with_retry(
        self,
        url: str,
        request: urllib.request.Request,
        opener: Optional[urllib.request.OpenerDirector] = None
    ) -> ScrapeResult:
        """Make request with retry logic."""
        last_error = None
        delay = self.config.retry_delay
        retries = 0

        for attempt in range(self.config.max_retries + 1):
            try:
                start_time = time.time()

                # Use proxy if configured
                proxy = self._get_next_proxy()
                if proxy and not opener:
                    proxy_handler = urllib.request.ProxyHandler({
                        'http': proxy.url,
                        'https': proxy.url
                    })
                    opener = urllib.request.build_opener(proxy_handler)

                # Make request
                if opener:
                    response = opener.open(request, timeout=self.config.timeout_seconds)
                else:
                    response = urllib.request.urlopen(request, timeout=self.config.timeout_seconds)

                elapsed = (time.time() - start_time) * 1000

                # Read content
                content = response.read()

                # Handle encoding
                encoding = response.headers.get_content_charset() or 'utf-8'
                try:
                    content_str = content.decode(encoding)
                except UnicodeDecodeError:
                    content_str = content.decode('utf-8', errors='replace')

                return ScrapeResult(
                    url=url,
                    status_code=response.status,
                    content=content_str,
                    headers=dict(response.headers),
                    elapsed_ms=elapsed,
                    final_url=response.url,
                    retries_used=retries
                )

            except urllib.error.HTTPError as e:
                elapsed = (time.time() - start_time) * 1000

                # Some errors shouldn't be retried
                if e.code in (400, 401, 403, 404, 410):
                    content = ""
                    try:
                        content = e.read().decode('utf-8', errors='replace')
                    except Exception:
                        pass

                    return ScrapeResult(
                        url=url,
                        status_code=e.code,
                        content=content,
                        headers=dict(e.headers) if e.headers else {},
                        elapsed_ms=elapsed,
                        retries_used=retries
                    )

                last_error = e
                retries += 1

            except Exception as e:
                last_error = e
                retries += 1

            # Retry delay with exponential backoff
            if attempt < self.config.max_retries:
                time.sleep(delay)
                delay *= 2

        # All retries failed
        raise ScraperError(f"Failed after {self.config.max_retries + 1} attempts: {last_error}", url)

    def _get_cached(self, url: str) -> Optional[ScrapeResult]:
        """Get cached result if not expired."""
        with self._cache_lock:
            if url in self._cache:
                result, cached_at = self._cache[url]
                if datetime.now() - cached_at < timedelta(seconds=self._cache_ttl):
                    result.from_cache = True
                    return result
                del self._cache[url]
        return None

    def _set_cached(self, url: str, result: ScrapeResult, ttl: int):
        """Cache result."""
        with self._cache_lock:
            self._cache_ttl = ttl
            self._cache[url] = (result, datetime.now())

    def clear_cache(self):
        """Clear response cache."""
        with self._cache_lock:
            self._cache.clear()


class ScraperError(Exception):
    """Scraper error with URL."""

    def __init__(self, message: str, url: str = ""):
        super().__init__(message)
        self.url = url


class PageScraper(ScraperBase):
    """
    Extended scraper with pagination support.

    Example:
    ```python
    scraper = PageScraper()

    # Scrape paginated content
    for page_content in scraper.iter_pages(
        base_url="https://example.com/items",
        page_param="page",
        max_pages=10
    ):
        # Process each page
        soup = BeautifulSoup(page_content, 'html.parser')
    ```
    """

    def iter_pages(
        self,
        base_url: str,
        page_param: str = "page",
        start_page: int = 1,
        max_pages: int = 100,
        stop_on_empty: bool = True,
        empty_check: Optional[Callable[[str], bool]] = None
    ) -> Iterator[str]:
        """
        Iterate over paginated content.

        Args:
            base_url: Base URL (can include other params)
            page_param: Name of page parameter
            start_page: Starting page number
            max_pages: Maximum pages to fetch
            stop_on_empty: Stop if page appears empty
            empty_check: Custom function to check if page is empty

        Yields:
            Page content for each page
        """
        # Parse base URL
        parsed = urllib.parse.urlparse(base_url)
        base_params = dict(urllib.parse.parse_qsl(parsed.query))

        for page_num in range(start_page, start_page + max_pages):
            # Build page URL
            params = {**base_params, page_param: str(page_num)}
            query = urllib.parse.urlencode(params)
            url = urllib.parse.urlunparse((
                parsed.scheme, parsed.netloc, parsed.path,
                parsed.params, query, parsed.fragment
            ))

            # Fetch page
            result = self.get(url)

            if not result.ok:
                break

            # Check if empty
            if stop_on_empty:
                if empty_check:
                    if empty_check(result.content):
                        break
                elif len(result.content.strip()) < 100:
                    break

            yield result.content

    def scrape_with_links(
        self,
        start_url: str,
        link_selector: str,
        max_depth: int = 2,
        max_pages: int = 100,
        same_domain: bool = True
    ) -> Dict[str, ScrapeResult]:
        """
        Scrape pages by following links.

        Requires beautifulsoup4.

        Args:
            start_url: Starting URL
            link_selector: CSS selector for links
            max_depth: Maximum link depth
            max_pages: Maximum pages to scrape
            same_domain: Only follow same-domain links

        Returns:
            Dict of URL -> ScrapeResult
        """
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError("beautifulsoup4 required for scrape_with_links()")

        results = {}
        visited = set()
        to_visit = [(start_url, 0)]  # (url, depth)
        start_domain = urllib.parse.urlparse(start_url).netloc

        while to_visit and len(results) < max_pages:
            url, depth = to_visit.pop(0)

            if url in visited:
                continue
            visited.add(url)

            # Fetch page
            try:
                result = self.get(url)
                if not result.ok:
                    continue
                results[url] = result
            except Exception:
                continue

            # Extract links if not at max depth
            if depth < max_depth:
                soup = BeautifulSoup(result.content, 'html.parser')
                for link in soup.select(link_selector):
                    href = link.get('href')
                    if not href:
                        continue

                    # Resolve relative URLs
                    full_url = urllib.parse.urljoin(url, href)

                    # Filter by domain
                    if same_domain:
                        link_domain = urllib.parse.urlparse(full_url).netloc
                        if link_domain != start_domain:
                            continue

                    if full_url not in visited:
                        to_visit.append((full_url, depth + 1))

        return results


# Factory functions

def create_gentle_scraper(delay: float = 2.0) -> ScraperBase:
    """
    Create a gentle scraper with conservative settings.

    Args:
        delay: Delay between requests in seconds

    Returns:
        Configured ScraperBase
    """
    return ScraperBase(ScraperConfig(
        rate_limit_seconds=delay,
        random_delay=True,
        random_delay_range=(1.0, 2.0),
        rotate_user_agents=True,
        max_retries=2
    ))


def create_aggressive_scraper(delay: float = 0.5) -> ScraperBase:
    """
    Create a faster scraper (use with caution).

    Args:
        delay: Delay between requests in seconds

    Returns:
        Configured ScraperBase
    """
    return ScraperBase(ScraperConfig(
        rate_limit_seconds=delay,
        random_delay=True,
        random_delay_range=(0.8, 1.2),
        rotate_user_agents=True,
        max_retries=3
    ))


def create_proxy_scraper(proxies: List[Dict[str, Any]]) -> ScraperBase:
    """
    Create a scraper with proxy rotation.

    Args:
        proxies: List of proxy configs [{"host": "x", "port": 8080}, ...]

    Returns:
        Configured ScraperBase with proxy support
    """
    proxy_configs = [
        ProxyConfig(
            host=p["host"],
            port=p["port"],
            proxy_type=ProxyType(p.get("type", "http")),
            username=p.get("username"),
            password=p.get("password")
        )
        for p in proxies
    ]

    return ScraperBase(ScraperConfig(
        rate_limit_seconds=1.0,
        rotate_user_agents=True,
        proxies=proxy_configs,
        rotate_proxies=True,
        max_retries=3
    ))
