"""
API Client Builder - Universal Business Solution Framework

Robust API client with rate limiting, retry logic, caching, and pagination.
Handles common API patterns like OAuth, API keys, and bearer tokens.

Example:
```python
from patterns.connectors import APIClient, RateLimiter, RetryConfig

# Create client with all features
client = APIClient(
    base_url="https://api.example.com/v1",
    auth=BearerAuth(token="your-token"),
    rate_limiter=RateLimiter(requests_per_second=10),
    retry_config=RetryConfig(max_retries=3),
    cache_ttl_seconds=300
)

# Make requests
response = client.get("/users", params={"page": 1})
data = client.get_all_pages("/items", page_param="page", per_page=100)
```
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Iterator, Union
import json
import time
import threading
import hashlib
import urllib.request
import urllib.error
import urllib.parse
from pathlib import Path


class AuthMethod(Enum):
    """Authentication methods."""
    NONE = "none"
    API_KEY = "api_key"
    BEARER = "bearer"
    BASIC = "basic"
    OAUTH2 = "oauth2"
    CUSTOM = "custom"


@dataclass
class RetryConfig:
    """Retry configuration."""
    max_retries: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    retry_on_status: List[int] = field(default_factory=lambda: [429, 500, 502, 503, 504])
    retry_on_exceptions: bool = True


@dataclass
class CacheEntry:
    """Cache entry with TTL."""
    data: Any
    created_at: datetime
    ttl_seconds: int

    def is_expired(self) -> bool:
        return datetime.now() - self.created_at > timedelta(seconds=self.ttl_seconds)


class RateLimiter:
    """
    Token bucket rate limiter.

    Controls request rate to avoid API limits.
    """

    def __init__(
        self,
        requests_per_second: float = 10.0,
        burst_size: int = 10
    ):
        """
        Initialize rate limiter.

        Args:
            requests_per_second: Sustained request rate
            burst_size: Max burst requests
        """
        self.rate = requests_per_second
        self.burst_size = burst_size
        self._tokens = float(burst_size)
        self._last_update = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, timeout: float = 30.0) -> bool:
        """
        Acquire a token (block until available).

        Args:
            timeout: Max time to wait

        Returns:
            True if token acquired, False if timeout
        """
        start = time.monotonic()

        while True:
            with self._lock:
                self._refill()

                if self._tokens >= 1:
                    self._tokens -= 1
                    return True

            # Check timeout
            if time.monotonic() - start > timeout:
                return False

            # Wait for refill
            time.sleep(1.0 / self.rate)

    def _refill(self):
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self._last_update
        self._tokens = min(self.burst_size, self._tokens + elapsed * self.rate)
        self._last_update = now

    def get_wait_time(self) -> float:
        """Get estimated wait time for next token."""
        with self._lock:
            self._refill()
            if self._tokens >= 1:
                return 0.0
            return (1 - self._tokens) / self.rate


class AuthHandler(ABC):
    """Base class for authentication handlers."""

    @abstractmethod
    def apply(self, headers: Dict[str, str], params: Dict[str, str]) -> None:
        """Apply authentication to request."""
        pass


class NoAuth(AuthHandler):
    """No authentication."""

    def apply(self, headers: Dict[str, str], params: Dict[str, str]) -> None:
        pass


class APIKeyAuth(AuthHandler):
    """API key authentication."""

    def __init__(
        self,
        api_key: str,
        key_name: str = "api_key",
        location: str = "header"  # "header" or "query"
    ):
        self.api_key = api_key
        self.key_name = key_name
        self.location = location

    def apply(self, headers: Dict[str, str], params: Dict[str, str]) -> None:
        if self.location == "header":
            headers[self.key_name] = self.api_key
        else:
            params[self.key_name] = self.api_key


class BearerAuth(AuthHandler):
    """Bearer token authentication."""

    def __init__(self, token: str):
        self.token = token

    def apply(self, headers: Dict[str, str], params: Dict[str, str]) -> None:
        headers["Authorization"] = f"Bearer {self.token}"


class BasicAuth(AuthHandler):
    """Basic authentication."""

    def __init__(self, username: str, password: str):
        import base64
        credentials = f"{username}:{password}"
        self.encoded = base64.b64encode(credentials.encode()).decode()

    def apply(self, headers: Dict[str, str], params: Dict[str, str]) -> None:
        headers["Authorization"] = f"Basic {self.encoded}"


@dataclass
class APIResponse:
    """API response wrapper."""
    status_code: int
    data: Any
    headers: Dict[str, str]
    url: str
    elapsed_ms: float
    from_cache: bool = False

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300

    def json(self) -> Any:
        """Get data (already parsed)."""
        return self.data

    def raise_for_status(self):
        """Raise exception if status code indicates error."""
        if not self.ok:
            raise APIError(f"HTTP {self.status_code}", self.status_code, self.data)


class APIError(Exception):
    """API error with status code."""

    def __init__(self, message: str, status_code: int = 0, response_data: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data


class APIClient:
    """
    Robust API client with rate limiting, retry, and caching.

    Features:
    - Multiple authentication methods
    - Rate limiting (token bucket)
    - Automatic retry with exponential backoff
    - Response caching
    - Pagination support
    - Request/response hooks

    Example:
    ```python
    client = APIClient(
        base_url="https://api.example.com",
        auth=BearerAuth("token"),
        rate_limiter=RateLimiter(requests_per_second=5)
    )

    # Simple GET
    users = client.get("/users")

    # POST with JSON body
    result = client.post("/users", json={"name": "John"})

    # Paginated request
    all_items = client.get_all_pages("/items")
    ```
    """

    def __init__(
        self,
        base_url: str,
        auth: Optional[AuthHandler] = None,
        rate_limiter: Optional[RateLimiter] = None,
        retry_config: Optional[RetryConfig] = None,
        cache_ttl_seconds: int = 0,
        timeout_seconds: int = 30,
        default_headers: Optional[Dict[str, str]] = None,
        user_agent: str = "APIClient/1.0"
    ):
        """
        Initialize API client.

        Args:
            base_url: Base URL for all requests
            auth: Authentication handler
            rate_limiter: Rate limiter instance
            retry_config: Retry configuration
            cache_ttl_seconds: Cache TTL (0 = no cache)
            timeout_seconds: Request timeout
            default_headers: Default headers for all requests
            user_agent: User agent string
        """
        self.base_url = base_url.rstrip('/')
        self.auth = auth or NoAuth()
        self.rate_limiter = rate_limiter
        self.retry_config = retry_config or RetryConfig()
        self.cache_ttl = cache_ttl_seconds
        self.timeout = timeout_seconds
        self.user_agent = user_agent

        self.default_headers = {
            "User-Agent": user_agent,
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        if default_headers:
            self.default_headers.update(default_headers)

        self._cache: Dict[str, CacheEntry] = {}
        self._cache_lock = threading.Lock()

        # Hooks
        self._request_hooks: List[Callable] = []
        self._response_hooks: List[Callable] = []

    def add_request_hook(self, hook: Callable[[str, str, Dict], None]) -> 'APIClient':
        """Add request hook (called before each request)."""
        self._request_hooks.append(hook)
        return self

    def add_response_hook(self, hook: Callable[[APIResponse], None]) -> 'APIClient':
        """Add response hook (called after each response)."""
        self._response_hooks.append(hook)
        return self

    def get(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        cache: bool = True
    ) -> APIResponse:
        """Make GET request."""
        return self._request("GET", path, params=params, headers=headers, cache=cache)

    def post(
        self,
        path: str,
        json: Optional[Dict] = None,
        data: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> APIResponse:
        """Make POST request."""
        return self._request("POST", path, json=json, data=data, params=params, headers=headers)

    def put(
        self,
        path: str,
        json: Optional[Dict] = None,
        data: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> APIResponse:
        """Make PUT request."""
        return self._request("PUT", path, json=json, data=data, params=params, headers=headers)

    def patch(
        self,
        path: str,
        json: Optional[Dict] = None,
        data: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> APIResponse:
        """Make PATCH request."""
        return self._request("PATCH", path, json=json, data=data, params=params, headers=headers)

    def delete(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> APIResponse:
        """Make DELETE request."""
        return self._request("DELETE", path, params=params, headers=headers)

    def get_all_pages(
        self,
        path: str,
        page_param: str = "page",
        per_page_param: str = "per_page",
        per_page: int = 100,
        data_key: Optional[str] = None,
        max_pages: int = 100,
        params: Optional[Dict[str, Any]] = None
    ) -> List[Any]:
        """
        Fetch all pages of a paginated endpoint.

        Args:
            path: API endpoint path
            page_param: Name of page parameter
            per_page_param: Name of per-page parameter
            per_page: Items per page
            data_key: Key in response containing items (None = response is list)
            max_pages: Maximum pages to fetch
            params: Additional parameters

        Returns:
            Combined list of all items
        """
        all_items = []
        page = 1
        params = params or {}

        while page <= max_pages:
            page_params = {**params, page_param: page, per_page_param: per_page}
            response = self.get(path, params=page_params, cache=False)

            if not response.ok:
                break

            # Extract items
            data = response.data
            if data_key:
                items = data.get(data_key, [])
            elif isinstance(data, list):
                items = data
            else:
                items = data.get("results", data.get("data", data.get("items", [])))

            if not items:
                break

            all_items.extend(items)

            # Check if more pages
            if len(items) < per_page:
                break

            page += 1

        return all_items

    def iter_pages(
        self,
        path: str,
        page_param: str = "page",
        per_page_param: str = "per_page",
        per_page: int = 100,
        data_key: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Iterator[List[Any]]:
        """
        Iterate over pages (generator).

        Yields:
            List of items for each page
        """
        page = 1
        params = params or {}

        while True:
            page_params = {**params, page_param: page, per_page_param: per_page}
            response = self.get(path, params=page_params, cache=False)

            if not response.ok:
                break

            data = response.data
            if data_key:
                items = data.get(data_key, [])
            elif isinstance(data, list):
                items = data
            else:
                items = data.get("results", data.get("data", data.get("items", [])))

            if not items:
                break

            yield items

            if len(items) < per_page:
                break

            page += 1

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        json: Optional[Dict] = None,
        data: Optional[str] = None,
        cache: bool = False
    ) -> APIResponse:
        """Make HTTP request with retry and caching."""
        # Build URL
        url = f"{self.base_url}{path}" if path.startswith('/') else f"{self.base_url}/{path}"

        # Build params
        params = params or {}

        # Apply auth
        request_headers = self.default_headers.copy()
        if headers:
            request_headers.update(headers)
        self.auth.apply(request_headers, params)

        # Add query params to URL
        if params:
            query_string = urllib.parse.urlencode(params)
            url = f"{url}?{query_string}"

        # Check cache for GET requests
        cache_key = None
        if cache and method == "GET" and self.cache_ttl > 0:
            cache_key = self._cache_key(url)
            cached = self._get_cached(cache_key)
            if cached:
                return APIResponse(
                    status_code=200,
                    data=cached,
                    headers={},
                    url=url,
                    elapsed_ms=0,
                    from_cache=True
                )

        # Call request hooks
        for hook in self._request_hooks:
            try:
                hook(method, url, request_headers)
            except Exception:
                pass

        # Rate limiting
        if self.rate_limiter:
            self.rate_limiter.acquire()

        # Make request with retry
        response = self._request_with_retry(method, url, request_headers, json, data)

        # Cache successful GET responses
        if cache_key and response.ok:
            self._set_cached(cache_key, response.data)

        # Call response hooks
        for hook in self._response_hooks:
            try:
                hook(response)
            except Exception:
                pass

        return response

    def _request_with_retry(
        self,
        method: str,
        url: str,
        headers: Dict[str, str],
        json_data: Optional[Dict],
        data: Optional[str]
    ) -> APIResponse:
        """Make request with retry logic."""
        last_error = None
        delay = self.retry_config.initial_delay

        for attempt in range(self.retry_config.max_retries + 1):
            try:
                start_time = time.time()

                # Prepare request body
                body = None
                if json_data:
                    body = json.dumps(json_data).encode('utf-8')
                elif data:
                    body = data.encode('utf-8')

                # Create request
                req = urllib.request.Request(
                    url,
                    data=body,
                    headers=headers,
                    method=method
                )

                # Make request
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    elapsed = (time.time() - start_time) * 1000

                    # Parse response
                    response_data = resp.read().decode('utf-8')
                    try:
                        parsed_data = json.loads(response_data) if response_data else None
                    except json.JSONDecodeError:
                        parsed_data = response_data

                    return APIResponse(
                        status_code=resp.status,
                        data=parsed_data,
                        headers=dict(resp.headers),
                        url=url,
                        elapsed_ms=elapsed
                    )

            except urllib.error.HTTPError as e:
                elapsed = (time.time() - start_time) * 1000

                # Read error response
                try:
                    error_body = e.read().decode('utf-8')
                    error_data = json.loads(error_body) if error_body else None
                except Exception:
                    error_data = None

                # Check if should retry
                if e.code in self.retry_config.retry_on_status and attempt < self.retry_config.max_retries:
                    last_error = e
                    time.sleep(delay)
                    delay = min(delay * self.retry_config.exponential_base, self.retry_config.max_delay)
                    continue

                return APIResponse(
                    status_code=e.code,
                    data=error_data,
                    headers=dict(e.headers) if e.headers else {},
                    url=url,
                    elapsed_ms=elapsed
                )

            except Exception as e:
                if self.retry_config.retry_on_exceptions and attempt < self.retry_config.max_retries:
                    last_error = e
                    time.sleep(delay)
                    delay = min(delay * self.retry_config.exponential_base, self.retry_config.max_delay)
                    continue
                raise APIError(str(e))

        raise APIError(f"Max retries exceeded: {last_error}")

    def _cache_key(self, url: str) -> str:
        """Generate cache key for URL."""
        return hashlib.md5(url.encode()).hexdigest()

    def _get_cached(self, key: str) -> Optional[Any]:
        """Get cached response if not expired."""
        with self._cache_lock:
            entry = self._cache.get(key)
            if entry and not entry.is_expired():
                return entry.data
            elif entry:
                del self._cache[key]
            return None

    def _set_cached(self, key: str, data: Any):
        """Cache response data."""
        with self._cache_lock:
            self._cache[key] = CacheEntry(
                data=data,
                created_at=datetime.now(),
                ttl_seconds=self.cache_ttl
            )

    def clear_cache(self):
        """Clear all cached responses."""
        with self._cache_lock:
            self._cache.clear()


class APIClientBuilder:
    """
    Fluent builder for APIClient.

    Example:
    ```python
    client = (APIClientBuilder("https://api.example.com")
        .with_bearer_token("token")
        .with_rate_limit(10)
        .with_retry(max_retries=3)
        .with_cache(ttl=300)
        .build())
    ```
    """

    def __init__(self, base_url: str):
        self.base_url = base_url
        self._auth: AuthHandler = NoAuth()
        self._rate_limiter: Optional[RateLimiter] = None
        self._retry_config: Optional[RetryConfig] = None
        self._cache_ttl: int = 0
        self._timeout: int = 30
        self._headers: Dict[str, str] = {}
        self._user_agent: str = "APIClient/1.0"

    def with_api_key(self, key: str, name: str = "api_key", in_header: bool = True) -> 'APIClientBuilder':
        """Add API key authentication."""
        self._auth = APIKeyAuth(key, name, "header" if in_header else "query")
        return self

    def with_bearer_token(self, token: str) -> 'APIClientBuilder':
        """Add bearer token authentication."""
        self._auth = BearerAuth(token)
        return self

    def with_basic_auth(self, username: str, password: str) -> 'APIClientBuilder':
        """Add basic authentication."""
        self._auth = BasicAuth(username, password)
        return self

    def with_rate_limit(self, requests_per_second: float, burst: int = 10) -> 'APIClientBuilder':
        """Add rate limiting."""
        self._rate_limiter = RateLimiter(requests_per_second, burst)
        return self

    def with_retry(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        retry_on_status: Optional[List[int]] = None
    ) -> 'APIClientBuilder':
        """Add retry configuration."""
        self._retry_config = RetryConfig(
            max_retries=max_retries,
            initial_delay=initial_delay,
            retry_on_status=retry_on_status or [429, 500, 502, 503, 504]
        )
        return self

    def with_cache(self, ttl: int) -> 'APIClientBuilder':
        """Add response caching."""
        self._cache_ttl = ttl
        return self

    def with_timeout(self, seconds: int) -> 'APIClientBuilder':
        """Set request timeout."""
        self._timeout = seconds
        return self

    def with_headers(self, headers: Dict[str, str]) -> 'APIClientBuilder':
        """Add default headers."""
        self._headers.update(headers)
        return self

    def with_user_agent(self, user_agent: str) -> 'APIClientBuilder':
        """Set user agent."""
        self._user_agent = user_agent
        return self

    def build(self) -> APIClient:
        """Build the API client."""
        return APIClient(
            base_url=self.base_url,
            auth=self._auth,
            rate_limiter=self._rate_limiter,
            retry_config=self._retry_config,
            cache_ttl_seconds=self._cache_ttl,
            timeout_seconds=self._timeout,
            default_headers=self._headers,
            user_agent=self._user_agent
        )


# Factory functions

def create_simple_client(base_url: str, api_key: Optional[str] = None) -> APIClient:
    """
    Create a simple API client with sensible defaults.

    Args:
        base_url: Base URL
        api_key: Optional API key

    Returns:
        Configured APIClient
    """
    builder = APIClientBuilder(base_url)

    if api_key:
        builder.with_api_key(api_key)

    return builder.with_retry().with_rate_limit(10).build()


def create_rate_limited_client(
    base_url: str,
    requests_per_second: float,
    auth: Optional[AuthHandler] = None
) -> APIClient:
    """
    Create a rate-limited API client.

    Args:
        base_url: Base URL
        requests_per_second: Request rate
        auth: Authentication handler

    Returns:
        Configured APIClient
    """
    return APIClient(
        base_url=base_url,
        auth=auth,
        rate_limiter=RateLimiter(requests_per_second),
        retry_config=RetryConfig(max_retries=3)
    )


def create_cached_client(
    base_url: str,
    cache_ttl_seconds: int = 300,
    auth: Optional[AuthHandler] = None
) -> APIClient:
    """
    Create a caching API client.

    Args:
        base_url: Base URL
        cache_ttl_seconds: Cache TTL
        auth: Authentication handler

    Returns:
        Configured APIClient
    """
    return APIClient(
        base_url=base_url,
        auth=auth,
        cache_ttl_seconds=cache_ttl_seconds,
        retry_config=RetryConfig(max_retries=2)
    )
