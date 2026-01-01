"""
Connectors Patterns - Universal Business Solution Framework

Data connection and fetching patterns for APIs, web scraping, and data sources.
Includes rate limiting, retry logic, caching, and session management.

Usage:
```python
from patterns.connectors import (
    # API Client
    APIClient, APIClientBuilder, RateLimiter, RetryConfig,
    BearerAuth, APIKeyAuth, BasicAuth,
    create_simple_client, create_rate_limited_client,

    # Web Scraping
    ScraperBase, ScraperConfig, PageScraper,
    create_gentle_scraper, create_aggressive_scraper,

    # Data Source Registry
    DataSourceRegistry, DataSource, SourceType,
    APISource, FileSource, MemorySource,
    CachedSource, FallbackSource,
    create_file_registry, create_api_registry,
)
```
"""

from .api_client import (
    APIClient,
    APIClientBuilder,
    APIResponse,
    APIError,
    RateLimiter,
    RetryConfig,
    CacheEntry,
    # Auth handlers
    AuthHandler,
    NoAuth,
    APIKeyAuth,
    BearerAuth,
    BasicAuth,
    AuthMethod,
    # Factory functions
    create_simple_client,
    create_rate_limited_client,
    create_cached_client,
)

from .scraper_base import (
    ScraperBase,
    ScraperConfig,
    ScraperSession,
    ScrapeResult,
    ScraperError,
    PageScraper,
    ProxyConfig,
    ProxyType,
    # Factory functions
    create_gentle_scraper,
    create_aggressive_scraper,
    create_proxy_scraper,
)

from .data_source_registry import (
    DataSourceRegistry,
    DataSource,
    SourceType,
    SourceStatus,
    SourceMetadata,
    FetchResult,
    # Source types
    MemorySource,
    FileSource,
    APISource,
    CachedSource,
    FallbackSource,
    # Factory functions
    create_simple_registry,
    create_file_registry,
    create_api_registry,
)

__all__ = [
    # API Client
    "APIClient",
    "APIClientBuilder",
    "APIResponse",
    "APIError",
    "RateLimiter",
    "RetryConfig",
    "CacheEntry",
    "AuthHandler",
    "NoAuth",
    "APIKeyAuth",
    "BearerAuth",
    "BasicAuth",
    "AuthMethod",
    "create_simple_client",
    "create_rate_limited_client",
    "create_cached_client",

    # Scraper
    "ScraperBase",
    "ScraperConfig",
    "ScraperSession",
    "ScrapeResult",
    "ScraperError",
    "PageScraper",
    "ProxyConfig",
    "ProxyType",
    "create_gentle_scraper",
    "create_aggressive_scraper",
    "create_proxy_scraper",

    # Data Source Registry
    "DataSourceRegistry",
    "DataSource",
    "SourceType",
    "SourceStatus",
    "SourceMetadata",
    "FetchResult",
    "MemorySource",
    "FileSource",
    "APISource",
    "CachedSource",
    "FallbackSource",
    "create_simple_registry",
    "create_file_registry",
    "create_api_registry",
]
