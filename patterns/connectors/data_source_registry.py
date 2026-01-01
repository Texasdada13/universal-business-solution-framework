"""
Data Source Registry - Universal Business Solution Framework

Unified interface for managing multiple data sources.
Supports APIs, databases, files, and scrapers with consistent access patterns.

Example:
```python
from patterns.connectors import DataSourceRegistry, DataSource, SourceType

# Create registry
registry = DataSourceRegistry()

# Register sources
registry.register("users_api", APIDataSource(
    base_url="https://api.example.com",
    auth_token="token"
))

registry.register("contracts_db", DatabaseSource(
    connection_string="postgresql://..."
))

registry.register("legacy_csv", FileSource(
    path="data/legacy.csv"
))

# Use sources
users = registry.get("users_api").fetch("/users")
contracts = registry.get("contracts_db").query("SELECT * FROM contracts")
legacy = registry.get("legacy_csv").read()
```
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Iterator, Union, Type
import json
import threading
from pathlib import Path


class SourceType(Enum):
    """Data source types."""
    API = "api"
    DATABASE = "database"
    FILE = "file"
    SCRAPER = "scraper"
    MEMORY = "memory"
    CUSTOM = "custom"


class SourceStatus(Enum):
    """Data source status."""
    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"


@dataclass
class SourceMetadata:
    """Metadata about a data source."""
    name: str
    source_type: SourceType
    description: str = ""
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: Optional[datetime] = None
    access_count: int = 0
    error_count: int = 0
    last_error: Optional[str] = None
    status: SourceStatus = SourceStatus.UNKNOWN
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FetchResult:
    """Result from data source fetch."""
    success: bool
    data: Any
    source_name: str
    elapsed_ms: float
    from_cache: bool = False
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class DataSource(ABC):
    """Base class for data sources."""

    def __init__(self, name: str = "", description: str = ""):
        self.name = name
        self.description = description
        self._metadata: Optional[SourceMetadata] = None

    @property
    @abstractmethod
    def source_type(self) -> SourceType:
        """Get source type."""
        pass

    @abstractmethod
    def fetch(self, *args, **kwargs) -> FetchResult:
        """Fetch data from source."""
        pass

    @abstractmethod
    def health_check(self) -> SourceStatus:
        """Check source health."""
        pass

    def get_metadata(self) -> SourceMetadata:
        """Get source metadata."""
        if not self._metadata:
            self._metadata = SourceMetadata(
                name=self.name,
                source_type=self.source_type,
                description=self.description
            )
        return self._metadata


class MemorySource(DataSource):
    """
    In-memory data source for testing or caching.

    Example:
    ```python
    source = MemorySource(data={"key": "value"})
    result = source.fetch()
    print(result.data)  # {"key": "value"}
    ```
    """

    def __init__(self, data: Any = None, name: str = "memory"):
        super().__init__(name)
        self._data = data

    @property
    def source_type(self) -> SourceType:
        return SourceType.MEMORY

    def fetch(self, key: Optional[str] = None, **kwargs) -> FetchResult:
        """Fetch data from memory."""
        import time
        start = time.time()

        try:
            if key and isinstance(self._data, dict):
                data = self._data.get(key)
            else:
                data = self._data

            return FetchResult(
                success=True,
                data=data,
                source_name=self.name,
                elapsed_ms=(time.time() - start) * 1000
            )
        except Exception as e:
            return FetchResult(
                success=False,
                data=None,
                source_name=self.name,
                elapsed_ms=(time.time() - start) * 1000,
                error=str(e)
            )

    def set_data(self, data: Any):
        """Set source data."""
        self._data = data

    def health_check(self) -> SourceStatus:
        return SourceStatus.HEALTHY


class FileSource(DataSource):
    """
    File-based data source.

    Supports JSON, CSV, and text files.

    Example:
    ```python
    source = FileSource("data/users.json")
    result = source.fetch()
    users = result.data
    ```
    """

    def __init__(
        self,
        path: str,
        format: Optional[str] = None,
        encoding: str = "utf-8",
        name: str = ""
    ):
        super().__init__(name or Path(path).name)
        self.path = Path(path)
        self.format = format or self._detect_format()
        self.encoding = encoding

    @property
    def source_type(self) -> SourceType:
        return SourceType.FILE

    def _detect_format(self) -> str:
        """Detect file format from extension."""
        ext = self.path.suffix.lower()
        formats = {
            '.json': 'json',
            '.csv': 'csv',
            '.txt': 'text',
            '.xml': 'xml',
            '.yaml': 'yaml',
            '.yml': 'yaml'
        }
        return formats.get(ext, 'text')

    def fetch(self, **kwargs) -> FetchResult:
        """Read file data."""
        import time
        start = time.time()

        try:
            if not self.path.exists():
                return FetchResult(
                    success=False,
                    data=None,
                    source_name=self.name,
                    elapsed_ms=(time.time() - start) * 1000,
                    error=f"File not found: {self.path}"
                )

            content = self.path.read_text(encoding=self.encoding)

            # Parse based on format
            if self.format == 'json':
                data = json.loads(content)
            elif self.format == 'csv':
                import csv
                from io import StringIO
                reader = csv.DictReader(StringIO(content))
                data = list(reader)
            else:
                data = content

            return FetchResult(
                success=True,
                data=data,
                source_name=self.name,
                elapsed_ms=(time.time() - start) * 1000,
                metadata={"path": str(self.path), "format": self.format}
            )

        except Exception as e:
            return FetchResult(
                success=False,
                data=None,
                source_name=self.name,
                elapsed_ms=(time.time() - start) * 1000,
                error=str(e)
            )

    def write(self, data: Any) -> bool:
        """Write data to file."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)

            if self.format == 'json':
                content = json.dumps(data, indent=2)
            elif self.format == 'csv':
                import csv
                from io import StringIO
                if data and isinstance(data, list) and isinstance(data[0], dict):
                    output = StringIO()
                    writer = csv.DictWriter(output, fieldnames=data[0].keys())
                    writer.writeheader()
                    writer.writerows(data)
                    content = output.getvalue()
                else:
                    content = str(data)
            else:
                content = str(data)

            self.path.write_text(content, encoding=self.encoding)
            return True

        except Exception:
            return False

    def health_check(self) -> SourceStatus:
        if self.path.exists():
            return SourceStatus.HEALTHY
        return SourceStatus.OFFLINE


class APISource(DataSource):
    """
    API data source.

    Wraps API client for registry integration.

    Example:
    ```python
    source = APISource(
        base_url="https://api.example.com",
        auth_token="token"
    )
    result = source.fetch("/users")
    users = result.data
    ```
    """

    def __init__(
        self,
        base_url: str,
        auth_token: Optional[str] = None,
        api_key: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: int = 30,
        name: str = ""
    ):
        super().__init__(name or base_url.split('//')[-1].split('/')[0])
        self.base_url = base_url.rstrip('/')
        self.auth_token = auth_token
        self.api_key = api_key
        self.timeout = timeout
        self.headers = headers or {}

    @property
    def source_type(self) -> SourceType:
        return SourceType.API

    def _get_headers(self) -> Dict[str, str]:
        """Build request headers."""
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            **self.headers
        }
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        return headers

    def fetch(
        self,
        endpoint: str = "",
        method: str = "GET",
        params: Optional[Dict] = None,
        data: Optional[Dict] = None,
        **kwargs
    ) -> FetchResult:
        """Fetch from API endpoint."""
        import time
        import urllib.request
        import urllib.error
        import urllib.parse

        start = time.time()

        try:
            # Build URL
            url = f"{self.base_url}{endpoint}" if endpoint.startswith('/') else f"{self.base_url}/{endpoint}"
            if params:
                url = f"{url}?{urllib.parse.urlencode(params)}"

            # Build request
            body = json.dumps(data).encode() if data else None
            req = urllib.request.Request(
                url,
                data=body,
                headers=self._get_headers(),
                method=method
            )

            # Make request
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                response_data = json.loads(response.read().decode())

                return FetchResult(
                    success=True,
                    data=response_data,
                    source_name=self.name,
                    elapsed_ms=(time.time() - start) * 1000,
                    metadata={"url": url, "status": response.status}
                )

        except urllib.error.HTTPError as e:
            return FetchResult(
                success=False,
                data=None,
                source_name=self.name,
                elapsed_ms=(time.time() - start) * 1000,
                error=f"HTTP {e.code}: {e.reason}"
            )
        except Exception as e:
            return FetchResult(
                success=False,
                data=None,
                source_name=self.name,
                elapsed_ms=(time.time() - start) * 1000,
                error=str(e)
            )

    def health_check(self) -> SourceStatus:
        """Check API health."""
        try:
            result = self.fetch("/health")
            if result.success:
                return SourceStatus.HEALTHY
            return SourceStatus.DEGRADED
        except Exception:
            return SourceStatus.UNHEALTHY


class DataSourceRegistry:
    """
    Registry for managing multiple data sources.

    Features:
    - Register and retrieve data sources
    - Health monitoring
    - Access tracking
    - Source discovery by type/tags

    Example:
    ```python
    registry = DataSourceRegistry()

    # Register sources
    registry.register("api", APISource(base_url="..."))
    registry.register("cache", MemorySource())
    registry.register("backup", FileSource("backup.json"))

    # Get source
    api = registry.get("api")
    result = api.fetch("/endpoint")

    # Get all API sources
    api_sources = registry.get_by_type(SourceType.API)

    # Health check all
    health = registry.health_check_all()
    ```
    """

    def __init__(self):
        self._sources: Dict[str, DataSource] = {}
        self._metadata: Dict[str, SourceMetadata] = {}
        self._lock = threading.Lock()

    def register(
        self,
        name: str,
        source: DataSource,
        description: str = "",
        tags: Optional[List[str]] = None
    ) -> 'DataSourceRegistry':
        """
        Register a data source.

        Args:
            name: Unique source name
            source: Data source instance
            description: Source description
            tags: Source tags for discovery

        Returns:
            Self for chaining
        """
        with self._lock:
            source.name = name
            if description:
                source.description = description

            self._sources[name] = source
            self._metadata[name] = SourceMetadata(
                name=name,
                source_type=source.source_type,
                description=description or source.description,
                tags=tags or []
            )

        return self

    def unregister(self, name: str) -> bool:
        """
        Remove a data source.

        Args:
            name: Source name

        Returns:
            True if source was removed
        """
        with self._lock:
            if name in self._sources:
                del self._sources[name]
                del self._metadata[name]
                return True
            return False

    def get(self, name: str) -> Optional[DataSource]:
        """
        Get a data source by name.

        Args:
            name: Source name

        Returns:
            DataSource or None if not found
        """
        source = self._sources.get(name)
        if source:
            with self._lock:
                meta = self._metadata[name]
                meta.last_accessed = datetime.now()
                meta.access_count += 1
        return source

    def fetch(self, name: str, *args, **kwargs) -> FetchResult:
        """
        Fetch data from a named source.

        Args:
            name: Source name
            *args, **kwargs: Args for source.fetch()

        Returns:
            FetchResult
        """
        source = self.get(name)
        if not source:
            return FetchResult(
                success=False,
                data=None,
                source_name=name,
                elapsed_ms=0,
                error=f"Source not found: {name}"
            )

        try:
            result = source.fetch(*args, **kwargs)

            # Track errors
            if not result.success:
                with self._lock:
                    meta = self._metadata[name]
                    meta.error_count += 1
                    meta.last_error = result.error

            return result

        except Exception as e:
            with self._lock:
                meta = self._metadata[name]
                meta.error_count += 1
                meta.last_error = str(e)

            return FetchResult(
                success=False,
                data=None,
                source_name=name,
                elapsed_ms=0,
                error=str(e)
            )

    def get_by_type(self, source_type: SourceType) -> List[DataSource]:
        """Get all sources of a specific type."""
        return [s for s in self._sources.values() if s.source_type == source_type]

    def get_by_tag(self, tag: str) -> List[DataSource]:
        """Get all sources with a specific tag."""
        return [
            self._sources[name]
            for name, meta in self._metadata.items()
            if tag in meta.tags
        ]

    def list_sources(self) -> List[str]:
        """Get list of all source names."""
        return list(self._sources.keys())

    def get_metadata(self, name: str) -> Optional[SourceMetadata]:
        """Get metadata for a source."""
        return self._metadata.get(name)

    def get_all_metadata(self) -> Dict[str, SourceMetadata]:
        """Get metadata for all sources."""
        return dict(self._metadata)

    def health_check(self, name: str) -> SourceStatus:
        """
        Check health of a specific source.

        Args:
            name: Source name

        Returns:
            SourceStatus
        """
        source = self._sources.get(name)
        if not source:
            return SourceStatus.UNKNOWN

        try:
            status = source.health_check()
            with self._lock:
                self._metadata[name].status = status
            return status
        except Exception:
            with self._lock:
                self._metadata[name].status = SourceStatus.UNHEALTHY
            return SourceStatus.UNHEALTHY

    def health_check_all(self) -> Dict[str, SourceStatus]:
        """
        Check health of all sources.

        Returns:
            Dict of source name -> status
        """
        results = {}
        for name in self._sources:
            results[name] = self.health_check(name)
        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        with self._lock:
            total_access = sum(m.access_count for m in self._metadata.values())
            total_errors = sum(m.error_count for m in self._metadata.values())

            by_type = {}
            for source_type in SourceType:
                count = len([s for s in self._sources.values() if s.source_type == source_type])
                if count > 0:
                    by_type[source_type.value] = count

            by_status = {}
            for status in SourceStatus:
                count = len([m for m in self._metadata.values() if m.status == status])
                if count > 0:
                    by_status[status.value] = count

            return {
                "total_sources": len(self._sources),
                "total_access_count": total_access,
                "total_error_count": total_errors,
                "by_type": by_type,
                "by_status": by_status
            }


class CachedSource(DataSource):
    """
    Wrapper that adds caching to any data source.

    Example:
    ```python
    api = APISource(base_url="...")
    cached_api = CachedSource(api, ttl_seconds=300)
    result = cached_api.fetch("/expensive-endpoint")  # Cached for 5 min
    ```
    """

    def __init__(
        self,
        source: DataSource,
        ttl_seconds: int = 300,
        name: str = ""
    ):
        super().__init__(name or f"cached_{source.name}")
        self._source = source
        self._ttl = ttl_seconds
        self._cache: Dict[str, tuple] = {}  # key -> (data, timestamp)
        self._lock = threading.Lock()

    @property
    def source_type(self) -> SourceType:
        return self._source.source_type

    def _cache_key(self, *args, **kwargs) -> str:
        """Generate cache key from args."""
        import hashlib
        content = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True, default=str)
        return hashlib.md5(content.encode()).hexdigest()

    def fetch(self, *args, **kwargs) -> FetchResult:
        """Fetch with caching."""
        key = self._cache_key(*args, **kwargs)

        # Check cache
        with self._lock:
            if key in self._cache:
                data, timestamp = self._cache[key]
                if datetime.now() - timestamp < timedelta(seconds=self._ttl):
                    return FetchResult(
                        success=True,
                        data=data,
                        source_name=self.name,
                        elapsed_ms=0,
                        from_cache=True
                    )

        # Fetch from source
        result = self._source.fetch(*args, **kwargs)

        # Cache if successful
        if result.success:
            with self._lock:
                self._cache[key] = (result.data, datetime.now())

        return result

    def invalidate(self, *args, **kwargs):
        """Invalidate specific cache entry."""
        key = self._cache_key(*args, **kwargs)
        with self._lock:
            self._cache.pop(key, None)

    def clear_cache(self):
        """Clear all cached data."""
        with self._lock:
            self._cache.clear()

    def health_check(self) -> SourceStatus:
        return self._source.health_check()


class FallbackSource(DataSource):
    """
    Source that falls back to alternatives on failure.

    Example:
    ```python
    primary = APISource(base_url="https://primary.api.com")
    backup = APISource(base_url="https://backup.api.com")
    cache = FileSource("cache.json")

    resilient = FallbackSource([primary, backup, cache])
    result = resilient.fetch("/data")  # Tries each until success
    ```
    """

    def __init__(
        self,
        sources: List[DataSource],
        name: str = "fallback"
    ):
        super().__init__(name)
        self._sources = sources

    @property
    def source_type(self) -> SourceType:
        return SourceType.CUSTOM

    def fetch(self, *args, **kwargs) -> FetchResult:
        """Fetch with fallback to alternatives."""
        errors = []

        for source in self._sources:
            try:
                result = source.fetch(*args, **kwargs)
                if result.success:
                    result.metadata["fallback_source"] = source.name
                    return result
                errors.append(f"{source.name}: {result.error}")
            except Exception as e:
                errors.append(f"{source.name}: {str(e)}")

        return FetchResult(
            success=False,
            data=None,
            source_name=self.name,
            elapsed_ms=0,
            error=f"All sources failed: {'; '.join(errors)}"
        )

    def health_check(self) -> SourceStatus:
        """Check if any source is healthy."""
        for source in self._sources:
            if source.health_check() == SourceStatus.HEALTHY:
                return SourceStatus.HEALTHY
        return SourceStatus.UNHEALTHY


# Factory functions

def create_simple_registry() -> DataSourceRegistry:
    """Create an empty registry."""
    return DataSourceRegistry()


def create_file_registry(data_dir: str) -> DataSourceRegistry:
    """
    Create a registry pre-populated with file sources from a directory.

    Args:
        data_dir: Directory containing data files

    Returns:
        Registry with file sources
    """
    registry = DataSourceRegistry()
    data_path = Path(data_dir)

    if data_path.exists():
        for file_path in data_path.glob("*"):
            if file_path.is_file() and file_path.suffix in ('.json', '.csv', '.txt'):
                name = file_path.stem
                registry.register(name, FileSource(str(file_path)))

    return registry


def create_api_registry(apis: Dict[str, Dict[str, Any]]) -> DataSourceRegistry:
    """
    Create a registry with multiple API sources.

    Args:
        apis: Dict of name -> API config
              {"name": {"base_url": "...", "auth_token": "..."}}

    Returns:
        Registry with API sources
    """
    registry = DataSourceRegistry()

    for name, config in apis.items():
        source = APISource(
            base_url=config["base_url"],
            auth_token=config.get("auth_token"),
            api_key=config.get("api_key"),
            headers=config.get("headers"),
            name=name
        )
        registry.register(name, source, tags=["api"])

    return registry
