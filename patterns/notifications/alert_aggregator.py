"""
Alert Aggregator - Universal Business Solution Framework

Aggregate, deduplicate, and rate-limit alerts before sending notifications.
Prevents alert fatigue by grouping related alerts and respecting quiet hours.

Example:
```python
from patterns.notifications import AlertAggregator, AlertPriority

# Create aggregator with rate limiting
aggregator = AlertAggregator(
    dedup_window_minutes=60,
    max_alerts_per_hour=10,
    quiet_hours=(22, 7)  # 10 PM - 7 AM
)

# Add alerts (duplicates within window are grouped)
aggregator.add_alert("Contract Expiring", "Contract #123 expires soon", AlertPriority.HIGH)
aggregator.add_alert("Contract Expiring", "Contract #123 expires soon", AlertPriority.HIGH)  # Deduped

# Get aggregated alerts ready to send
ready_alerts = aggregator.get_pending_alerts()
```
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Tuple
from collections import defaultdict
import hashlib
import json
import threading


class AlertPriority(Enum):
    """Alert priority levels."""
    CRITICAL = 5    # Immediate notification, bypass rate limits
    HIGH = 4        # High priority, limited rate limiting
    MEDIUM = 3      # Normal priority
    LOW = 2         # Low priority, aggressive rate limiting
    INFO = 1        # Informational, can be batched


class AlertStatus(Enum):
    """Alert lifecycle status."""
    PENDING = "pending"
    AGGREGATED = "aggregated"
    SENT = "sent"
    SUPPRESSED = "suppressed"
    EXPIRED = "expired"


@dataclass
class Alert:
    """Individual alert."""
    id: str
    title: str
    message: str
    priority: AlertPriority
    source: str = ""
    category: str = ""
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    status: AlertStatus = AlertStatus.PENDING
    occurrence_count: int = 1
    first_occurrence: datetime = field(default_factory=datetime.now)
    last_occurrence: datetime = field(default_factory=datetime.now)
    fingerprint: str = ""

    def __post_init__(self):
        if not self.fingerprint:
            self.fingerprint = self._generate_fingerprint()

    def _generate_fingerprint(self) -> str:
        """Generate unique fingerprint for deduplication."""
        content = f"{self.title}|{self.message}|{self.source}|{self.category}"
        return hashlib.md5(content.encode()).hexdigest()[:16]


@dataclass
class AlertGroup:
    """Group of related alerts."""
    key: str
    alerts: List[Alert] = field(default_factory=list)
    priority: AlertPriority = AlertPriority.MEDIUM
    total_count: int = 0
    first_occurrence: datetime = field(default_factory=datetime.now)
    last_occurrence: datetime = field(default_factory=datetime.now)

    def add_alert(self, alert: Alert):
        """Add alert to group."""
        self.alerts.append(alert)
        self.total_count += alert.occurrence_count
        self.last_occurrence = alert.last_occurrence

        # Escalate priority to highest in group
        if alert.priority.value > self.priority.value:
            self.priority = alert.priority

        # Track first occurrence
        if alert.first_occurrence < self.first_occurrence:
            self.first_occurrence = alert.first_occurrence


class AlertAggregator:
    """
    Aggregate and manage alerts with deduplication and rate limiting.

    Features:
    - Deduplication within configurable time window
    - Rate limiting per priority level
    - Quiet hours suppression
    - Alert grouping by category/source
    - Priority escalation
    - Expiration handling

    Example:
    ```python
    aggregator = AlertAggregator(
        dedup_window_minutes=60,
        max_alerts_per_hour=20,
        quiet_hours=(22, 7)
    )

    # Add alerts
    aggregator.add_alert("Server Down", "Web server not responding", AlertPriority.CRITICAL)

    # Get ready alerts
    for alert in aggregator.get_pending_alerts():
        send_notification(alert)
        aggregator.mark_sent(alert.id)
    ```
    """

    def __init__(
        self,
        dedup_window_minutes: int = 60,
        max_alerts_per_hour: int = 20,
        quiet_hours: Optional[Tuple[int, int]] = None,
        expiration_hours: int = 24,
        group_by: Optional[List[str]] = None,
        priority_rate_limits: Optional[Dict[AlertPriority, int]] = None
    ):
        """
        Initialize alert aggregator.

        Args:
            dedup_window_minutes: Time window for deduplication
            max_alerts_per_hour: Maximum alerts per hour (0 = unlimited)
            quiet_hours: Tuple of (start_hour, end_hour) for quiet period
            expiration_hours: Hours until unsent alerts expire
            group_by: Fields to group alerts by (e.g., ['category', 'source'])
            priority_rate_limits: Per-priority rate limits (alerts/hour)
        """
        self.dedup_window = timedelta(minutes=dedup_window_minutes)
        self.max_alerts_per_hour = max_alerts_per_hour
        self.quiet_hours = quiet_hours
        self.expiration_hours = expiration_hours
        self.group_by = group_by or ['category']

        # Priority-specific rate limits
        self.priority_rate_limits = priority_rate_limits or {
            AlertPriority.CRITICAL: 0,   # No limit for critical
            AlertPriority.HIGH: 50,
            AlertPriority.MEDIUM: 20,
            AlertPriority.LOW: 10,
            AlertPriority.INFO: 5
        }

        # Alert storage
        self._alerts: Dict[str, Alert] = {}  # id -> Alert
        self._fingerprints: Dict[str, str] = {}  # fingerprint -> alert_id
        self._sent_counts: Dict[str, int] = defaultdict(int)  # hour_key -> count
        self._priority_counts: Dict[Tuple[str, AlertPriority], int] = defaultdict(int)

        self._lock = threading.Lock()

    def add_alert(
        self,
        title: str,
        message: str,
        priority: AlertPriority = AlertPriority.MEDIUM,
        source: str = "",
        category: str = "",
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        alert_id: Optional[str] = None
    ) -> Alert:
        """
        Add an alert to the aggregator.

        Args:
            title: Alert title
            message: Alert message
            priority: Alert priority level
            source: Source system/component
            category: Alert category for grouping
            tags: Additional tags
            metadata: Additional metadata
            alert_id: Optional custom ID

        Returns:
            The created or updated Alert
        """
        with self._lock:
            # Create alert
            alert = Alert(
                id=alert_id or self._generate_id(),
                title=title,
                message=message,
                priority=priority,
                source=source,
                category=category,
                tags=tags or [],
                metadata=metadata or {}
            )

            # Check for duplicate
            existing_id = self._fingerprints.get(alert.fingerprint)
            if existing_id and existing_id in self._alerts:
                existing = self._alerts[existing_id]

                # Check if within dedup window
                if datetime.now() - existing.last_occurrence < self.dedup_window:
                    # Aggregate into existing
                    existing.occurrence_count += 1
                    existing.last_occurrence = datetime.now()
                    existing.status = AlertStatus.AGGREGATED

                    # Escalate priority if needed
                    if priority.value > existing.priority.value:
                        existing.priority = priority

                    return existing

            # New alert
            self._alerts[alert.id] = alert
            self._fingerprints[alert.fingerprint] = alert.id

            return alert

    def get_pending_alerts(
        self,
        respect_rate_limits: bool = True,
        respect_quiet_hours: bool = True
    ) -> List[Alert]:
        """
        Get alerts ready to be sent.

        Args:
            respect_rate_limits: Apply rate limiting
            respect_quiet_hours: Suppress during quiet hours

        Returns:
            List of alerts ready to send
        """
        with self._lock:
            now = datetime.now()

            # Check quiet hours
            if respect_quiet_hours and self._is_quiet_hours(now):
                # Only return critical alerts during quiet hours
                return [
                    a for a in self._alerts.values()
                    if a.status == AlertStatus.PENDING and a.priority == AlertPriority.CRITICAL
                ]

            # Get pending alerts
            pending = [
                a for a in self._alerts.values()
                if a.status in (AlertStatus.PENDING, AlertStatus.AGGREGATED)
            ]

            # Sort by priority (highest first) and time
            pending.sort(key=lambda a: (-a.priority.value, a.created_at))

            # Apply rate limiting
            if respect_rate_limits:
                pending = self._apply_rate_limits(pending, now)

            # Expire old alerts
            self._expire_old_alerts(now)

            return pending

    def get_grouped_alerts(self) -> Dict[str, AlertGroup]:
        """
        Get alerts grouped by configured fields.

        Returns:
            Dict of group_key -> AlertGroup
        """
        groups: Dict[str, AlertGroup] = {}

        for alert in self._alerts.values():
            if alert.status not in (AlertStatus.PENDING, AlertStatus.AGGREGATED):
                continue

            # Generate group key
            key_parts = []
            for field in self.group_by:
                if field == 'category':
                    key_parts.append(alert.category or 'uncategorized')
                elif field == 'source':
                    key_parts.append(alert.source or 'unknown')
                elif field == 'priority':
                    key_parts.append(alert.priority.name)
                else:
                    key_parts.append(alert.metadata.get(field, 'unknown'))

            group_key = "|".join(key_parts)

            if group_key not in groups:
                groups[group_key] = AlertGroup(key=group_key)

            groups[group_key].add_alert(alert)

        return groups

    def mark_sent(self, alert_id: str) -> bool:
        """
        Mark an alert as sent.

        Args:
            alert_id: Alert ID to mark

        Returns:
            True if alert was found and marked
        """
        with self._lock:
            if alert_id in self._alerts:
                alert = self._alerts[alert_id]
                alert.status = AlertStatus.SENT

                # Track sent counts
                hour_key = datetime.now().strftime("%Y%m%d%H")
                self._sent_counts[hour_key] += 1
                self._priority_counts[(hour_key, alert.priority)] += 1

                return True
            return False

    def mark_suppressed(self, alert_id: str, reason: str = "") -> bool:
        """
        Mark an alert as suppressed (not sent).

        Args:
            alert_id: Alert ID to suppress
            reason: Reason for suppression

        Returns:
            True if alert was found and suppressed
        """
        with self._lock:
            if alert_id in self._alerts:
                alert = self._alerts[alert_id]
                alert.status = AlertStatus.SUPPRESSED
                alert.metadata['suppression_reason'] = reason
                return True
            return False

    def get_alert(self, alert_id: str) -> Optional[Alert]:
        """Get an alert by ID."""
        return self._alerts.get(alert_id)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get aggregator statistics.

        Returns:
            Dict with stats including counts by status, priority, etc.
        """
        stats = {
            "total_alerts": len(self._alerts),
            "by_status": defaultdict(int),
            "by_priority": defaultdict(int),
            "sent_last_hour": 0,
            "is_quiet_hours": self._is_quiet_hours(datetime.now()),
            "rate_limit_remaining": self._get_rate_limit_remaining()
        }

        for alert in self._alerts.values():
            stats["by_status"][alert.status.value] += 1
            stats["by_priority"][alert.priority.name] += 1

        # Sent last hour
        hour_key = datetime.now().strftime("%Y%m%d%H")
        stats["sent_last_hour"] = self._sent_counts.get(hour_key, 0)

        return dict(stats)

    def clear_expired(self) -> int:
        """
        Remove expired alerts.

        Returns:
            Number of alerts removed
        """
        with self._lock:
            now = datetime.now()
            expired_ids = []

            for alert_id, alert in self._alerts.items():
                if alert.status == AlertStatus.EXPIRED:
                    expired_ids.append(alert_id)
                elif now - alert.created_at > timedelta(hours=self.expiration_hours):
                    if alert.status in (AlertStatus.PENDING, AlertStatus.AGGREGATED):
                        alert.status = AlertStatus.EXPIRED
                        expired_ids.append(alert_id)

            for alert_id in expired_ids:
                alert = self._alerts.pop(alert_id, None)
                if alert:
                    self._fingerprints.pop(alert.fingerprint, None)

            return len(expired_ids)

    def clear_all(self):
        """Clear all alerts."""
        with self._lock:
            self._alerts.clear()
            self._fingerprints.clear()
            self._sent_counts.clear()
            self._priority_counts.clear()

    def _generate_id(self) -> str:
        """Generate unique alert ID."""
        import uuid
        return str(uuid.uuid4())[:8]

    def _is_quiet_hours(self, now: datetime) -> bool:
        """Check if current time is in quiet hours."""
        if not self.quiet_hours:
            return False

        start, end = self.quiet_hours
        current_hour = now.hour

        if start < end:
            # Same day range (e.g., 9-17)
            return start <= current_hour < end
        else:
            # Overnight range (e.g., 22-7)
            return current_hour >= start or current_hour < end

    def _apply_rate_limits(self, alerts: List[Alert], now: datetime) -> List[Alert]:
        """Apply rate limits to alert list."""
        hour_key = now.strftime("%Y%m%d%H")
        current_count = self._sent_counts.get(hour_key, 0)

        # Check global limit
        if self.max_alerts_per_hour > 0 and current_count >= self.max_alerts_per_hour:
            # Only allow critical
            return [a for a in alerts if a.priority == AlertPriority.CRITICAL]

        result = []
        for alert in alerts:
            # Check priority-specific limit
            priority_limit = self.priority_rate_limits.get(alert.priority, 0)
            priority_count = self._priority_counts.get((hour_key, alert.priority), 0)

            if priority_limit == 0 or priority_count < priority_limit:
                result.append(alert)

                # Check global limit
                if self.max_alerts_per_hour > 0 and len(result) + current_count >= self.max_alerts_per_hour:
                    break

        return result

    def _get_rate_limit_remaining(self) -> Dict[str, int]:
        """Get remaining alerts allowed per priority."""
        hour_key = datetime.now().strftime("%Y%m%d%H")
        remaining = {}

        for priority, limit in self.priority_rate_limits.items():
            if limit == 0:
                remaining[priority.name] = -1  # Unlimited
            else:
                current = self._priority_counts.get((hour_key, priority), 0)
                remaining[priority.name] = max(0, limit - current)

        return remaining

    def _expire_old_alerts(self, now: datetime):
        """Mark old alerts as expired."""
        expiration_threshold = now - timedelta(hours=self.expiration_hours)

        for alert in self._alerts.values():
            if alert.status in (AlertStatus.PENDING, AlertStatus.AGGREGATED):
                if alert.created_at < expiration_threshold:
                    alert.status = AlertStatus.EXPIRED


class AlertBuffer:
    """
    Buffer alerts for batch processing.

    Collects alerts and flushes them at intervals or when threshold reached.

    Example:
    ```python
    def send_batch(alerts):
        # Send all alerts at once
        email_service.send_digest(alerts)

    buffer = AlertBuffer(
        flush_callback=send_batch,
        max_size=50,
        flush_interval_seconds=300
    )

    # Add alerts (will auto-flush when conditions met)
    buffer.add(alert1)
    buffer.add(alert2)

    # Manual flush
    buffer.flush()
    ```
    """

    def __init__(
        self,
        flush_callback: Callable[[List[Alert]], None],
        max_size: int = 100,
        flush_interval_seconds: int = 300,
        auto_flush: bool = True
    ):
        """
        Initialize alert buffer.

        Args:
            flush_callback: Function to call with buffered alerts
            max_size: Max alerts before auto-flush
            flush_interval_seconds: Max seconds between flushes
            auto_flush: Whether to auto-flush on size/time thresholds
        """
        self.flush_callback = flush_callback
        self.max_size = max_size
        self.flush_interval = timedelta(seconds=flush_interval_seconds)
        self.auto_flush = auto_flush

        self._buffer: List[Alert] = []
        self._last_flush = datetime.now()
        self._lock = threading.Lock()

    def add(self, alert: Alert) -> bool:
        """
        Add alert to buffer.

        Args:
            alert: Alert to add

        Returns:
            True if buffer was flushed
        """
        flushed = False

        with self._lock:
            self._buffer.append(alert)

            if self.auto_flush:
                # Check size threshold
                if len(self._buffer) >= self.max_size:
                    self._flush_internal()
                    flushed = True

                # Check time threshold
                elif datetime.now() - self._last_flush > self.flush_interval:
                    self._flush_internal()
                    flushed = True

                # Immediate flush for critical
                elif alert.priority == AlertPriority.CRITICAL:
                    self._flush_internal()
                    flushed = True

        return flushed

    def flush(self) -> int:
        """
        Manually flush buffer.

        Returns:
            Number of alerts flushed
        """
        with self._lock:
            return self._flush_internal()

    def _flush_internal(self) -> int:
        """Internal flush (must hold lock)."""
        if not self._buffer:
            return 0

        alerts = self._buffer.copy()
        self._buffer.clear()
        self._last_flush = datetime.now()

        # Call callback outside lock to avoid deadlock
        try:
            self.flush_callback(alerts)
        except Exception:
            # Re-add alerts on failure
            self._buffer.extend(alerts)
            raise

        return len(alerts)

    def size(self) -> int:
        """Get current buffer size."""
        return len(self._buffer)

    def clear(self):
        """Clear buffer without flushing."""
        with self._lock:
            self._buffer.clear()


# Factory functions

def create_standard_aggregator(
    max_per_hour: int = 20,
    quiet_start: int = 22,
    quiet_end: int = 7
) -> AlertAggregator:
    """
    Create a standard alert aggregator with common defaults.

    Args:
        max_per_hour: Max alerts per hour
        quiet_start: Quiet hours start (24h)
        quiet_end: Quiet hours end (24h)

    Returns:
        Configured AlertAggregator
    """
    return AlertAggregator(
        dedup_window_minutes=60,
        max_alerts_per_hour=max_per_hour,
        quiet_hours=(quiet_start, quiet_end),
        expiration_hours=24,
        group_by=['category', 'source']
    )


def create_critical_only_aggregator() -> AlertAggregator:
    """
    Create aggregator that only passes critical alerts.

    Returns:
        Configured AlertAggregator
    """
    return AlertAggregator(
        dedup_window_minutes=15,
        max_alerts_per_hour=0,  # No global limit
        quiet_hours=None,  # No quiet hours
        priority_rate_limits={
            AlertPriority.CRITICAL: 0,  # Unlimited
            AlertPriority.HIGH: 0,
            AlertPriority.MEDIUM: 0,
            AlertPriority.LOW: 0,
            AlertPriority.INFO: 0
        }
    )


def create_digest_aggregator(
    digest_interval_hours: int = 24
) -> AlertAggregator:
    """
    Create aggregator optimized for daily/weekly digests.

    Args:
        digest_interval_hours: Hours between digests

    Returns:
        Configured AlertAggregator for digest mode
    """
    return AlertAggregator(
        dedup_window_minutes=digest_interval_hours * 60,
        max_alerts_per_hour=0,  # Collect everything
        quiet_hours=None,
        expiration_hours=digest_interval_hours * 2,
        group_by=['category', 'priority']
    )
