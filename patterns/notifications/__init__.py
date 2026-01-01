"""
Notifications Patterns - Universal Business Solution Framework

Alert management, email digests, and multi-channel notification routing.
Supports deduplication, rate limiting, and professional email templates.

Usage:
```python
from patterns.notifications import (
    # Email Digests
    EmailDigestBuilder, DigestTheme, AlertEmailBuilder,
    create_daily_digest, create_alert_digest,

    # Alert Aggregation
    AlertAggregator, AlertPriority, AlertStatus, AlertBuffer,
    create_standard_aggregator, create_digest_aggregator,

    # Notification Routing
    NotificationRouter, Channel, Notification, NotificationPriority,
    SlackHandler, TeamsHandler, WebhookHandler,
    create_slack_router, create_multi_channel_router,
)
```
"""

from .email_digest import (
    EmailDigestBuilder,
    DigestTheme,
    AlertSeverity,
    ThemeConfig,
    DigestSection,
    AlertEmailBuilder,
    # Factory functions
    create_daily_digest,
    create_alert_digest,
    create_report_email,
)

from .alert_aggregator import (
    AlertAggregator,
    AlertPriority,
    AlertStatus,
    Alert,
    AlertGroup,
    AlertBuffer,
    # Factory functions
    create_standard_aggregator,
    create_critical_only_aggregator,
    create_digest_aggregator,
)

from .notification_router import (
    NotificationRouter,
    Channel,
    Notification,
    NotificationPriority,
    DeliveryResult,
    RoutingRule,
    # Handlers
    ChannelHandler,
    ConsoleHandler,
    FileHandler,
    SlackHandler,
    TeamsHandler,
    WebhookHandler,
    # Factory functions
    create_slack_router,
    create_teams_router,
    create_multi_channel_router,
    create_development_router,
)

__all__ = [
    # Email Digest
    "EmailDigestBuilder",
    "DigestTheme",
    "AlertSeverity",
    "ThemeConfig",
    "DigestSection",
    "AlertEmailBuilder",
    "create_daily_digest",
    "create_alert_digest",
    "create_report_email",

    # Alert Aggregator
    "AlertAggregator",
    "AlertPriority",
    "AlertStatus",
    "Alert",
    "AlertGroup",
    "AlertBuffer",
    "create_standard_aggregator",
    "create_critical_only_aggregator",
    "create_digest_aggregator",

    # Notification Router
    "NotificationRouter",
    "Channel",
    "Notification",
    "NotificationPriority",
    "DeliveryResult",
    "RoutingRule",
    "ChannelHandler",
    "ConsoleHandler",
    "FileHandler",
    "SlackHandler",
    "TeamsHandler",
    "WebhookHandler",
    "create_slack_router",
    "create_teams_router",
    "create_multi_channel_router",
    "create_development_router",
]
