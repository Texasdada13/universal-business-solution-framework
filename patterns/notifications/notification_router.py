"""
Notification Router - Universal Business Solution Framework

Route notifications to multiple channels (Email, Slack, Teams, Webhook).
Supports routing rules, fallbacks, and channel-specific formatting.

Example:
```python
from patterns.notifications import NotificationRouter, Channel, RoutingRule

# Configure router
router = NotificationRouter()

# Add channels
router.add_channel(Channel.EMAIL, EmailHandler(smtp_config))
router.add_channel(Channel.SLACK, SlackHandler(webhook_url))
router.add_channel(Channel.TEAMS, TeamsHandler(webhook_url))

# Add routing rules
router.add_rule(RoutingRule(
    condition=lambda n: n.priority == 'critical',
    channels=[Channel.EMAIL, Channel.SLACK],
    immediate=True
))

# Route notification
router.send(Notification(title="Alert", message="Server down", priority="critical"))
```
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Union
import json
import threading
from pathlib import Path


class Channel(Enum):
    """Notification channel types."""
    EMAIL = "email"
    SLACK = "slack"
    TEAMS = "teams"
    WEBHOOK = "webhook"
    SMS = "sms"
    CONSOLE = "console"  # For testing/debugging
    FILE = "file"        # Log to file


class NotificationPriority(Enum):
    """Notification priority levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class Notification:
    """Notification payload."""
    title: str
    message: str
    priority: NotificationPriority = NotificationPriority.MEDIUM
    source: str = ""
    category: str = ""
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    url: str = ""  # Action URL
    recipients: List[str] = field(default_factory=list)  # For email/SMS


@dataclass
class DeliveryResult:
    """Result of notification delivery."""
    success: bool
    channel: Channel
    timestamp: datetime = field(default_factory=datetime.now)
    message: str = ""
    error: Optional[str] = None
    response: Optional[Any] = None


class ChannelHandler(ABC):
    """Base class for channel handlers."""

    @abstractmethod
    def send(self, notification: Notification) -> DeliveryResult:
        """Send notification through this channel."""
        pass

    @abstractmethod
    def validate_config(self) -> bool:
        """Validate handler configuration."""
        pass


class ConsoleHandler(ChannelHandler):
    """
    Console handler for testing/debugging.

    Prints notifications to stdout.
    """

    def __init__(self, prefix: str = "[NOTIFICATION]"):
        self.prefix = prefix

    def send(self, notification: Notification) -> DeliveryResult:
        """Print notification to console."""
        print(f"{self.prefix} [{notification.priority.value.upper()}] {notification.title}")
        print(f"  {notification.message}")
        if notification.metadata:
            print(f"  Metadata: {notification.metadata}")

        return DeliveryResult(
            success=True,
            channel=Channel.CONSOLE,
            message="Printed to console"
        )

    def validate_config(self) -> bool:
        return True


class FileHandler(ChannelHandler):
    """
    File handler for logging notifications.

    Appends notifications to a log file.
    """

    def __init__(self, file_path: str, format: str = "json"):
        self.file_path = Path(file_path)
        self.format = format
        self._lock = threading.Lock()

    def send(self, notification: Notification) -> DeliveryResult:
        """Append notification to file."""
        try:
            with self._lock:
                with open(self.file_path, 'a', encoding='utf-8') as f:
                    if self.format == "json":
                        entry = {
                            "timestamp": notification.created_at.isoformat(),
                            "priority": notification.priority.value,
                            "title": notification.title,
                            "message": notification.message,
                            "source": notification.source,
                            "category": notification.category,
                            "metadata": notification.metadata
                        }
                        f.write(json.dumps(entry) + "\n")
                    else:
                        f.write(f"[{notification.created_at.isoformat()}] [{notification.priority.value.upper()}] {notification.title}: {notification.message}\n")

            return DeliveryResult(
                success=True,
                channel=Channel.FILE,
                message=f"Written to {self.file_path}"
            )

        except Exception as e:
            return DeliveryResult(
                success=False,
                channel=Channel.FILE,
                error=str(e)
            )

    def validate_config(self) -> bool:
        try:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            return True
        except Exception:
            return False


class SlackHandler(ChannelHandler):
    """
    Slack webhook handler.

    Sends notifications to Slack via incoming webhooks.
    """

    def __init__(
        self,
        webhook_url: str,
        username: str = "Alert Bot",
        icon_emoji: str = ":bell:",
        default_channel: str = ""
    ):
        self.webhook_url = webhook_url
        self.username = username
        self.icon_emoji = icon_emoji
        self.default_channel = default_channel

    def _get_color(self, priority: NotificationPriority) -> str:
        """Get Slack attachment color for priority."""
        colors = {
            NotificationPriority.CRITICAL: "#dc3545",  # Red
            NotificationPriority.HIGH: "#fd7e14",      # Orange
            NotificationPriority.MEDIUM: "#ffc107",    # Yellow
            NotificationPriority.LOW: "#17a2b8"        # Blue
        }
        return colors.get(priority, "#6c757d")

    def _build_payload(self, notification: Notification) -> Dict:
        """Build Slack message payload."""
        # Build attachment fields
        fields = []
        if notification.source:
            fields.append({"title": "Source", "value": notification.source, "short": True})
        if notification.category:
            fields.append({"title": "Category", "value": notification.category, "short": True})

        # Add custom metadata fields
        for key, value in notification.metadata.items():
            if isinstance(value, (str, int, float)):
                fields.append({"title": key, "value": str(value), "short": True})

        # Build attachment
        attachment = {
            "color": self._get_color(notification.priority),
            "title": notification.title,
            "text": notification.message,
            "fields": fields,
            "footer": f"Priority: {notification.priority.value.upper()}",
            "ts": int(notification.created_at.timestamp())
        }

        # Add action button if URL provided
        if notification.url:
            attachment["actions"] = [{
                "type": "button",
                "text": "View Details",
                "url": notification.url
            }]

        payload = {
            "username": self.username,
            "icon_emoji": self.icon_emoji,
            "attachments": [attachment]
        }

        if self.default_channel:
            payload["channel"] = self.default_channel

        return payload

    def send(self, notification: Notification) -> DeliveryResult:
        """Send notification to Slack."""
        try:
            import urllib.request
            import urllib.error

            payload = self._build_payload(notification)
            data = json.dumps(payload).encode('utf-8')

            req = urllib.request.Request(
                self.webhook_url,
                data=data,
                headers={'Content-Type': 'application/json'}
            )

            with urllib.request.urlopen(req, timeout=10) as response:
                return DeliveryResult(
                    success=True,
                    channel=Channel.SLACK,
                    message="Sent to Slack",
                    response=response.read().decode()
                )

        except urllib.error.HTTPError as e:
            return DeliveryResult(
                success=False,
                channel=Channel.SLACK,
                error=f"HTTP {e.code}: {e.reason}"
            )
        except Exception as e:
            return DeliveryResult(
                success=False,
                channel=Channel.SLACK,
                error=str(e)
            )

    def validate_config(self) -> bool:
        return bool(self.webhook_url and self.webhook_url.startswith('https://'))


class TeamsHandler(ChannelHandler):
    """
    Microsoft Teams webhook handler.

    Sends notifications to Teams via incoming webhooks.
    """

    def __init__(self, webhook_url: str, theme_color: str = "0076D7"):
        self.webhook_url = webhook_url
        self.theme_color = theme_color

    def _get_color(self, priority: NotificationPriority) -> str:
        """Get Teams card color for priority."""
        colors = {
            NotificationPriority.CRITICAL: "dc3545",
            NotificationPriority.HIGH: "fd7e14",
            NotificationPriority.MEDIUM: "ffc107",
            NotificationPriority.LOW: "17a2b8"
        }
        return colors.get(priority, self.theme_color)

    def _build_payload(self, notification: Notification) -> Dict:
        """Build Teams message card payload."""
        # Build facts
        facts = [
            {"name": "Priority", "value": notification.priority.value.upper()}
        ]
        if notification.source:
            facts.append({"name": "Source", "value": notification.source})
        if notification.category:
            facts.append({"name": "Category", "value": notification.category})

        # Add metadata
        for key, value in notification.metadata.items():
            if isinstance(value, (str, int, float)):
                facts.append({"name": key, "value": str(value)})

        # Build card
        card = {
            "@type": "MessageCard",
            "@context": "http://schema.org/extensions",
            "themeColor": self._get_color(notification.priority),
            "summary": notification.title,
            "sections": [{
                "activityTitle": notification.title,
                "text": notification.message,
                "facts": facts
            }]
        }

        # Add action button if URL provided
        if notification.url:
            card["potentialAction"] = [{
                "@type": "OpenUri",
                "name": "View Details",
                "targets": [{"os": "default", "uri": notification.url}]
            }]

        return card

    def send(self, notification: Notification) -> DeliveryResult:
        """Send notification to Teams."""
        try:
            import urllib.request
            import urllib.error

            payload = self._build_payload(notification)
            data = json.dumps(payload).encode('utf-8')

            req = urllib.request.Request(
                self.webhook_url,
                data=data,
                headers={'Content-Type': 'application/json'}
            )

            with urllib.request.urlopen(req, timeout=10) as response:
                return DeliveryResult(
                    success=True,
                    channel=Channel.TEAMS,
                    message="Sent to Teams",
                    response=response.read().decode()
                )

        except urllib.error.HTTPError as e:
            return DeliveryResult(
                success=False,
                channel=Channel.TEAMS,
                error=f"HTTP {e.code}: {e.reason}"
            )
        except Exception as e:
            return DeliveryResult(
                success=False,
                channel=Channel.TEAMS,
                error=str(e)
            )

    def validate_config(self) -> bool:
        return bool(self.webhook_url and self.webhook_url.startswith('https://'))


class WebhookHandler(ChannelHandler):
    """
    Generic webhook handler.

    Sends notifications to any HTTP endpoint.
    """

    def __init__(
        self,
        url: str,
        method: str = "POST",
        headers: Optional[Dict[str, str]] = None,
        transform: Optional[Callable[[Notification], Dict]] = None
    ):
        self.url = url
        self.method = method
        self.headers = headers or {"Content-Type": "application/json"}
        self.transform = transform

    def _build_payload(self, notification: Notification) -> Dict:
        """Build webhook payload."""
        if self.transform:
            return self.transform(notification)

        return {
            "title": notification.title,
            "message": notification.message,
            "priority": notification.priority.value,
            "source": notification.source,
            "category": notification.category,
            "tags": notification.tags,
            "metadata": notification.metadata,
            "timestamp": notification.created_at.isoformat(),
            "url": notification.url
        }

    def send(self, notification: Notification) -> DeliveryResult:
        """Send notification to webhook."""
        try:
            import urllib.request
            import urllib.error

            payload = self._build_payload(notification)
            data = json.dumps(payload).encode('utf-8')

            req = urllib.request.Request(
                self.url,
                data=data,
                headers=self.headers,
                method=self.method
            )

            with urllib.request.urlopen(req, timeout=30) as response:
                return DeliveryResult(
                    success=True,
                    channel=Channel.WEBHOOK,
                    message=f"Sent to {self.url}",
                    response=response.read().decode()
                )

        except urllib.error.HTTPError as e:
            return DeliveryResult(
                success=False,
                channel=Channel.WEBHOOK,
                error=f"HTTP {e.code}: {e.reason}"
            )
        except Exception as e:
            return DeliveryResult(
                success=False,
                channel=Channel.WEBHOOK,
                error=str(e)
            )

    def validate_config(self) -> bool:
        return bool(self.url)


@dataclass
class RoutingRule:
    """
    Routing rule for notifications.

    Defines which channels to use based on notification attributes.
    """
    name: str = ""
    condition: Callable[[Notification], bool] = lambda n: True
    channels: List[Channel] = field(default_factory=list)
    immediate: bool = False  # Bypass batching
    stop_on_match: bool = False  # Don't check further rules
    transform: Optional[Callable[[Notification], Notification]] = None  # Modify notification


class NotificationRouter:
    """
    Route notifications to appropriate channels.

    Features:
    - Multiple channel support (Email, Slack, Teams, Webhook)
    - Routing rules with conditions
    - Fallback channels
    - Delivery tracking
    - Retry logic

    Example:
    ```python
    router = NotificationRouter()

    # Add channels
    router.add_channel(Channel.SLACK, SlackHandler(webhook_url))
    router.add_channel(Channel.CONSOLE, ConsoleHandler())

    # Add rules
    router.add_rule(RoutingRule(
        name="critical_alerts",
        condition=lambda n: n.priority == NotificationPriority.CRITICAL,
        channels=[Channel.SLACK, Channel.CONSOLE],
        immediate=True
    ))

    # Default channels for anything not matching rules
    router.set_default_channels([Channel.CONSOLE])

    # Send
    result = router.send(Notification(title="Test", message="Hello"))
    ```
    """

    def __init__(
        self,
        default_channels: Optional[List[Channel]] = None,
        retry_count: int = 3,
        retry_delay_seconds: float = 1.0
    ):
        """
        Initialize notification router.

        Args:
            default_channels: Channels to use when no rules match
            retry_count: Number of retries on failure
            retry_delay_seconds: Delay between retries
        """
        self.default_channels = default_channels or []
        self.retry_count = retry_count
        self.retry_delay = retry_delay_seconds

        self._handlers: Dict[Channel, ChannelHandler] = {}
        self._rules: List[RoutingRule] = []
        self._delivery_history: List[DeliveryResult] = []
        self._lock = threading.Lock()

    def add_channel(self, channel: Channel, handler: ChannelHandler) -> 'NotificationRouter':
        """
        Register a channel handler.

        Args:
            channel: Channel type
            handler: Channel handler instance

        Returns:
            Self for chaining
        """
        if not handler.validate_config():
            raise ValueError(f"Invalid configuration for {channel.value} handler")

        self._handlers[channel] = handler
        return self

    def add_rule(self, rule: RoutingRule) -> 'NotificationRouter':
        """
        Add a routing rule.

        Args:
            rule: Routing rule to add

        Returns:
            Self for chaining
        """
        self._rules.append(rule)
        return self

    def set_default_channels(self, channels: List[Channel]) -> 'NotificationRouter':
        """Set default channels for unmatched notifications."""
        self.default_channels = channels
        return self

    def get_channels_for(self, notification: Notification) -> List[Channel]:
        """
        Determine which channels to use for a notification.

        Args:
            notification: Notification to route

        Returns:
            List of channels to send to
        """
        matched_channels = set()

        for rule in self._rules:
            try:
                if rule.condition(notification):
                    matched_channels.update(rule.channels)

                    if rule.stop_on_match:
                        break
            except Exception:
                continue

        if not matched_channels:
            matched_channels = set(self.default_channels)

        # Filter to configured channels
        return [c for c in matched_channels if c in self._handlers]

    def send(
        self,
        notification: Notification,
        channels: Optional[List[Channel]] = None
    ) -> List[DeliveryResult]:
        """
        Send notification to appropriate channels.

        Args:
            notification: Notification to send
            channels: Override channels (skip routing rules)

        Returns:
            List of delivery results
        """
        # Determine channels
        target_channels = channels or self.get_channels_for(notification)

        if not target_channels:
            return [DeliveryResult(
                success=False,
                channel=Channel.CONSOLE,
                error="No channels configured for this notification"
            )]

        results = []

        for channel in target_channels:
            handler = self._handlers.get(channel)
            if not handler:
                continue

            # Apply transforms from matching rules
            transformed = notification
            for rule in self._rules:
                if rule.transform and rule.condition(notification):
                    try:
                        transformed = rule.transform(transformed)
                    except Exception:
                        pass

            # Send with retry
            result = self._send_with_retry(handler, transformed, channel)
            results.append(result)

            with self._lock:
                self._delivery_history.append(result)

        return results

    def _send_with_retry(
        self,
        handler: ChannelHandler,
        notification: Notification,
        channel: Channel
    ) -> DeliveryResult:
        """Send with retry logic."""
        import time

        last_error = None

        for attempt in range(self.retry_count + 1):
            try:
                result = handler.send(notification)
                if result.success:
                    return result
                last_error = result.error
            except Exception as e:
                last_error = str(e)

            if attempt < self.retry_count:
                time.sleep(self.retry_delay * (attempt + 1))

        return DeliveryResult(
            success=False,
            channel=channel,
            error=f"Failed after {self.retry_count + 1} attempts: {last_error}"
        )

    def send_batch(
        self,
        notifications: List[Notification]
    ) -> Dict[str, List[DeliveryResult]]:
        """
        Send multiple notifications.

        Args:
            notifications: List of notifications

        Returns:
            Dict mapping notification title to results
        """
        results = {}

        for notification in notifications:
            results[notification.title] = self.send(notification)

        return results

    def get_delivery_history(
        self,
        limit: int = 100,
        channel: Optional[Channel] = None,
        success_only: bool = False
    ) -> List[DeliveryResult]:
        """
        Get delivery history.

        Args:
            limit: Max results to return
            channel: Filter by channel
            success_only: Only successful deliveries

        Returns:
            List of delivery results
        """
        with self._lock:
            history = self._delivery_history.copy()

        # Filter
        if channel:
            history = [r for r in history if r.channel == channel]
        if success_only:
            history = [r for r in history if r.success]

        # Sort by timestamp descending
        history.sort(key=lambda r: r.timestamp, reverse=True)

        return history[:limit]

    def get_stats(self) -> Dict[str, Any]:
        """Get router statistics."""
        with self._lock:
            history = self._delivery_history.copy()

        stats = {
            "total_sent": len(history),
            "successful": sum(1 for r in history if r.success),
            "failed": sum(1 for r in history if not r.success),
            "by_channel": {},
            "configured_channels": list(self._handlers.keys()),
            "rules_count": len(self._rules)
        }

        for channel in Channel:
            channel_results = [r for r in history if r.channel == channel]
            if channel_results:
                stats["by_channel"][channel.value] = {
                    "total": len(channel_results),
                    "successful": sum(1 for r in channel_results if r.success),
                    "failed": sum(1 for r in channel_results if not r.success)
                }

        return stats

    def clear_history(self):
        """Clear delivery history."""
        with self._lock:
            self._delivery_history.clear()


# Factory functions

def create_slack_router(webhook_url: str, fallback_to_console: bool = True) -> NotificationRouter:
    """
    Create a router for Slack notifications.

    Args:
        webhook_url: Slack webhook URL
        fallback_to_console: Also log to console

    Returns:
        Configured NotificationRouter
    """
    router = NotificationRouter()
    router.add_channel(Channel.SLACK, SlackHandler(webhook_url))

    if fallback_to_console:
        router.add_channel(Channel.CONSOLE, ConsoleHandler())

    router.set_default_channels([Channel.SLACK])

    return router


def create_teams_router(webhook_url: str) -> NotificationRouter:
    """
    Create a router for Teams notifications.

    Args:
        webhook_url: Teams webhook URL

    Returns:
        Configured NotificationRouter
    """
    router = NotificationRouter()
    router.add_channel(Channel.TEAMS, TeamsHandler(webhook_url))
    router.set_default_channels([Channel.TEAMS])

    return router


def create_multi_channel_router(
    slack_url: Optional[str] = None,
    teams_url: Optional[str] = None,
    log_file: Optional[str] = None
) -> NotificationRouter:
    """
    Create a router with multiple channels.

    Args:
        slack_url: Slack webhook URL
        teams_url: Teams webhook URL
        log_file: Path to log file

    Returns:
        Configured NotificationRouter
    """
    router = NotificationRouter()
    default_channels = []

    if slack_url:
        router.add_channel(Channel.SLACK, SlackHandler(slack_url))
        default_channels.append(Channel.SLACK)

    if teams_url:
        router.add_channel(Channel.TEAMS, TeamsHandler(teams_url))
        default_channels.append(Channel.TEAMS)

    if log_file:
        router.add_channel(Channel.FILE, FileHandler(log_file))
        default_channels.append(Channel.FILE)

    # Always add console
    router.add_channel(Channel.CONSOLE, ConsoleHandler())

    router.set_default_channels(default_channels or [Channel.CONSOLE])

    # Add critical rule to notify all channels
    if len(default_channels) > 1:
        router.add_rule(RoutingRule(
            name="critical_all_channels",
            condition=lambda n: n.priority == NotificationPriority.CRITICAL,
            channels=default_channels,
            immediate=True
        ))

    return router


def create_development_router() -> NotificationRouter:
    """
    Create a router for development (console only).

    Returns:
        Configured NotificationRouter
    """
    router = NotificationRouter()
    router.add_channel(Channel.CONSOLE, ConsoleHandler("[DEV]"))
    router.set_default_channels([Channel.CONSOLE])

    return router
