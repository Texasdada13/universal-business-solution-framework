"""
Email Digest Builder - Universal Business Solution Framework

Professional HTML email templates for digests, alerts, and summaries.
Supports multiple themes, responsive design, and plain-text fallback.

Example:
```python
from patterns.notifications import EmailDigestBuilder, DigestTheme

# Create daily digest
digest = EmailDigestBuilder(
    title="Daily Contract Alerts",
    theme=DigestTheme.PROFESSIONAL
)

digest.add_summary({
    "New Alerts": 12,
    "Critical": 3,
    "Resolved": 5
})

digest.add_alert_section("Critical Alerts", critical_alerts)
digest.add_table("Expiring Contracts", contracts_data)
digest.add_action_button("View Dashboard", "https://app.example.com")

html, text = digest.build()
```
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import html as html_escape


class DigestTheme(Enum):
    """Email theme presets."""
    PROFESSIONAL = "professional"  # Blue corporate theme
    ALERT = "alert"                # Red/orange for urgent notifications
    SUCCESS = "success"            # Green for positive reports
    MINIMAL = "minimal"            # Clean, minimal styling
    DARK = "dark"                  # Dark theme


class AlertSeverity(Enum):
    """Alert severity levels for styling."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class ThemeConfig:
    """Theme color configuration."""
    primary_color: str
    secondary_color: str
    background_color: str
    text_color: str
    header_bg: str
    border_color: str
    success_color: str = "#28a745"
    warning_color: str = "#ffc107"
    danger_color: str = "#dc3545"
    info_color: str = "#17a2b8"


# Predefined themes
THEMES: Dict[DigestTheme, ThemeConfig] = {
    DigestTheme.PROFESSIONAL: ThemeConfig(
        primary_color="#0066cc",
        secondary_color="#004499",
        background_color="#f8f9fa",
        text_color="#333333",
        header_bg="#0066cc",
        border_color="#dee2e6"
    ),
    DigestTheme.ALERT: ThemeConfig(
        primary_color="#dc3545",
        secondary_color="#c82333",
        background_color="#fff5f5",
        text_color="#333333",
        header_bg="#dc3545",
        border_color="#f5c6cb"
    ),
    DigestTheme.SUCCESS: ThemeConfig(
        primary_color="#28a745",
        secondary_color="#1e7e34",
        background_color="#f0fff4",
        text_color="#333333",
        header_bg="#28a745",
        border_color="#c3e6cb"
    ),
    DigestTheme.MINIMAL: ThemeConfig(
        primary_color="#333333",
        secondary_color="#666666",
        background_color="#ffffff",
        text_color="#333333",
        header_bg="#f8f9fa",
        border_color="#e9ecef"
    ),
    DigestTheme.DARK: ThemeConfig(
        primary_color="#6c757d",
        secondary_color="#495057",
        background_color="#212529",
        text_color="#f8f9fa",
        header_bg="#343a40",
        border_color="#495057"
    ),
}


@dataclass
class DigestSection:
    """A section in the email digest."""
    section_type: str
    title: str
    content: Any
    severity: Optional[AlertSeverity] = None


class EmailDigestBuilder:
    """
    Build professional HTML email digests.

    Features:
    - Multiple theme presets
    - Responsive design for mobile
    - Summary cards with metrics
    - Alert sections with severity styling
    - Data tables
    - Action buttons
    - Plain-text fallback generation

    Example:
    ```python
    digest = EmailDigestBuilder("Weekly Report", theme=DigestTheme.PROFESSIONAL)
    digest.add_summary({"Total": 100, "Active": 85})
    digest.add_section("Details", "Here are the details...")
    html, text = digest.build()
    ```
    """

    def __init__(
        self,
        title: str,
        subtitle: str = "",
        theme: DigestTheme = DigestTheme.PROFESSIONAL,
        logo_url: Optional[str] = None,
        footer_text: str = "",
        custom_theme: Optional[ThemeConfig] = None
    ):
        self.title = title
        self.subtitle = subtitle
        self.theme = theme
        self.theme_config = custom_theme or THEMES[theme]
        self.logo_url = logo_url
        self.footer_text = footer_text or f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        self.sections: List[DigestSection] = []
        self.preheader: str = ""  # Preview text in email clients

    def set_preheader(self, text: str) -> 'EmailDigestBuilder':
        """Set email preview text (shown in inbox before opening)."""
        self.preheader = text
        return self

    def add_summary(
        self,
        metrics: Dict[str, Any],
        title: str = "Summary"
    ) -> 'EmailDigestBuilder':
        """
        Add a summary section with metric cards.

        Args:
            metrics: Dict of metric name -> value
            title: Section title
        """
        self.sections.append(DigestSection(
            section_type="summary",
            title=title,
            content=metrics
        ))
        return self

    def add_section(
        self,
        title: str,
        content: str,
        severity: Optional[AlertSeverity] = None
    ) -> 'EmailDigestBuilder':
        """Add a text section."""
        self.sections.append(DigestSection(
            section_type="text",
            title=title,
            content=content,
            severity=severity
        ))
        return self

    def add_alert_section(
        self,
        title: str,
        alerts: List[Dict[str, Any]],
        severity: AlertSeverity = AlertSeverity.MEDIUM
    ) -> 'EmailDigestBuilder':
        """
        Add an alerts section with styled alert items.

        Args:
            title: Section title
            alerts: List of alert dicts with 'title', 'message', optional 'severity'
            severity: Default severity for styling
        """
        self.sections.append(DigestSection(
            section_type="alerts",
            title=title,
            content=alerts,
            severity=severity
        ))
        return self

    def add_table(
        self,
        title: str,
        data: List[Dict[str, Any]],
        columns: Optional[List[str]] = None,
        max_rows: int = 10
    ) -> 'EmailDigestBuilder':
        """
        Add a data table section.

        Args:
            title: Section title
            data: List of row dicts
            columns: Column order (defaults to all keys)
            max_rows: Maximum rows to show
        """
        self.sections.append(DigestSection(
            section_type="table",
            title=title,
            content={
                "data": data[:max_rows],
                "columns": columns,
                "total_rows": len(data),
                "truncated": len(data) > max_rows
            }
        ))
        return self

    def add_list(
        self,
        title: str,
        items: List[str],
        ordered: bool = False
    ) -> 'EmailDigestBuilder':
        """Add a bullet or numbered list section."""
        self.sections.append(DigestSection(
            section_type="list",
            title=title,
            content={"items": items, "ordered": ordered}
        ))
        return self

    def add_action_button(
        self,
        text: str,
        url: str,
        secondary: bool = False
    ) -> 'EmailDigestBuilder':
        """Add a call-to-action button."""
        self.sections.append(DigestSection(
            section_type="button",
            title=text,
            content={"url": url, "secondary": secondary}
        ))
        return self

    def add_divider(self) -> 'EmailDigestBuilder':
        """Add a horizontal divider."""
        self.sections.append(DigestSection(
            section_type="divider",
            title="",
            content=None
        ))
        return self

    def add_key_value(
        self,
        title: str,
        data: Dict[str, Any]
    ) -> 'EmailDigestBuilder':
        """Add a key-value pairs section."""
        self.sections.append(DigestSection(
            section_type="key_value",
            title=title,
            content=data
        ))
        return self

    def _get_severity_color(self, severity: AlertSeverity) -> str:
        """Get color for severity level."""
        colors = {
            AlertSeverity.CRITICAL: self.theme_config.danger_color,
            AlertSeverity.HIGH: "#fd7e14",  # Orange
            AlertSeverity.MEDIUM: self.theme_config.warning_color,
            AlertSeverity.LOW: self.theme_config.info_color,
            AlertSeverity.INFO: "#6c757d",  # Gray
        }
        return colors.get(severity, self.theme_config.primary_color)

    def _escape(self, text: Any) -> str:
        """HTML escape text."""
        return html_escape.escape(str(text))

    def _render_summary(self, section: DigestSection) -> str:
        """Render summary metrics as cards."""
        metrics = section.content
        cards_html = ""

        for name, value in metrics.items():
            cards_html += f'''
            <td style="padding: 10px; text-align: center; background: {self.theme_config.background_color}; border-radius: 8px; border: 1px solid {self.theme_config.border_color};">
                <div style="font-size: 24px; font-weight: bold; color: {self.theme_config.primary_color};">{self._escape(value)}</div>
                <div style="font-size: 12px; color: #666; margin-top: 4px;">{self._escape(name)}</div>
            </td>
            '''

        return f'''
        <table width="100%" cellpadding="0" cellspacing="8" style="margin: 16px 0;">
            <tr>{cards_html}</tr>
        </table>
        '''

    def _render_alerts(self, section: DigestSection) -> str:
        """Render alert items."""
        alerts = section.content
        html = ""

        for alert in alerts:
            alert_severity = AlertSeverity(alert.get('severity', section.severity.value)) if alert.get('severity') else section.severity
            color = self._get_severity_color(alert_severity)

            html += f'''
            <div style="padding: 12px; margin: 8px 0; border-left: 4px solid {color}; background: {self.theme_config.background_color}; border-radius: 0 4px 4px 0;">
                <div style="font-weight: bold; color: {self.theme_config.text_color};">{self._escape(alert.get('title', ''))}</div>
                <div style="font-size: 14px; color: #666; margin-top: 4px;">{self._escape(alert.get('message', ''))}</div>
            </div>
            '''

        return html

    def _render_table(self, section: DigestSection) -> str:
        """Render data table."""
        data = section.content["data"]
        columns = section.content["columns"]

        if not data:
            return "<p>No data available.</p>"

        # Determine columns
        if not columns:
            columns = list(data[0].keys())

        # Build header
        header_cells = "".join(
            f'<th style="padding: 10px; text-align: left; background: {self.theme_config.header_bg}; color: white; font-size: 12px; text-transform: uppercase;">{self._escape(col)}</th>'
            for col in columns
        )

        # Build rows
        rows_html = ""
        for i, row in enumerate(data):
            bg = self.theme_config.background_color if i % 2 == 0 else "#ffffff"
            cells = "".join(
                f'<td style="padding: 10px; border-bottom: 1px solid {self.theme_config.border_color};">{self._escape(row.get(col, ""))}</td>'
                for col in columns
            )
            rows_html += f'<tr style="background: {bg};">{cells}</tr>'

        # Truncation notice
        footer = ""
        if section.content.get("truncated"):
            total = section.content["total_rows"]
            shown = len(data)
            footer = f'<tr><td colspan="{len(columns)}" style="padding: 8px; text-align: center; color: #666; font-size: 12px;">Showing {shown} of {total} rows</td></tr>'

        return f'''
        <table width="100%" cellpadding="0" cellspacing="0" style="border-collapse: collapse; margin: 16px 0; border: 1px solid {self.theme_config.border_color}; border-radius: 4px;">
            <thead><tr>{header_cells}</tr></thead>
            <tbody>{rows_html}{footer}</tbody>
        </table>
        '''

    def _render_list(self, section: DigestSection) -> str:
        """Render list items."""
        items = section.content["items"]
        ordered = section.content["ordered"]

        tag = "ol" if ordered else "ul"
        items_html = "".join(f"<li style='margin: 4px 0;'>{self._escape(item)}</li>" for item in items)

        return f"<{tag} style='margin: 8px 0; padding-left: 24px;'>{items_html}</{tag}>"

    def _render_button(self, section: DigestSection) -> str:
        """Render action button."""
        url = section.content["url"]
        secondary = section.content["secondary"]

        bg_color = self.theme_config.secondary_color if secondary else self.theme_config.primary_color

        return f'''
        <table cellpadding="0" cellspacing="0" style="margin: 16px 0;">
            <tr>
                <td style="background: {bg_color}; border-radius: 4px;">
                    <a href="{self._escape(url)}" style="display: inline-block; padding: 12px 24px; color: white; text-decoration: none; font-weight: bold;">{self._escape(section.title)}</a>
                </td>
            </tr>
        </table>
        '''

    def _render_key_value(self, section: DigestSection) -> str:
        """Render key-value pairs."""
        data = section.content
        rows = ""

        for key, value in data.items():
            rows += f'''
            <tr>
                <td style="padding: 8px; font-weight: bold; color: #666; width: 40%;">{self._escape(key)}</td>
                <td style="padding: 8px;">{self._escape(value)}</td>
            </tr>
            '''

        return f'''
        <table width="100%" style="margin: 8px 0; border: 1px solid {self.theme_config.border_color}; border-radius: 4px;">
            {rows}
        </table>
        '''

    def _render_section(self, section: DigestSection) -> str:
        """Render a single section."""
        # Section title
        title_html = ""
        if section.title and section.section_type not in ("button", "divider"):
            title_html = f'<h2 style="margin: 24px 0 12px 0; font-size: 18px; color: {self.theme_config.text_color}; border-bottom: 2px solid {self.theme_config.primary_color}; padding-bottom: 8px;">{self._escape(section.title)}</h2>'

        # Section content
        content_html = ""
        if section.section_type == "summary":
            content_html = self._render_summary(section)
        elif section.section_type == "text":
            content_html = f'<p style="margin: 8px 0; line-height: 1.6;">{self._escape(section.content)}</p>'
        elif section.section_type == "alerts":
            content_html = self._render_alerts(section)
        elif section.section_type == "table":
            content_html = self._render_table(section)
        elif section.section_type == "list":
            content_html = self._render_list(section)
        elif section.section_type == "button":
            content_html = self._render_button(section)
        elif section.section_type == "divider":
            content_html = f'<hr style="border: none; border-top: 1px solid {self.theme_config.border_color}; margin: 24px 0;" />'
        elif section.section_type == "key_value":
            content_html = self._render_key_value(section)

        return title_html + content_html

    def build(self) -> Tuple[str, str]:
        """
        Build the email digest.

        Returns:
            Tuple of (html_content, plain_text_content)
        """
        # Render all sections
        sections_html = "\n".join(self._render_section(s) for s in self.sections)

        # Logo
        logo_html = ""
        if self.logo_url:
            logo_html = f'<img src="{self._escape(self.logo_url)}" alt="Logo" style="max-height: 50px; margin-bottom: 16px;" />'

        # Preheader (hidden preview text)
        preheader_html = ""
        if self.preheader:
            preheader_html = f'<span style="display: none !important; visibility: hidden; mso-hide: all; font-size: 1px; color: #ffffff; line-height: 1px; max-height: 0px; max-width: 0px; opacity: 0; overflow: hidden;">{self._escape(self.preheader)}</span>'

        # Build full HTML
        html = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self._escape(self.title)}</title>
</head>
<body style="margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background: #f0f0f0;">
    {preheader_html}
    <table width="100%" cellpadding="0" cellspacing="0" style="background: #f0f0f0; padding: 20px 0;">
        <tr>
            <td align="center">
                <table width="600" cellpadding="0" cellspacing="0" style="background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                    <!-- Header -->
                    <tr>
                        <td style="background: {self.theme_config.header_bg}; padding: 24px; text-align: center;">
                            {logo_html}
                            <h1 style="margin: 0; color: white; font-size: 24px;">{self._escape(self.title)}</h1>
                            {f'<p style="margin: 8px 0 0 0; color: rgba(255,255,255,0.8); font-size: 14px;">{self._escape(self.subtitle)}</p>' if self.subtitle else ''}
                        </td>
                    </tr>
                    <!-- Content -->
                    <tr>
                        <td style="padding: 24px; color: {self.theme_config.text_color};">
                            {sections_html}
                        </td>
                    </tr>
                    <!-- Footer -->
                    <tr>
                        <td style="background: {self.theme_config.background_color}; padding: 16px; text-align: center; font-size: 12px; color: #666; border-top: 1px solid {self.theme_config.border_color};">
                            {self._escape(self.footer_text)}
                        </td>
                    </tr>
                </table>
            </td>
        </tr>
    </table>
</body>
</html>'''

        # Build plain text version
        text = self._build_plain_text()

        return html, text

    def _build_plain_text(self) -> str:
        """Build plain text version of the digest."""
        lines = [
            self.title.upper(),
            "=" * len(self.title),
            ""
        ]

        if self.subtitle:
            lines.extend([self.subtitle, ""])

        for section in self.sections:
            if section.section_type == "divider":
                lines.extend(["", "-" * 40, ""])
                continue

            if section.title:
                lines.extend(["", section.title, "-" * len(section.title)])

            if section.section_type == "summary":
                for name, value in section.content.items():
                    lines.append(f"  {name}: {value}")

            elif section.section_type == "text":
                lines.append(section.content)

            elif section.section_type == "alerts":
                for alert in section.content:
                    severity = alert.get('severity', section.severity.value if section.severity else 'info')
                    lines.append(f"  [{severity.upper()}] {alert.get('title', '')}")
                    if alert.get('message'):
                        lines.append(f"    {alert.get('message')}")

            elif section.section_type == "table":
                data = section.content["data"]
                columns = section.content["columns"] or (list(data[0].keys()) if data else [])

                # Header
                lines.append("  " + " | ".join(columns))
                lines.append("  " + "-+-".join("-" * len(c) for c in columns))

                # Rows
                for row in data:
                    lines.append("  " + " | ".join(str(row.get(c, ""))[:len(c)] for c in columns))

            elif section.section_type == "list":
                for i, item in enumerate(section.content["items"], 1):
                    prefix = f"{i}." if section.content["ordered"] else "*"
                    lines.append(f"  {prefix} {item}")

            elif section.section_type == "button":
                lines.append(f"  [{section.title}] {section.content['url']}")

            elif section.section_type == "key_value":
                for key, value in section.content.items():
                    lines.append(f"  {key}: {value}")

        lines.extend(["", "-" * 40, self.footer_text])

        return "\n".join(lines)

    def save(self, path: str, include_text: bool = True) -> Dict[str, Path]:
        """
        Save digest to file(s).

        Args:
            path: Base path (without extension)
            include_text: Also save plain text version

        Returns:
            Dict of format -> file path
        """
        html, text = self.build()
        paths = {}

        html_path = Path(f"{path}.html")
        html_path.write_text(html, encoding='utf-8')
        paths['html'] = html_path

        if include_text:
            text_path = Path(f"{path}.txt")
            text_path.write_text(text, encoding='utf-8')
            paths['text'] = text_path

        return paths


class AlertEmailBuilder:
    """
    Quick builder for single-alert notification emails.

    Example:
    ```python
    email = AlertEmailBuilder(
        title="Critical: Contract Expiring",
        message="Contract #12345 expires in 7 days.",
        severity=AlertSeverity.CRITICAL
    )
    email.add_details({"Contract": "#12345", "Vendor": "ABC Corp"})
    email.set_action("View Contract", "https://app.example.com/contract/12345")
    html, text = email.build()
    ```
    """

    def __init__(
        self,
        title: str,
        message: str,
        severity: AlertSeverity = AlertSeverity.MEDIUM,
        theme: DigestTheme = DigestTheme.ALERT
    ):
        self.builder = EmailDigestBuilder(title, theme=theme)
        self.message = message
        self.severity = severity
        self.details: Dict[str, Any] = {}
        self.action_text: Optional[str] = None
        self.action_url: Optional[str] = None

    def add_details(self, details: Dict[str, Any]) -> 'AlertEmailBuilder':
        """Add key-value details."""
        self.details.update(details)
        return self

    def set_action(self, text: str, url: str) -> 'AlertEmailBuilder':
        """Set call-to-action button."""
        self.action_text = text
        self.action_url = url
        return self

    def build(self) -> Tuple[str, str]:
        """Build the alert email."""
        # Add alert
        self.builder.add_alert_section(
            "Alert",
            [{"title": self.builder.title, "message": self.message, "severity": self.severity.value}],
            self.severity
        )

        # Add details
        if self.details:
            self.builder.add_key_value("Details", self.details)

        # Add action
        if self.action_text and self.action_url:
            self.builder.add_action_button(self.action_text, self.action_url)

        return self.builder.build()


# Factory functions

def create_daily_digest(
    title: str,
    metrics: Dict[str, Any],
    items: List[Dict[str, Any]],
    dashboard_url: str = ""
) -> EmailDigestBuilder:
    """
    Create a standard daily digest email.

    Args:
        title: Email title
        metrics: Summary metrics to display
        items: Data items for table
        dashboard_url: Optional link to dashboard

    Returns:
        Configured EmailDigestBuilder
    """
    digest = EmailDigestBuilder(title, theme=DigestTheme.PROFESSIONAL)
    digest.add_summary(metrics)

    if items:
        digest.add_table("Recent Activity", items)

    if dashboard_url:
        digest.add_action_button("View Dashboard", dashboard_url)

    return digest


def create_alert_digest(
    title: str,
    alerts: List[Dict[str, Any]],
    dashboard_url: str = ""
) -> EmailDigestBuilder:
    """
    Create an alert digest email.

    Args:
        title: Email title
        alerts: List of alert dicts with 'title', 'message', 'severity'
        dashboard_url: Optional link to dashboard

    Returns:
        Configured EmailDigestBuilder
    """
    # Determine theme based on max severity
    severities = [a.get('severity', 'medium') for a in alerts]
    if 'critical' in severities:
        theme = DigestTheme.ALERT
    elif 'high' in severities:
        theme = DigestTheme.ALERT
    else:
        theme = DigestTheme.PROFESSIONAL

    digest = EmailDigestBuilder(title, theme=theme)

    # Group by severity
    critical = [a for a in alerts if a.get('severity') == 'critical']
    high = [a for a in alerts if a.get('severity') == 'high']
    medium = [a for a in alerts if a.get('severity') == 'medium']
    low = [a for a in alerts if a.get('severity') in ('low', 'info') or not a.get('severity')]

    # Summary
    digest.add_summary({
        "Critical": len(critical),
        "High": len(high),
        "Medium": len(medium),
        "Low": len(low)
    })

    # Add sections
    if critical:
        digest.add_alert_section("Critical Alerts", critical, AlertSeverity.CRITICAL)
    if high:
        digest.add_alert_section("High Priority", high, AlertSeverity.HIGH)
    if medium:
        digest.add_alert_section("Medium Priority", medium, AlertSeverity.MEDIUM)
    if low:
        digest.add_alert_section("Low Priority", low, AlertSeverity.LOW)

    if dashboard_url:
        digest.add_action_button("View All Alerts", dashboard_url)

    return digest


def create_report_email(
    title: str,
    subtitle: str,
    sections: List[Dict[str, Any]],
    footer: str = ""
) -> EmailDigestBuilder:
    """
    Create a report-style email with custom sections.

    Args:
        title: Email title
        subtitle: Email subtitle
        sections: List of section dicts with 'type', 'title', 'content'
        footer: Custom footer text

    Returns:
        Configured EmailDigestBuilder
    """
    digest = EmailDigestBuilder(title, subtitle, theme=DigestTheme.PROFESSIONAL)

    if footer:
        digest.footer_text = footer

    for section in sections:
        section_type = section.get('type', 'text')
        section_title = section.get('title', '')
        content = section.get('content')

        if section_type == 'text':
            digest.add_section(section_title, content)
        elif section_type == 'metrics':
            digest.add_summary(content, section_title)
        elif section_type == 'table':
            digest.add_table(section_title, content)
        elif section_type == 'list':
            digest.add_list(section_title, content)
        elif section_type == 'key_value':
            digest.add_key_value(section_title, content)
        elif section_type == 'button':
            digest.add_action_button(section_title, content)
        elif section_type == 'divider':
            digest.add_divider()

    return digest
