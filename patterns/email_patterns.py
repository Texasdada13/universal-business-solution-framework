"""
Universal Email Alert Patterns
Send professional email notifications for any event

Based on: NYPD Procurement Tracker
"""

import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime


class EmailAlerter:
    """
    Send email alerts with HTML and plain-text formatting

    Usage:
        alerter = EmailAlerter()
        alerter.send_alert(
            subject="Alert: New Items Detected",
            to_email="recipient@example.com",
            items=[...],
            template_type="new_items"
        )
    """

    def __init__(self, smtp_server=None, smtp_port=None, username=None, password=None):
        """
        Initialize email alerter

        Args:
            smtp_server (str): SMTP server address
            smtp_port (int): SMTP port
            username (str): Email username
            password (str): Email password (or app password)
        """
        self.smtp_server = smtp_server or os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = smtp_port or int(os.getenv('SMTP_PORT', '587'))
        self.username = username or os.getenv('EMAIL_USERNAME')
        self.password = password or os.getenv('EMAIL_PASSWORD')

    def send_email(self, subject, to_email, text_content, html_content, from_email=None):
        """
        Send email with both HTML and plain-text versions

        Args:
            subject (str): Email subject
            to_email (str): Recipient email
            text_content (str): Plain-text version
            html_content (str): HTML version
            from_email (str): Sender email (uses username if not provided)

        Returns:
            bool: True if sent successfully, False otherwise
        """
        if from_email is None:
            from_email = self.username

        if not self.username or not self.password:
            print("WARNING: Email credentials not configured")
            print("Set EMAIL_USERNAME and EMAIL_PASSWORD environment variables")
            return False

        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = from_email
            msg['To'] = to_email

            # Attach both versions
            part1 = MIMEText(text_content, 'plain')
            part2 = MIMEText(html_content, 'html')
            msg.attach(part1)
            msg.attach(part2)

            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)

            print(f"[OK] Email sent to {to_email}")
            return True

        except Exception as e:
            print(f"[ERROR] Failed to send email: {e}")
            return False

    def create_new_items_email(self, items, title="New Items Detected", item_name_field='name'):
        """
        Create email for new items alert

        Args:
            items (list): List of new items
            title (str): Email title
            item_name_field (str): Field to use for item name

        Returns:
            tuple: (text_content, html_content)
        """
        # Plain text version
        text_lines = []
        text_lines.append(f"[ALERT] {len(items)} {title}")
        text_lines.append("")
        text_lines.append(f"Detected at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        text_lines.append("")

        for i, item in enumerate(items, 1):
            if item_name_field in item:
                text_lines.append(f"{i}. {item[item_name_field]}")
            else:
                text_lines.append(f"{i}. {item}")

        text_lines.append("")
        text_lines.append("ACTION REQUIRED: Review these items immediately")

        text_content = "\n".join(text_lines)

        # HTML version
        html_content = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; }}
                .header {{ background-color: #003366; color: white; padding: 20px; text-align: center; }}
                .alert {{ background-color: #ff6b6b; color: white; padding: 15px; margin: 20px 0; border-radius: 5px; }}
                .item {{ background-color: #f8f9fa; padding: 15px; margin: 10px 0; border-left: 4px solid #0066cc; }}
                .action {{ background-color: #28a745; color: white; padding: 15px; margin: 20px 0; text-align: center; border-radius: 5px; }}
                .footer {{ background-color: #f8f9fa; padding: 15px; margin-top: 20px; text-align: center; font-size: 12px; color: #666; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{title}</h1>
            </div>

            <div class="alert">
                <strong>{len(items)} New Items Detected</strong><br>
                Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            </div>

            <h2>New Items:</h2>
        """

        for i, item in enumerate(items, 1):
            item_text = item.get(item_name_field, str(item))
            html_content += f"""
            <div class="item">
                <strong>{i}. {item_text}</strong>
            </div>
            """

        html_content += """
            <div class="action">
                <h3>ACTION REQUIRED</h3>
                <p>Review these items immediately</p>
            </div>

            <div class="footer">
                <p>This is an automated alert</p>
            </div>
        </body>
        </html>
        """

        return text_content, html_content

    def send_new_items_alert(self, items, to_email, subject=None, title="New Items", item_name_field='name'):
        """
        Send alert for new items

        Args:
            items (list): New items to alert about
            to_email (str): Recipient email
            subject (str): Email subject
            title (str): Alert title
            item_name_field (str): Field to use for item name

        Returns:
            bool: True if sent successfully
        """
        if not items:
            print("No items to send - skipping email")
            return False

        subject = subject or f"[ALERT] {len(items)} {title} Detected"
        text_content, html_content = self.create_new_items_email(items, title, item_name_field)

        return self.send_email(subject, to_email, text_content, html_content)

    def create_summary_email(self, stats, title="Summary Report"):
        """
        Create summary email with statistics

        Args:
            stats (dict): Statistics to include
            title (str): Email title

        Returns:
            tuple: (text_content, html_content)
        """
        # Plain text version
        text_lines = []
        text_lines.append(title)
        text_lines.append("=" * 70)
        for key, value in stats.items():
            text_lines.append(f"{key}: {value}")
        text_lines.append("=" * 70)

        text_content = "\n".join(text_lines)

        # HTML version
        html_content = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; }}
                .header {{ background-color: #003366; color: white; padding: 20px; text-align: center; }}
                .stats {{ background-color: #f8f9fa; padding: 20px; margin: 20px 0; }}
                .stat-row {{ padding: 10px; border-bottom: 1px solid #ddd; }}
                .stat-label {{ font-weight: bold; color: #003366; }}
                .stat-value {{ float: right; color: #666; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{title}</h1>
            </div>
            <div class="stats">
        """

        for key, value in stats.items():
            html_content += f"""
                <div class="stat-row">
                    <span class="stat-label">{key}</span>
                    <span class="stat-value">{value}</span>
                    <div style="clear:both;"></div>
                </div>
            """

        html_content += """
            </div>
        </body>
        </html>
        """

        return text_content, html_content

    def send_summary_report(self, stats, to_email, subject=None, title="Summary Report"):
        """
        Send summary report email

        Args:
            stats (dict): Statistics to include
            to_email (str): Recipient email
            subject (str): Email subject
            title (str): Report title

        Returns:
            bool: True if sent successfully
        """
        subject = subject or f"{title} - {datetime.now().strftime('%Y-%m-%d')}"
        text_content, html_content = self.create_summary_email(stats, title)

        return self.send_email(subject, to_email, text_content, html_content)


# Example usage
if __name__ == "__main__":
    # Initialize alerter
    alerter = EmailAlerter()

    # Example 1: Send new items alert
    new_items = [
        {'name': 'Item A', 'value': '100'},
        {'name': 'Item B', 'value': '200'},
        {'name': 'Item C', 'value': '300'},
    ]

    alerter.send_new_items_alert(
        items=new_items,
        to_email="recipient@example.com",
        title="New Opportunities",
        item_name_field='name'
    )

    # Example 2: Send summary report
    stats = {
        'Total Items': 150,
        'New Today': 5,
        'Removed Today': 2,
        'Last Check': '2025-12-18 14:30:00'
    }

    alerter.send_summary_report(
        stats=stats,
        to_email="recipient@example.com",
        title="Daily Summary Report"
    )
