"""
Universal Web Scraping Patterns
Reusable scraping components for any project

Based on: NYPD Procurement Tracker
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
from datetime import datetime
from pathlib import Path


class UniversalScraper:
    """
    Base web scraper with common functionality

    Usage:
        scraper = UniversalScraper(url="https://example.com/data")
        html = scraper.fetch_html()
        soup = scraper.parse_html(html)
        data = scraper.parse_table(soup)
        scraper.save_to_csv(data, "output.csv")
    """

    def __init__(self, url, headers=None, output_dir=None):
        self.url = url
        self.headers = headers or self._default_headers()
        self.output_dir = Path(output_dir) if output_dir else Path('output')
        self.output_dir.mkdir(exist_ok=True)
        self.data = []

    def _default_headers(self):
        """Default user agent to avoid blocks"""
        return {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def fetch_html(self, url=None):
        """
        Fetch HTML from URL with error handling

        Args:
            url (str): URL to fetch (uses self.url if not provided)

        Returns:
            str: HTML content or None if failed
        """
        url = url or self.url

        try:
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            print(f"Successfully fetched: {url}")
            return response.text
        except requests.exceptions.RequestException as e:
            print(f"Error fetching {url}: {e}")
            return None

    def parse_html(self, html):
        """
        Parse HTML into BeautifulSoup object

        Args:
            html (str): HTML content

        Returns:
            BeautifulSoup: Parsed HTML
        """
        return BeautifulSoup(html, 'lxml')

    def parse_table(self, soup, table_id=None, table_class=None, headers_row=0):
        """
        Parse HTML table into list of dictionaries

        Args:
            soup (BeautifulSoup): Parsed HTML
            table_id (str): ID of table element
            table_class (str): Class of table element
            headers_row (int): Row index containing headers

        Returns:
            list: List of dictionaries with table data
        """
        # Find table
        if table_id:
            table = soup.find('table', id=table_id)
        elif table_class:
            table = soup.find('table', class_=table_class)
        else:
            table = soup.find('table')

        if not table:
            print("No table found")
            return []

        # Extract headers
        headers = []
        header_row = table.find_all('tr')[headers_row]
        for th in header_row.find_all(['th', 'td']):
            headers.append(th.get_text(strip=True))

        # Extract data rows
        data = []
        rows = table.find_all('tr')[headers_row + 1:]

        for row in rows:
            cells = row.find_all(['td', 'th'])
            if len(cells) == len(headers):
                row_data = {}
                for header, cell in zip(headers, cells):
                    row_data[header] = cell.get_text(strip=True)

                # Add metadata
                row_data['scraped_date'] = datetime.now().strftime('%Y-%m-%d')
                row_data['source_url'] = self.url

                data.append(row_data)

        print(f"Extracted {len(data)} rows from table")
        return data

    def scrape(self):
        """
        Main scraping method - override in subclass

        Returns:
            list: Scraped data
        """
        html = self.fetch_html()
        if not html:
            return []

        soup = self.parse_html(html)
        data = self.parse_table(soup)
        self.data = data
        return data

    def save_to_csv(self, data=None, filename=None):
        """
        Save data to CSV file

        Args:
            data (list): Data to save (uses self.data if not provided)
            filename (str): Output filename

        Returns:
            Path: Path to saved file
        """
        data = data or self.data
        if not data:
            print("No data to save")
            return None

        filename = filename or f"scraped_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = self.output_dir / filename

        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False, encoding='utf-8')
        print(f"Saved to: {filepath}")
        return filepath

    def save_to_json(self, data=None, filename=None):
        """
        Save data to JSON file

        Args:
            data (list): Data to save (uses self.data if not provided)
            filename (str): Output filename

        Returns:
            Path: Path to saved file
        """
        data = data or self.data
        if not data:
            print("No data to save")
            return None

        filename = filename or f"scraped_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.output_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"Saved to: {filepath}")
        return filepath

    def print_summary(self, data=None):
        """
        Print summary statistics of scraped data

        Args:
            data (list): Data to summarize (uses self.data if not provided)
        """
        data = data or self.data
        if not data:
            print("No data available")
            return

        print("=" * 70)
        print("SCRAPING SUMMARY")
        print("=" * 70)
        print(f"Total items: {len(data)}")
        print(f"Fields: {', '.join(data[0].keys())}")
        print(f"Source: {self.url}")
        print(f"Scraped: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)


# Example usage and customization
if __name__ == "__main__":
    # Example 1: Basic table scraping
    scraper = UniversalScraper(url="https://example.com/table-page")
    data = scraper.scrape()
    scraper.save_to_csv()
    scraper.save_to_json()
    scraper.print_summary()

    # Example 2: Custom scraper (inherit and override)
    class CustomScraper(UniversalScraper):
        def scrape(self):
            """Custom scraping logic"""
            html = self.fetch_html()
            if not html:
                return []

            soup = self.parse_html(html)

            # Custom parsing logic here
            # e.g., extract specific elements, process data, etc.

            return self.data

    # Example 3: Multiple tables
    scraper = UniversalScraper(url="https://example.com/data")
    html = scraper.fetch_html()
    soup = scraper.parse_html(html)

    # Parse multiple tables
    table1 = scraper.parse_table(soup, table_id="data-table-1")
    table2 = scraper.parse_table(soup, table_class="results-table")
