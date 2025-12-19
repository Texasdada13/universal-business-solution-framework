"""
Universal Change Detection Patterns
Track changes in any dataset over time

Based on: NYPD Procurement Tracker
"""

import json
import hashlib
from datetime import datetime
from pathlib import Path


class ChangeDetector:
    """
    Generic change detection with baseline comparison

    Usage:
        detector = ChangeDetector(data_dir='data')
        current_data = [...] # Your current data
        changes = detector.detect_changes(current_data, key_fields=['id', 'name'])

        if changes['new_count'] > 0:
            print(f"Found {changes['new_count']} new items!")
    """

    def __init__(self, data_dir=None):
        """
        Initialize change detector

        Args:
            data_dir (str|Path): Directory for baseline and history files
        """
        if data_dir is None:
            self.data_dir = Path('data')
        else:
            self.data_dir = Path(data_dir)

        self.data_dir.mkdir(exist_ok=True)
        self.baseline_file = self.data_dir / 'baseline.json'
        self.history_file = self.data_dir / 'history.json'

    def create_hash(self, item, key_fields):
        """
        Create unique hash for an item based on key fields

        Args:
            item (dict): Data item
            key_fields (list): Fields to use for creating unique hash

        Returns:
            str: MD5 hash of concatenated key fields
        """
        # Concatenate key field values
        key_values = [str(item.get(field, '')) for field in key_fields]
        key_string = '|'.join(key_values)

        # Create MD5 hash
        return hashlib.md5(key_string.encode()).hexdigest()

    def load_baseline(self):
        """
        Load the baseline (last known state)

        Returns:
            list: Baseline data or empty list if doesn't exist
        """
        if self.baseline_file.exists():
            with open(self.baseline_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []

    def save_baseline(self, data):
        """
        Save new baseline

        Args:
            data (list): Data to save as new baseline
        """
        with open(self.baseline_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load_history(self):
        """
        Load change history

        Returns:
            list: History of changes or empty list if doesn't exist
        """
        if self.history_file.exists():
            with open(self.history_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []

    def save_history(self, history):
        """
        Save change history

        Args:
            history (list): Change history to save
        """
        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

    def detect_changes(self, current_data, key_fields=None):
        """
        Detect new and removed items by comparing to baseline

        Args:
            current_data (list): Current dataset
            key_fields (list): Fields to use for uniqueness (default: all fields)

        Returns:
            dict: Changes detected with new/removed items and counts
        """
        baseline = self.load_baseline()

        # If no key fields specified, use all fields
        if key_fields is None:
            if current_data:
                key_fields = list(current_data[0].keys())
            else:
                key_fields = []

        # Create hash sets for comparison
        baseline_hashes = {self.create_hash(item, key_fields): item for item in baseline}
        current_hashes = {self.create_hash(item, key_fields): item for item in current_data}

        # Find new items (in current but not in baseline)
        new_hashes = set(current_hashes.keys()) - set(baseline_hashes.keys())
        new_items = [current_hashes[h] for h in new_hashes]

        # Find removed items (in baseline but not in current)
        removed_hashes = set(baseline_hashes.keys()) - set(current_hashes.keys())
        removed_items = [baseline_hashes[h] for h in removed_hashes]

        # Build changes object
        changes = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'new_items': new_items,
            'removed_items': removed_items,
            'new_count': len(new_items),
            'removed_count': len(removed_items),
            'total_current': len(current_data),
            'total_baseline': len(baseline),
            'key_fields': key_fields
        }

        # Update history if there are changes
        if new_items or removed_items:
            history = self.load_history()
            history.append({
                'timestamp': changes['timestamp'],
                'new_count': changes['new_count'],
                'removed_count': changes['removed_count'],
                'total_current': changes['total_current']
            })

            # Keep last 100 changes
            if len(history) > 100:
                history = history[-100:]

            self.save_history(history)

        # Always save new baseline
        self.save_baseline(current_data)

        return changes

    def format_change_report(self, changes, item_description_field=None):
        """
        Format changes into readable text report

        Args:
            changes (dict): Changes from detect_changes()
            item_description_field (str): Field to use for describing items

        Returns:
            str: Formatted report
        """
        report = []
        report.append("=" * 70)
        report.append("CHANGE DETECTION REPORT")
        report.append("=" * 70)
        report.append(f"Timestamp: {changes['timestamp']}")
        report.append(f"Total Items: {changes['total_current']} (was {changes['total_baseline']})")
        report.append("")

        if changes['new_count'] > 0:
            report.append(f"[NEW] ITEMS: {changes['new_count']}")
            report.append("-" * 70)
            for item in changes['new_items']:
                if item_description_field and item_description_field in item:
                    report.append(f"  {item[item_description_field]}")
                else:
                    report.append(f"  {item}")
                report.append("")
        else:
            report.append("[OK] No new items detected")
            report.append("")

        if changes['removed_count'] > 0:
            report.append(f"[REMOVED] ITEMS: {changes['removed_count']}")
            report.append("-" * 70)
            for item in changes['removed_items']:
                if item_description_field and item_description_field in item:
                    report.append(f"  {item[item_description_field]}")
                else:
                    report.append(f"  {item}")
                report.append("")

        report.append("=" * 70)
        return "\n".join(report)

    def get_stats(self):
        """
        Get statistics about tracked changes

        Returns:
            dict: Statistics about change history
        """
        baseline = self.load_baseline()
        history = self.load_history()

        return {
            'current_total': len(baseline),
            'total_checks': len(history),
            'total_new_ever': sum(h.get('new_count', 0) for h in history),
            'total_removed_ever': sum(h.get('removed_count', 0) for h in history),
            'recent_new': sum(h.get('new_count', 0) for h in history[-10:]),
            'recent_removed': sum(h.get('removed_count', 0) for h in history[-10:]),
            'last_check': history[-1].get('timestamp') if history else 'Never'
        }


# Example usage
if __name__ == "__main__":
    # Example 1: Basic change detection
    detector = ChangeDetector(data_dir='data')

    # Simulate current data
    current_data = [
        {'id': '1', 'name': 'Item A', 'value': '100'},
        {'id': '2', 'name': 'Item B', 'value': '200'},
        {'id': '3', 'name': 'Item C', 'value': '300'},  # NEW
    ]

    # Detect changes (uses 'id' and 'name' to identify unique items)
    changes = detector.detect_changes(current_data, key_fields=['id', 'name'])

    # Print report
    report = detector.format_change_report(changes, item_description_field='name')
    print(report)

    # Check if there are new items
    if changes['new_count'] > 0:
        print(f"\n[ALERT] {changes['new_count']} new items detected!")

    # Get statistics
    stats = detector.get_stats()
    print(f"\nTotal items tracked: {stats['current_total']}")
    print(f"Total checks performed: {stats['total_checks']}")
