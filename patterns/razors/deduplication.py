"""
Deduplication Patterns - Universal Business Solution Framework

Multi-key matching and change detection for data deduplication.
Supports fuzzy matching, multi-field keys, and change tracking.

Usage:
```python
from patterns.razors import Deduplicator, ChangeDetector, MatchStrategy

# Simple deduplication
dedup = Deduplicator(key_fields=['agency', 'bid_number'])
unique_items = dedup.deduplicate(items)

# With change detection
detector = ChangeDetector(
    key_fields=['id'],
    compare_fields=['status', 'price', 'description']
)
changes = detector.detect_changes(new_data, existing_data)
print(f"New: {len(changes['new'])}, Modified: {len(changes['modified'])}")
```
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Set, Tuple, Union
from datetime import datetime
import hashlib
import json
import re


class MatchStrategy(Enum):
    """Strategies for matching records"""
    EXACT = "exact"  # Exact match on all key fields
    ANY = "any"  # Match if ANY key combination matches
    FUZZY = "fuzzy"  # Fuzzy string matching
    NORMALIZED = "normalized"  # Normalize before matching


@dataclass
class DuplicateMatch:
    """Information about a duplicate match"""
    original_index: int
    duplicate_index: int
    matched_keys: List[str]
    similarity: float = 1.0  # 1.0 = exact match
    original_item: Optional[Dict] = None
    duplicate_item: Optional[Dict] = None


@dataclass
class ChangeRecord:
    """Record of a detected change"""
    key: str
    change_type: str  # 'new', 'modified', 'removed'
    old_values: Optional[Dict] = None
    new_values: Optional[Dict] = None
    changed_fields: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            'key': self.key,
            'change_type': self.change_type,
            'old_values': self.old_values,
            'new_values': self.new_values,
            'changed_fields': self.changed_fields,
            'timestamp': self.timestamp.isoformat()
        }


class Deduplicator:
    """
    Multi-key deduplication engine.

    Supports multiple key field combinations, fuzzy matching,
    and flexible matching strategies.

    Example - Simple deduplication:
    ```python
    dedup = Deduplicator(key_fields=['email'])
    unique = dedup.deduplicate(contacts)
    ```

    Example - Multi-key with fallback:
    ```python
    dedup = Deduplicator(
        key_fields=[
            ['agency', 'bid_number'],  # Primary key
            ['agency', 'title']         # Fallback key
        ],
        strategy=MatchStrategy.ANY
    )
    unique = dedup.deduplicate(bids)
    ```
    """

    def __init__(
        self,
        key_fields: Union[List[str], List[List[str]]],
        strategy: MatchStrategy = MatchStrategy.EXACT,
        normalize_strings: bool = True,
        case_sensitive: bool = False
    ):
        """
        Initialize deduplicator.

        Args:
            key_fields: Single list of fields or multiple key combinations
            strategy: Matching strategy
            normalize_strings: Normalize whitespace and special chars
            case_sensitive: Case-sensitive string matching
        """
        # Normalize key_fields to list of lists
        if key_fields and isinstance(key_fields[0], str):
            self.key_combinations = [key_fields]
        else:
            self.key_combinations = key_fields

        self.strategy = strategy
        self.normalize_strings = normalize_strings
        self.case_sensitive = case_sensitive
        self._seen_keys: Dict[str, int] = {}
        self._duplicates: List[DuplicateMatch] = []

    def _normalize_value(self, value: Any) -> str:
        """Normalize a value for comparison"""
        if value is None:
            return ""

        value_str = str(value)

        if not self.case_sensitive:
            value_str = value_str.lower()

        if self.normalize_strings:
            # Remove extra whitespace
            value_str = ' '.join(value_str.split())
            # Remove special characters for comparison
            value_str = re.sub(r'[^\w\s]', '', value_str)

        return value_str.strip()

    def _generate_key(self, item: Dict, key_fields: List[str]) -> str:
        """Generate a composite key from fields"""
        parts = []
        for field in key_fields:
            value = item.get(field, '')
            parts.append(self._normalize_value(value))
        return '|'.join(parts)

    def _generate_hash(self, item: Dict, key_fields: List[str]) -> str:
        """Generate a hash key for the item"""
        key = self._generate_key(item, key_fields)
        return hashlib.md5(key.encode()).hexdigest()

    def deduplicate(
        self,
        items: List[Dict],
        keep: str = 'first'
    ) -> List[Dict]:
        """
        Remove duplicates from items.

        Args:
            items: List of dictionaries
            keep: 'first' or 'last' - which duplicate to keep

        Returns:
            Deduplicated list
        """
        self._seen_keys.clear()
        self._duplicates.clear()

        if keep == 'last':
            items = list(reversed(items))

        unique = []

        for idx, item in enumerate(items):
            is_duplicate = False

            for key_fields in self.key_combinations:
                key = self._generate_key(item, key_fields)

                if key and key in self._seen_keys:
                    # Found duplicate
                    self._duplicates.append(DuplicateMatch(
                        original_index=self._seen_keys[key],
                        duplicate_index=idx,
                        matched_keys=key_fields,
                        original_item=unique[self._seen_keys[key]] if self._seen_keys[key] < len(unique) else None,
                        duplicate_item=item
                    ))
                    is_duplicate = True

                    if self.strategy == MatchStrategy.EXACT:
                        break  # Stop checking other key combinations
                    elif self.strategy == MatchStrategy.ANY:
                        break  # Found match, stop

                elif key:
                    self._seen_keys[key] = len(unique)

            if not is_duplicate:
                unique.append(item)

        if keep == 'last':
            unique = list(reversed(unique))

        return unique

    def find_duplicates(self, items: List[Dict]) -> List[DuplicateMatch]:
        """Find all duplicates without removing them"""
        self.deduplicate(items)
        return self._duplicates

    def get_duplicate_groups(self, items: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Group items by their duplicate key.

        Returns dict of key -> list of matching items
        """
        groups: Dict[str, List[Dict]] = {}

        for item in items:
            for key_fields in self.key_combinations:
                key = self._generate_key(item, key_fields)
                if key:
                    if key not in groups:
                        groups[key] = []
                    groups[key].append(item)
                    break  # Use first matching key combination

        # Filter to only groups with duplicates
        return {k: v for k, v in groups.items() if len(v) > 1}

    def merge_duplicates(
        self,
        items: List[Dict],
        merge_func: Optional[Callable[[List[Dict]], Dict]] = None
    ) -> List[Dict]:
        """
        Merge duplicates using a custom function.

        Args:
            items: Items to merge
            merge_func: Function that takes list of duplicates and returns merged item

        Returns:
            List with merged items
        """
        if merge_func is None:
            # Default: keep first, but collect all unique values
            def merge_func(dups):
                merged = dups[0].copy()
                for d in dups[1:]:
                    for k, v in d.items():
                        if k not in merged or merged[k] is None:
                            merged[k] = v
                return merged

        groups = self.get_duplicate_groups(items)
        processed_keys = set()
        result = []

        for item in items:
            for key_fields in self.key_combinations:
                key = self._generate_key(item, key_fields)
                if key in groups and key not in processed_keys:
                    merged = merge_func(groups[key])
                    result.append(merged)
                    processed_keys.add(key)
                    break
                elif key not in groups:
                    result.append(item)
                    break

        return result


class ChangeDetector:
    """
    Detect changes between datasets.

    Tracks new items, modified items, and removed items.
    Useful for scraping, API polling, and data synchronization.

    Example:
    ```python
    detector = ChangeDetector(
        key_fields=['contract_id'],
        compare_fields=['status', 'amount', 'end_date']
    )

    changes = detector.detect_changes(new_data, existing_data)
    print(f"New: {len(changes['new'])}")
    print(f"Modified: {len(changes['modified'])}")
    print(f"Removed: {len(changes['removed'])}")
    ```
    """

    def __init__(
        self,
        key_fields: List[str],
        compare_fields: Optional[List[str]] = None,
        ignore_fields: Optional[List[str]] = None,
        track_history: bool = True,
        max_history: int = 100
    ):
        """
        Initialize change detector.

        Args:
            key_fields: Fields that uniquely identify a record
            compare_fields: Fields to compare for changes (None = all)
            ignore_fields: Fields to ignore when comparing
            track_history: Whether to maintain change history
            max_history: Maximum history entries to keep
        """
        self.key_fields = key_fields
        self.compare_fields = compare_fields
        self.ignore_fields = set(ignore_fields or [])
        self.track_history = track_history
        self.max_history = max_history
        self._history: List[ChangeRecord] = []

    def _generate_key(self, item: Dict) -> str:
        """Generate unique key for an item"""
        parts = [str(item.get(f, '')) for f in self.key_fields]
        return '|'.join(parts)

    def _get_compare_fields(self, item: Dict) -> List[str]:
        """Get fields to compare for an item"""
        if self.compare_fields:
            return [f for f in self.compare_fields if f not in self.ignore_fields]
        else:
            return [f for f in item.keys() if f not in self.ignore_fields and f not in self.key_fields]

    def _items_differ(self, old: Dict, new: Dict) -> Tuple[bool, List[str]]:
        """Check if two items differ, return (differs, changed_fields)"""
        changed_fields = []
        fields = self._get_compare_fields(new)

        for field in fields:
            old_val = old.get(field)
            new_val = new.get(field)

            # Normalize for comparison
            if isinstance(old_val, str) and isinstance(new_val, str):
                if old_val.strip().lower() != new_val.strip().lower():
                    changed_fields.append(field)
            elif old_val != new_val:
                changed_fields.append(field)

        return len(changed_fields) > 0, changed_fields

    def detect_changes(
        self,
        new_data: List[Dict],
        existing_data: List[Dict]
    ) -> Dict[str, List]:
        """
        Detect changes between new and existing data.

        Args:
            new_data: Current/new dataset
            existing_data: Previous/existing dataset

        Returns:
            Dictionary with 'new', 'modified', 'removed', 'unchanged' lists
        """
        # Index existing data by key
        existing_index = {}
        for item in existing_data:
            key = self._generate_key(item)
            existing_index[key] = item

        # Index new data by key
        new_index = {}
        for item in new_data:
            key = self._generate_key(item)
            new_index[key] = item

        changes = {
            'new': [],
            'modified': [],
            'removed': [],
            'unchanged': [],
            'new_count': 0,
            'modified_count': 0,
            'removed_count': 0,
            'unchanged_count': 0
        }

        # Find new and modified items
        for key, item in new_index.items():
            if key not in existing_index:
                # New item
                changes['new'].append(item)
                if self.track_history:
                    self._add_history(ChangeRecord(
                        key=key,
                        change_type='new',
                        new_values=item
                    ))
            else:
                # Check for modifications
                old_item = existing_index[key]
                differs, changed_fields = self._items_differ(old_item, item)

                if differs:
                    changes['modified'].append({
                        'item': item,
                        'old_item': old_item,
                        'changed_fields': changed_fields
                    })
                    if self.track_history:
                        self._add_history(ChangeRecord(
                            key=key,
                            change_type='modified',
                            old_values=old_item,
                            new_values=item,
                            changed_fields=changed_fields
                        ))
                else:
                    changes['unchanged'].append(item)

        # Find removed items
        for key, item in existing_index.items():
            if key not in new_index:
                changes['removed'].append(item)
                if self.track_history:
                    self._add_history(ChangeRecord(
                        key=key,
                        change_type='removed',
                        old_values=item
                    ))

        # Update counts
        changes['new_count'] = len(changes['new'])
        changes['modified_count'] = len(changes['modified'])
        changes['removed_count'] = len(changes['removed'])
        changes['unchanged_count'] = len(changes['unchanged'])

        return changes

    def _add_history(self, record: ChangeRecord) -> None:
        """Add record to history"""
        self._history.append(record)
        # Trim history if needed
        if len(self._history) > self.max_history:
            self._history = self._history[-self.max_history:]

    def get_history(self) -> List[ChangeRecord]:
        """Get change history"""
        return self._history

    def clear_history(self) -> None:
        """Clear change history"""
        self._history = []

    def get_change_summary(self, changes: Dict) -> str:
        """Get human-readable change summary"""
        lines = [
            f"New items: {changes['new_count']}",
            f"Modified items: {changes['modified_count']}",
            f"Removed items: {changes['removed_count']}",
            f"Unchanged items: {changes['unchanged_count']}"
        ]

        if changes['modified']:
            lines.append("\nModified fields:")
            all_changed = set()
            for mod in changes['modified']:
                all_changed.update(mod.get('changed_fields', []))
            for field in sorted(all_changed):
                lines.append(f"  - {field}")

        return '\n'.join(lines)


class FingerprintGenerator:
    """
    Generate fingerprints/hashes for change detection.

    Useful for tracking document or record versions.

    Example:
    ```python
    fingerprint = FingerprintGenerator(
        include_fields=['title', 'content', 'status'],
        exclude_fields=['updated_at', 'id']
    )

    # Generate fingerprint
    fp = fingerprint.generate(document)

    # Compare
    if fingerprint.compare(old_doc, new_doc):
        print("Documents are identical")
    ```
    """

    def __init__(
        self,
        include_fields: Optional[List[str]] = None,
        exclude_fields: Optional[List[str]] = None,
        algorithm: str = 'md5'
    ):
        """
        Initialize fingerprint generator.

        Args:
            include_fields: Only include these fields (None = all)
            exclude_fields: Exclude these fields
            algorithm: Hash algorithm ('md5', 'sha1', 'sha256')
        """
        self.include_fields = include_fields
        self.exclude_fields = set(exclude_fields or [])
        self.algorithm = algorithm

    def _get_hasher(self):
        """Get hash object"""
        if self.algorithm == 'md5':
            return hashlib.md5()
        elif self.algorithm == 'sha1':
            return hashlib.sha1()
        elif self.algorithm == 'sha256':
            return hashlib.sha256()
        else:
            return hashlib.md5()

    def generate(self, item: Dict) -> str:
        """Generate fingerprint for an item"""
        # Select fields
        if self.include_fields:
            fields = self.include_fields
        else:
            fields = sorted(item.keys())

        fields = [f for f in fields if f not in self.exclude_fields]

        # Build content to hash
        content_parts = []
        for field in fields:
            value = item.get(field, '')
            # Normalize value
            if value is None:
                value = ''
            elif not isinstance(value, str):
                value = json.dumps(value, sort_keys=True, default=str)
            content_parts.append(f"{field}:{value}")

        content = '|'.join(content_parts)

        # Generate hash
        hasher = self._get_hasher()
        hasher.update(content.encode('utf-8'))
        return hasher.hexdigest()

    def compare(self, item1: Dict, item2: Dict) -> bool:
        """Compare two items by fingerprint"""
        return self.generate(item1) == self.generate(item2)

    def batch_generate(
        self,
        items: List[Dict],
        output_field: str = '_fingerprint'
    ) -> List[Dict]:
        """Generate fingerprints for all items"""
        for item in items:
            item[output_field] = self.generate(item)
        return items


# ==================== Factory Functions ====================


def create_simple_deduplicator(key_field: str) -> Deduplicator:
    """Create a simple single-field deduplicator"""
    return Deduplicator(key_fields=[key_field])


def create_multi_key_deduplicator(
    primary_keys: List[str],
    fallback_keys: Optional[List[str]] = None
) -> Deduplicator:
    """
    Create deduplicator with primary and fallback keys.

    Matches on primary keys first, falls back to secondary.
    """
    key_combinations = [primary_keys]
    if fallback_keys:
        key_combinations.append(fallback_keys)
    return Deduplicator(
        key_fields=key_combinations,
        strategy=MatchStrategy.ANY
    )


def create_procurement_deduplicator() -> Deduplicator:
    """
    Create deduplicator for procurement/bid data.

    Primary: agency + bid_number
    Fallback: agency + title
    """
    return Deduplicator(
        key_fields=[
            ['agency', 'bid_number'],
            ['agency', 'title']
        ],
        strategy=MatchStrategy.ANY,
        normalize_strings=True,
        case_sensitive=False
    )


def create_contact_deduplicator() -> Deduplicator:
    """
    Create deduplicator for contact/person data.

    Primary: email
    Fallback: first_name + last_name + company
    """
    return Deduplicator(
        key_fields=[
            ['email'],
            ['first_name', 'last_name', 'company']
        ],
        strategy=MatchStrategy.ANY,
        normalize_strings=True,
        case_sensitive=False
    )


def create_document_change_detector(
    id_field: str = 'id'
) -> ChangeDetector:
    """
    Create change detector for documents.

    Ignores metadata fields, compares content.
    """
    return ChangeDetector(
        key_fields=[id_field],
        ignore_fields=['id', 'created_at', 'updated_at', '_id', 'timestamp']
    )


def create_contract_change_detector() -> ChangeDetector:
    """
    Create change detector for contract data.

    Tracks key contract fields for changes.
    """
    return ChangeDetector(
        key_fields=['contract_id'],
        compare_fields=[
            'status', 'amount', 'current_amount', 'end_date',
            'vendor', 'department', 'description', 'contact_info'
        ]
    )


# ==================== Exports ====================

__all__ = [
    'Deduplicator',
    'ChangeDetector',
    'FingerprintGenerator',
    'MatchStrategy',
    'DuplicateMatch',
    'ChangeRecord',
    'create_simple_deduplicator',
    'create_multi_key_deduplicator',
    'create_procurement_deduplicator',
    'create_contact_deduplicator',
    'create_document_change_detector',
    'create_contract_change_detector',
]
