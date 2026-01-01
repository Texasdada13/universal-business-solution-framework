"""
Filter Chains - Universal Business Solution Framework

Composable filter logic for building complex data filtering pipelines.
Supports AND/OR logic, negative filters, threshold-based filtering.

Usage:
```python
from patterns.razors import FilterChain, Filter, FilterOperator

# Build a filter chain
chain = FilterChain()
chain.add_filter(Filter('status', FilterOperator.EQUALS, 'Active'))
chain.add_filter(Filter('score', FilterOperator.GREATER_THAN, 50))
chain.add_negative_filter(Filter('category', FilterOperator.IN, ['spam', 'test']))

# Apply to data
filtered = chain.apply(items)

# Or use fluent API
filtered = (FilterChain()
    .where('status').equals('Active')
    .where('score').greater_than(50)
    .exclude('category').is_in(['spam', 'test'])
    .apply(items))
```
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Union
import re
from datetime import datetime


class FilterOperator(Enum):
    """Filter comparison operators"""
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    GREATER_THAN = "greater_than"
    GREATER_THAN_OR_EQUAL = "greater_than_or_equal"
    LESS_THAN = "less_than"
    LESS_THAN_OR_EQUAL = "less_than_or_equal"
    IN = "in"
    NOT_IN = "not_in"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    REGEX = "regex"
    IS_NULL = "is_null"
    IS_NOT_NULL = "is_not_null"
    BETWEEN = "between"
    CUSTOM = "custom"


class FilterLogic(Enum):
    """Logic for combining multiple filters"""
    AND = "and"
    OR = "or"


@dataclass
class Filter:
    """
    Single filter definition.

    Args:
        field: Field name to filter on
        operator: Comparison operator
        value: Value to compare against
        case_sensitive: For string comparisons
    """
    field: str
    operator: FilterOperator
    value: Any = None
    case_sensitive: bool = False
    custom_func: Optional[Callable[[Any], bool]] = None

    def evaluate(self, item: Dict) -> bool:
        """Evaluate filter against an item"""
        field_value = self._get_nested_value(item, self.field)

        # Handle null checks first
        if self.operator == FilterOperator.IS_NULL:
            return field_value is None
        if self.operator == FilterOperator.IS_NOT_NULL:
            return field_value is not None

        # If field is None and we're not checking for null, filter fails
        if field_value is None:
            return False

        # Custom function
        if self.operator == FilterOperator.CUSTOM and self.custom_func:
            return self.custom_func(field_value)

        # Normalize strings for case-insensitive comparison
        if isinstance(field_value, str) and not self.case_sensitive:
            field_value = field_value.lower()
            if isinstance(self.value, str):
                compare_value = self.value.lower()
            elif isinstance(self.value, list):
                compare_value = [v.lower() if isinstance(v, str) else v for v in self.value]
            else:
                compare_value = self.value
        else:
            compare_value = self.value

        # Evaluate based on operator
        if self.operator == FilterOperator.EQUALS:
            return field_value == compare_value

        elif self.operator == FilterOperator.NOT_EQUALS:
            return field_value != compare_value

        elif self.operator == FilterOperator.GREATER_THAN:
            return field_value > compare_value

        elif self.operator == FilterOperator.GREATER_THAN_OR_EQUAL:
            return field_value >= compare_value

        elif self.operator == FilterOperator.LESS_THAN:
            return field_value < compare_value

        elif self.operator == FilterOperator.LESS_THAN_OR_EQUAL:
            return field_value <= compare_value

        elif self.operator == FilterOperator.IN:
            return field_value in compare_value

        elif self.operator == FilterOperator.NOT_IN:
            return field_value not in compare_value

        elif self.operator == FilterOperator.CONTAINS:
            if isinstance(field_value, str):
                return compare_value in field_value
            elif isinstance(field_value, list):
                return compare_value in field_value
            return False

        elif self.operator == FilterOperator.NOT_CONTAINS:
            if isinstance(field_value, str):
                return compare_value not in field_value
            elif isinstance(field_value, list):
                return compare_value not in field_value
            return True

        elif self.operator == FilterOperator.STARTS_WITH:
            return isinstance(field_value, str) and field_value.startswith(compare_value)

        elif self.operator == FilterOperator.ENDS_WITH:
            return isinstance(field_value, str) and field_value.endswith(compare_value)

        elif self.operator == FilterOperator.REGEX:
            if isinstance(field_value, str):
                flags = 0 if self.case_sensitive else re.IGNORECASE
                return bool(re.search(self.value, field_value, flags))
            return False

        elif self.operator == FilterOperator.BETWEEN:
            if isinstance(compare_value, (list, tuple)) and len(compare_value) == 2:
                return compare_value[0] <= field_value <= compare_value[1]
            return False

        return False

    def _get_nested_value(self, item: Dict, field: str) -> Any:
        """Get value from nested dictionary using dot notation"""
        keys = field.split('.')
        value = item
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return None
        return value


@dataclass
class FilterGroup:
    """Group of filters with shared logic"""
    filters: List[Filter] = field(default_factory=list)
    logic: FilterLogic = FilterLogic.AND
    is_negative: bool = False  # If True, inverts the result

    def evaluate(self, item: Dict) -> bool:
        """Evaluate all filters in group"""
        if not self.filters:
            return True

        if self.logic == FilterLogic.AND:
            result = all(f.evaluate(item) for f in self.filters)
        else:  # OR
            result = any(f.evaluate(item) for f in self.filters)

        return not result if self.is_negative else result


class FilterBuilder:
    """Fluent builder for creating filters"""

    def __init__(self, chain: 'FilterChain', field: str, is_negative: bool = False):
        self.chain = chain
        self.field = field
        self.is_negative = is_negative

    def equals(self, value: Any) -> 'FilterChain':
        return self._add_filter(FilterOperator.EQUALS, value)

    def not_equals(self, value: Any) -> 'FilterChain':
        return self._add_filter(FilterOperator.NOT_EQUALS, value)

    def greater_than(self, value: Any) -> 'FilterChain':
        return self._add_filter(FilterOperator.GREATER_THAN, value)

    def greater_than_or_equal(self, value: Any) -> 'FilterChain':
        return self._add_filter(FilterOperator.GREATER_THAN_OR_EQUAL, value)

    def less_than(self, value: Any) -> 'FilterChain':
        return self._add_filter(FilterOperator.LESS_THAN, value)

    def less_than_or_equal(self, value: Any) -> 'FilterChain':
        return self._add_filter(FilterOperator.LESS_THAN_OR_EQUAL, value)

    def is_in(self, values: List[Any]) -> 'FilterChain':
        return self._add_filter(FilterOperator.IN, values)

    def not_in(self, values: List[Any]) -> 'FilterChain':
        return self._add_filter(FilterOperator.NOT_IN, values)

    def contains(self, value: str) -> 'FilterChain':
        return self._add_filter(FilterOperator.CONTAINS, value)

    def not_contains(self, value: str) -> 'FilterChain':
        return self._add_filter(FilterOperator.NOT_CONTAINS, value)

    def starts_with(self, value: str) -> 'FilterChain':
        return self._add_filter(FilterOperator.STARTS_WITH, value)

    def ends_with(self, value: str) -> 'FilterChain':
        return self._add_filter(FilterOperator.ENDS_WITH, value)

    def matches(self, pattern: str) -> 'FilterChain':
        return self._add_filter(FilterOperator.REGEX, pattern)

    def is_null(self) -> 'FilterChain':
        return self._add_filter(FilterOperator.IS_NULL, None)

    def is_not_null(self) -> 'FilterChain':
        return self._add_filter(FilterOperator.IS_NOT_NULL, None)

    def between(self, min_val: Any, max_val: Any) -> 'FilterChain':
        return self._add_filter(FilterOperator.BETWEEN, [min_val, max_val])

    def _add_filter(self, operator: FilterOperator, value: Any) -> 'FilterChain':
        f = Filter(field=self.field, operator=operator, value=value)
        if self.is_negative:
            self.chain.add_negative_filter(f)
        else:
            self.chain.add_filter(f)
        return self.chain


class FilterChain:
    """
    Composable filter chain for complex data filtering.

    Supports:
    - AND/OR logic between filters
    - Negative filters (exclusion lists)
    - Nested field access (dot notation)
    - Fluent API for building filters
    - Statistics on filtering

    Example:
    ```python
    # Traditional API
    chain = FilterChain()
    chain.add_filter(Filter('status', FilterOperator.EQUALS, 'Active'))
    chain.add_filter(Filter('score', FilterOperator.GREATER_THAN, 50))
    results = chain.apply(items)

    # Fluent API
    results = (FilterChain()
        .where('status').equals('Active')
        .where('score').greater_than(50)
        .apply(items))
    ```
    """

    def __init__(self, logic: FilterLogic = FilterLogic.AND):
        """
        Initialize filter chain.

        Args:
            logic: Default logic for combining filters (AND/OR)
        """
        self.groups: List[FilterGroup] = []
        self.default_logic = logic
        self._current_group: Optional[FilterGroup] = None
        self._last_stats: Dict = {}

    def add_filter(self, filter_obj: Filter) -> 'FilterChain':
        """Add a filter to the current group"""
        if self._current_group is None:
            self._current_group = FilterGroup(logic=self.default_logic)
            self.groups.append(self._current_group)
        self._current_group.filters.append(filter_obj)
        return self

    def add_negative_filter(self, filter_obj: Filter) -> 'FilterChain':
        """Add a filter that excludes matches"""
        group = FilterGroup(filters=[filter_obj], is_negative=True)
        self.groups.append(group)
        return self

    def add_or_group(self, filters: List[Filter]) -> 'FilterChain':
        """Add a group of filters with OR logic"""
        group = FilterGroup(filters=filters, logic=FilterLogic.OR)
        self.groups.append(group)
        return self

    def where(self, field: str) -> FilterBuilder:
        """Start building a filter with fluent API"""
        return FilterBuilder(self, field, is_negative=False)

    def exclude(self, field: str) -> FilterBuilder:
        """Start building a negative filter with fluent API"""
        return FilterBuilder(self, field, is_negative=True)

    def or_where(self, field: str) -> FilterBuilder:
        """Start a new OR group"""
        self._current_group = FilterGroup(logic=FilterLogic.OR)
        self.groups.append(self._current_group)
        return FilterBuilder(self, field, is_negative=False)

    def new_group(self, logic: FilterLogic = FilterLogic.AND) -> 'FilterChain':
        """Start a new filter group"""
        self._current_group = FilterGroup(logic=logic)
        self.groups.append(self._current_group)
        return self

    def apply(self, items: List[Dict]) -> List[Dict]:
        """
        Apply filter chain to items.

        Args:
            items: List of dictionaries to filter

        Returns:
            Filtered list
        """
        original_count = len(items)
        results = []

        for item in items:
            if self._evaluate_item(item):
                results.append(item)

        # Track statistics
        self._last_stats = {
            'original_count': original_count,
            'filtered_count': len(results),
            'removed_count': original_count - len(results),
            'pass_rate': len(results) / original_count if original_count > 0 else 0
        }

        return results

    def _evaluate_item(self, item: Dict) -> bool:
        """Evaluate all filter groups against an item"""
        if not self.groups:
            return True

        # All groups must pass (groups are AND'd together)
        return all(group.evaluate(item) for group in self.groups)

    def get_stats(self) -> Dict:
        """Get statistics from last apply() call"""
        return self._last_stats

    def clear(self) -> 'FilterChain':
        """Clear all filters"""
        self.groups = []
        self._current_group = None
        return self

    def describe(self) -> str:
        """Get human-readable description of filters"""
        descriptions = []
        for i, group in enumerate(self.groups):
            prefix = "EXCLUDE" if group.is_negative else "INCLUDE"
            logic = group.logic.value.upper()
            filters = []
            for f in group.filters:
                filters.append(f"{f.field} {f.operator.value} {f.value}")
            group_desc = f" {logic} ".join(filters)
            descriptions.append(f"{prefix}: ({group_desc})")
        return " AND ".join(descriptions)


class NegativeKeywordFilter:
    """
    Keyword-based exclusion filter.

    Common pattern from RFP/procurement filtering.

    Example:
    ```python
    filter = NegativeKeywordFilter([
        'construction', 'janitorial', 'lawn care',
        'office supplies', 'furniture'
    ])

    # Filter items by title or description
    filtered = filter.apply(items, fields=['title', 'description'])
    ```
    """

    def __init__(
        self,
        keywords: List[str],
        case_sensitive: bool = False,
        match_whole_word: bool = False
    ):
        """
        Initialize negative keyword filter.

        Args:
            keywords: List of keywords to exclude
            case_sensitive: Whether matching is case-sensitive
            match_whole_word: Only match whole words (not substrings)
        """
        self.keywords = keywords
        self.case_sensitive = case_sensitive
        self.match_whole_word = match_whole_word

        # Compile regex patterns for efficiency
        self._patterns = []
        for keyword in keywords:
            if match_whole_word:
                pattern = r'\b' + re.escape(keyword) + r'\b'
            else:
                pattern = re.escape(keyword)
            flags = 0 if case_sensitive else re.IGNORECASE
            self._patterns.append(re.compile(pattern, flags))

    def matches(self, text: str) -> bool:
        """Check if text matches any negative keyword"""
        if not text:
            return False
        return any(p.search(text) for p in self._patterns)

    def get_matched_keywords(self, text: str) -> List[str]:
        """Get list of matched keywords"""
        if not text:
            return []
        matched = []
        for keyword, pattern in zip(self.keywords, self._patterns):
            if pattern.search(text):
                matched.append(keyword)
        return matched

    def apply(
        self,
        items: List[Dict],
        fields: List[str] = None,
        any_field: bool = True
    ) -> List[Dict]:
        """
        Filter items based on negative keywords.

        Args:
            items: List of dictionaries
            fields: Fields to check for keywords
            any_field: If True, exclude if ANY field matches; if False, only if ALL match

        Returns:
            Filtered list (items NOT matching keywords)
        """
        if fields is None:
            fields = ['title', 'description', 'name']

        results = []
        for item in items:
            matches = []
            for field in fields:
                value = item.get(field, '')
                if value and self.matches(str(value)):
                    matches.append(field)

            # Decide whether to include
            if any_field:
                # Exclude if ANY field matches
                if not matches:
                    results.append(item)
            else:
                # Exclude only if ALL fields match
                if len(matches) < len(fields):
                    results.append(item)

        return results


class ThresholdFilter:
    """
    Multi-threshold filtering with priority classification.

    Common pattern for score-based filtering.

    Example:
    ```python
    filter = ThresholdFilter(
        field='score',
        thresholds={
            'urgent': 85,
            'high': 70,
            'medium': 40,
            'low': 0
        }
    )

    # Classify items
    classified = filter.classify(items)

    # Filter to specific priority
    urgent_items = filter.filter_by_priority(items, 'urgent')
    ```
    """

    def __init__(
        self,
        field: str,
        thresholds: Dict[str, float],
        higher_is_better: bool = True
    ):
        """
        Initialize threshold filter.

        Args:
            field: Field to check
            thresholds: Dict of priority_name -> threshold_value
            higher_is_better: If True, higher values get higher priorities
        """
        self.field = field
        self.thresholds = thresholds
        self.higher_is_better = higher_is_better

        # Sort thresholds for evaluation
        self._sorted_thresholds = sorted(
            thresholds.items(),
            key=lambda x: x[1],
            reverse=higher_is_better
        )

    def get_priority(self, value: float) -> str:
        """Get priority level for a value"""
        for priority, threshold in self._sorted_thresholds:
            if self.higher_is_better:
                if value >= threshold:
                    return priority
            else:
                if value <= threshold:
                    return priority
        return list(self.thresholds.keys())[-1]  # Lowest priority

    def classify(self, items: List[Dict], output_field: str = 'priority') -> List[Dict]:
        """Classify all items by priority"""
        for item in items:
            value = item.get(self.field, 0)
            item[output_field] = self.get_priority(value)
        return items

    def filter_by_priority(
        self,
        items: List[Dict],
        priority: str,
        include_higher: bool = True
    ) -> List[Dict]:
        """
        Filter to specific priority level.

        Args:
            items: Items to filter
            priority: Target priority
            include_higher: Include higher priorities too
        """
        if priority not in self.thresholds:
            raise ValueError(f"Unknown priority: {priority}")

        target_threshold = self.thresholds[priority]
        results = []

        for item in items:
            value = item.get(self.field, 0)
            item_priority = self.get_priority(value)

            if item_priority == priority:
                results.append(item)
            elif include_higher:
                # Check if item's priority is higher
                item_threshold = self.thresholds.get(item_priority, 0)
                if self.higher_is_better:
                    if item_threshold > target_threshold:
                        results.append(item)
                else:
                    if item_threshold < target_threshold:
                        results.append(item)

        return results

    def get_distribution(self, items: List[Dict]) -> Dict[str, int]:
        """Get count of items at each priority level"""
        distribution = {priority: 0 for priority in self.thresholds.keys()}
        for item in items:
            value = item.get(self.field, 0)
            priority = self.get_priority(value)
            distribution[priority] = distribution.get(priority, 0) + 1
        return distribution


# ==================== Factory Functions ====================


def create_status_filter(
    field: str = 'status',
    include: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None
) -> FilterChain:
    """Create a simple status filter"""
    chain = FilterChain()
    if include:
        chain.add_filter(Filter(field, FilterOperator.IN, include))
    if exclude:
        chain.add_negative_filter(Filter(field, FilterOperator.IN, exclude))
    return chain


def create_date_range_filter(
    field: str = 'date',
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> FilterChain:
    """Create a date range filter"""
    chain = FilterChain()
    if start_date:
        chain.add_filter(Filter(field, FilterOperator.GREATER_THAN_OR_EQUAL, start_date))
    if end_date:
        chain.add_filter(Filter(field, FilterOperator.LESS_THAN_OR_EQUAL, end_date))
    return chain


def create_score_filter(
    field: str = 'score',
    min_score: Optional[float] = None,
    max_score: Optional[float] = None
) -> FilterChain:
    """Create a score range filter"""
    chain = FilterChain()
    if min_score is not None:
        chain.add_filter(Filter(field, FilterOperator.GREATER_THAN_OR_EQUAL, min_score))
    if max_score is not None:
        chain.add_filter(Filter(field, FilterOperator.LESS_THAN_OR_EQUAL, max_score))
    return chain


def create_procurement_keyword_filter() -> NegativeKeywordFilter:
    """
    Create standard procurement negative keyword filter.

    Excludes common non-IT procurement categories.
    """
    return NegativeKeywordFilter([
        'construction', 'janitorial', 'custodial', 'lawn care',
        'landscaping', 'office supplies', 'furniture', 'moving services',
        'food service', 'catering', 'vehicle', 'fleet', 'plumbing',
        'electrical', 'hvac', 'roofing', 'painting', 'cleaning'
    ])


def create_priority_filter(
    field: str = 'score',
    urgent: float = 85,
    high: float = 70,
    medium: float = 40
) -> ThresholdFilter:
    """Create standard priority threshold filter"""
    return ThresholdFilter(
        field=field,
        thresholds={
            'urgent': urgent,
            'high': high,
            'medium': medium,
            'low': 0
        }
    )


# ==================== Exports ====================

__all__ = [
    'FilterChain',
    'Filter',
    'FilterOperator',
    'FilterLogic',
    'FilterGroup',
    'FilterBuilder',
    'NegativeKeywordFilter',
    'ThresholdFilter',
    'create_status_filter',
    'create_date_range_filter',
    'create_score_filter',
    'create_procurement_keyword_filter',
    'create_priority_filter',
]
