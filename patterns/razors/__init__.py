"""
Razors Patterns - Universal Business Solution Framework

Analysis and filtering patterns (razors) for business data processing.
Provides temporal decay, filter chains, and deduplication.

Usage:
```python
from patterns.razors import (
    # Temporal Decay
    TemporalDecay, DecayFunction, ExpirationTracker,
    create_recency_scorer,

    # Filter Chains
    FilterChain, Filter, FilterOperator,
    NegativeKeywordFilter, ThresholdFilter,
    create_priority_filter,

    # Deduplication
    Deduplicator, ChangeDetector, FingerprintGenerator,
    create_procurement_deduplicator,
)
```
"""

from .temporal_decay import (
    TemporalDecay,
    DecayFunction,
    DecayConfig,
    TemporalScore,
    ExpirationTracker,
    # Factory functions
    create_recency_scorer,
    create_contract_expiration_tracker,
    create_exponential_decay,
    create_activity_freshness_scorer,
)

from .filter_chains import (
    FilterChain,
    Filter,
    FilterOperator,
    FilterLogic,
    FilterGroup,
    FilterBuilder,
    NegativeKeywordFilter,
    ThresholdFilter,
    # Factory functions
    create_status_filter,
    create_date_range_filter,
    create_score_filter,
    create_procurement_keyword_filter,
    create_priority_filter,
)

from .deduplication import (
    Deduplicator,
    ChangeDetector,
    FingerprintGenerator,
    MatchStrategy,
    DuplicateMatch,
    ChangeRecord,
    # Factory functions
    create_simple_deduplicator,
    create_multi_key_deduplicator,
    create_procurement_deduplicator,
    create_contact_deduplicator,
    create_document_change_detector,
    create_contract_change_detector,
)

__all__ = [
    # Temporal Decay
    "TemporalDecay",
    "DecayFunction",
    "DecayConfig",
    "TemporalScore",
    "ExpirationTracker",
    "create_recency_scorer",
    "create_contract_expiration_tracker",
    "create_exponential_decay",
    "create_activity_freshness_scorer",

    # Filter Chains
    "FilterChain",
    "Filter",
    "FilterOperator",
    "FilterLogic",
    "FilterGroup",
    "FilterBuilder",
    "NegativeKeywordFilter",
    "ThresholdFilter",
    "create_status_filter",
    "create_date_range_filter",
    "create_score_filter",
    "create_procurement_keyword_filter",
    "create_priority_filter",

    # Deduplication
    "Deduplicator",
    "ChangeDetector",
    "FingerprintGenerator",
    "MatchStrategy",
    "DuplicateMatch",
    "ChangeRecord",
    "create_simple_deduplicator",
    "create_multi_key_deduplicator",
    "create_procurement_deduplicator",
    "create_contact_deduplicator",
    "create_document_change_detector",
    "create_contract_change_detector",
]
