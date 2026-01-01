"""
Temporal Decay Scoring - Universal Business Solution Framework

Time-based scoring patterns that adjust values based on freshness/recency.
Supports multiple decay functions and configurable time brackets.

Usage:
```python
from patterns.razors import TemporalDecay, DecayFunction

# Create recency scorer with brackets
decay = TemporalDecay(
    brackets=[
        (7, 100),   # Within 7 days = 100
        (14, 90),   # Within 14 days = 90
        (30, 75),   # Within 30 days = 75
        (60, 50),   # Within 60 days = 50
        (90, 30),   # Within 90 days = 30
    ],
    default_score=10  # Older than 90 days
)

# Score based on age
score = decay.score(days_old=25)  # Returns 75
```
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
import math


class DecayFunction(Enum):
    """Types of decay functions"""
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    STEP = "step"  # Bracket-based
    LOGARITHMIC = "logarithmic"
    SIGMOID = "sigmoid"
    CUSTOM = "custom"


@dataclass
class DecayConfig:
    """Configuration for decay behavior"""
    max_score: float = 100.0
    min_score: float = 0.0
    half_life_days: float = 30.0  # For exponential decay
    decay_rate: float = 0.1  # For linear decay per day

    def validate(self) -> None:
        """Validate configuration"""
        if self.max_score <= self.min_score:
            raise ValueError("max_score must be greater than min_score")
        if self.half_life_days <= 0:
            raise ValueError("half_life_days must be positive")


@dataclass
class TemporalScore:
    """Result of temporal scoring"""
    score: float
    days_old: float
    bracket: Optional[str] = None
    decay_function: str = ""
    freshness_label: str = ""

    def to_dict(self) -> Dict:
        return {
            'score': self.score,
            'days_old': self.days_old,
            'bracket': self.bracket,
            'decay_function': self.decay_function,
            'freshness_label': self.freshness_label
        }


class TemporalDecay:
    """
    Time-based scoring with configurable decay functions.

    Supports bracket-based scoring (most common in business apps),
    as well as continuous decay functions.

    Example - Bracket-based (like Procurement Intel Tool):
    ```python
    decay = TemporalDecay(brackets=[
        (7, 100),   # Fresh: within 7 days
        (14, 90),   # Recent: 8-14 days
        (30, 75),   # Current: 15-30 days
        (60, 50),   # Aging: 31-60 days
        (90, 30),   # Stale: 61-90 days
    ])
    score = decay.score(days_old=45)  # Returns 50
    ```

    Example - Exponential decay:
    ```python
    decay = TemporalDecay(
        function=DecayFunction.EXPONENTIAL,
        config=DecayConfig(max_score=100, half_life_days=14)
    )
    score = decay.score(days_old=14)  # Returns ~50
    ```
    """

    def __init__(
        self,
        brackets: Optional[List[Tuple[int, float]]] = None,
        default_score: float = 10.0,
        function: DecayFunction = DecayFunction.STEP,
        config: Optional[DecayConfig] = None,
        labels: Optional[Dict[str, Tuple[int, int]]] = None
    ):
        """
        Initialize temporal decay scorer.

        Args:
            brackets: List of (days, score) tuples for step function
            default_score: Score for items older than all brackets
            function: Type of decay function
            config: Configuration for continuous decay functions
            labels: Optional freshness labels {label: (min_days, max_days)}
        """
        self.brackets = sorted(brackets or [], key=lambda x: x[0])
        self.default_score = default_score
        self.function = function
        self.config = config or DecayConfig()
        self.labels = labels or self._default_labels()

        if function == DecayFunction.STEP and not brackets:
            # Default brackets if none provided
            self.brackets = [
                (7, 100),
                (14, 90),
                (30, 75),
                (60, 50),
                (90, 30),
            ]

    def _default_labels(self) -> Dict[str, Tuple[int, int]]:
        """Default freshness labels"""
        return {
            'Fresh': (0, 7),
            'Recent': (8, 14),
            'Current': (15, 30),
            'Aging': (31, 60),
            'Stale': (61, 90),
            'Outdated': (91, 9999)
        }

    def score(
        self,
        days_old: Optional[float] = None,
        date: Optional[datetime] = None,
        reference_date: Optional[datetime] = None
    ) -> float:
        """
        Calculate decay score.

        Args:
            days_old: Age in days (direct)
            date: Date to score (calculates days from reference)
            reference_date: Reference date (default: now)

        Returns:
            Decay-adjusted score
        """
        if days_old is None:
            if date is None:
                raise ValueError("Either days_old or date must be provided")
            reference = reference_date or datetime.now()
            days_old = (reference - date).total_seconds() / 86400

        days_old = max(0, days_old)  # No negative days

        if self.function == DecayFunction.STEP:
            return self._step_decay(days_old)
        elif self.function == DecayFunction.LINEAR:
            return self._linear_decay(days_old)
        elif self.function == DecayFunction.EXPONENTIAL:
            return self._exponential_decay(days_old)
        elif self.function == DecayFunction.LOGARITHMIC:
            return self._logarithmic_decay(days_old)
        elif self.function == DecayFunction.SIGMOID:
            return self._sigmoid_decay(days_old)
        else:
            return self._step_decay(days_old)

    def score_with_details(
        self,
        days_old: Optional[float] = None,
        date: Optional[datetime] = None,
        reference_date: Optional[datetime] = None
    ) -> TemporalScore:
        """Calculate score with full details"""
        if days_old is None:
            if date is None:
                raise ValueError("Either days_old or date must be provided")
            reference = reference_date or datetime.now()
            days_old = (reference - date).total_seconds() / 86400

        days_old = max(0, days_old)
        score = self.score(days_old=days_old)
        bracket = self._get_bracket_name(days_old)
        label = self._get_freshness_label(days_old)

        return TemporalScore(
            score=score,
            days_old=days_old,
            bracket=bracket,
            decay_function=self.function.value,
            freshness_label=label
        )

    def _step_decay(self, days_old: float) -> float:
        """Bracket-based step function decay"""
        for max_days, score in self.brackets:
            if days_old <= max_days:
                return score
        return self.default_score

    def _linear_decay(self, days_old: float) -> float:
        """Linear decay over time"""
        score = self.config.max_score - (self.config.decay_rate * days_old)
        return max(self.config.min_score, score)

    def _exponential_decay(self, days_old: float) -> float:
        """Exponential decay with half-life"""
        decay_factor = math.pow(0.5, days_old / self.config.half_life_days)
        score = self.config.min_score + (self.config.max_score - self.config.min_score) * decay_factor
        return score

    def _logarithmic_decay(self, days_old: float) -> float:
        """Logarithmic decay (fast initial, slow later)"""
        if days_old == 0:
            return self.config.max_score
        # Slower decay than linear
        decay = self.config.decay_rate * math.log(days_old + 1)
        score = self.config.max_score - decay
        return max(self.config.min_score, score)

    def _sigmoid_decay(self, days_old: float) -> float:
        """Sigmoid decay (S-curve, stable at extremes)"""
        # Midpoint at half_life_days
        midpoint = self.config.half_life_days
        steepness = 0.1
        sigmoid = 1 / (1 + math.exp(steepness * (days_old - midpoint)))
        score = self.config.min_score + (self.config.max_score - self.config.min_score) * sigmoid
        return score

    def _get_bracket_name(self, days_old: float) -> str:
        """Get descriptive bracket name"""
        for i, (max_days, score) in enumerate(self.brackets):
            if days_old <= max_days:
                if i == 0:
                    return f"0-{max_days} days"
                else:
                    prev_max = self.brackets[i-1][0]
                    return f"{prev_max+1}-{max_days} days"
        if self.brackets:
            return f">{self.brackets[-1][0]} days"
        return "unknown"

    def _get_freshness_label(self, days_old: float) -> str:
        """Get human-readable freshness label"""
        for label, (min_days, max_days) in self.labels.items():
            if min_days <= days_old <= max_days:
                return label
        return "Unknown"

    def batch_score(
        self,
        items: List[Dict],
        date_field: str = 'date',
        output_field: str = 'recency_score',
        reference_date: Optional[datetime] = None
    ) -> List[Dict]:
        """
        Score multiple items by date field.

        Args:
            items: List of dictionaries with date field
            date_field: Name of date field in items
            output_field: Name for output score field
            reference_date: Reference date for scoring

        Returns:
            Items with added score field
        """
        reference = reference_date or datetime.now()

        for item in items:
            date_value = item.get(date_field)
            if date_value:
                if isinstance(date_value, str):
                    date_value = datetime.fromisoformat(date_value.replace('Z', '+00:00'))
                days_old = (reference - date_value).total_seconds() / 86400
                item[output_field] = self.score(days_old=days_old)
            else:
                item[output_field] = self.default_score

        return items

    def get_decay_curve(self, max_days: int = 100, step: int = 1) -> List[Tuple[int, float]]:
        """
        Generate decay curve for visualization.

        Args:
            max_days: Maximum days to generate
            step: Day increment

        Returns:
            List of (days, score) tuples
        """
        return [(d, self.score(days_old=d)) for d in range(0, max_days + 1, step)]


class ExpirationTracker:
    """
    Track items approaching expiration with tiered alerts.

    Common use case: Contract renewals, license expirations, certifications.

    Example:
    ```python
    tracker = ExpirationTracker(
        warning_days=90,
        critical_days=30,
        expired_grace_days=0
    )

    status = tracker.check(expiration_date)
    # Returns: ExpirationStatus with level, days_remaining, urgency_score
    ```
    """

    def __init__(
        self,
        warning_days: int = 90,
        critical_days: int = 30,
        urgent_days: int = 7,
        expired_grace_days: int = 0
    ):
        """
        Initialize expiration tracker.

        Args:
            warning_days: Days before expiration for warning level
            critical_days: Days before expiration for critical level
            urgent_days: Days before expiration for urgent level
            expired_grace_days: Grace period after expiration
        """
        self.warning_days = warning_days
        self.critical_days = critical_days
        self.urgent_days = urgent_days
        self.expired_grace_days = expired_grace_days

    def check(
        self,
        expiration_date: Union[datetime, str],
        reference_date: Optional[datetime] = None
    ) -> Dict:
        """
        Check expiration status.

        Args:
            expiration_date: Date of expiration
            reference_date: Reference date (default: now)

        Returns:
            Dictionary with status details
        """
        if isinstance(expiration_date, str):
            expiration_date = datetime.fromisoformat(expiration_date.replace('Z', '+00:00'))

        reference = reference_date or datetime.now()
        days_remaining = (expiration_date - reference).days

        # Determine level
        if days_remaining < -self.expired_grace_days:
            level = "expired"
            urgency_score = 100
        elif days_remaining < 0:
            level = "grace_period"
            urgency_score = 95
        elif days_remaining <= self.urgent_days:
            level = "urgent"
            urgency_score = 90
        elif days_remaining <= self.critical_days:
            level = "critical"
            urgency_score = 75
        elif days_remaining <= self.warning_days:
            level = "warning"
            urgency_score = 50
        else:
            level = "ok"
            urgency_score = 0

        return {
            'level': level,
            'days_remaining': days_remaining,
            'urgency_score': urgency_score,
            'expiration_date': expiration_date.isoformat(),
            'action_required': level in ['expired', 'grace_period', 'urgent', 'critical'],
            'color': self._get_color(level),
            'icon': self._get_icon(level)
        }

    def _get_color(self, level: str) -> str:
        """Get color for level"""
        colors = {
            'expired': '#dc3545',      # Red
            'grace_period': '#dc3545', # Red
            'urgent': '#fd7e14',       # Orange
            'critical': '#ffc107',     # Yellow
            'warning': '#17a2b8',      # Cyan
            'ok': '#28a745'            # Green
        }
        return colors.get(level, '#6c757d')

    def _get_icon(self, level: str) -> str:
        """Get icon for level"""
        icons = {
            'expired': '🔴',
            'grace_period': '🔴',
            'urgent': '🟠',
            'critical': '🟡',
            'warning': '🔵',
            'ok': '🟢'
        }
        return icons.get(level, '⚪')

    def batch_check(
        self,
        items: List[Dict],
        date_field: str = 'expiration_date',
        reference_date: Optional[datetime] = None
    ) -> List[Dict]:
        """Check expiration for multiple items"""
        for item in items:
            exp_date = item.get(date_field)
            if exp_date:
                status = self.check(exp_date, reference_date)
                item['expiration_status'] = status
        return items

    def get_summary(self, items: List[Dict], date_field: str = 'expiration_date') -> Dict:
        """Get summary of expiration statuses"""
        items = self.batch_check(items, date_field)

        summary = {
            'total': len(items),
            'expired': 0,
            'urgent': 0,
            'critical': 0,
            'warning': 0,
            'ok': 0,
            'action_required_count': 0,
            'items_by_level': {}
        }

        for item in items:
            status = item.get('expiration_status', {})
            level = status.get('level', 'unknown')

            if level in summary:
                summary[level] += 1

            if status.get('action_required'):
                summary['action_required_count'] += 1

            if level not in summary['items_by_level']:
                summary['items_by_level'][level] = []
            summary['items_by_level'][level].append(item)

        return summary


# ==================== Factory Functions ====================


def create_recency_scorer(
    fresh_days: int = 7,
    recent_days: int = 14,
    current_days: int = 30,
    aging_days: int = 60,
    stale_days: int = 90
) -> TemporalDecay:
    """
    Create a standard recency scorer with common brackets.

    Good for: News articles, opportunities, leads, activity tracking.
    """
    return TemporalDecay(
        brackets=[
            (fresh_days, 100),
            (recent_days, 90),
            (current_days, 75),
            (aging_days, 50),
            (stale_days, 30),
        ],
        default_score=10
    )


def create_contract_expiration_tracker() -> ExpirationTracker:
    """
    Create tracker for contract/license expirations.

    90 days = warning, 30 days = critical, 7 days = urgent.
    """
    return ExpirationTracker(
        warning_days=90,
        critical_days=30,
        urgent_days=7,
        expired_grace_days=0
    )


def create_exponential_decay(half_life_days: float = 14) -> TemporalDecay:
    """
    Create exponential decay scorer.

    Good for: Social media relevance, news decay, interest tracking.
    """
    return TemporalDecay(
        function=DecayFunction.EXPONENTIAL,
        config=DecayConfig(
            max_score=100,
            min_score=0,
            half_life_days=half_life_days
        )
    )


def create_activity_freshness_scorer() -> TemporalDecay:
    """
    Create scorer for user/system activity freshness.

    Good for: Last login tracking, activity monitoring, engagement.
    """
    return TemporalDecay(
        brackets=[
            (1, 100),    # Active today
            (3, 95),     # Active this week
            (7, 85),     # Active this week
            (14, 70),    # Active recently
            (30, 50),    # Active this month
            (60, 30),    # Inactive
            (90, 15),    # Very inactive
        ],
        default_score=5,
        labels={
            'Active': (0, 1),
            'Recent': (2, 7),
            'This Month': (8, 30),
            'Inactive': (31, 60),
            'Dormant': (61, 90),
            'Lost': (91, 9999)
        }
    )


# ==================== Exports ====================

__all__ = [
    'TemporalDecay',
    'DecayFunction',
    'DecayConfig',
    'TemporalScore',
    'ExpirationTracker',
    'create_recency_scorer',
    'create_contract_expiration_tracker',
    'create_exponential_decay',
    'create_activity_freshness_scorer',
]
