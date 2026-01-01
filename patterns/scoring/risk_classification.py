"""
Risk Classification Pattern - Universal Business Solution Framework

A flexible risk classification system that converts continuous scores
into discrete risk levels with configurable thresholds. Useful for:
- Risk dashboards and heat maps
- Portfolio risk assessment
- Compliance risk ratings
- Project health indicators
- Vendor risk tiers

Extracted from: Contract Oversight System
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Standard risk levels with associated properties."""
    CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"
    MINIMAL = "Minimal"

    @property
    def priority(self) -> int:
        """Numeric priority (lower = more urgent)."""
        return {
            RiskLevel.CRITICAL: 1,
            RiskLevel.HIGH: 2,
            RiskLevel.MEDIUM: 3,
            RiskLevel.LOW: 4,
            RiskLevel.MINIMAL: 5
        }[self]

    @property
    def color(self) -> str:
        """Standard color for visualization."""
        return {
            RiskLevel.CRITICAL: "#dc3545",  # Red
            RiskLevel.HIGH: "#fd7e14",      # Orange
            RiskLevel.MEDIUM: "#ffc107",    # Yellow
            RiskLevel.LOW: "#28a745",       # Green
            RiskLevel.MINIMAL: "#17a2b8"    # Blue/Teal
        }[self]

    @property
    def icon(self) -> str:
        """Unicode icon for display."""
        return {
            RiskLevel.CRITICAL: "🔴",
            RiskLevel.HIGH: "🟠",
            RiskLevel.MEDIUM: "🟡",
            RiskLevel.LOW: "🟢",
            RiskLevel.MINIMAL: "🔵"
        }[self]


@dataclass
class RiskThreshold:
    """Configuration for a single risk threshold."""
    level: RiskLevel
    min_score: float  # Minimum score for this level (inclusive)
    max_score: float  # Maximum score for this level (exclusive, except for last)
    description: str = ""
    action_required: str = ""


@dataclass
class RiskClassification:
    """Result of classifying an entity's risk."""
    entity_id: str
    score: float
    level: RiskLevel
    description: str
    action_required: str
    threshold_details: Dict[str, Any] = field(default_factory=dict)
    factors: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "score": self.score,
            "level": self.level.value,
            "level_priority": self.level.priority,
            "level_color": self.level.color,
            "level_icon": self.level.icon,
            "description": self.description,
            "action_required": self.action_required,
            "factors": self.factors,
            "metadata": self.metadata
        }


class RiskClassifier:
    """
    Classifies continuous scores into discrete risk levels.

    Supports two scoring directions:
    - HIGHER_IS_RISKIER: High scores = high risk (default for raw risk scores)
    - LOWER_IS_RISKIER: Low scores = high risk (for health scores)

    Example usage:
    ```python
    # For health scores (low = bad, high = good)
    classifier = RiskClassifier(
        direction="lower_is_riskier",
        thresholds=[
            RiskThreshold(RiskLevel.CRITICAL, 0, 30, "Critical issues", "Immediate intervention"),
            RiskThreshold(RiskLevel.HIGH, 30, 50, "Significant concerns", "Priority attention"),
            RiskThreshold(RiskLevel.MEDIUM, 50, 70, "Moderate issues", "Monitor closely"),
            RiskThreshold(RiskLevel.LOW, 70, 85, "Minor concerns", "Routine monitoring"),
            RiskThreshold(RiskLevel.MINIMAL, 85, 100, "Healthy", "Continue monitoring"),
        ]
    )

    result = classifier.classify(score=45, entity_id="CONTRACT-001")
    print(f"Risk: {result.level.value}")  # "High"
    ```
    """

    # Default thresholds for health scores (0-100 where 100 is best)
    DEFAULT_HEALTH_THRESHOLDS = [
        RiskThreshold(RiskLevel.CRITICAL, 0, 30, "Critical health issues requiring immediate attention",
                     "Escalate to leadership; develop immediate remediation plan"),
        RiskThreshold(RiskLevel.HIGH, 30, 50, "Significant concerns that may impact success",
                     "Prioritize for intervention; weekly monitoring"),
        RiskThreshold(RiskLevel.MEDIUM, 50, 70, "Moderate issues requiring attention",
                     "Monitor closely; address in next review cycle"),
        RiskThreshold(RiskLevel.LOW, 70, 85, "Minor concerns within acceptable range",
                     "Routine monitoring; no immediate action needed"),
        RiskThreshold(RiskLevel.MINIMAL, 85, 100, "Excellent health with minimal risk",
                     "Continue standard monitoring"),
    ]

    # Default thresholds for risk scores (0-100 where 100 is highest risk)
    DEFAULT_RISK_THRESHOLDS = [
        RiskThreshold(RiskLevel.MINIMAL, 0, 15, "Very low risk exposure",
                     "Standard procedures apply"),
        RiskThreshold(RiskLevel.LOW, 15, 30, "Low risk with minor concerns",
                     "Monitor as part of routine oversight"),
        RiskThreshold(RiskLevel.MEDIUM, 30, 50, "Moderate risk requiring attention",
                     "Implement additional controls; regular review"),
        RiskThreshold(RiskLevel.HIGH, 50, 70, "High risk requiring active management",
                     "Develop mitigation plan; frequent monitoring"),
        RiskThreshold(RiskLevel.CRITICAL, 70, 100, "Critical risk threatening objectives",
                     "Immediate executive attention; crisis response"),
    ]

    def __init__(
        self,
        direction: str = "lower_is_riskier",  # or "higher_is_riskier"
        thresholds: Optional[List[RiskThreshold]] = None
    ):
        """
        Initialize risk classifier.

        Args:
            direction: "lower_is_riskier" for health scores, "higher_is_riskier" for risk scores
            thresholds: Custom thresholds, or None to use defaults
        """
        self.direction = direction

        if thresholds:
            self.thresholds = sorted(thresholds, key=lambda t: t.min_score)
        elif direction == "lower_is_riskier":
            self.thresholds = self.DEFAULT_HEALTH_THRESHOLDS
        else:
            self.thresholds = self.DEFAULT_RISK_THRESHOLDS

        # Validate thresholds
        self._validate_thresholds()

    def _validate_thresholds(self) -> None:
        """Validate threshold configuration."""
        if not self.thresholds:
            raise ValueError("At least one threshold must be defined")

        # Check for gaps or overlaps
        for i in range(len(self.thresholds) - 1):
            current = self.thresholds[i]
            next_t = self.thresholds[i + 1]
            if current.max_score != next_t.min_score:
                logger.warning(
                    f"Threshold gap/overlap between {current.level.value} "
                    f"({current.max_score}) and {next_t.level.value} ({next_t.min_score})"
                )

    def classify(
        self,
        score: float,
        entity_id: str = "unknown",
        factors: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> RiskClassification:
        """
        Classify a score into a risk level.

        Args:
            score: The score to classify (typically 0-100)
            entity_id: Identifier for the entity being classified
            factors: Optional list of contributing factors
            metadata: Optional additional metadata

        Returns:
            RiskClassification with level and details
        """
        # Clamp score to valid range
        min_score = self.thresholds[0].min_score
        max_score = self.thresholds[-1].max_score
        clamped_score = max(min_score, min(max_score, score))

        # Find matching threshold
        matched_threshold = None
        for threshold in self.thresholds:
            if threshold.min_score <= clamped_score < threshold.max_score:
                matched_threshold = threshold
                break

        # Handle edge case for max value
        if matched_threshold is None and clamped_score == max_score:
            matched_threshold = self.thresholds[-1]

        if matched_threshold is None:
            logger.error(f"No threshold found for score {score}")
            matched_threshold = self.thresholds[-1]  # Default to last

        return RiskClassification(
            entity_id=entity_id,
            score=round(score, 2),
            level=matched_threshold.level,
            description=matched_threshold.description,
            action_required=matched_threshold.action_required,
            threshold_details={
                "min_score": matched_threshold.min_score,
                "max_score": matched_threshold.max_score,
                "position_in_band": self._calculate_band_position(score, matched_threshold)
            },
            factors=factors or [],
            metadata=metadata or {}
        )

    def classify_batch(
        self,
        entities: List[Dict[str, Any]],
        score_field: str = "score",
        id_field: str = "id"
    ) -> List[RiskClassification]:
        """
        Classify multiple entities.

        Args:
            entities: List of entity dictionaries
            score_field: Field containing the score
            id_field: Field containing the entity ID

        Returns:
            List of RiskClassification objects
        """
        results = []
        for entity in entities:
            score = entity.get(score_field)
            if score is None:
                logger.warning(f"Missing score for entity {entity.get(id_field)}")
                continue

            entity_id = str(entity.get(id_field, "unknown"))
            metadata = {k: v for k, v in entity.items() if k not in [score_field, id_field]}

            result = self.classify(score, entity_id=entity_id, metadata=metadata)
            results.append(result)

        return results

    def _calculate_band_position(self, score: float, threshold: RiskThreshold) -> float:
        """Calculate position within the threshold band (0-100%)."""
        band_size = threshold.max_score - threshold.min_score
        if band_size == 0:
            return 100.0
        position = ((score - threshold.min_score) / band_size) * 100
        return round(position, 1)

    def get_risk_distribution(
        self,
        classifications: List[RiskClassification]
    ) -> Dict[str, Any]:
        """
        Calculate risk distribution statistics.

        Args:
            classifications: List of classification results

        Returns:
            Dictionary with distribution stats
        """
        if not classifications:
            return {"total": 0, "distribution": {}}

        distribution = {}
        for level in RiskLevel:
            count = sum(1 for c in classifications if c.level == level)
            distribution[level.value] = {
                "count": count,
                "percentage": round(count / len(classifications) * 100, 1),
                "color": level.color
            }

        return {
            "total": len(classifications),
            "distribution": distribution,
            "highest_risk": max(classifications, key=lambda c: c.level.priority).level.value,
            "average_score": round(sum(c.score for c in classifications) / len(classifications), 2)
        }

    def get_threshold_summary(self) -> List[Dict[str, Any]]:
        """Get summary of configured thresholds."""
        return [
            {
                "level": t.level.value,
                "color": t.level.color,
                "icon": t.level.icon,
                "min_score": t.min_score,
                "max_score": t.max_score,
                "description": t.description,
                "action_required": t.action_required
            }
            for t in self.thresholds
        ]


class MultiDimensionalRiskClassifier:
    """
    Classifies risk based on multiple dimensions/factors.

    Useful for comprehensive risk assessments that consider
    multiple independent risk categories.

    Example:
    ```python
    classifier = MultiDimensionalRiskClassifier(
        dimensions={
            "financial": RiskClassifier(direction="lower_is_riskier"),
            "operational": RiskClassifier(direction="lower_is_riskier"),
            "compliance": RiskClassifier(direction="lower_is_riskier"),
        },
        weights={"financial": 0.4, "operational": 0.35, "compliance": 0.25}
    )

    result = classifier.classify({
        "financial": 75,
        "operational": 45,
        "compliance": 90
    }, entity_id="PROJECT-001")
    ```
    """

    def __init__(
        self,
        dimensions: Dict[str, RiskClassifier],
        weights: Optional[Dict[str, float]] = None,
        aggregation: str = "weighted_average"  # or "worst_case", "best_case"
    ):
        self.dimensions = dimensions
        self.aggregation = aggregation

        # Default to equal weights
        if weights:
            self.weights = weights
        else:
            equal_weight = 1.0 / len(dimensions)
            self.weights = {name: equal_weight for name in dimensions}

        # Normalize weights
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}

    def classify(
        self,
        scores: Dict[str, float],
        entity_id: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None
    ) -> RiskClassification:
        """
        Classify based on multiple dimension scores.

        Args:
            scores: Dict mapping dimension names to scores
            entity_id: Entity identifier
            metadata: Additional metadata

        Returns:
            Aggregated RiskClassification
        """
        dimension_results = {}
        factors = []

        for dim_name, classifier in self.dimensions.items():
            score = scores.get(dim_name)
            if score is None:
                logger.warning(f"Missing score for dimension '{dim_name}'")
                continue

            result = classifier.classify(score, entity_id=f"{entity_id}_{dim_name}")
            dimension_results[dim_name] = result

            factors.append({
                "dimension": dim_name,
                "score": score,
                "level": result.level.value,
                "weight": self.weights.get(dim_name, 0),
                "weighted_contribution": score * self.weights.get(dim_name, 0)
            })

        # Calculate aggregate score
        if self.aggregation == "weighted_average":
            aggregate_score = sum(
                scores.get(dim, 0) * self.weights.get(dim, 0)
                for dim in self.dimensions.keys()
            )
        elif self.aggregation == "worst_case":
            aggregate_score = min(
                scores.get(dim, 100) for dim in self.dimensions.keys()
            )
        else:  # best_case
            aggregate_score = max(
                scores.get(dim, 0) for dim in self.dimensions.keys()
            )

        # Use first dimension's classifier for final classification
        first_classifier = list(self.dimensions.values())[0]
        final_result = first_classifier.classify(
            aggregate_score,
            entity_id=entity_id,
            factors=factors,
            metadata={
                **(metadata or {}),
                "dimension_results": {
                    k: v.to_dict() for k, v in dimension_results.items()
                },
                "aggregation_method": self.aggregation
            }
        )

        return final_result


# =============================================================================
# Factory Functions for Common Use Cases
# =============================================================================

def create_health_score_classifier() -> RiskClassifier:
    """Create a classifier for health scores (0-100, higher = healthier)."""
    return RiskClassifier(direction="lower_is_riskier")


def create_risk_score_classifier() -> RiskClassifier:
    """Create a classifier for risk scores (0-100, higher = riskier)."""
    return RiskClassifier(direction="higher_is_riskier")


def create_financial_risk_classifier() -> RiskClassifier:
    """Create a classifier for financial risk metrics."""
    thresholds = [
        RiskThreshold(RiskLevel.MINIMAL, 0, 10,
                     "Excellent financial health",
                     "Standard monitoring"),
        RiskThreshold(RiskLevel.LOW, 10, 25,
                     "Good financial position with minor variances",
                     "Track budget adherence"),
        RiskThreshold(RiskLevel.MEDIUM, 25, 40,
                     "Moderate financial pressure",
                     "Review spending; identify cost reduction opportunities"),
        RiskThreshold(RiskLevel.HIGH, 40, 60,
                     "Significant budget concerns",
                     "Implement cost controls; escalate to finance"),
        RiskThreshold(RiskLevel.CRITICAL, 60, 100,
                     "Severe financial distress",
                     "Emergency budget review; executive intervention"),
    ]
    return RiskClassifier(direction="higher_is_riskier", thresholds=thresholds)


def create_project_risk_classifier() -> MultiDimensionalRiskClassifier:
    """Create a multi-dimensional project risk classifier."""
    return MultiDimensionalRiskClassifier(
        dimensions={
            "schedule": create_health_score_classifier(),
            "budget": create_health_score_classifier(),
            "scope": create_health_score_classifier(),
            "quality": create_health_score_classifier(),
        },
        weights={
            "schedule": 0.30,
            "budget": 0.30,
            "scope": 0.20,
            "quality": 0.20,
        },
        aggregation="weighted_average"
    )


# =============================================================================
# Demo / Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("RISK CLASSIFICATION DEMO")
    print("=" * 60)

    # Create classifier for health scores
    classifier = create_health_score_classifier()

    print("\nThreshold Configuration:")
    for t in classifier.get_threshold_summary():
        print(f"  {t['icon']} {t['level']}: {t['min_score']}-{t['max_score']}")

    # Classify sample entities
    entities = [
        {"id": "P001", "name": "Alpha Project", "score": 85},
        {"id": "P002", "name": "Beta Project", "score": 55},
        {"id": "P003", "name": "Gamma Project", "score": 25},
        {"id": "P004", "name": "Delta Project", "score": 72},
    ]

    print("\n" + "-" * 40)
    print("Classification Results:")

    results = classifier.classify_batch(entities)
    for result in results:
        print(f"\n  {result.entity_id}:")
        print(f"    Score: {result.score}")
        print(f"    Risk: {result.level.icon} {result.level.value}")
        print(f"    Action: {result.action_required}")

    # Distribution
    dist = classifier.get_risk_distribution(results)
    print("\n" + "-" * 40)
    print("Risk Distribution:")
    for level, stats in dist["distribution"].items():
        if stats["count"] > 0:
            print(f"  {level}: {stats['count']} ({stats['percentage']}%)")

    # Multi-dimensional example
    print("\n" + "-" * 40)
    print("Multi-Dimensional Classification:")

    multi_classifier = create_project_risk_classifier()

    result = multi_classifier.classify(
        scores={
            "schedule": 80,
            "budget": 45,
            "scope": 70,
            "quality": 90
        },
        entity_id="PROJECT-X"
    )

    print(f"\n  Entity: {result.entity_id}")
    print(f"  Aggregate Score: {result.score}")
    print(f"  Overall Risk: {result.level.icon} {result.level.value}")
    print("\n  Dimension Breakdown:")
    for factor in result.factors:
        print(f"    {factor['dimension']}: {factor['score']} ({factor['level']})")
