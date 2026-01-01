"""
Weighted Scoring Pattern - Universal Business Solution Framework

A flexible, configurable multi-component scoring engine that calculates
weighted aggregate scores from multiple metrics. Useful for:
- Health scores (contracts, vendors, projects, employees)
- Risk assessments
- Performance evaluations
- Quality metrics
- Compliance scoring

Extracted from: Contract Oversight System
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Union
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ScoreDirection(Enum):
    """Whether higher values are better or worse."""
    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"


@dataclass
class ScoreComponent:
    """Definition of a single scoring component."""
    name: str
    weight: float  # 0.0 to 1.0, all weights should sum to 1.0
    direction: ScoreDirection = ScoreDirection.HIGHER_IS_BETTER
    min_value: float = 0.0
    max_value: float = 100.0
    description: str = ""

    def normalize(self, value: float) -> float:
        """Normalize a value to 0-100 scale."""
        if self.max_value == self.min_value:
            return 100.0 if value >= self.max_value else 0.0

        # Clamp value to range
        value = max(self.min_value, min(self.max_value, value))

        # Normalize to 0-100
        normalized = ((value - self.min_value) / (self.max_value - self.min_value)) * 100

        # Invert if lower is better
        if self.direction == ScoreDirection.LOWER_IS_BETTER:
            normalized = 100 - normalized

        return round(normalized, 2)


@dataclass
class ScoreResult:
    """Result of scoring an entity."""
    entity_id: str
    overall_score: float
    grade: str
    component_scores: Dict[str, float]
    component_details: Dict[str, Dict[str, Any]]
    risk_level: str
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class WeightedScoringEngine:
    """
    A configurable multi-component weighted scoring engine.

    Example usage:
    ```python
    # Define scoring components
    components = [
        ScoreComponent("cost_variance", weight=0.30, direction=ScoreDirection.LOWER_IS_BETTER,
                      min_value=0, max_value=50, description="Budget variance percentage"),
        ScoreComponent("schedule_variance", weight=0.25, direction=ScoreDirection.LOWER_IS_BETTER,
                      min_value=0, max_value=30, description="Schedule delay in days"),
        ScoreComponent("performance", weight=0.25, direction=ScoreDirection.HIGHER_IS_BETTER,
                      min_value=0, max_value=100, description="Milestone completion %"),
        ScoreComponent("compliance", weight=0.20, direction=ScoreDirection.HIGHER_IS_BETTER,
                      min_value=0, max_value=100, description="Compliance checklist %"),
    ]

    # Create engine
    engine = WeightedScoringEngine(components)

    # Score an entity
    result = engine.score({
        "cost_variance": 15,      # 15% over budget
        "schedule_variance": 5,   # 5 days behind
        "performance": 85,        # 85% milestones complete
        "compliance": 100,        # Full compliance
    }, entity_id="CONTRACT-001")

    print(f"Overall Score: {result.overall_score}")  # e.g., 78.5
    print(f"Grade: {result.grade}")                  # e.g., "C"
    print(f"Risk Level: {result.risk_level}")        # e.g., "Medium"
    ```
    """

    # Default grade thresholds
    DEFAULT_GRADE_THRESHOLDS = {
        90: "A",
        80: "B",
        70: "C",
        60: "D",
        0: "F"
    }

    # Default risk thresholds (based on inverse of score)
    DEFAULT_RISK_THRESHOLDS = {
        80: "Low",
        60: "Medium",
        40: "High",
        0: "Critical"
    }

    def __init__(
        self,
        components: List[ScoreComponent],
        grade_thresholds: Optional[Dict[int, str]] = None,
        risk_thresholds: Optional[Dict[int, str]] = None,
        recommendation_rules: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Initialize the scoring engine.

        Args:
            components: List of ScoreComponent definitions
            grade_thresholds: Dict mapping minimum scores to grades (descending order)
            risk_thresholds: Dict mapping minimum scores to risk levels (descending order)
            recommendation_rules: List of rules for generating recommendations
        """
        self.components = {c.name: c for c in components}
        self.grade_thresholds = grade_thresholds or self.DEFAULT_GRADE_THRESHOLDS
        self.risk_thresholds = risk_thresholds or self.DEFAULT_RISK_THRESHOLDS
        self.recommendation_rules = recommendation_rules or []

        # Validate weights sum to 1.0
        total_weight = sum(c.weight for c in components)
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(f"Component weights sum to {total_weight}, not 1.0. Normalizing...")
            for c in components:
                c.weight = c.weight / total_weight

    def score(
        self,
        values: Dict[str, float],
        entity_id: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None
    ) -> ScoreResult:
        """
        Calculate weighted score for an entity.

        Args:
            values: Dict mapping component names to raw values
            entity_id: Identifier for the entity being scored
            metadata: Optional additional metadata to include in result

        Returns:
            ScoreResult with overall score, grade, component breakdown
        """
        component_scores = {}
        component_details = {}
        weighted_sum = 0.0

        for name, component in self.components.items():
            raw_value = values.get(name)

            if raw_value is None:
                logger.warning(f"Missing value for component '{name}', using min value")
                raw_value = component.min_value

            # Normalize the score
            normalized = component.normalize(raw_value)
            component_scores[name] = normalized

            # Calculate weighted contribution
            weighted_contribution = normalized * component.weight
            weighted_sum += weighted_contribution

            # Store details
            component_details[name] = {
                "raw_value": raw_value,
                "normalized_score": normalized,
                "weight": component.weight,
                "weighted_contribution": round(weighted_contribution, 2),
                "direction": component.direction.value,
                "description": component.description
            }

        overall_score = round(weighted_sum, 2)
        grade = self._determine_grade(overall_score)
        risk_level = self._determine_risk_level(overall_score)
        recommendations = self._generate_recommendations(component_scores, overall_score)

        return ScoreResult(
            entity_id=entity_id,
            overall_score=overall_score,
            grade=grade,
            component_scores=component_scores,
            component_details=component_details,
            risk_level=risk_level,
            recommendations=recommendations,
            metadata=metadata or {}
        )

    def score_batch(
        self,
        entities: List[Dict[str, Any]],
        id_field: str = "id",
        value_fields: Optional[List[str]] = None
    ) -> List[ScoreResult]:
        """
        Score multiple entities at once.

        Args:
            entities: List of dicts containing entity data
            id_field: Name of the field containing entity ID
            value_fields: List of fields to use for scoring (defaults to component names)

        Returns:
            List of ScoreResult objects
        """
        results = []
        value_fields = value_fields or list(self.components.keys())

        for entity in entities:
            entity_id = str(entity.get(id_field, "unknown"))
            values = {field: entity.get(field) for field in value_fields}

            # Extract non-scoring fields as metadata
            metadata = {k: v for k, v in entity.items()
                       if k not in value_fields and k != id_field}

            try:
                result = self.score(values, entity_id=entity_id, metadata=metadata)
                results.append(result)
            except Exception as e:
                logger.error(f"Error scoring entity {entity_id}: {e}")
                continue

        return results

    def _determine_grade(self, score: float) -> str:
        """Determine letter grade from score."""
        for threshold, grade in sorted(self.grade_thresholds.items(), reverse=True):
            if score >= threshold:
                return grade
        return "F"

    def _determine_risk_level(self, score: float) -> str:
        """Determine risk level from score."""
        for threshold, level in sorted(self.risk_thresholds.items(), reverse=True):
            if score >= threshold:
                return level
        return "Critical"

    def _generate_recommendations(
        self,
        component_scores: Dict[str, float],
        overall_score: float
    ) -> List[str]:
        """Generate recommendations based on scores."""
        recommendations = []

        # Default recommendations based on low component scores
        for name, score in component_scores.items():
            if score < 50:
                component = self.components[name]
                recommendations.append(
                    f"Critical: Improve {name} (currently {score:.0f}/100) - {component.description}"
                )
            elif score < 70:
                component = self.components[name]
                recommendations.append(
                    f"Warning: Monitor {name} (currently {score:.0f}/100) - {component.description}"
                )

        # Apply custom recommendation rules
        for rule in self.recommendation_rules:
            condition = rule.get("condition")
            message = rule.get("message")

            if condition and message:
                try:
                    if condition(component_scores, overall_score):
                        recommendations.append(message)
                except Exception as e:
                    logger.warning(f"Error evaluating recommendation rule: {e}")

        return recommendations

    def get_component_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary of all scoring components."""
        return {
            name: {
                "weight": c.weight,
                "weight_percent": f"{c.weight * 100:.0f}%",
                "direction": c.direction.value,
                "range": f"{c.min_value} - {c.max_value}",
                "description": c.description
            }
            for name, c in self.components.items()
        }


class AggregatedScoringEngine:
    """
    Aggregates scores across multiple entities (e.g., vendor score from contracts).

    Example usage:
    ```python
    # Create base scoring engine
    contract_scorer = WeightedScoringEngine(contract_components)

    # Create aggregator
    aggregator = AggregatedScoringEngine(
        base_engine=contract_scorer,
        aggregation_weights={"value": 0.7, "count": 0.3}  # Weight by contract value
    )

    # Score vendor across multiple contracts
    contracts = [
        {"id": "C1", "value": 100000, "cost_variance": 5, ...},
        {"id": "C2", "value": 50000, "cost_variance": 20, ...},
    ]

    vendor_score = aggregator.aggregate(
        entities=contracts,
        group_id="VENDOR-001",
        weight_field="value"
    )
    ```
    """

    def __init__(
        self,
        base_engine: WeightedScoringEngine,
        aggregation_method: str = "weighted_average"  # or "simple_average", "min", "max"
    ):
        self.base_engine = base_engine
        self.aggregation_method = aggregation_method

    def aggregate(
        self,
        entities: List[Dict[str, Any]],
        group_id: str,
        weight_field: Optional[str] = None,
        id_field: str = "id"
    ) -> ScoreResult:
        """
        Aggregate scores across multiple entities.

        Args:
            entities: List of entities to score and aggregate
            group_id: ID for the aggregated result (e.g., vendor ID)
            weight_field: Field to use for weighting (e.g., "contract_value")
            id_field: Field containing entity IDs

        Returns:
            Aggregated ScoreResult
        """
        if not entities:
            return ScoreResult(
                entity_id=group_id,
                overall_score=0.0,
                grade="F",
                component_scores={},
                component_details={},
                risk_level="Unknown",
                recommendations=["No data available for scoring"]
            )

        # Score all entities
        individual_scores = self.base_engine.score_batch(entities, id_field=id_field)

        # Get weights
        if weight_field and self.aggregation_method == "weighted_average":
            weights = [e.get(weight_field, 1) for e in entities]
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]
        else:
            weights = [1 / len(entities)] * len(entities)

        # Aggregate overall scores
        if self.aggregation_method in ["weighted_average", "simple_average"]:
            overall_score = sum(
                score.overall_score * weight
                for score, weight in zip(individual_scores, weights)
            )
        elif self.aggregation_method == "min":
            overall_score = min(s.overall_score for s in individual_scores)
        elif self.aggregation_method == "max":
            overall_score = max(s.overall_score for s in individual_scores)
        else:
            overall_score = sum(s.overall_score for s in individual_scores) / len(individual_scores)

        # Aggregate component scores
        component_scores = {}
        for name in self.base_engine.components.keys():
            if self.aggregation_method in ["weighted_average", "simple_average"]:
                component_scores[name] = sum(
                    s.component_scores.get(name, 0) * weight
                    for s, weight in zip(individual_scores, weights)
                )
            else:
                values = [s.component_scores.get(name, 0) for s in individual_scores]
                component_scores[name] = min(values) if self.aggregation_method == "min" else max(values)

        overall_score = round(overall_score, 2)

        return ScoreResult(
            entity_id=group_id,
            overall_score=overall_score,
            grade=self.base_engine._determine_grade(overall_score),
            component_scores=component_scores,
            component_details={
                "aggregation_method": self.aggregation_method,
                "entity_count": len(entities),
                "individual_scores": [
                    {"id": s.entity_id, "score": s.overall_score, "grade": s.grade}
                    for s in individual_scores
                ]
            },
            risk_level=self.base_engine._determine_risk_level(overall_score),
            metadata={"weight_field": weight_field}
        )


# =============================================================================
# Example Usage and Factory Functions
# =============================================================================

def create_contract_scoring_engine() -> WeightedScoringEngine:
    """Factory function: Create a contract health scoring engine."""
    components = [
        ScoreComponent(
            name="cost_variance",
            weight=0.30,
            direction=ScoreDirection.LOWER_IS_BETTER,
            min_value=0,
            max_value=50,
            description="Budget variance percentage (0% = on budget, 50% = 50% over)"
        ),
        ScoreComponent(
            name="schedule_variance",
            weight=0.25,
            direction=ScoreDirection.LOWER_IS_BETTER,
            min_value=0,
            max_value=60,
            description="Schedule delay in days"
        ),
        ScoreComponent(
            name="performance",
            weight=0.25,
            direction=ScoreDirection.HIGHER_IS_BETTER,
            min_value=0,
            max_value=100,
            description="Milestone completion percentage"
        ),
        ScoreComponent(
            name="compliance",
            weight=0.20,
            direction=ScoreDirection.HIGHER_IS_BETTER,
            min_value=0,
            max_value=100,
            description="Compliance checklist completion"
        ),
    ]

    return WeightedScoringEngine(components)


def create_employee_performance_engine() -> WeightedScoringEngine:
    """Factory function: Create an employee performance scoring engine."""
    components = [
        ScoreComponent(
            name="productivity",
            weight=0.30,
            direction=ScoreDirection.HIGHER_IS_BETTER,
            min_value=0,
            max_value=100,
            description="Task completion rate"
        ),
        ScoreComponent(
            name="quality",
            weight=0.25,
            direction=ScoreDirection.HIGHER_IS_BETTER,
            min_value=0,
            max_value=100,
            description="Work quality score"
        ),
        ScoreComponent(
            name="attendance",
            weight=0.20,
            direction=ScoreDirection.HIGHER_IS_BETTER,
            min_value=0,
            max_value=100,
            description="Attendance rate"
        ),
        ScoreComponent(
            name="collaboration",
            weight=0.15,
            direction=ScoreDirection.HIGHER_IS_BETTER,
            min_value=0,
            max_value=100,
            description="Team collaboration score"
        ),
        ScoreComponent(
            name="initiative",
            weight=0.10,
            direction=ScoreDirection.HIGHER_IS_BETTER,
            min_value=0,
            max_value=100,
            description="Proactive contribution score"
        ),
    ]

    return WeightedScoringEngine(components)


def create_vendor_scoring_engine() -> WeightedScoringEngine:
    """Factory function: Create a vendor evaluation scoring engine."""
    components = [
        ScoreComponent(
            name="delivery_performance",
            weight=0.25,
            direction=ScoreDirection.HIGHER_IS_BETTER,
            min_value=0,
            max_value=100,
            description="On-time delivery rate"
        ),
        ScoreComponent(
            name="quality_rating",
            weight=0.25,
            direction=ScoreDirection.HIGHER_IS_BETTER,
            min_value=0,
            max_value=100,
            description="Product/service quality"
        ),
        ScoreComponent(
            name="price_competitiveness",
            weight=0.20,
            direction=ScoreDirection.HIGHER_IS_BETTER,
            min_value=0,
            max_value=100,
            description="Price vs market benchmark"
        ),
        ScoreComponent(
            name="responsiveness",
            weight=0.15,
            direction=ScoreDirection.HIGHER_IS_BETTER,
            min_value=0,
            max_value=100,
            description="Communication and issue resolution"
        ),
        ScoreComponent(
            name="compliance",
            weight=0.15,
            direction=ScoreDirection.HIGHER_IS_BETTER,
            min_value=0,
            max_value=100,
            description="Contract and regulatory compliance"
        ),
    ]

    return WeightedScoringEngine(components)


# =============================================================================
# Demo / Test
# =============================================================================

if __name__ == "__main__":
    # Demo: Contract scoring
    print("=" * 60)
    print("WEIGHTED SCORING ENGINE DEMO")
    print("=" * 60)

    # Create engine
    engine = create_contract_scoring_engine()

    # Show component summary
    print("\nScoring Components:")
    for name, info in engine.get_component_summary().items():
        print(f"  {name}: {info['weight_percent']} weight, {info['direction']}")

    # Score a sample contract
    print("\n" + "-" * 40)
    print("Scoring Sample Contract:")

    result = engine.score(
        values={
            "cost_variance": 15,      # 15% over budget
            "schedule_variance": 10,  # 10 days behind
            "performance": 85,        # 85% milestones complete
            "compliance": 100,        # Full compliance
        },
        entity_id="CONTRACT-2024-001"
    )

    print(f"\nEntity: {result.entity_id}")
    print(f"Overall Score: {result.overall_score}")
    print(f"Grade: {result.grade}")
    print(f"Risk Level: {result.risk_level}")

    print("\nComponent Breakdown:")
    for name, details in result.component_details.items():
        print(f"  {name}:")
        print(f"    Raw: {details['raw_value']}, Normalized: {details['normalized_score']}")
        print(f"    Contribution: {details['weighted_contribution']} ({details['weight']*100:.0f}% weight)")

    if result.recommendations:
        print("\nRecommendations:")
        for rec in result.recommendations:
            print(f"  - {rec}")

    # Batch scoring demo
    print("\n" + "-" * 40)
    print("Batch Scoring Multiple Contracts:")

    contracts = [
        {"id": "C001", "cost_variance": 5, "schedule_variance": 0, "performance": 95, "compliance": 100},
        {"id": "C002", "cost_variance": 25, "schedule_variance": 20, "performance": 60, "compliance": 80},
        {"id": "C003", "cost_variance": 10, "schedule_variance": 5, "performance": 90, "compliance": 95},
    ]

    results = engine.score_batch(contracts)

    for r in results:
        print(f"  {r.entity_id}: Score={r.overall_score}, Grade={r.grade}, Risk={r.risk_level}")
