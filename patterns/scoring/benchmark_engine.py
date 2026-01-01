"""
Benchmark Engine Pattern - Universal Business Solution Framework

A KPI benchmarking system that compares actual performance against
industry standards, targets, or historical baselines. Generates
gap analysis, scores, and recommendations. Useful for:
- Procurement benchmarking
- Financial KPI analysis
- Operational efficiency metrics
- Industry comparison reports
- Performance gap analysis
- Continuous improvement tracking

Extracted from: Contract Oversight System (Coupa 2025 Benchmarks)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Union
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class KPIDirection(Enum):
    """Whether higher or lower values are better."""
    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"


class KPICategory(Enum):
    """Categories for organizing KPIs."""
    FINANCIAL = "Financial"
    OPERATIONAL = "Operational"
    QUALITY = "Quality"
    EFFICIENCY = "Efficiency"
    COMPLIANCE = "Compliance"
    CUSTOMER = "Customer"
    PROCUREMENT = "Procurement"
    HR = "Human Resources"
    CUSTOM = "Custom"


@dataclass
class KPIDefinition:
    """
    Definition of a Key Performance Indicator.

    Example:
    ```python
    KPIDefinition(
        kpi_id="on_time_delivery",
        name="On-Time Delivery Rate",
        category=KPICategory.OPERATIONAL,
        direction=KPIDirection.HIGHER_IS_BETTER,
        benchmark_value=95.0,
        unit="%",
        description="Percentage of deliveries completed on schedule"
    )
    ```
    """
    kpi_id: str
    name: str
    benchmark_value: float
    direction: KPIDirection = KPIDirection.HIGHER_IS_BETTER
    category: KPICategory = KPICategory.CUSTOM
    unit: str = ""
    description: str = ""
    weight: float = 1.0  # For weighted aggregation
    threshold_excellent: Optional[float] = None  # Better than benchmark by this %
    threshold_poor: Optional[float] = None  # Worse than benchmark by this %


@dataclass
class KPIScore:
    """Score for a single KPI."""
    kpi_id: str
    kpi_name: str
    actual_value: float
    benchmark_value: float
    score: float  # 0-100
    gap: float  # Actual - Benchmark
    gap_percent: float
    direction: KPIDirection
    rating: str  # "Excellent", "Good", "Fair", "Poor", "Critical"
    unit: str = ""
    recommendation: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CategoryScore:
    """Aggregated score for a category of KPIs."""
    category: str
    score: float
    kpi_count: int
    kpi_scores: List[KPIScore]
    strengths: List[str]
    improvements: List[str]


@dataclass
class BenchmarkReport:
    """Complete benchmark analysis report."""
    entity_id: str
    overall_score: float
    overall_rating: str
    grade: str
    category_scores: Dict[str, CategoryScore]
    kpi_scores: List[KPIScore]
    top_strengths: List[str]
    top_improvements: List[str]
    recommendations: List[str]
    percentile: Optional[float] = None  # Percentile ranking if peer data available
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "overall_score": self.overall_score,
            "overall_rating": self.overall_rating,
            "grade": self.grade,
            "category_scores": {
                cat: {
                    "score": cs.score,
                    "kpi_count": cs.kpi_count,
                    "strengths": cs.strengths,
                    "improvements": cs.improvements
                }
                for cat, cs in self.category_scores.items()
            },
            "kpi_count": len(self.kpi_scores),
            "top_strengths": self.top_strengths,
            "top_improvements": self.top_improvements,
            "recommendations": self.recommendations,
            "percentile": self.percentile
        }


class BenchmarkEngine:
    """
    KPI benchmarking engine that compares actual values against benchmarks.

    Example usage:
    ```python
    # Define KPIs
    kpis = [
        KPIDefinition("cost_savings", "Cost Savings Rate", 15.0,
                     KPIDirection.HIGHER_IS_BETTER, KPICategory.FINANCIAL, "%"),
        KPIDefinition("cycle_time", "Procurement Cycle Time", 30.0,
                     KPIDirection.LOWER_IS_BETTER, KPICategory.EFFICIENCY, "days"),
        KPIDefinition("compliance_rate", "Contract Compliance", 95.0,
                     KPIDirection.HIGHER_IS_BETTER, KPICategory.COMPLIANCE, "%"),
    ]

    engine = BenchmarkEngine(kpis)

    # Analyze actual performance
    report = engine.analyze({
        "cost_savings": 18.5,
        "cycle_time": 25,
        "compliance_rate": 92
    }, entity_id="DEPT-001")

    print(f"Overall Score: {report.overall_score}")
    print(f"Grade: {report.grade}")
    ```
    """

    # Rating thresholds (score -> rating)
    RATING_THRESHOLDS = {
        90: "Excellent",
        75: "Good",
        60: "Fair",
        40: "Poor",
        0: "Critical"
    }

    # Grade thresholds
    GRADE_THRESHOLDS = {
        90: "A",
        80: "B",
        70: "C",
        60: "D",
        0: "F"
    }

    def __init__(
        self,
        kpis: List[KPIDefinition],
        category_weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize benchmark engine.

        Args:
            kpis: List of KPI definitions
            category_weights: Optional weights for category aggregation
        """
        self.kpis = {kpi.kpi_id: kpi for kpi in kpis}
        self.category_weights = category_weights or {}

        # Group KPIs by category
        self.kpis_by_category: Dict[str, List[KPIDefinition]] = {}
        for kpi in kpis:
            cat = kpi.category.value
            if cat not in self.kpis_by_category:
                self.kpis_by_category[cat] = []
            self.kpis_by_category[cat].append(kpi)

    def score_kpi(
        self,
        kpi: KPIDefinition,
        actual_value: float
    ) -> KPIScore:
        """
        Score a single KPI against its benchmark.

        Args:
            kpi: KPI definition
            actual_value: Actual measured value

        Returns:
            KPIScore with score, gap, and rating
        """
        benchmark = kpi.benchmark_value

        # Calculate gap
        gap = actual_value - benchmark

        # Calculate gap percentage (relative to benchmark)
        if benchmark != 0:
            gap_percent = (gap / abs(benchmark)) * 100
        else:
            gap_percent = 100 if actual_value > 0 else 0

        # Calculate score (0-100)
        score = self._calculate_score(actual_value, benchmark, kpi.direction)

        # Determine rating
        rating = self._determine_rating(score)

        # Generate recommendation
        recommendation = self._generate_recommendation(kpi, actual_value, benchmark, rating)

        return KPIScore(
            kpi_id=kpi.kpi_id,
            kpi_name=kpi.name,
            actual_value=round(actual_value, 2),
            benchmark_value=benchmark,
            score=round(score, 1),
            gap=round(gap, 2),
            gap_percent=round(gap_percent, 1),
            direction=kpi.direction,
            rating=rating,
            unit=kpi.unit,
            recommendation=recommendation,
            metadata={
                "category": kpi.category.value,
                "weight": kpi.weight,
                "description": kpi.description
            }
        )

    def _calculate_score(
        self,
        actual: float,
        benchmark: float,
        direction: KPIDirection
    ) -> float:
        """Calculate score based on actual vs benchmark."""
        if benchmark == 0:
            return 100 if actual >= 0 else 0

        if direction == KPIDirection.HIGHER_IS_BETTER:
            if actual >= benchmark:
                # Exceeds benchmark: score 100 + bonus (capped at 120)
                bonus = min(20, ((actual - benchmark) / benchmark) * 20)
                return min(120, 100 + bonus)
            else:
                # Below benchmark: proportional score
                return max(0, (actual / benchmark) * 100)
        else:  # LOWER_IS_BETTER
            if actual <= benchmark:
                # Better than benchmark: score 100 + bonus
                if actual == 0:
                    return 120
                bonus = min(20, ((benchmark - actual) / benchmark) * 20)
                return min(120, 100 + bonus)
            else:
                # Worse than benchmark: proportional score
                # Score decreases as actual exceeds benchmark
                excess_ratio = actual / benchmark
                return max(0, 100 - ((excess_ratio - 1) * 100))

    def _determine_rating(self, score: float) -> str:
        """Determine rating from score."""
        for threshold, rating in sorted(self.RATING_THRESHOLDS.items(), reverse=True):
            if score >= threshold:
                return rating
        return "Critical"

    def _determine_grade(self, score: float) -> str:
        """Determine letter grade from score."""
        for threshold, grade in sorted(self.GRADE_THRESHOLDS.items(), reverse=True):
            if score >= threshold:
                return grade
        return "F"

    def _generate_recommendation(
        self,
        kpi: KPIDefinition,
        actual: float,
        benchmark: float,
        rating: str
    ) -> str:
        """Generate improvement recommendation."""
        if rating in ["Excellent", "Good"]:
            return f"Maintain current performance in {kpi.name}"

        direction_text = "increase" if kpi.direction == KPIDirection.HIGHER_IS_BETTER else "reduce"
        gap = abs(actual - benchmark)

        if rating == "Fair":
            return f"Minor improvement needed: {direction_text} {kpi.name} by {gap:.1f}{kpi.unit}"
        elif rating == "Poor":
            return f"Priority action: {direction_text} {kpi.name} significantly (gap: {gap:.1f}{kpi.unit})"
        else:
            return f"CRITICAL: Immediate intervention required for {kpi.name}"

    def analyze(
        self,
        actual_values: Dict[str, float],
        entity_id: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None
    ) -> BenchmarkReport:
        """
        Perform complete benchmark analysis.

        Args:
            actual_values: Dict mapping KPI IDs to actual values
            entity_id: Entity identifier
            metadata: Additional metadata

        Returns:
            Complete BenchmarkReport
        """
        kpi_scores = []
        category_kpis: Dict[str, List[KPIScore]] = {}

        # Score each KPI
        for kpi_id, kpi in self.kpis.items():
            actual = actual_values.get(kpi_id)
            if actual is None:
                logger.warning(f"Missing value for KPI '{kpi_id}'")
                continue

            score = self.score_kpi(kpi, actual)
            kpi_scores.append(score)

            # Group by category
            cat = kpi.category.value
            if cat not in category_kpis:
                category_kpis[cat] = []
            category_kpis[cat].append(score)

        # Calculate category scores
        category_scores = {}
        for cat, scores in category_kpis.items():
            cat_score = self._calculate_category_score(cat, scores)
            category_scores[cat] = cat_score

        # Calculate overall score (weighted by category)
        overall_score = self._calculate_overall_score(category_scores)
        overall_rating = self._determine_rating(overall_score)
        grade = self._determine_grade(overall_score)

        # Identify strengths and improvements
        sorted_kpis = sorted(kpi_scores, key=lambda k: k.score, reverse=True)
        top_strengths = [
            f"{k.kpi_name}: {k.actual_value}{k.unit} ({k.rating})"
            for k in sorted_kpis[:3] if k.rating in ["Excellent", "Good"]
        ]
        top_improvements = [
            f"{k.kpi_name}: {k.actual_value}{k.unit} vs benchmark {k.benchmark_value}{k.unit}"
            for k in sorted_kpis[-3:] if k.rating in ["Poor", "Critical"]
        ]

        # Compile recommendations
        recommendations = [k.recommendation for k in kpi_scores if k.rating in ["Poor", "Critical"]]

        return BenchmarkReport(
            entity_id=entity_id,
            overall_score=round(overall_score, 1),
            overall_rating=overall_rating,
            grade=grade,
            category_scores=category_scores,
            kpi_scores=kpi_scores,
            top_strengths=top_strengths,
            top_improvements=top_improvements,
            recommendations=recommendations[:5],  # Top 5 recommendations
            metadata=metadata or {}
        )

    def _calculate_category_score(
        self,
        category: str,
        kpi_scores: List[KPIScore]
    ) -> CategoryScore:
        """Calculate aggregated category score."""
        if not kpi_scores:
            return CategoryScore(
                category=category,
                score=0,
                kpi_count=0,
                kpi_scores=[],
                strengths=[],
                improvements=[]
            )

        # Weighted average of KPI scores
        total_weight = sum(k.metadata.get("weight", 1) for k in kpi_scores)
        weighted_sum = sum(
            k.score * k.metadata.get("weight", 1) for k in kpi_scores
        )
        avg_score = weighted_sum / total_weight if total_weight > 0 else 0

        # Identify category strengths and improvements
        strengths = [k.kpi_name for k in kpi_scores if k.rating in ["Excellent", "Good"]]
        improvements = [k.kpi_name for k in kpi_scores if k.rating in ["Poor", "Critical"]]

        return CategoryScore(
            category=category,
            score=round(avg_score, 1),
            kpi_count=len(kpi_scores),
            kpi_scores=kpi_scores,
            strengths=strengths,
            improvements=improvements
        )

    def _calculate_overall_score(
        self,
        category_scores: Dict[str, CategoryScore]
    ) -> float:
        """Calculate overall score from category scores."""
        if not category_scores:
            return 0

        # Use category weights if provided, otherwise equal weighting
        total_weight = 0
        weighted_sum = 0

        for cat, cat_score in category_scores.items():
            weight = self.category_weights.get(cat, 1.0)
            weighted_sum += cat_score.score * weight
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0

    def compare_entities(
        self,
        entities: List[Dict[str, Any]],
        id_field: str = "id"
    ) -> List[BenchmarkReport]:
        """
        Compare multiple entities against benchmarks.

        Args:
            entities: List of entity data dictionaries
            id_field: Field containing entity ID

        Returns:
            List of BenchmarkReports, sorted by overall score
        """
        reports = []
        for entity in entities:
            entity_id = str(entity.get(id_field, "unknown"))
            values = {k: v for k, v in entity.items() if k in self.kpis}

            report = self.analyze(values, entity_id=entity_id)
            reports.append(report)

        # Sort by score (best first)
        reports.sort(key=lambda r: r.overall_score, reverse=True)

        # Add percentile rankings
        for i, report in enumerate(reports):
            report.percentile = round((1 - (i / len(reports))) * 100, 1)

        return reports

    def get_kpi_summary(self) -> List[Dict[str, Any]]:
        """Get summary of all configured KPIs."""
        return [
            {
                "kpi_id": kpi.kpi_id,
                "name": kpi.name,
                "category": kpi.category.value,
                "benchmark": kpi.benchmark_value,
                "unit": kpi.unit,
                "direction": kpi.direction.value,
                "weight": kpi.weight,
                "description": kpi.description
            }
            for kpi in self.kpis.values()
        ]


# =============================================================================
# Factory Functions for Common Benchmark Sets
# =============================================================================

def create_procurement_benchmarks() -> BenchmarkEngine:
    """Create procurement/sourcing benchmark engine (Coupa-style)."""
    kpis = [
        # Spend Analysis
        KPIDefinition("spend_under_management", "Spend Under Management",
                     80.0, KPIDirection.HIGHER_IS_BETTER, KPICategory.PROCUREMENT, "%",
                     "Percentage of total spend managed through procurement"),
        KPIDefinition("contract_coverage", "Contract Coverage",
                     75.0, KPIDirection.HIGHER_IS_BETTER, KPICategory.PROCUREMENT, "%",
                     "Spend covered by contracts"),

        # Supplier Management
        KPIDefinition("supplier_consolidation", "Supplier Consolidation Rate",
                     15.0, KPIDirection.HIGHER_IS_BETTER, KPICategory.EFFICIENCY, "%",
                     "Year-over-year reduction in supplier count"),
        KPIDefinition("preferred_supplier_usage", "Preferred Supplier Usage",
                     70.0, KPIDirection.HIGHER_IS_BETTER, KPICategory.PROCUREMENT, "%"),

        # Source-to-Contract
        KPIDefinition("cost_savings", "Negotiated Savings Rate",
                     12.0, KPIDirection.HIGHER_IS_BETTER, KPICategory.FINANCIAL, "%",
                     "Average savings from negotiations"),
        KPIDefinition("cycle_time", "Sourcing Cycle Time",
                     45.0, KPIDirection.LOWER_IS_BETTER, KPICategory.EFFICIENCY, "days"),

        # Procurement Operations
        KPIDefinition("po_accuracy", "PO Accuracy Rate",
                     98.0, KPIDirection.HIGHER_IS_BETTER, KPICategory.QUALITY, "%"),
        KPIDefinition("maverick_spend", "Maverick Spend Rate",
                     10.0, KPIDirection.LOWER_IS_BETTER, KPICategory.COMPLIANCE, "%",
                     "Off-contract or non-compliant spend"),

        # Cash & Payments
        KPIDefinition("early_payment_discount", "Early Payment Discount Capture",
                     60.0, KPIDirection.HIGHER_IS_BETTER, KPICategory.FINANCIAL, "%"),
        KPIDefinition("invoice_exception_rate", "Invoice Exception Rate",
                     5.0, KPIDirection.LOWER_IS_BETTER, KPICategory.QUALITY, "%"),
    ]

    return BenchmarkEngine(
        kpis,
        category_weights={
            "Financial": 1.2,
            "Procurement": 1.0,
            "Efficiency": 0.9,
            "Quality": 0.8,
            "Compliance": 1.1
        }
    )


def create_hr_benchmarks() -> BenchmarkEngine:
    """Create HR/workforce benchmark engine."""
    kpis = [
        KPIDefinition("turnover_rate", "Voluntary Turnover Rate",
                     12.0, KPIDirection.LOWER_IS_BETTER, KPICategory.HR, "%"),
        KPIDefinition("time_to_fill", "Time to Fill Positions",
                     45.0, KPIDirection.LOWER_IS_BETTER, KPICategory.EFFICIENCY, "days"),
        KPIDefinition("employee_engagement", "Employee Engagement Score",
                     75.0, KPIDirection.HIGHER_IS_BETTER, KPICategory.HR, "%"),
        KPIDefinition("training_hours", "Training Hours per Employee",
                     40.0, KPIDirection.HIGHER_IS_BETTER, KPICategory.HR, "hours/year"),
        KPIDefinition("absenteeism_rate", "Absenteeism Rate",
                     3.0, KPIDirection.LOWER_IS_BETTER, KPICategory.OPERATIONAL, "%"),
        KPIDefinition("revenue_per_employee", "Revenue per Employee",
                     200000.0, KPIDirection.HIGHER_IS_BETTER, KPICategory.FINANCIAL, "$"),
    ]

    return BenchmarkEngine(kpis)


def create_manufacturing_benchmarks() -> BenchmarkEngine:
    """Create manufacturing/operations benchmark engine."""
    kpis = [
        KPIDefinition("oee", "Overall Equipment Effectiveness",
                     85.0, KPIDirection.HIGHER_IS_BETTER, KPICategory.OPERATIONAL, "%"),
        KPIDefinition("first_pass_yield", "First Pass Yield",
                     95.0, KPIDirection.HIGHER_IS_BETTER, KPICategory.QUALITY, "%"),
        KPIDefinition("scrap_rate", "Scrap Rate",
                     2.0, KPIDirection.LOWER_IS_BETTER, KPICategory.QUALITY, "%"),
        KPIDefinition("on_time_delivery", "On-Time Delivery",
                     95.0, KPIDirection.HIGHER_IS_BETTER, KPICategory.CUSTOMER, "%"),
        KPIDefinition("inventory_turns", "Inventory Turns",
                     8.0, KPIDirection.HIGHER_IS_BETTER, KPICategory.EFFICIENCY, "turns/year"),
        KPIDefinition("downtime", "Unplanned Downtime",
                     5.0, KPIDirection.LOWER_IS_BETTER, KPICategory.OPERATIONAL, "%"),
        KPIDefinition("defect_rate", "Defect Rate (DPMO)",
                     3400.0, KPIDirection.LOWER_IS_BETTER, KPICategory.QUALITY, "DPMO"),
    ]

    return BenchmarkEngine(kpis)


# =============================================================================
# Demo / Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("BENCHMARK ENGINE DEMO")
    print("=" * 60)

    # Create procurement benchmarks
    engine = create_procurement_benchmarks()

    print(f"\nLoaded {len(engine.kpis)} KPIs:")
    for kpi in engine.get_kpi_summary()[:5]:
        print(f"  {kpi['name']}: benchmark {kpi['benchmark']}{kpi['unit']}")
    print("  ...")

    # Analyze a department
    print("\n" + "-" * 40)
    print("Analyzing Procurement Department:")

    report = engine.analyze({
        "spend_under_management": 72,
        "contract_coverage": 68,
        "supplier_consolidation": 18,
        "preferred_supplier_usage": 65,
        "cost_savings": 14,
        "cycle_time": 52,
        "po_accuracy": 97,
        "maverick_spend": 15,
        "early_payment_discount": 45,
        "invoice_exception_rate": 8,
    }, entity_id="PROCUREMENT-2024")

    print(f"\n  Overall Score: {report.overall_score}")
    print(f"  Grade: {report.grade}")
    print(f"  Rating: {report.overall_rating}")

    print("\n  Category Breakdown:")
    for cat, cat_score in report.category_scores.items():
        print(f"    {cat}: {cat_score.score} ({len(cat_score.strengths)} strengths, "
              f"{len(cat_score.improvements)} improvements)")

    if report.top_strengths:
        print("\n  Top Strengths:")
        for s in report.top_strengths[:3]:
            print(f"    + {s}")

    if report.top_improvements:
        print("\n  Areas for Improvement:")
        for i in report.top_improvements[:3]:
            print(f"    - {i}")

    if report.recommendations:
        print("\n  Recommendations:")
        for r in report.recommendations[:3]:
            print(f"    > {r}")
