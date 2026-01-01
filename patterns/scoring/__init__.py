"""
Scoring & Risk Patterns - Universal Business Solution Framework

This package provides reusable patterns for:
- Multi-component weighted scoring
- Rule-based alert generation
- Risk level classification
- KPI benchmarking and gap analysis

Usage:
```python
from patterns.scoring import (
    WeightedScoringEngine, ScoreComponent,
    AlertRulesEngine, AlertRule, AlertSeverity,
    RiskClassifier, RiskLevel,
    BenchmarkEngine, KPIDefinition
)
```
"""

from .weighted_scoring import (
    WeightedScoringEngine,
    AggregatedScoringEngine,
    ScoreComponent,
    ScoreDirection,
    ScoreResult,
    # Factory functions
    create_contract_scoring_engine,
    create_employee_performance_engine,
    create_vendor_scoring_engine,
)

from .alert_rules_engine import (
    AlertRulesEngine,
    AlertRule,
    Alert,
    AlertSeverity,
    AlertCategory,
    # Factory functions
    create_contract_alert_rules,
    create_workforce_alert_rules,
    create_financial_alert_rules,
)

from .risk_classification import (
    RiskClassifier,
    MultiDimensionalRiskClassifier,
    RiskLevel,
    RiskThreshold,
    RiskClassification,
    # Factory functions
    create_health_score_classifier,
    create_risk_score_classifier,
    create_financial_risk_classifier,
    create_project_risk_classifier,
)

from .benchmark_engine import (
    BenchmarkEngine,
    KPIDefinition,
    KPIDirection,
    KPICategory,
    KPIScore,
    CategoryScore,
    BenchmarkReport,
    # Factory functions
    create_procurement_benchmarks,
    create_hr_benchmarks,
    create_manufacturing_benchmarks,
)

__all__ = [
    # Weighted Scoring
    "WeightedScoringEngine",
    "AggregatedScoringEngine",
    "ScoreComponent",
    "ScoreDirection",
    "ScoreResult",
    "create_contract_scoring_engine",
    "create_employee_performance_engine",
    "create_vendor_scoring_engine",

    # Alert Rules
    "AlertRulesEngine",
    "AlertRule",
    "Alert",
    "AlertSeverity",
    "AlertCategory",
    "create_contract_alert_rules",
    "create_workforce_alert_rules",
    "create_financial_alert_rules",

    # Risk Classification
    "RiskClassifier",
    "MultiDimensionalRiskClassifier",
    "RiskLevel",
    "RiskThreshold",
    "RiskClassification",
    "create_health_score_classifier",
    "create_risk_score_classifier",
    "create_financial_risk_classifier",
    "create_project_risk_classifier",

    # Benchmarking
    "BenchmarkEngine",
    "KPIDefinition",
    "KPIDirection",
    "KPICategory",
    "KPIScore",
    "CategoryScore",
    "BenchmarkReport",
    "create_procurement_benchmarks",
    "create_hr_benchmarks",
    "create_manufacturing_benchmarks",
]
