"""
ML/AI Patterns - Universal Business Solution Framework

This package provides reusable patterns for:
- Feature engineering pipelines
- Time series forecasting
- Ensemble prediction
- Model versioning and management

Usage:
```python
from patterns.ml import (
    # Feature Engineering
    FeaturePipeline, FeatureCategory,
    create_technical_analysis_pipeline,

    # Time Series
    TimeSeriesForecaster, ForecastMethod,
    create_ensemble_forecaster,

    # Ensemble
    EnsemblePredictor, EnsembleMethod,
    create_weighted_ensemble,

    # Model Management
    ModelRegistry, ModelStage,
    create_local_registry,
)
```
"""

from .feature_pipeline import (
    FeaturePipeline,
    FeatureCategory,
    FeatureDefinition,
    # Factory functions
    create_technical_analysis_pipeline,
    create_ml_prediction_pipeline,
    create_minimal_pipeline,
    create_volume_analysis_pipeline,
    create_volatility_pipeline,
)

from .time_series_models import (
    TimeSeriesForecaster,
    ForecastMethod,
    ForecastResult,
    SeasonalityType,
    BaseForecaster,
    MovingAverageForecaster,
    ExponentialSmoothingForecaster,
    LinearRegressionForecaster,
    ARIMAForecaster,
    # Factory functions
    create_simple_forecaster,
    create_trend_forecaster,
    create_arima_forecaster,
    create_seasonal_forecaster,
    create_ensemble_forecaster,
    create_auto_forecaster,
)

from .ensemble_predictor import (
    EnsemblePredictor,
    EnsembleMethod,
    UncertaintyMethod,
    PredictionResult,
    ModelInfo,
    BaseModel,
    ModelWrapper,
    SimpleMetaLearner,
    # Factory functions
    create_simple_ensemble,
    create_weighted_ensemble,
    create_stacking_ensemble,
    create_voting_ensemble,
    create_robust_ensemble,
)

from .model_registry import (
    ModelRegistry,
    ModelVersion,
    ModelMetadata,
    ModelStage,
    SerializationFormat,
    # Factory functions
    create_local_registry,
    create_json_registry,
)

__all__ = [
    # Feature Pipeline
    "FeaturePipeline",
    "FeatureCategory",
    "FeatureDefinition",
    "create_technical_analysis_pipeline",
    "create_ml_prediction_pipeline",
    "create_minimal_pipeline",
    "create_volume_analysis_pipeline",
    "create_volatility_pipeline",

    # Time Series Models
    "TimeSeriesForecaster",
    "ForecastMethod",
    "ForecastResult",
    "SeasonalityType",
    "BaseForecaster",
    "MovingAverageForecaster",
    "ExponentialSmoothingForecaster",
    "LinearRegressionForecaster",
    "ARIMAForecaster",
    "create_simple_forecaster",
    "create_trend_forecaster",
    "create_arima_forecaster",
    "create_seasonal_forecaster",
    "create_ensemble_forecaster",
    "create_auto_forecaster",

    # Ensemble Predictor
    "EnsemblePredictor",
    "EnsembleMethod",
    "UncertaintyMethod",
    "PredictionResult",
    "ModelInfo",
    "BaseModel",
    "ModelWrapper",
    "SimpleMetaLearner",
    "create_simple_ensemble",
    "create_weighted_ensemble",
    "create_stacking_ensemble",
    "create_voting_ensemble",
    "create_robust_ensemble",

    # Model Registry
    "ModelRegistry",
    "ModelVersion",
    "ModelMetadata",
    "ModelStage",
    "SerializationFormat",
    "create_local_registry",
    "create_json_registry",
]
