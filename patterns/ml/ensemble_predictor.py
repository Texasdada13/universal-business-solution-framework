"""
Ensemble Predictor - Universal Business Solution Framework

Multi-model prediction orchestration with weighted voting, stacking,
and uncertainty quantification.

Usage:
```python
from patterns.ml import EnsemblePredictor, EnsembleMethod

# Create ensemble
ensemble = EnsemblePredictor(method=EnsembleMethod.WEIGHTED_AVERAGE)

# Register models
ensemble.register_model('model_a', model_a, weight=0.3)
ensemble.register_model('model_b', model_b, weight=0.5)
ensemble.register_model('model_c', model_c, weight=0.2)

# Get ensemble prediction
result = ensemble.predict(X)

# Get prediction with uncertainty
result = ensemble.predict_with_uncertainty(X)
```
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Callable, Any, Union, Tuple
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from datetime import datetime


class EnsembleMethod(Enum):
    """Methods for combining model predictions"""
    SIMPLE_AVERAGE = "simple_average"
    WEIGHTED_AVERAGE = "weighted_average"
    MEDIAN = "median"
    VOTING = "voting"  # For classification
    STACKING = "stacking"
    BEST_MODEL = "best_model"


class UncertaintyMethod(Enum):
    """Methods for uncertainty quantification"""
    STD = "std"  # Standard deviation of predictions
    IQR = "iqr"  # Interquartile range
    CONFIDENCE_INTERVAL = "confidence_interval"
    BOOTSTRAP = "bootstrap"


@dataclass
class ModelInfo:
    """Information about a registered model"""
    name: str
    model: Any
    weight: float = 1.0
    validation_error: Optional[float] = None
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictionResult:
    """Container for ensemble prediction results"""
    prediction: Union[float, np.ndarray, pd.Series]
    uncertainty: Optional[Union[float, np.ndarray]] = None
    confidence: Optional[float] = None
    individual_predictions: Optional[Dict[str, Any]] = None
    method: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'prediction': self.prediction if not isinstance(self.prediction, np.ndarray) else self.prediction.tolist(),
            'uncertainty': self.uncertainty if not isinstance(self.uncertainty, np.ndarray) else self.uncertainty.tolist(),
            'confidence': self.confidence,
            'method': self.method,
            'metadata': self.metadata
        }


class BaseModel(ABC):
    """
    Abstract base class for models to be used in ensemble.

    Implement this interface for custom models.
    """

    @abstractmethod
    def predict(self, X: Any) -> np.ndarray:
        """Generate predictions"""
        pass

    def get_name(self) -> str:
        """Get model name"""
        return self.__class__.__name__


class EnsemblePredictor:
    """
    Multi-model prediction orchestrator.

    Combines predictions from multiple models using various ensemble methods.
    Provides uncertainty quantification and model performance tracking.

    Example:
    ```python
    ensemble = EnsemblePredictor(method=EnsembleMethod.WEIGHTED_AVERAGE)
    ensemble.register_model('rf', random_forest, weight=0.4)
    ensemble.register_model('xgb', xgboost, weight=0.6)
    result = ensemble.predict_with_uncertainty(X_test)
    ```
    """

    def __init__(
        self,
        method: EnsembleMethod = EnsembleMethod.WEIGHTED_AVERAGE,
        uncertainty_method: UncertaintyMethod = UncertaintyMethod.STD
    ):
        """
        Initialize ensemble predictor.

        Args:
            method: Method for combining predictions
            uncertainty_method: Method for uncertainty quantification
        """
        self.method = method
        self.uncertainty_method = uncertainty_method
        self._models: Dict[str, ModelInfo] = {}
        self._meta_learner: Optional[Any] = None
        self._performance_history: List[Dict[str, Any]] = []

    def register_model(
        self,
        name: str,
        model: Any,
        weight: float = 1.0,
        validation_error: Optional[float] = None,
        predict_method: str = 'predict',
        metadata: Optional[Dict[str, Any]] = None
    ) -> 'EnsemblePredictor':
        """
        Register a model in the ensemble.

        Args:
            name: Unique model identifier
            model: Model object with predict method
            weight: Model weight (used in weighted methods)
            validation_error: Optional validation error for auto-weighting
            predict_method: Name of prediction method (default: 'predict')
            metadata: Additional model metadata

        Returns:
            Self for chaining
        """
        # Wrap model to standardize interface
        wrapped_model = ModelWrapper(model, predict_method)

        self._models[name] = ModelInfo(
            name=name,
            model=wrapped_model,
            weight=weight,
            validation_error=validation_error,
            metadata=metadata or {}
        )
        return self

    def unregister_model(self, name: str) -> 'EnsemblePredictor':
        """Remove a model from the ensemble"""
        if name in self._models:
            del self._models[name]
        return self

    def enable_model(self, name: str) -> 'EnsemblePredictor':
        """Enable a model for predictions"""
        if name in self._models:
            self._models[name].enabled = True
        return self

    def disable_model(self, name: str) -> 'EnsemblePredictor':
        """Disable a model from predictions"""
        if name in self._models:
            self._models[name].enabled = False
        return self

    def set_weights(self, weights: Dict[str, float]) -> 'EnsemblePredictor':
        """
        Set model weights.

        Args:
            weights: Dictionary mapping model names to weights
        """
        for name, weight in weights.items():
            if name in self._models:
                self._models[name].weight = weight
        return self

    def auto_weight_by_error(self) -> 'EnsemblePredictor':
        """
        Automatically set weights based on validation errors.

        Models with lower validation error get higher weights.
        Weight = 1 / validation_error (normalized)
        """
        models_with_error = [
            (name, info) for name, info in self._models.items()
            if info.validation_error is not None and info.enabled
        ]

        if not models_with_error:
            return self

        # Calculate inverse error weights
        inverse_errors = [1 / info.validation_error for _, info in models_with_error]
        total = sum(inverse_errors)

        # Normalize weights
        for (name, _), inv_err in zip(models_with_error, inverse_errors):
            self._models[name].weight = inv_err / total

        return self

    def predict(self, X: Any) -> np.ndarray:
        """
        Generate ensemble prediction.

        Args:
            X: Input features

        Returns:
            Ensemble prediction
        """
        predictions = self._get_individual_predictions(X)

        if not predictions:
            raise ValueError("No models available for prediction")

        if self.method == EnsembleMethod.SIMPLE_AVERAGE:
            return self._simple_average(predictions)

        elif self.method == EnsembleMethod.WEIGHTED_AVERAGE:
            return self._weighted_average(predictions)

        elif self.method == EnsembleMethod.MEDIAN:
            return self._median(predictions)

        elif self.method == EnsembleMethod.VOTING:
            return self._voting(predictions)

        elif self.method == EnsembleMethod.STACKING:
            return self._stacking(predictions, X)

        elif self.method == EnsembleMethod.BEST_MODEL:
            return self._best_model(predictions)

        else:
            return self._simple_average(predictions)

    def predict_with_uncertainty(self, X: Any) -> PredictionResult:
        """
        Generate prediction with uncertainty quantification.

        Args:
            X: Input features

        Returns:
            PredictionResult with prediction, uncertainty, and confidence
        """
        predictions = self._get_individual_predictions(X)

        if not predictions:
            raise ValueError("No models available for prediction")

        # Get ensemble prediction
        ensemble_pred = self.predict(X)

        # Calculate uncertainty
        pred_values = [p for p in predictions.values()]
        pred_array = np.array(pred_values)

        if self.uncertainty_method == UncertaintyMethod.STD:
            uncertainty = np.std(pred_array, axis=0)
        elif self.uncertainty_method == UncertaintyMethod.IQR:
            q75 = np.percentile(pred_array, 75, axis=0)
            q25 = np.percentile(pred_array, 25, axis=0)
            uncertainty = q75 - q25
        else:
            uncertainty = np.std(pred_array, axis=0)

        # Calculate confidence (inverse of normalized uncertainty)
        if isinstance(ensemble_pred, np.ndarray):
            mean_abs_pred = np.mean(np.abs(ensemble_pred))
            mean_uncertainty = np.mean(uncertainty)
        else:
            mean_abs_pred = abs(ensemble_pred)
            mean_uncertainty = uncertainty

        if mean_abs_pred > 0:
            confidence = max(0, 1 - (mean_uncertainty / mean_abs_pred))
        else:
            confidence = 0.0

        return PredictionResult(
            prediction=ensemble_pred,
            uncertainty=uncertainty,
            confidence=float(confidence),
            individual_predictions=predictions,
            method=self.method.value,
            metadata={
                'model_count': len(predictions),
                'enabled_models': [name for name, info in self._models.items() if info.enabled]
            }
        )

    def _get_individual_predictions(self, X: Any) -> Dict[str, np.ndarray]:
        """Get predictions from all enabled models"""
        predictions = {}
        for name, info in self._models.items():
            if info.enabled:
                try:
                    pred = info.model.predict(X)
                    predictions[name] = np.array(pred).flatten()
                except Exception as e:
                    print(f"Warning: Model '{name}' prediction failed: {e}")
        return predictions

    def _simple_average(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Simple average of all predictions"""
        pred_array = np.array(list(predictions.values()))
        return np.mean(pred_array, axis=0)

    def _weighted_average(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Weighted average based on model weights"""
        weighted_sum = None
        total_weight = 0.0

        for name, pred in predictions.items():
            weight = self._models[name].weight
            if weighted_sum is None:
                weighted_sum = weight * pred
            else:
                weighted_sum += weight * pred
            total_weight += weight

        if total_weight > 0:
            return weighted_sum / total_weight
        else:
            return self._simple_average(predictions)

    def _median(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Median of all predictions"""
        pred_array = np.array(list(predictions.values()))
        return np.median(pred_array, axis=0)

    def _voting(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Majority voting for classification"""
        pred_array = np.array(list(predictions.values()))
        # Round predictions and find mode
        rounded = np.round(pred_array).astype(int)

        # For each sample, find most common prediction
        result = []
        for i in range(pred_array.shape[1] if pred_array.ndim > 1 else 1):
            if pred_array.ndim > 1:
                votes = rounded[:, i]
            else:
                votes = rounded
            values, counts = np.unique(votes, return_counts=True)
            result.append(values[np.argmax(counts)])

        return np.array(result)

    def _stacking(self, predictions: Dict[str, np.ndarray], X: Any) -> np.ndarray:
        """Use meta-learner for stacking"""
        if self._meta_learner is None:
            # Fall back to weighted average if no meta-learner
            return self._weighted_average(predictions)

        # Stack predictions as features for meta-learner
        stacked_features = np.column_stack(list(predictions.values()))
        return self._meta_learner.predict(stacked_features)

    def _best_model(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Use prediction from best performing model"""
        best_name = None
        best_error = float('inf')

        for name, info in self._models.items():
            if info.enabled and info.validation_error is not None:
                if info.validation_error < best_error:
                    best_error = info.validation_error
                    best_name = name

        if best_name and best_name in predictions:
            return predictions[best_name]
        else:
            # Fall back to first available
            return list(predictions.values())[0]

    def set_meta_learner(self, meta_learner: Any) -> 'EnsemblePredictor':
        """
        Set meta-learner for stacking ensemble.

        Args:
            meta_learner: Model to combine base model predictions
        """
        self._meta_learner = ModelWrapper(meta_learner)
        return self

    def train_meta_learner(
        self,
        X_train: Any,
        y_train: np.ndarray,
        meta_learner: Optional[Any] = None
    ) -> 'EnsemblePredictor':
        """
        Train meta-learner on base model predictions.

        Args:
            X_train: Training features
            y_train: Training targets
            meta_learner: Optional custom meta-learner (default: LinearRegression)
        """
        # Get predictions from base models
        base_predictions = self._get_individual_predictions(X_train)

        if not base_predictions:
            raise ValueError("No models available for meta-learner training")

        # Stack predictions
        stacked_features = np.column_stack(list(base_predictions.values()))

        # Use provided meta-learner or create simple one
        if meta_learner is not None:
            self._meta_learner = ModelWrapper(meta_learner)
        else:
            # Simple ridge regression meta-learner
            self._meta_learner = SimpleMetaLearner()

        # Train meta-learner
        self._meta_learner.fit(stacked_features, y_train)

        return self

    def update_validation_errors(self, X_val: Any, y_val: np.ndarray) -> 'EnsemblePredictor':
        """
        Update model validation errors.

        Args:
            X_val: Validation features
            y_val: Validation targets
        """
        predictions = self._get_individual_predictions(X_val)

        for name, pred in predictions.items():
            error = np.mean((y_val - pred) ** 2)  # MSE
            self._models[name].validation_error = float(error)

        return self

    def get_model_performance(self) -> pd.DataFrame:
        """Get DataFrame of model performance metrics"""
        data = []
        for name, info in self._models.items():
            data.append({
                'model': name,
                'weight': info.weight,
                'validation_error': info.validation_error,
                'enabled': info.enabled
            })
        return pd.DataFrame(data)

    def get_model_names(self) -> List[str]:
        """Get list of registered model names"""
        return list(self._models.keys())

    def get_enabled_models(self) -> List[str]:
        """Get list of enabled model names"""
        return [name for name, info in self._models.items() if info.enabled]


class ModelWrapper:
    """Wrapper to standardize model interfaces"""

    def __init__(self, model: Any, predict_method: str = 'predict'):
        self.model = model
        self.predict_method = predict_method

    def predict(self, X: Any) -> np.ndarray:
        """Generate predictions using the wrapped model"""
        method = getattr(self.model, self.predict_method, None)
        if method is None:
            raise ValueError(f"Model has no method '{self.predict_method}'")
        return np.array(method(X))

    def fit(self, X: Any, y: Any) -> 'ModelWrapper':
        """Fit the wrapped model"""
        if hasattr(self.model, 'fit'):
            self.model.fit(X, y)
        return self


class SimpleMetaLearner:
    """Simple ridge regression meta-learner"""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.coefficients: Optional[np.ndarray] = None
        self.intercept: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SimpleMetaLearner':
        """Fit using ridge regression"""
        n_samples, n_features = X.shape

        # Add regularization
        X_aug = np.column_stack([np.ones(n_samples), X])
        I = np.eye(X_aug.shape[1])
        I[0, 0] = 0  # Don't regularize intercept

        # Solve (X'X + αI)^-1 X'y
        XtX = X_aug.T @ X_aug
        Xty = X_aug.T @ y

        try:
            coeffs = np.linalg.solve(XtX + self.alpha * I, Xty)
            self.intercept = coeffs[0]
            self.coefficients = coeffs[1:]
        except Exception:
            # Fall back to equal weights
            self.intercept = np.mean(y)
            self.coefficients = np.ones(n_features) / n_features

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions"""
        if self.coefficients is None:
            raise ValueError("Meta-learner not fitted")
        return self.intercept + X @ self.coefficients


# ==================== Factory Functions ====================


def create_simple_ensemble() -> EnsemblePredictor:
    """Create a simple averaging ensemble"""
    return EnsemblePredictor(method=EnsembleMethod.SIMPLE_AVERAGE)


def create_weighted_ensemble() -> EnsemblePredictor:
    """Create a weighted averaging ensemble"""
    return EnsemblePredictor(method=EnsembleMethod.WEIGHTED_AVERAGE)


def create_stacking_ensemble() -> EnsemblePredictor:
    """Create a stacking ensemble with meta-learner"""
    return EnsemblePredictor(method=EnsembleMethod.STACKING)


def create_voting_ensemble() -> EnsemblePredictor:
    """Create a voting ensemble for classification"""
    return EnsemblePredictor(method=EnsembleMethod.VOTING)


def create_robust_ensemble() -> EnsemblePredictor:
    """Create an ensemble using median (robust to outliers)"""
    return EnsemblePredictor(
        method=EnsembleMethod.MEDIAN,
        uncertainty_method=UncertaintyMethod.IQR
    )


# ==================== Exports ====================

__all__ = [
    'EnsemblePredictor',
    'EnsembleMethod',
    'UncertaintyMethod',
    'PredictionResult',
    'ModelInfo',
    'BaseModel',
    'ModelWrapper',
    'SimpleMetaLearner',
    'create_simple_ensemble',
    'create_weighted_ensemble',
    'create_stacking_ensemble',
    'create_voting_ensemble',
    'create_robust_ensemble',
]
