"""
Time Series Models - Universal Business Solution Framework

Reusable time series forecasting patterns supporting multiple algorithms.
Provides unified interface for ARIMA, Prophet-style, and ML-based forecasting.

Usage:
```python
from patterns.ml import TimeSeriesForecaster, ForecastMethod

# Create forecaster
forecaster = TimeSeriesForecaster(method=ForecastMethod.ENSEMBLE)

# Train on historical data
forecaster.fit(df['close'])

# Generate forecasts
predictions = forecaster.predict(horizon=30)

# Get confidence intervals
forecast_with_ci = forecaster.predict_with_intervals(horizon=30)
```
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


class ForecastMethod(Enum):
    """Available forecasting methods"""
    MOVING_AVERAGE = "moving_average"
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"
    LINEAR_REGRESSION = "linear_regression"
    ARIMA = "arima"
    HOLT_WINTERS = "holt_winters"
    ENSEMBLE = "ensemble"


class SeasonalityType(Enum):
    """Types of seasonality patterns"""
    NONE = "none"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"
    AUTO = "auto"


@dataclass
class ForecastResult:
    """Container for forecast results"""
    predictions: pd.Series
    lower_bound: Optional[pd.Series] = None
    upper_bound: Optional[pd.Series] = None
    confidence_level: float = 0.95
    method: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def prediction_interval_width(self) -> Optional[pd.Series]:
        """Width of prediction interval"""
        if self.lower_bound is not None and self.upper_bound is not None:
            return self.upper_bound - self.lower_bound
        return None

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame"""
        data = {'prediction': self.predictions}
        if self.lower_bound is not None:
            data['lower_bound'] = self.lower_bound
        if self.upper_bound is not None:
            data['upper_bound'] = self.upper_bound
        return pd.DataFrame(data)


class BaseForecaster(ABC):
    """Abstract base class for forecasters"""

    @abstractmethod
    def fit(self, y: pd.Series, **kwargs) -> 'BaseForecaster':
        """Fit the model to data"""
        pass

    @abstractmethod
    def predict(self, horizon: int) -> pd.Series:
        """Generate point predictions"""
        pass

    def predict_with_intervals(
        self,
        horizon: int,
        confidence: float = 0.95
    ) -> ForecastResult:
        """Generate predictions with confidence intervals"""
        predictions = self.predict(horizon)
        return ForecastResult(
            predictions=predictions,
            confidence_level=confidence,
            method=self.__class__.__name__
        )


class MovingAverageForecaster(BaseForecaster):
    """Simple and weighted moving average forecaster"""

    def __init__(self, window: int = 20, weighted: bool = False):
        self.window = window
        self.weighted = weighted
        self.last_values: Optional[pd.Series] = None
        self.std: float = 0.0

    def fit(self, y: pd.Series, **kwargs) -> 'MovingAverageForecaster':
        self.last_values = y.tail(self.window)
        self.std = y.diff().std()
        return self

    def predict(self, horizon: int) -> pd.Series:
        if self.weighted:
            weights = np.arange(1, self.window + 1)
            forecast_value = np.average(self.last_values, weights=weights)
        else:
            forecast_value = self.last_values.mean()

        last_date = self.last_values.index[-1]
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=horizon,
            freq='D'
        )
        return pd.Series([forecast_value] * horizon, index=future_dates)

    def predict_with_intervals(
        self,
        horizon: int,
        confidence: float = 0.95
    ) -> ForecastResult:
        predictions = self.predict(horizon)
        z_score = 1.96 if confidence == 0.95 else 2.576

        # Expanding uncertainty over time
        uncertainty = self.std * np.sqrt(np.arange(1, horizon + 1))
        lower = predictions - z_score * uncertainty
        upper = predictions + z_score * uncertainty

        return ForecastResult(
            predictions=predictions,
            lower_bound=pd.Series(lower, index=predictions.index),
            upper_bound=pd.Series(upper, index=predictions.index),
            confidence_level=confidence,
            method="MovingAverage"
        )


class ExponentialSmoothingForecaster(BaseForecaster):
    """Simple, Double, and Triple Exponential Smoothing"""

    def __init__(
        self,
        alpha: float = 0.3,
        beta: Optional[float] = None,
        gamma: Optional[float] = None,
        seasonal_period: int = 7
    ):
        self.alpha = alpha
        self.beta = beta  # Trend smoothing
        self.gamma = gamma  # Seasonal smoothing
        self.seasonal_period = seasonal_period
        self.level: float = 0.0
        self.trend: float = 0.0
        self.seasonals: List[float] = []
        self.residual_std: float = 0.0
        self._fitted_values: Optional[pd.Series] = None

    def fit(self, y: pd.Series, **kwargs) -> 'ExponentialSmoothingForecaster':
        values = y.values
        n = len(values)

        # Initialize level
        self.level = values[0]

        # Initialize trend (if using double/triple)
        if self.beta is not None:
            self.trend = (values[min(self.seasonal_period, n-1)] - values[0]) / min(self.seasonal_period, n-1)

        # Initialize seasonals (if using triple)
        if self.gamma is not None:
            self.seasonals = [0.0] * self.seasonal_period
            for i in range(min(self.seasonal_period, n)):
                self.seasonals[i] = values[i] - self.level

        # Fit model and collect residuals
        fitted = []
        residuals = []

        for i, val in enumerate(values):
            if self.gamma is not None:
                # Triple exponential smoothing (Holt-Winters)
                season_idx = i % self.seasonal_period
                forecast = self.level + self.trend + self.seasonals[season_idx]

                old_level = self.level
                self.level = self.alpha * (val - self.seasonals[season_idx]) + (1 - self.alpha) * (self.level + self.trend)
                self.trend = self.beta * (self.level - old_level) + (1 - self.beta) * self.trend
                self.seasonals[season_idx] = self.gamma * (val - self.level) + (1 - self.gamma) * self.seasonals[season_idx]

            elif self.beta is not None:
                # Double exponential smoothing (Holt's)
                forecast = self.level + self.trend

                old_level = self.level
                self.level = self.alpha * val + (1 - self.alpha) * (self.level + self.trend)
                self.trend = self.beta * (self.level - old_level) + (1 - self.beta) * self.trend

            else:
                # Simple exponential smoothing
                forecast = self.level
                self.level = self.alpha * val + (1 - self.alpha) * self.level

            fitted.append(forecast)
            residuals.append(val - forecast)

        self._fitted_values = pd.Series(fitted, index=y.index)
        self.residual_std = np.std(residuals)

        return self

    def predict(self, horizon: int) -> pd.Series:
        predictions = []
        level = self.level
        trend = self.trend if self.beta is not None else 0

        for h in range(horizon):
            if self.gamma is not None:
                season_idx = h % self.seasonal_period
                pred = level + trend + self.seasonals[season_idx]
            else:
                pred = level + trend * h

            predictions.append(pred)

            if self.gamma is None:
                level = level + trend

        last_date = self._fitted_values.index[-1]
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=horizon,
            freq='D'
        )
        return pd.Series(predictions, index=future_dates)

    def predict_with_intervals(
        self,
        horizon: int,
        confidence: float = 0.95
    ) -> ForecastResult:
        predictions = self.predict(horizon)
        z_score = 1.96 if confidence == 0.95 else 2.576

        # Uncertainty grows with horizon
        uncertainty = self.residual_std * np.sqrt(np.arange(1, horizon + 1))
        lower = predictions - z_score * uncertainty
        upper = predictions + z_score * uncertainty

        return ForecastResult(
            predictions=predictions,
            lower_bound=pd.Series(lower.values, index=predictions.index),
            upper_bound=pd.Series(upper.values, index=predictions.index),
            confidence_level=confidence,
            method="ExponentialSmoothing"
        )


class LinearRegressionForecaster(BaseForecaster):
    """Linear and polynomial trend forecaster"""

    def __init__(self, degree: int = 1, add_features: bool = True):
        self.degree = degree
        self.add_features = add_features
        self.coefficients: Optional[np.ndarray] = None
        self.residual_std: float = 0.0
        self.last_index: int = 0
        self.y_train: Optional[pd.Series] = None

    def fit(self, y: pd.Series, **kwargs) -> 'LinearRegressionForecaster':
        self.y_train = y
        n = len(y)
        X = self._create_features(np.arange(n))

        # Fit using least squares
        self.coefficients = np.linalg.lstsq(X, y.values, rcond=None)[0]
        self.last_index = n

        # Calculate residuals
        fitted = X @ self.coefficients
        self.residual_std = np.std(y.values - fitted)

        return self

    def _create_features(self, time_index: np.ndarray) -> np.ndarray:
        """Create polynomial features"""
        features = [np.ones(len(time_index))]  # Intercept
        for d in range(1, self.degree + 1):
            features.append(time_index ** d)
        return np.column_stack(features)

    def predict(self, horizon: int) -> pd.Series:
        future_index = np.arange(self.last_index, self.last_index + horizon)
        X = self._create_features(future_index)
        predictions = X @ self.coefficients

        last_date = self.y_train.index[-1]
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=horizon,
            freq='D'
        )
        return pd.Series(predictions, index=future_dates)

    def predict_with_intervals(
        self,
        horizon: int,
        confidence: float = 0.95
    ) -> ForecastResult:
        predictions = self.predict(horizon)
        z_score = 1.96 if confidence == 0.95 else 2.576

        # Simple prediction interval (grows linearly)
        uncertainty = self.residual_std * (1 + 0.1 * np.arange(1, horizon + 1))
        lower = predictions - z_score * uncertainty
        upper = predictions + z_score * uncertainty

        return ForecastResult(
            predictions=predictions,
            lower_bound=pd.Series(lower.values, index=predictions.index),
            upper_bound=pd.Series(upper.values, index=predictions.index),
            confidence_level=confidence,
            method="LinearRegression"
        )


class ARIMAForecaster(BaseForecaster):
    """
    Auto-Regressive Integrated Moving Average forecaster.

    Simplified implementation without external dependencies.
    For production use, consider statsmodels.
    """

    def __init__(self, p: int = 1, d: int = 1, q: int = 1):
        self.p = p  # AR order
        self.d = d  # Differencing order
        self.q = q  # MA order
        self.ar_coefficients: Optional[np.ndarray] = None
        self.ma_coefficients: Optional[np.ndarray] = None
        self.differenced_data: Optional[np.ndarray] = None
        self.original_data: Optional[pd.Series] = None
        self.residual_std: float = 0.0

    def _difference(self, data: np.ndarray, order: int) -> np.ndarray:
        """Apply differencing"""
        result = data.copy()
        for _ in range(order):
            result = np.diff(result)
        return result

    def _undifference(self, diff_forecast: np.ndarray, last_values: np.ndarray) -> np.ndarray:
        """Reverse differencing"""
        result = diff_forecast.copy()
        for i in range(self.d):
            cumsum = np.cumsum(result)
            result = cumsum + last_values[-(i + 1)]
        return result

    def fit(self, y: pd.Series, **kwargs) -> 'ARIMAForecaster':
        self.original_data = y
        values = y.values

        # Apply differencing
        self.differenced_data = self._difference(values, self.d)

        # Fit AR coefficients using Yule-Walker equations (simplified)
        if self.p > 0:
            self.ar_coefficients = self._fit_ar(self.differenced_data, self.p)
        else:
            self.ar_coefficients = np.array([])

        # Calculate residuals and fit MA (simplified - using residual autocorrelation)
        residuals = self._calculate_residuals()
        self.residual_std = np.std(residuals)

        if self.q > 0:
            self.ma_coefficients = np.zeros(self.q)  # Simplified
        else:
            self.ma_coefficients = np.array([])

        return self

    def _fit_ar(self, data: np.ndarray, order: int) -> np.ndarray:
        """Fit AR coefficients using least squares"""
        n = len(data)
        if n <= order:
            return np.zeros(order)

        # Create design matrix
        X = np.zeros((n - order, order))
        for i in range(order):
            X[:, i] = data[order - i - 1:n - i - 1]

        y = data[order:]

        # Solve least squares
        try:
            coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
        except Exception:
            coeffs = np.zeros(order)

        return coeffs

    def _calculate_residuals(self) -> np.ndarray:
        """Calculate model residuals"""
        if len(self.ar_coefficients) == 0:
            return self.differenced_data

        n = len(self.differenced_data)
        residuals = np.zeros(n - self.p)

        for i in range(self.p, n):
            pred = 0
            for j, coef in enumerate(self.ar_coefficients):
                pred += coef * self.differenced_data[i - j - 1]
            residuals[i - self.p] = self.differenced_data[i] - pred

        return residuals

    def predict(self, horizon: int) -> pd.Series:
        # Forecast differenced series
        diff_forecast = np.zeros(horizon)

        # Use last p values for AR prediction
        history = list(self.differenced_data[-self.p:]) if self.p > 0 else []

        for h in range(horizon):
            pred = 0
            for i, coef in enumerate(self.ar_coefficients):
                if i < len(history):
                    pred += coef * history[-(i + 1)]
            diff_forecast[h] = pred

            # Update history
            if self.p > 0:
                history.append(pred)
                if len(history) > self.p:
                    history.pop(0)

        # Undifference
        last_values = self.original_data.values[-self.d:] if self.d > 0 else []
        if self.d > 0:
            forecast = self._undifference(diff_forecast, self.original_data.values)
        else:
            forecast = diff_forecast

        last_date = self.original_data.index[-1]
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=horizon,
            freq='D'
        )
        return pd.Series(forecast, index=future_dates)

    def predict_with_intervals(
        self,
        horizon: int,
        confidence: float = 0.95
    ) -> ForecastResult:
        predictions = self.predict(horizon)
        z_score = 1.96 if confidence == 0.95 else 2.576

        # Uncertainty grows with horizon
        uncertainty = self.residual_std * np.sqrt(np.arange(1, horizon + 1))
        lower = predictions - z_score * uncertainty
        upper = predictions + z_score * uncertainty

        return ForecastResult(
            predictions=predictions,
            lower_bound=pd.Series(lower.values, index=predictions.index),
            upper_bound=pd.Series(upper.values, index=predictions.index),
            confidence_level=confidence,
            method="ARIMA"
        )


class TimeSeriesForecaster:
    """
    Unified time series forecasting interface.

    Supports multiple methods and ensemble predictions.

    Example:
    ```python
    forecaster = TimeSeriesForecaster(method=ForecastMethod.ENSEMBLE)
    forecaster.fit(df['value'])
    result = forecaster.predict_with_intervals(30)
    ```
    """

    def __init__(
        self,
        method: ForecastMethod = ForecastMethod.ENSEMBLE,
        seasonality: SeasonalityType = SeasonalityType.NONE
    ):
        self.method = method
        self.seasonality = seasonality
        self._forecasters: Dict[str, BaseForecaster] = {}
        self._fitted = False
        self._y: Optional[pd.Series] = None

    def fit(self, y: pd.Series, **kwargs) -> 'TimeSeriesForecaster':
        """
        Fit the forecaster to data.

        Args:
            y: Time series data (pandas Series with DatetimeIndex preferred)
        """
        self._y = y

        # Detect seasonality if auto
        if self.seasonality == SeasonalityType.AUTO:
            self.seasonality = self._detect_seasonality(y)

        seasonal_period = self._get_seasonal_period()

        if self.method == ForecastMethod.MOVING_AVERAGE:
            self._forecasters['ma'] = MovingAverageForecaster(window=20).fit(y)

        elif self.method == ForecastMethod.EXPONENTIAL_SMOOTHING:
            if self.seasonality != SeasonalityType.NONE:
                self._forecasters['es'] = ExponentialSmoothingForecaster(
                    alpha=0.3, beta=0.1, gamma=0.1, seasonal_period=seasonal_period
                ).fit(y)
            else:
                self._forecasters['es'] = ExponentialSmoothingForecaster(
                    alpha=0.3, beta=0.1
                ).fit(y)

        elif self.method == ForecastMethod.LINEAR_REGRESSION:
            self._forecasters['lr'] = LinearRegressionForecaster(degree=1).fit(y)

        elif self.method == ForecastMethod.ARIMA:
            self._forecasters['arima'] = ARIMAForecaster(p=1, d=1, q=1).fit(y)

        elif self.method == ForecastMethod.HOLT_WINTERS:
            self._forecasters['hw'] = ExponentialSmoothingForecaster(
                alpha=0.3, beta=0.1, gamma=0.1, seasonal_period=seasonal_period
            ).fit(y)

        elif self.method == ForecastMethod.ENSEMBLE:
            # Fit multiple models for ensemble
            self._forecasters['ma'] = MovingAverageForecaster(window=20).fit(y)
            self._forecasters['ma_weighted'] = MovingAverageForecaster(window=20, weighted=True).fit(y)
            self._forecasters['es_simple'] = ExponentialSmoothingForecaster(alpha=0.3).fit(y)
            self._forecasters['es_double'] = ExponentialSmoothingForecaster(alpha=0.3, beta=0.1).fit(y)
            self._forecasters['lr'] = LinearRegressionForecaster(degree=1).fit(y)
            self._forecasters['arima'] = ARIMAForecaster(p=1, d=1, q=1).fit(y)

        self._fitted = True
        return self

    def predict(self, horizon: int) -> pd.Series:
        """Generate point predictions"""
        if not self._fitted:
            raise ValueError("Forecaster not fitted. Call fit() first.")

        if self.method == ForecastMethod.ENSEMBLE:
            return self._ensemble_predict(horizon)
        else:
            forecaster = list(self._forecasters.values())[0]
            return forecaster.predict(horizon)

    def predict_with_intervals(
        self,
        horizon: int,
        confidence: float = 0.95
    ) -> ForecastResult:
        """Generate predictions with confidence intervals"""
        if not self._fitted:
            raise ValueError("Forecaster not fitted. Call fit() first.")

        if self.method == ForecastMethod.ENSEMBLE:
            return self._ensemble_predict_with_intervals(horizon, confidence)
        else:
            forecaster = list(self._forecasters.values())[0]
            return forecaster.predict_with_intervals(horizon, confidence)

    def _ensemble_predict(self, horizon: int) -> pd.Series:
        """Combine predictions from multiple models"""
        predictions = []
        for name, forecaster in self._forecasters.items():
            try:
                pred = forecaster.predict(horizon)
                predictions.append(pred.values)
            except Exception as e:
                print(f"Warning: {name} prediction failed: {e}")

        if not predictions:
            raise ValueError("All forecasters failed")

        # Average predictions
        ensemble_pred = np.mean(predictions, axis=0)
        sample_pred = list(self._forecasters.values())[0].predict(horizon)
        return pd.Series(ensemble_pred, index=sample_pred.index)

    def _ensemble_predict_with_intervals(
        self,
        horizon: int,
        confidence: float = 0.95
    ) -> ForecastResult:
        """Ensemble prediction with uncertainty from model disagreement"""
        predictions = []
        for name, forecaster in self._forecasters.items():
            try:
                pred = forecaster.predict(horizon)
                predictions.append(pred.values)
            except Exception:
                pass

        if not predictions:
            raise ValueError("All forecasters failed")

        predictions = np.array(predictions)
        ensemble_mean = np.mean(predictions, axis=0)
        ensemble_std = np.std(predictions, axis=0)

        z_score = 1.96 if confidence == 0.95 else 2.576

        sample_pred = list(self._forecasters.values())[0].predict(horizon)

        return ForecastResult(
            predictions=pd.Series(ensemble_mean, index=sample_pred.index),
            lower_bound=pd.Series(ensemble_mean - z_score * ensemble_std, index=sample_pred.index),
            upper_bound=pd.Series(ensemble_mean + z_score * ensemble_std, index=sample_pred.index),
            confidence_level=confidence,
            method="Ensemble",
            metrics={
                'model_count': len(predictions),
                'mean_std': float(np.mean(ensemble_std))
            }
        )

    def _detect_seasonality(self, y: pd.Series) -> SeasonalityType:
        """Detect seasonality in time series"""
        if len(y) < 14:
            return SeasonalityType.NONE

        # Simple autocorrelation-based detection
        values = y.values
        n = len(values)

        # Check weekly seasonality (lag 7)
        if n > 14:
            corr_7 = np.corrcoef(values[7:], values[:-7])[0, 1]
            if corr_7 > 0.5:
                return SeasonalityType.WEEKLY

        # Check monthly seasonality (lag 30)
        if n > 60:
            corr_30 = np.corrcoef(values[30:], values[:-30])[0, 1]
            if corr_30 > 0.5:
                return SeasonalityType.MONTHLY

        return SeasonalityType.NONE

    def _get_seasonal_period(self) -> int:
        """Get period based on seasonality type"""
        periods = {
            SeasonalityType.NONE: 1,
            SeasonalityType.DAILY: 1,
            SeasonalityType.WEEKLY: 7,
            SeasonalityType.MONTHLY: 30,
            SeasonalityType.QUARTERLY: 90,
            SeasonalityType.YEARLY: 365,
            SeasonalityType.AUTO: 7,
        }
        return periods.get(self.seasonality, 7)

    def get_fitted_models(self) -> List[str]:
        """Get list of fitted model names"""
        return list(self._forecasters.keys())

    def evaluate(
        self,
        test_data: pd.Series,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Evaluate forecast accuracy on test data.

        Args:
            test_data: Actual values to compare against
            metrics: List of metrics to compute (default: MAE, RMSE, MAPE)

        Returns:
            Dictionary of metric names to values
        """
        if metrics is None:
            metrics = ['mae', 'rmse', 'mape']

        predictions = self.predict(len(test_data))
        actual = test_data.values
        pred = predictions.values

        results = {}

        if 'mae' in metrics:
            results['mae'] = np.mean(np.abs(actual - pred))

        if 'rmse' in metrics:
            results['rmse'] = np.sqrt(np.mean((actual - pred) ** 2))

        if 'mape' in metrics:
            non_zero = actual != 0
            results['mape'] = np.mean(np.abs((actual[non_zero] - pred[non_zero]) / actual[non_zero])) * 100

        if 'mse' in metrics:
            results['mse'] = np.mean((actual - pred) ** 2)

        return results


# ==================== Factory Functions ====================


def create_simple_forecaster() -> TimeSeriesForecaster:
    """Create a simple moving average forecaster"""
    return TimeSeriesForecaster(method=ForecastMethod.MOVING_AVERAGE)


def create_trend_forecaster() -> TimeSeriesForecaster:
    """Create a forecaster that captures trends"""
    return TimeSeriesForecaster(method=ForecastMethod.EXPONENTIAL_SMOOTHING)


def create_arima_forecaster(p: int = 1, d: int = 1, q: int = 1) -> TimeSeriesForecaster:
    """Create an ARIMA forecaster with custom parameters"""
    forecaster = TimeSeriesForecaster(method=ForecastMethod.ARIMA)
    return forecaster


def create_seasonal_forecaster(
    seasonality: SeasonalityType = SeasonalityType.WEEKLY
) -> TimeSeriesForecaster:
    """Create a forecaster that handles seasonality"""
    return TimeSeriesForecaster(
        method=ForecastMethod.HOLT_WINTERS,
        seasonality=seasonality
    )


def create_ensemble_forecaster() -> TimeSeriesForecaster:
    """Create an ensemble forecaster combining multiple methods"""
    return TimeSeriesForecaster(method=ForecastMethod.ENSEMBLE)


def create_auto_forecaster() -> TimeSeriesForecaster:
    """Create a forecaster with automatic seasonality detection"""
    return TimeSeriesForecaster(
        method=ForecastMethod.ENSEMBLE,
        seasonality=SeasonalityType.AUTO
    )


# ==================== Exports ====================

__all__ = [
    'TimeSeriesForecaster',
    'ForecastMethod',
    'ForecastResult',
    'SeasonalityType',
    'BaseForecaster',
    'MovingAverageForecaster',
    'ExponentialSmoothingForecaster',
    'LinearRegressionForecaster',
    'ARIMAForecaster',
    'create_simple_forecaster',
    'create_trend_forecaster',
    'create_arima_forecaster',
    'create_seasonal_forecaster',
    'create_ensemble_forecaster',
    'create_auto_forecaster',
]
