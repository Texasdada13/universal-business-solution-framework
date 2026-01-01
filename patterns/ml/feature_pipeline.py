"""
Feature Pipeline - Universal Business Solution Framework

Reusable feature engineering patterns for time series and tabular data.
Supports technical indicators, statistical features, and custom transformations.

Usage:
```python
from patterns.ml import FeaturePipeline, FeatureCategory

# Create pipeline with selected categories
pipeline = FeaturePipeline(categories=[
    FeatureCategory.PRICE,
    FeatureCategory.MOMENTUM,
    FeatureCategory.VOLATILITY
])

# Generate features from OHLCV data
features_df = pipeline.generate_features(df)

# Get feature importance
importance = pipeline.get_feature_importance()
```
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Callable, Any, Union
import pandas as pd
import numpy as np
from datetime import datetime


class FeatureCategory(Enum):
    """Categories of features for organization and selection"""
    PRICE = "price"
    VOLUME = "volume"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    TREND = "trend"
    STATISTICAL = "statistical"
    CYCLICAL = "cyclical"
    LAG = "lag"
    CUSTOM = "custom"


@dataclass
class FeatureDefinition:
    """Definition of a single feature"""
    name: str
    category: FeatureCategory
    compute_func: Callable[[pd.DataFrame], pd.Series]
    description: str = ""
    dependencies: List[str] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)


class FeaturePipeline:
    """
    Configurable feature engineering pipeline.

    Generates technical indicators, statistical features, and custom
    transformations from time series data.

    Example:
    ```python
    pipeline = FeaturePipeline()
    pipeline.add_price_features()
    pipeline.add_momentum_features()
    features = pipeline.generate_features(df)
    ```
    """

    def __init__(
        self,
        categories: Optional[List[FeatureCategory]] = None,
        include_all: bool = False
    ):
        """
        Initialize feature pipeline.

        Args:
            categories: List of feature categories to include
            include_all: If True, include all available features
        """
        self.features: Dict[str, FeatureDefinition] = {}
        self.feature_importance: Dict[str, float] = {}
        self._generated_columns: List[str] = []

        if include_all:
            self.add_all_features()
        elif categories:
            for category in categories:
                self._add_category_features(category)

    def _add_category_features(self, category: FeatureCategory) -> None:
        """Add all features for a given category"""
        category_map = {
            FeatureCategory.PRICE: self.add_price_features,
            FeatureCategory.VOLUME: self.add_volume_features,
            FeatureCategory.MOMENTUM: self.add_momentum_features,
            FeatureCategory.VOLATILITY: self.add_volatility_features,
            FeatureCategory.TREND: self.add_trend_features,
            FeatureCategory.STATISTICAL: self.add_statistical_features,
            FeatureCategory.CYCLICAL: self.add_cyclical_features,
            FeatureCategory.LAG: self.add_lag_features,
        }
        if category in category_map:
            category_map[category]()

    def add_all_features(self) -> 'FeaturePipeline':
        """Add all available feature categories"""
        self.add_price_features()
        self.add_volume_features()
        self.add_momentum_features()
        self.add_volatility_features()
        self.add_trend_features()
        self.add_statistical_features()
        self.add_cyclical_features()
        self.add_lag_features()
        return self

    # ==================== Price Features ====================

    def add_price_features(self) -> 'FeaturePipeline':
        """Add price-based features"""

        def returns(df):
            return df['close'].pct_change()

        def log_returns(df):
            return np.log(df['close'] / df['close'].shift(1))

        def high_low_ratio(df):
            return df['high'] / df['low']

        def close_open_ratio(df):
            return df['close'] / df['open']

        def true_range(df):
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift(1))
            low_close = abs(df['low'] - df['close'].shift(1))
            return pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        self._register_feature('returns', FeatureCategory.PRICE, returns,
                              "Simple percentage returns")
        self._register_feature('log_returns', FeatureCategory.PRICE, log_returns,
                              "Logarithmic returns")
        self._register_feature('high_low_ratio', FeatureCategory.PRICE, high_low_ratio,
                              "High/Low price ratio")
        self._register_feature('close_open_ratio', FeatureCategory.PRICE, close_open_ratio,
                              "Close/Open price ratio")
        self._register_feature('true_range', FeatureCategory.PRICE, true_range,
                              "True range (ATR component)")

        # Moving averages
        for period in [5, 10, 20, 50, 100, 200]:
            self._register_feature(
                f'sma_{period}', FeatureCategory.PRICE,
                lambda df, p=period: df['close'].rolling(window=p).mean(),
                f"{period}-period simple moving average"
            )
            self._register_feature(
                f'ema_{period}', FeatureCategory.PRICE,
                lambda df, p=period: df['close'].ewm(span=p, adjust=False).mean(),
                f"{period}-period exponential moving average"
            )

        # Bollinger Bands
        for period in [20]:
            self._register_feature(
                f'bb_upper_{period}', FeatureCategory.PRICE,
                lambda df, p=period: df['close'].rolling(p).mean() + 2 * df['close'].rolling(p).std(),
                f"Bollinger Band upper ({period})"
            )
            self._register_feature(
                f'bb_lower_{period}', FeatureCategory.PRICE,
                lambda df, p=period: df['close'].rolling(p).mean() - 2 * df['close'].rolling(p).std(),
                f"Bollinger Band lower ({period})"
            )
            self._register_feature(
                f'bb_width_{period}', FeatureCategory.PRICE,
                lambda df, p=period: (
                    (df['close'].rolling(p).mean() + 2 * df['close'].rolling(p).std()) -
                    (df['close'].rolling(p).mean() - 2 * df['close'].rolling(p).std())
                ) / df['close'].rolling(p).mean(),
                f"Bollinger Band width ({period})"
            )
            self._register_feature(
                f'bb_position_{period}', FeatureCategory.PRICE,
                lambda df, p=period: (
                    (df['close'] - (df['close'].rolling(p).mean() - 2 * df['close'].rolling(p).std())) /
                    ((df['close'].rolling(p).mean() + 2 * df['close'].rolling(p).std()) -
                     (df['close'].rolling(p).mean() - 2 * df['close'].rolling(p).std()))
                ),
                f"Position within Bollinger Bands ({period})"
            )

        return self

    # ==================== Volume Features ====================

    def add_volume_features(self) -> 'FeaturePipeline':
        """Add volume-based features"""

        # Volume moving average ratios
        for period in [5, 10, 20, 50]:
            self._register_feature(
                f'volume_sma_ratio_{period}', FeatureCategory.VOLUME,
                lambda df, p=period: df['volume'] / df['volume'].rolling(p).mean(),
                f"Volume to {period}-period SMA ratio"
            )

        # On-Balance Volume (OBV)
        def obv(df):
            direction = np.sign(df['close'].diff())
            return (direction * df['volume']).cumsum()

        self._register_feature('obv', FeatureCategory.VOLUME, obv,
                              "On-Balance Volume")

        # Volume-Price Trend (VPT)
        def vpt(df):
            return ((df['close'].diff() / df['close'].shift(1)) * df['volume']).cumsum()

        self._register_feature('vpt', FeatureCategory.VOLUME, vpt,
                              "Volume-Price Trend")

        # Money Flow Index components
        def money_flow(df):
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            return typical_price * df['volume']

        self._register_feature('money_flow', FeatureCategory.VOLUME, money_flow,
                              "Money flow (typical price * volume)")

        return self

    # ==================== Momentum Features ====================

    def add_momentum_features(self) -> 'FeaturePipeline':
        """Add momentum indicators"""

        # Rate of Change (ROC)
        for period in [3, 5, 10, 20, 60]:
            self._register_feature(
                f'roc_{period}', FeatureCategory.MOMENTUM,
                lambda df, p=period: (df['close'] - df['close'].shift(p)) / df['close'].shift(p) * 100,
                f"{period}-period Rate of Change"
            )

        # RSI
        for period in [7, 14, 21, 28]:
            def rsi(df, p=period):
                delta = df['close'].diff()
                gain = delta.where(delta > 0, 0).rolling(window=p).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=p).mean()
                rs = gain / loss
                return 100 - (100 / (1 + rs))

            self._register_feature(f'rsi_{period}', FeatureCategory.MOMENTUM, rsi,
                                  f"{period}-period Relative Strength Index")

        # MACD
        def macd(df):
            ema_12 = df['close'].ewm(span=12, adjust=False).mean()
            ema_26 = df['close'].ewm(span=26, adjust=False).mean()
            return ema_12 - ema_26

        def macd_signal(df):
            macd_line = df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26, adjust=False).mean()
            return macd_line.ewm(span=9, adjust=False).mean()

        def macd_histogram(df):
            macd_line = df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26, adjust=False).mean()
            signal = macd_line.ewm(span=9, adjust=False).mean()
            return macd_line - signal

        self._register_feature('macd', FeatureCategory.MOMENTUM, macd, "MACD line")
        self._register_feature('macd_signal', FeatureCategory.MOMENTUM, macd_signal, "MACD signal line")
        self._register_feature('macd_histogram', FeatureCategory.MOMENTUM, macd_histogram, "MACD histogram")

        # Stochastic Oscillator
        def stoch_k(df, period=14):
            lowest_low = df['low'].rolling(window=period).min()
            highest_high = df['high'].rolling(window=period).max()
            return 100 * (df['close'] - lowest_low) / (highest_high - lowest_low)

        self._register_feature('stoch_k', FeatureCategory.MOMENTUM, stoch_k,
                              "Stochastic %K (14)")
        self._register_feature('stoch_d', FeatureCategory.MOMENTUM,
                              lambda df: stoch_k(df).rolling(3).mean(),
                              "Stochastic %D (3-period SMA of %K)")

        # Williams %R
        def williams_r(df, period=14):
            highest_high = df['high'].rolling(window=period).max()
            lowest_low = df['low'].rolling(window=period).min()
            return -100 * (highest_high - df['close']) / (highest_high - lowest_low)

        self._register_feature('williams_r', FeatureCategory.MOMENTUM, williams_r,
                              "Williams %R (14)")

        return self

    # ==================== Volatility Features ====================

    def add_volatility_features(self) -> 'FeaturePipeline':
        """Add volatility indicators"""

        # Rolling volatility
        for period in [5, 10, 20, 30, 60]:
            self._register_feature(
                f'volatility_{period}', FeatureCategory.VOLATILITY,
                lambda df, p=period: df['close'].pct_change().rolling(p).std() * np.sqrt(252),
                f"{period}-period annualized volatility"
            )

        # ATR
        for period in [7, 14, 21]:
            def atr(df, p=period):
                high_low = df['high'] - df['low']
                high_close = abs(df['high'] - df['close'].shift(1))
                low_close = abs(df['low'] - df['close'].shift(1))
                tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                return tr.rolling(window=p).mean()

            self._register_feature(f'atr_{period}', FeatureCategory.VOLATILITY, atr,
                                  f"{period}-period Average True Range")

        # Parkinson volatility (uses high-low range)
        def parkinson_volatility(df, period=20):
            factor = 1 / (4 * np.log(2))
            return np.sqrt(factor * (np.log(df['high'] / df['low']) ** 2).rolling(period).mean())

        self._register_feature('parkinson_vol', FeatureCategory.VOLATILITY,
                              parkinson_volatility, "Parkinson volatility estimator")

        # Garman-Klass volatility
        def garman_klass_volatility(df, period=20):
            log_hl = np.log(df['high'] / df['low']) ** 2
            log_co = np.log(df['close'] / df['open']) ** 2
            return np.sqrt((0.5 * log_hl - (2 * np.log(2) - 1) * log_co).rolling(period).mean())

        self._register_feature('garman_klass_vol', FeatureCategory.VOLATILITY,
                              garman_klass_volatility, "Garman-Klass volatility estimator")

        return self

    # ==================== Trend Features ====================

    def add_trend_features(self) -> 'FeaturePipeline':
        """Add trend indicators"""

        # ADX (Average Directional Index)
        def adx(df, period=14):
            plus_dm = df['high'].diff()
            minus_dm = df['low'].diff().abs()
            plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
            minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

            tr = pd.concat([
                df['high'] - df['low'],
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            ], axis=1).max(axis=1)

            atr = tr.rolling(period).mean()
            plus_di = 100 * plus_dm.rolling(period).mean() / atr
            minus_di = 100 * minus_dm.rolling(period).mean() / atr
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            return dx.rolling(period).mean()

        self._register_feature('adx', FeatureCategory.TREND, adx,
                              "Average Directional Index (14)")

        # Aroon indicators
        def aroon_up(df, period=25):
            return 100 * df['high'].rolling(period + 1).apply(
                lambda x: period - x.argmax(), raw=True
            ) / period

        def aroon_down(df, period=25):
            return 100 * df['low'].rolling(period + 1).apply(
                lambda x: period - x.argmin(), raw=True
            ) / period

        self._register_feature('aroon_up', FeatureCategory.TREND, aroon_up, "Aroon Up (25)")
        self._register_feature('aroon_down', FeatureCategory.TREND, aroon_down, "Aroon Down (25)")
        self._register_feature('aroon_oscillator', FeatureCategory.TREND,
                              lambda df: aroon_up(df) - aroon_down(df), "Aroon Oscillator")

        # Linear regression slope
        def linreg_slope(df, period=20):
            x = np.arange(period)
            slopes = df['close'].rolling(period).apply(
                lambda y: np.polyfit(x, y, 1)[0] if len(y) == period else np.nan,
                raw=True
            )
            return slopes

        self._register_feature('linreg_slope', FeatureCategory.TREND, linreg_slope,
                              "Linear regression slope (20)")

        # Trend strength (SMA crossover)
        self._register_feature('trend_strength', FeatureCategory.TREND,
                              lambda df: (df['close'].rolling(20).mean() - df['close'].rolling(50).mean()) / df['close'].rolling(50).mean(),
                              "Trend strength (SMA20 - SMA50 normalized)")

        return self

    # ==================== Statistical Features ====================

    def add_statistical_features(self) -> 'FeaturePipeline':
        """Add statistical features"""

        # Skewness
        self._register_feature('skewness_20', FeatureCategory.STATISTICAL,
                              lambda df: df['close'].pct_change().rolling(20).skew(),
                              "20-period return skewness")

        # Kurtosis
        self._register_feature('kurtosis_20', FeatureCategory.STATISTICAL,
                              lambda df: df['close'].pct_change().rolling(20).kurt(),
                              "20-period return kurtosis")

        # Z-score
        self._register_feature('zscore_20', FeatureCategory.STATISTICAL,
                              lambda df: (df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).std(),
                              "20-period price z-score")

        # Percentile rank
        def percentile_rank(df, period=100):
            return df['close'].rolling(period).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1],
                raw=False
            )

        self._register_feature('percentile_100', FeatureCategory.STATISTICAL,
                              percentile_rank, "100-period percentile rank")

        return self

    # ==================== Cyclical Features ====================

    def add_cyclical_features(self) -> 'FeaturePipeline':
        """Add cyclical time-based features"""

        # Day of week (sine/cosine encoding)
        self._register_feature('day_of_week_sin', FeatureCategory.CYCLICAL,
                              lambda df: np.sin(2 * np.pi * df.index.dayofweek / 7) if hasattr(df.index, 'dayofweek') else pd.Series(0, index=df.index),
                              "Day of week (sine encoding)")
        self._register_feature('day_of_week_cos', FeatureCategory.CYCLICAL,
                              lambda df: np.cos(2 * np.pi * df.index.dayofweek / 7) if hasattr(df.index, 'dayofweek') else pd.Series(0, index=df.index),
                              "Day of week (cosine encoding)")

        # Month (sine/cosine encoding)
        self._register_feature('month_sin', FeatureCategory.CYCLICAL,
                              lambda df: np.sin(2 * np.pi * df.index.month / 12) if hasattr(df.index, 'month') else pd.Series(0, index=df.index),
                              "Month (sine encoding)")
        self._register_feature('month_cos', FeatureCategory.CYCLICAL,
                              lambda df: np.cos(2 * np.pi * df.index.month / 12) if hasattr(df.index, 'month') else pd.Series(0, index=df.index),
                              "Month (cosine encoding)")

        # Quarter
        self._register_feature('quarter_sin', FeatureCategory.CYCLICAL,
                              lambda df: np.sin(2 * np.pi * df.index.quarter / 4) if hasattr(df.index, 'quarter') else pd.Series(0, index=df.index),
                              "Quarter (sine encoding)")

        return self

    # ==================== Lag Features ====================

    def add_lag_features(self, lags: Optional[List[int]] = None) -> 'FeaturePipeline':
        """Add lagged features for time series modeling"""
        if lags is None:
            lags = [1, 2, 3, 5, 10, 20]

        for lag in lags:
            self._register_feature(f'close_lag_{lag}', FeatureCategory.LAG,
                                  lambda df, l=lag: df['close'].shift(l),
                                  f"Close price lagged {lag} periods")
            self._register_feature(f'returns_lag_{lag}', FeatureCategory.LAG,
                                  lambda df, l=lag: df['close'].pct_change().shift(l),
                                  f"Returns lagged {lag} periods")
            self._register_feature(f'volume_lag_{lag}', FeatureCategory.LAG,
                                  lambda df, l=lag: df['volume'].shift(l) if 'volume' in df.columns else pd.Series(np.nan, index=df.index),
                                  f"Volume lagged {lag} periods")

        return self

    # ==================== Custom Features ====================

    def add_custom_feature(
        self,
        name: str,
        compute_func: Callable[[pd.DataFrame], pd.Series],
        description: str = ""
    ) -> 'FeaturePipeline':
        """
        Add a custom feature to the pipeline.

        Args:
            name: Feature name
            compute_func: Function that takes DataFrame and returns Series
            description: Feature description
        """
        self._register_feature(name, FeatureCategory.CUSTOM, compute_func, description)
        return self

    # ==================== Core Methods ====================

    def _register_feature(
        self,
        name: str,
        category: FeatureCategory,
        compute_func: Callable,
        description: str = ""
    ) -> None:
        """Register a feature in the pipeline"""
        self.features[name] = FeatureDefinition(
            name=name,
            category=category,
            compute_func=compute_func,
            description=description
        )

    def generate_features(
        self,
        df: pd.DataFrame,
        drop_na: bool = True,
        prefix: str = ""
    ) -> pd.DataFrame:
        """
        Generate all registered features.

        Args:
            df: Input DataFrame with OHLCV columns (open, high, low, close, volume)
            drop_na: Whether to drop rows with NaN values
            prefix: Optional prefix for feature names

        Returns:
            DataFrame with generated features
        """
        result = df.copy()
        self._generated_columns = []

        # Normalize column names
        result.columns = result.columns.str.lower()

        for name, feature in self.features.items():
            try:
                col_name = f"{prefix}{name}" if prefix else name
                result[col_name] = feature.compute_func(result)
                self._generated_columns.append(col_name)
            except Exception as e:
                print(f"Warning: Could not compute feature '{name}': {e}")

        if drop_na:
            result = result.dropna()

        return result

    def get_feature_names(self) -> List[str]:
        """Get list of all registered feature names"""
        return list(self.features.keys())

    def get_generated_columns(self) -> List[str]:
        """Get list of columns generated in last run"""
        return self._generated_columns

    def get_features_by_category(self, category: FeatureCategory) -> List[str]:
        """Get feature names for a specific category"""
        return [
            name for name, feat in self.features.items()
            if feat.category == category
        ]

    def describe_features(self) -> pd.DataFrame:
        """Get DataFrame describing all registered features"""
        data = []
        for name, feat in self.features.items():
            data.append({
                'name': name,
                'category': feat.category.value,
                'description': feat.description
            })
        return pd.DataFrame(data)

    def set_feature_importance(self, importance: Dict[str, float]) -> None:
        """Set feature importance scores (e.g., from model training)"""
        self.feature_importance = importance

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        return self.feature_importance

    def get_top_features(self, n: int = 10) -> List[str]:
        """Get top N features by importance"""
        sorted_features = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [f[0] for f in sorted_features[:n]]


# ==================== Factory Functions ====================


def create_technical_analysis_pipeline() -> FeaturePipeline:
    """
    Create a pipeline with standard technical analysis features.

    Includes: Price, Momentum, Volatility, Trend indicators.
    Good for stock/crypto analysis.
    """
    return FeaturePipeline(categories=[
        FeatureCategory.PRICE,
        FeatureCategory.MOMENTUM,
        FeatureCategory.VOLATILITY,
        FeatureCategory.TREND
    ])


def create_ml_prediction_pipeline() -> FeaturePipeline:
    """
    Create a pipeline optimized for ML prediction models.

    Includes all features + lag features for sequence modeling.
    """
    pipeline = FeaturePipeline(include_all=True)
    return pipeline


def create_minimal_pipeline() -> FeaturePipeline:
    """
    Create a minimal pipeline with core features only.

    Good for quick analysis or when computational resources are limited.
    """
    pipeline = FeaturePipeline()
    pipeline.add_price_features()
    pipeline.add_momentum_features()
    return pipeline


def create_volume_analysis_pipeline() -> FeaturePipeline:
    """
    Create a pipeline focused on volume analysis.

    Good for detecting accumulation/distribution patterns.
    """
    pipeline = FeaturePipeline(categories=[
        FeatureCategory.PRICE,
        FeatureCategory.VOLUME
    ])
    return pipeline


def create_volatility_pipeline() -> FeaturePipeline:
    """
    Create a pipeline focused on volatility analysis.

    Good for options pricing, risk management.
    """
    return FeaturePipeline(categories=[
        FeatureCategory.PRICE,
        FeatureCategory.VOLATILITY,
        FeatureCategory.STATISTICAL
    ])


# ==================== Exports ====================

__all__ = [
    'FeaturePipeline',
    'FeatureCategory',
    'FeatureDefinition',
    'create_technical_analysis_pipeline',
    'create_ml_prediction_pipeline',
    'create_minimal_pipeline',
    'create_volume_analysis_pipeline',
    'create_volatility_pipeline',
]
