"""
Feature engineering module for Port Decision AI.

Implements rolling window features, lag features, and derived metrics
for time-series delay prediction.
"""

import pandas as pd
import numpy as np
from typing import List, Optional

from src.config import FeatureConfig


def create_rolling_features(
    df: pd.DataFrame,
    windows_hours: List[float],
    timestamp_col: str = "timestamp",
    numeric_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Create rolling window features for numeric columns.
    
    Computes rolling mean, sum, and std for specified windows.
    
    Args:
        df: DataFrame with timestamp column.
        windows_hours: List of window sizes in hours.
        timestamp_col: Name of timestamp column.
        numeric_cols: List of numeric columns to process.
            If None, processes all numeric columns except target.
    
    Returns:
        DataFrame with added rolling features.
    """
    df = df.copy()
    
    if numeric_cols is None:
        numeric_cols = [
            col
            for col in df.select_dtypes(include=[np.number]).columns
            if col not in [timestamp_col, "delay_minutes"]
        ]
    
    # Ensure sorted by timestamp
    df = df.sort_values(timestamp_col).reset_index(drop=True)
    
    # Convert windows to number of periods (assuming 15-min intervals)
    freq_minutes = (df[timestamp_col].iloc[1] - df[timestamp_col].iloc[0]).total_seconds() / 60
    windows_periods = [int(w * 60 / freq_minutes) for w in windows_hours]
    
    # Collect all new feature columns to avoid DataFrame fragmentation
    new_features = {}
    
    for col in numeric_cols:
        for window_periods in windows_periods:
            window_str = f"{int(window_periods * freq_minutes / 60)}h"
            
            # Rolling mean
            new_features[f"{col}_rolling_mean_{window_str}"] = (
                df[col].rolling(window=window_periods, min_periods=1).mean()
            )
            
            # Rolling sum
            new_features[f"{col}_rolling_sum_{window_str}"] = (
                df[col].rolling(window=window_periods, min_periods=1).sum()
            )
            
            # Rolling std (volatility)
            new_features[f"{col}_rolling_std_{window_str}"] = (
                df[col].rolling(window=window_periods, min_periods=1).std().fillna(0)
            )
    
    # Concatenate all new features at once to avoid fragmentation
    new_features_df = pd.DataFrame(new_features, index=df.index)
    df = pd.concat([df, new_features_df], axis=1)
    
    return df


def create_lag_features(
    df: pd.DataFrame,
    lags: List[int],
    numeric_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Create lag features for numeric columns.
    
    Args:
        df: DataFrame sorted by timestamp.
        lags: List of lag periods to create.
        numeric_cols: List of numeric columns to process.
            If None, processes key operational columns.
    
    Returns:
        DataFrame with added lag features.
    """
    df = df.copy()
    
    if numeric_cols is None:
        numeric_cols = ["workload", "resource_allocation", "congestion", "delay_minutes"]
    
    for col in numeric_cols:
        if col not in df.columns:
            continue
        for lag in lags:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)
    
    return df


def create_trend_features(df: pd.DataFrame, window: int = 12) -> pd.DataFrame:
    """
    Create trend slope features using linear regression over rolling window.
    
    Args:
        df: DataFrame sorted by timestamp.
        window: Window size for trend calculation.
    
    Returns:
        DataFrame with added trend features.
    """
    df = df.copy()
    
    numeric_cols = ["workload", "resource_allocation", "congestion"]
    
    for col in numeric_cols:
        if col not in df.columns:
            continue
        
        # Calculate trend slope using rolling linear regression
        trend_values = []
        for i in range(len(df)):
            start_idx = max(0, i - window + 1)
            y_window = df[col].iloc[start_idx : i + 1].values
            x_window = np.arange(len(y_window))
            
            if len(y_window) > 1:
                slope = np.polyfit(x_window, y_window, 1)[0]
            else:
                slope = 0.0
            
            trend_values.append(slope)
        
        df[f"{col}_trend_slope"] = trend_values
    
    return df


def create_utilization_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create utilization and ratio features.
    
    Args:
        df: DataFrame with workload and resource_allocation.
    
    Returns:
        DataFrame with added utilization features.
    """
    df = df.copy()
    
    if "workload" in df.columns and "resource_allocation" in df.columns:
        df["utilization_ratio"] = df["workload"] / (df["resource_allocation"] + 1e-6)
        df["resource_gap"] = df["workload"] - df["resource_allocation"]
        df["resource_surplus"] = np.maximum(0, df["resource_allocation"] - df["workload"])
        df["resource_deficit"] = np.maximum(0, df["workload"] - df["resource_allocation"])
    
    return df


def create_time_features(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """
    Create time-based features from timestamp.
    
    Args:
        df: DataFrame with timestamp column.
        timestamp_col: Name of timestamp column.
    
    Returns:
        DataFrame with added time features.
    """
    df = df.copy()
    
    if timestamp_col not in df.columns:
        return df
    
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    
    df["hour"] = df[timestamp_col].dt.hour
    df["day_of_week"] = df[timestamp_col].dt.dayofweek
    df["day_of_month"] = df[timestamp_col].dt.day
    df["month"] = df[timestamp_col].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    
    # Cyclical encoding for hour and day_of_week
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    
    return df


def engineer_features(
    df: pd.DataFrame,
    config: Optional[FeatureConfig] = None,
) -> pd.DataFrame:
    """
    Complete feature engineering pipeline.
    
    Applies all feature engineering steps in correct order.
    
    Args:
        df: Cleaned DataFrame ready for feature engineering.
        config: FeatureConfig instance. If None, uses defaults.
    
    Returns:
        DataFrame with all engineered features.
    """
    if config is None:
        config = FeatureConfig()
    
    df = df.copy()
    
    # 1. Time features
    df = create_time_features(df, config.timestamp_col)
    
    # 2. Utilization features
    df = create_utilization_features(df)
    
    # 3. Rolling window features
    df = create_rolling_features(
        df,
        config.rolling_windows_hours,
        config.timestamp_col,
    )
    
    # 4. Lag features
    df = create_lag_features(df, config.lag_features)
    
    # 5. Trend features
    df = create_trend_features(df, window=12)
    
    # Drop rows with NaN from lag features
    max_lag = max(config.lag_features) if config.lag_features else 0
    if max_lag > 0:
        df = df.iloc[max_lag:].reset_index(drop=True)
    
    return df


def get_feature_columns(
    df: pd.DataFrame,
    exclude_cols: Optional[List[str]] = None,
) -> List[str]:
    """
    Get list of feature columns (excluding target and metadata).
    
    Args:
        df: DataFrame with features.
        exclude_cols: Additional columns to exclude.
    
    Returns:
        List of feature column names.
    """
    if exclude_cols is None:
        exclude_cols = []
    
    exclude = {
        "timestamp",
        "delay_minutes",
        "target",
        "y",
        *exclude_cols,
    }
    
    feature_cols = [col for col in df.columns if col not in exclude]
    return feature_cols

