"""
Data preprocessing utilities for Port Decision AI.

Handles data cleaning, validation, and preparation for feature engineering.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Tuple

from src.config import FeatureConfig


def validate_data(df: pd.DataFrame, required_cols: List[str]) -> bool:
    """
    Validate that DataFrame contains required columns.
    
    Args:
        df: DataFrame to validate.
        required_cols: List of required column names.
    
    Returns:
        True if all columns present, raises ValueError otherwise.
    
    Raises:
        ValueError: If required columns are missing.
    """
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    return True


def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = "forward_fill",
    numeric_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Handle missing values in DataFrame.
    
    Args:
        df: DataFrame to process.
        strategy: Strategy for handling missing values.
            Options: 'forward_fill', 'backward_fill', 'mean', 'median', 'drop'.
        numeric_cols: List of numeric columns to process.
            If None, processes all numeric columns.
    
    Returns:
        DataFrame with missing values handled.
    """
    df = df.copy()
    
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if strategy == "forward_fill":
        df[numeric_cols] = df[numeric_cols].ffill()
        df[numeric_cols] = df[numeric_cols].bfill()
    elif strategy == "backward_fill":
        df[numeric_cols] = df[numeric_cols].bfill()
        df[numeric_cols] = df[numeric_cols].ffill()
    elif strategy == "mean":
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif strategy == "median":
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    elif strategy == "drop":
        df = df.dropna(subset=numeric_cols)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    return df


def remove_outliers(
    df: pd.DataFrame,
    columns: List[str],
    method: str = "iqr",
    factor: float = 1.5,
) -> pd.DataFrame:
    """
    Remove outliers from specified columns.
    
    Args:
        df: DataFrame to process.
        columns: List of column names to process.
        method: Outlier detection method ('iqr' or 'zscore').
        factor: Factor for IQR method or threshold for z-score.
    
    Returns:
        DataFrame with outliers removed.
    """
    df = df.copy()
    
    for col in columns:
        if method == "iqr":
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
        elif method == "zscore":
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            mask = z_scores < factor
        else:
            raise ValueError(f"Unknown method: {method}")
        
        df = df[mask]
    
    return df.reset_index(drop=True)


def prepare_data_for_training(
    df: pd.DataFrame,
    config: Optional[FeatureConfig] = None,
    remove_outliers_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Prepare data for training by cleaning and validating.
    
    Args:
        df: Raw DataFrame.
        config: FeatureConfig instance. If None, uses defaults.
        remove_outliers_cols: Optional list of columns to remove outliers from.
    
    Returns:
        Cleaned DataFrame ready for feature engineering.
    """
    if config is None:
        config = FeatureConfig()
    
    # Validate required columns
    required_cols = [
        config.timestamp_col,
        config.target_col,
        "workload",
        "resource_allocation",
        "congestion",
    ]
    validate_data(df, required_cols)
    
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[config.timestamp_col]):
        df[config.timestamp_col] = pd.to_datetime(df[config.timestamp_col])
    
    # Sort by timestamp
    df = df.sort_values(config.timestamp_col).reset_index(drop=True)
    
    # Handle missing values
    df = handle_missing_values(df, strategy="forward_fill")
    
    # Remove outliers if specified
    if remove_outliers_cols:
        df = remove_outliers(df, remove_outliers_cols, method="iqr")
    
    return df


def train_test_split_temporal(
    df: pd.DataFrame,
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    timestamp_col: str = "timestamp",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data temporally (no shuffling) for time-series.
    
    Args:
        df: DataFrame sorted by timestamp.
        train_split: Proportion for training set.
        val_split: Proportion for validation set.
        test_split: Proportion for test set.
        timestamp_col: Name of timestamp column.
    
    Returns:
        Tuple of (train_df, val_df, test_df).
    
    Raises:
        ValueError: If splits don't sum to 1.0.
    """
    if abs(train_split + val_split + test_split - 1.0) > 1e-6:
        raise ValueError("Splits must sum to 1.0")
    
    n = len(df)
    train_end = int(n * train_split)
    val_end = int(n * (train_split + val_split))
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    return train_df, val_df, test_df

