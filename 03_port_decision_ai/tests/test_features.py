"""
Unit tests for feature engineering module.
"""

import pandas as pd
import numpy as np
import pytest
from datetime import datetime, timedelta

from src.data.features import (
    create_rolling_features,
    create_lag_features,
    create_trend_features,
    create_utilization_features,
    engineer_features,
)


@pytest.fixture
def sample_data():
    """Create sample time-series data."""
    timestamps = pd.date_range(
        start="2024-01-01", periods=100, freq="15min"
    )
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "workload": np.random.uniform(20, 80, 100),
            "resource_allocation": np.random.uniform(30, 70, 100),
            "congestion": np.random.uniform(0, 1, 100),
            "delay_minutes": np.random.uniform(0, 30, 100),
        }
    )


def test_create_rolling_features(sample_data):
    """Test rolling window feature creation."""
    result = create_rolling_features(
        sample_data,
        windows_hours=[1.0, 3.0],
        timestamp_col="timestamp",
    )
    
    assert len(result) == len(sample_data)
    assert "workload_rolling_mean_1h" in result.columns
    assert "workload_rolling_std_3h" in result.columns


def test_create_lag_features(sample_data):
    """Test lag feature creation."""
    result = create_lag_features(sample_data, lags=[1, 2, 4])
    
    assert "workload_lag_1" in result.columns
    assert "workload_lag_2" in result.columns
    assert "workload_lag_4" in result.columns


def test_create_utilization_features(sample_data):
    """Test utilization feature creation."""
    result = create_utilization_features(sample_data)
    
    assert "utilization_ratio" in result.columns
    assert "resource_gap" in result.columns


def test_engineer_features(sample_data):
    """Test complete feature engineering pipeline."""
    result = engineer_features(sample_data)
    
    assert len(result) > 0
    assert "hour" in result.columns
    assert "utilization_ratio" in result.columns

