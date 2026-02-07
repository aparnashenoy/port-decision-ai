"""
Unit tests for metrics module.
"""

import numpy as np
import pytest

from src.utils.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    prediction_interval_coverage,
    calculate_regression_metrics,
)


def test_mean_absolute_error():
    """Test MAE calculation."""
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
    
    mae = mean_absolute_error(y_true, y_pred)
    assert mae == 0.5


def test_root_mean_squared_error():
    """Test RMSE calculation."""
    y_true = np.array([1, 2, 3])
    y_pred = np.array([2, 3, 4])
    
    rmse = root_mean_squared_error(y_true, y_pred)
    assert abs(rmse - np.sqrt(3)) < 1e-6


def test_prediction_interval_coverage():
    """Test interval coverage calculation."""
    y_true = np.array([5, 10, 15, 20, 25])
    lower = np.array([3, 8, 13, 18, 23])
    upper = np.array([7, 12, 17, 22, 27])
    
    coverage = prediction_interval_coverage(y_true, lower, upper)
    assert coverage == 100.0


def test_calculate_regression_metrics():
    """Test comprehensive metrics calculation."""
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1.2, 2.1, 2.9, 4.2, 4.8])
    y_lower = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    y_upper = np.array([2.0, 2.8, 3.5, 5.0, 5.5])
    
    metrics = calculate_regression_metrics(y_true, y_pred, y_lower, y_upper)
    
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "interval_coverage" in metrics
    assert metrics["mae"] > 0

