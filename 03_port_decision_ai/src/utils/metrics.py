"""
Metrics utilities for model evaluation and monitoring.

Provides regression metrics and uncertainty quantification measures.
"""

import numpy as np
from typing import Dict, Tuple


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error (MAE).
    
    Args:
        y_true: True target values.
        y_pred: Predicted values.
    
    Returns:
        MAE score.
    """
    return float(np.mean(np.abs(y_true - y_pred)))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Error (RMSE).
    
    Args:
        y_true: True target values.
        y_pred: Predicted values.
    
    Returns:
        RMSE score.
    """
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mean_absolute_percentage_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    epsilon: float = 1e-8,
) -> float:
    """
    Calculate Mean Absolute Percentage Error (MAPE).
    
    Args:
        y_true: True target values.
        y_pred: Predicted values.
        epsilon: Small value to avoid division by zero.
    
    Returns:
        MAPE score.
    """
    return float(
        np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    )


def prediction_interval_coverage(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """
    Calculate coverage of prediction intervals.
    
    Args:
        y_true: True target values.
        lower: Lower bound of prediction intervals.
        upper: Upper bound of prediction intervals.
    
    Returns:
        Coverage percentage (0-100).
    """
    coverage = np.mean((y_true >= lower) & (y_true <= upper))
    return float(coverage * 100)


def mean_prediction_interval_width(
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """
    Calculate mean width of prediction intervals.
    
    Args:
        lower: Lower bound of prediction intervals.
        upper: Upper bound of prediction intervals.
    
    Returns:
        Mean interval width.
    """
    return float(np.mean(upper - lower))


def calculate_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_lower: np.ndarray = None,
    y_upper: np.ndarray = None,
) -> Dict[str, float]:
    """
    Calculate comprehensive regression metrics.
    
    Args:
        y_true: True target values.
        y_pred: Predicted values.
        y_lower: Optional lower bound for uncertainty metrics.
        y_upper: Optional upper bound for uncertainty metrics.
    
    Returns:
        Dictionary of metric names and values.
    """
    metrics = {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "mape": mean_absolute_percentage_error(y_true, y_pred),
    }
    
    if y_lower is not None and y_upper is not None:
        metrics["interval_coverage"] = prediction_interval_coverage(
            y_true, y_lower, y_upper
        )
        metrics["mean_interval_width"] = mean_prediction_interval_width(
            y_lower, y_upper
        )
    
    return metrics


def calculate_quantile_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    quantile: float,
) -> float:
    """
    Calculate quantile loss (pinball loss).
    
    Args:
        y_true: True target values.
        y_pred: Predicted quantile values.
        quantile: Quantile level (0-1).
    
    Returns:
        Quantile loss.
    """
    error = y_true - y_pred
    loss = np.maximum(quantile * error, (quantile - 1) * error)
    return float(np.mean(loss))

