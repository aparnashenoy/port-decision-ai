"""
Model evaluation module for Port Decision AI.

Provides comprehensive evaluation metrics and analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple

from src.models.train_delay_model import DelayPredictor
from src.utils.metrics import calculate_regression_metrics
from src.utils.logging import get_logger

logger = get_logger("evaluate")


def evaluate_model(
    model: DelayPredictor,
    X: pd.DataFrame,
    y: pd.Series,
    dataset_name: str = "test",
) -> Dict[str, float]:
    """
    Evaluate model performance on a dataset.
    
    Args:
        model: Trained DelayPredictor instance.
        X: Feature DataFrame.
        y: True target values.
        dataset_name: Name of dataset for logging.
    
    Returns:
        Dictionary of evaluation metrics.
    """
    logger.info(f"Evaluating model on {dataset_name} set ({len(X)} samples)")
    
    # Get predictions with uncertainty
    predictions = model.predict(X, return_uncertainty=True)
    
    # Calculate metrics
    metrics = calculate_regression_metrics(
        y.values,
        predictions["prediction"],
        predictions.get("lower"),
        predictions.get("upper"),
    )
    
    # Add dataset prefix
    metrics = {f"{dataset_name}_{k}": v for k, v in metrics.items()}
    
    logger.info(f"{dataset_name} MAE: {metrics[f'{dataset_name}_mae']:.2f}")
    logger.info(f"{dataset_name} RMSE: {metrics[f'{dataset_name}_rmse']:.2f}")
    
    if f"{dataset_name}_interval_coverage" in metrics:
        logger.info(
            f"{dataset_name} Interval Coverage: "
            f"{metrics[f'{dataset_name}_interval_coverage']:.2f}%"
        )
    
    return metrics


def evaluate_by_time_period(
    model: DelayPredictor,
    X: pd.DataFrame,
    y: pd.Series,
    timestamps: pd.Series,
    period: str = "hour",
) -> pd.DataFrame:
    """
    Evaluate model performance by time period.
    
    Args:
        model: Trained DelayPredictor instance.
        X: Feature DataFrame.
        y: True target values.
        timestamps: Timestamp series.
        period: Time period to group by ('hour', 'day', 'week').
    
    Returns:
        DataFrame with metrics per time period.
    """
    predictions = model.predict(X, return_uncertainty=True)
    
    df_results = pd.DataFrame(
        {
            "timestamp": timestamps,
            "y_true": y.values,
            "y_pred": predictions["prediction"],
            "y_lower": predictions.get("lower"),
            "y_upper": predictions.get("upper"),
        }
    )
    
    # Map period names to pandas frequency strings
    # Note: Use lowercase 'h' for hourly (some pandas versions require lowercase)
    period_map = {
        "hour": "h",
        "day": "D",
        "week": "W",
        "month": "M",
    }
    
    # Convert period name to pandas frequency
    pandas_freq = period_map.get(period.lower(), period)
    
    # Group by time period
    df_results["period"] = pd.to_datetime(df_results["timestamp"]).dt.to_period(pandas_freq)
    
    # Calculate metrics per period
    period_metrics = []
    for period_val, group in df_results.groupby("period"):
        period_mae = np.mean(np.abs(group["y_true"] - group["y_pred"]))
        period_rmse = np.sqrt(np.mean((group["y_true"] - group["y_pred"]) ** 2))
        
        if "y_lower" in group.columns and "y_upper" in group.columns:
            coverage = np.mean(
                (group["y_true"] >= group["y_lower"])
                & (group["y_true"] <= group["y_upper"])
            )
        else:
            coverage = np.nan
        
        period_metrics.append(
            {
                "period": str(period_val),
                "n_samples": len(group),
                "mae": period_mae,
                "rmse": period_rmse,
                "coverage": coverage,
            }
        )
    
    return pd.DataFrame(period_metrics)


def analyze_prediction_errors(
    model: DelayPredictor,
    X: pd.DataFrame,
    y: pd.Series,
    feature_cols: list,
) -> pd.DataFrame:
    """
    Analyze prediction errors by feature values.
    
    Args:
        model: Trained DelayPredictor instance.
        X: Feature DataFrame.
        y: True target values.
        feature_cols: List of feature columns to analyze.
    
    Returns:
        DataFrame with error analysis by feature.
    """
    predictions = model.predict(X, return_uncertainty=False)
    errors = y.values - predictions["prediction"]
    abs_errors = np.abs(errors)
    
    error_analysis = []
    
    for col in feature_cols[:10]:  # Limit to top 10 features
        if col not in X.columns:
            continue
        
        # Bin feature values
        feature_values = X[col].values
        if pd.api.types.is_numeric_dtype(X[col]):
            bins = np.quantile(feature_values, [0, 0.25, 0.5, 0.75, 1.0])
            labels = ["Q1", "Q2", "Q3", "Q4"]
            binned = pd.cut(feature_values, bins=bins, labels=labels, include_lowest=True)
        else:
            binned = feature_values
        
        # Calculate mean error per bin
        for bin_val in pd.unique(binned):
            mask = binned == bin_val
            if mask.sum() > 0:
                error_analysis.append(
                    {
                        "feature": col,
                        "bin": str(bin_val),
                        "mean_error": np.mean(errors[mask]),
                        "mean_abs_error": np.mean(abs_errors[mask]),
                        "n_samples": mask.sum(),
                    }
                )
    
    return pd.DataFrame(error_analysis)


def create_evaluation_report(
    model: DelayPredictor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    timestamps: Optional[pd.Series] = None,
) -> Dict:
    """
    Create comprehensive evaluation report.
    
    Args:
        model: Trained DelayPredictor instance.
        X_test: Test feature DataFrame.
        y_test: Test target values.
        timestamps: Optional timestamp series.
    
    Returns:
        Dictionary with evaluation report.
    """
    # Overall metrics
    metrics = evaluate_model(model, X_test, y_test, dataset_name="test")
    
    report = {
        "overall_metrics": metrics,
        "feature_importance": model.get_feature_importance().to_dict("records"),
    }
    
    # Time-based analysis if timestamps provided
    if timestamps is not None:
        report["hourly_metrics"] = evaluate_by_time_period(
            model, X_test, y_test, timestamps, period="hour"
        ).to_dict("records")
    
    return report

