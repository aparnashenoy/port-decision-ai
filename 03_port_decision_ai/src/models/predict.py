"""
Prediction module for Port Decision AI.

Provides prediction interface for trained models.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional

from src.models.train_delay_model import DelayPredictor
from src.utils.logging import get_logger

logger = get_logger("predict")


def predict_delay(
    model: DelayPredictor,
    X: pd.DataFrame,
    return_uncertainty: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Predict delays using trained model.
    
    Args:
        model: Trained DelayPredictor instance.
        X: Feature DataFrame.
        return_uncertainty: Whether to return uncertainty intervals.
    
    Returns:
        Dictionary with predictions and optional uncertainty bounds.
    """
    return model.predict(X, return_uncertainty=return_uncertainty)


def predict_batch(
    model: DelayPredictor,
    X_batch: pd.DataFrame,
    batch_size: int = 1000,
) -> Dict[str, np.ndarray]:
    """
    Predict delays in batches for large datasets.
    
    Args:
        model: Trained DelayPredictor instance.
        X_batch: Feature DataFrame.
        batch_size: Number of samples per batch.
    
    Returns:
        Dictionary with predictions and uncertainty bounds.
    """
    n_samples = len(X_batch)
    all_predictions = []
    all_lower = []
    all_upper = []
    
    for i in range(0, n_samples, batch_size):
        batch = X_batch.iloc[i : i + batch_size]
        batch_result = model.predict(batch, return_uncertainty=True)
        
        all_predictions.append(batch_result["prediction"])
        all_lower.append(batch_result["lower"])
        all_upper.append(batch_result["upper"])
    
    return {
        "prediction": np.concatenate(all_predictions),
        "lower": np.concatenate(all_lower),
        "upper": np.concatenate(all_upper),
    }


def format_prediction_result(
    predictions: Dict[str, np.ndarray],
    include_confidence: bool = True,
) -> pd.DataFrame:
    """
    Format prediction results as DataFrame.
    
    Args:
        predictions: Dictionary with prediction results.
        include_confidence: Whether to include confidence metrics.
    
    Returns:
        Formatted DataFrame with predictions.
    """
    df = pd.DataFrame(
        {
            "predicted_delay": predictions["prediction"],
            "lower_bound": predictions.get("lower", predictions["prediction"]),
            "upper_bound": predictions.get("upper", predictions["prediction"]),
        }
    )
    
    if include_confidence:
        df["confidence_interval_width"] = df["upper_bound"] - df["lower_bound"]
        df["uncertainty_level"] = pd.cut(
            df["confidence_interval_width"],
            bins=[0, 5, 10, 20, float("inf")],
            labels=["Low", "Medium", "High", "Very High"],
        )
    
    return df

