"""
Model training module for Port Decision AI.

Trains LightGBM regression model for delay prediction with uncertainty estimation.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple

import lightgbm as lgb
from sklearn.model_selection import train_test_split

from src.config import ModelConfig, ProjectConfig
from src.utils.logging import get_logger
from src.utils.metrics import calculate_regression_metrics

logger = get_logger("train_delay_model")


class DelayPredictor:
    """
    LightGBM-based delay prediction model with uncertainty estimation.
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize delay predictor.
        
        Args:
            config: ModelConfig instance. If None, uses defaults.
        """
        self.config = config or ModelConfig()
        self.model = None
        self.quantile_models = {}
        self.feature_names = None
        self.residual_std = None
        self.is_fitted = False
    
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> Dict[str, float]:
        """
        Train the delay prediction model.
        
        Args:
            X_train: Training features.
            y_train: Training targets.
            X_val: Optional validation features.
            y_val: Optional validation targets.
        
        Returns:
            Dictionary of training metrics.
        """
        logger.info(f"Training model on {len(X_train)} samples")
        
        self.feature_names = X_train.columns.tolist()
        
        # Convert to numpy arrays
        X_train_arr = X_train.values
        y_train_arr = y_train.values
        
        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train_arr, label=y_train_arr)
        
        if X_val is not None and y_val is not None:
            X_val_arr = X_val.values
            y_val_arr = y_val.values
            val_data = lgb.Dataset(X_val_arr, label=y_val_arr, reference=train_data)
            valid_sets = [train_data, val_data]
            valid_names = ["train", "valid"]
            # Early stopping requires validation set
            callbacks = [lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=50)]
        else:
            valid_sets = [train_data]
            valid_names = ["train"]
            # No early stopping without validation set
            callbacks = [lgb.log_evaluation(period=50)]
        
        # Train main model
        self.model = lgb.train(
            self.config.lgbm_params,
            train_data,
            num_boost_round=self.config.n_estimators,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )
        
        # Calculate residual standard deviation for uncertainty
        y_pred_train = self.model.predict(X_train_arr)
        residuals = y_train_arr - y_pred_train
        self.residual_std = np.std(residuals)
        
        # Train quantile models if configured
        if self.config.quantile_levels:
            logger.info("Training quantile models for uncertainty estimation")
            for quantile in self.config.quantile_levels:
                quantile_params = self.config.lgbm_params.copy()
                quantile_params["objective"] = "quantile"
                quantile_params["alpha"] = quantile
                
                # Use same callback setup as main model (early stopping only if validation set exists)
                quantile_callbacks = callbacks
                
                quantile_model = lgb.train(
                    quantile_params,
                    train_data,
                    num_boost_round=self.config.n_estimators,
                    valid_sets=valid_sets,
                    valid_names=valid_names,
                    callbacks=quantile_callbacks,
                )
                self.quantile_models[quantile] = quantile_model
        
        self.is_fitted = True
        
        # Calculate training metrics
        metrics = self._calculate_metrics(X_train, y_train, X_val, y_val)
        
        logger.info(f"Training completed. Train MAE: {metrics['train_mae']:.2f}")
        if "val_mae" in metrics:
            logger.info(f"Validation MAE: {metrics['val_mae']:.2f}")
        
        return metrics
    
    def predict(
        self,
        X: pd.DataFrame,
        return_uncertainty: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Predict delays with uncertainty estimates.
        
        Args:
            X: Feature DataFrame.
            return_uncertainty: Whether to return uncertainty intervals.
        
        Returns:
            Dictionary with 'prediction', 'lower', 'upper' keys.
        
        Raises:
            ValueError: If model is not fitted.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_arr = X.values
        
        # Main prediction
        prediction = self.model.predict(X_arr)
        
        result = {"prediction": prediction}
        
        if return_uncertainty:
            if self.quantile_models:
                # Use quantile models
                quantile_predictions = {}
                for quantile, model in self.quantile_models.items():
                    quantile_predictions[quantile] = model.predict(X_arr)
                
                # Get lower and upper bounds
                if 0.1 in quantile_predictions and 0.9 in quantile_predictions:
                    result["lower"] = quantile_predictions[0.1]
                    result["upper"] = quantile_predictions[0.9]
                elif 0.5 in quantile_predictions:
                    # Use median and residual std
                    median = quantile_predictions[0.5]
                    result["lower"] = median - 1.96 * self.residual_std
                    result["upper"] = median + 1.96 * self.residual_std
            else:
                # Use residual-based intervals
                result["lower"] = prediction - 1.96 * self.residual_std
                result["upper"] = prediction + 1.96 * self.residual_std
            
            # Ensure non-negative delays
            result["lower"] = np.maximum(0, result["lower"])
            result["upper"] = np.maximum(result["lower"], result["upper"])
        
        return result
    
    def get_feature_importance(self, importance_type: str = "gain") -> pd.DataFrame:
        """
        Get feature importance from trained model.
        
        Args:
            importance_type: Type of importance ('gain', 'split', 'gain_by_feature').
        
        Returns:
            DataFrame with feature names and importance scores.
        
        Raises:
            ValueError: If model is not fitted.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        importance = self.model.feature_importance(importance_type=importance_type)
        
        df_importance = pd.DataFrame(
            {
                "feature": self.feature_names,
                "importance": importance,
            }
        ).sort_values("importance", ascending=False)
        
        return df_importance
    
    def _calculate_metrics(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> Dict[str, float]:
        """Calculate training and validation metrics."""
        metrics = {}
        
        # Training metrics
        train_pred = self.model.predict(X_train.values)
        train_metrics = calculate_regression_metrics(y_train.values, train_pred)
        metrics.update({f"train_{k}": v for k, v in train_metrics.items()})
        
        # Validation metrics
        if X_val is not None and y_val is not None:
            val_pred_dict = self.predict(X_val, return_uncertainty=True)
            val_pred = val_pred_dict["prediction"]
            val_metrics = calculate_regression_metrics(
                y_val.values,
                val_pred,
                val_pred_dict.get("lower"),
                val_pred_dict.get("upper"),
            )
            metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
        
        return metrics
    
    def save(self, filepath: str) -> None:
        """
        Save model to file.
        
        Args:
            filepath: Path to save model.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            "model": self.model,
            "quantile_models": self.quantile_models,
            "feature_names": self.feature_names,
            "residual_std": self.residual_std,
            "config": self.config,
        }
        
        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> "DelayPredictor":
        """
        Load model from file.
        
        Args:
            filepath: Path to model file.
        
        Returns:
            Loaded DelayPredictor instance.
        """
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)
        
        predictor = cls(config=model_data["config"])
        predictor.model = model_data["model"]
        predictor.quantile_models = model_data["quantile_models"]
        predictor.feature_names = model_data["feature_names"]
        predictor.residual_std = model_data["residual_std"]
        predictor.is_fitted = True
        
        logger.info(f"Model loaded from {filepath}")
        return predictor


def train_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    config: Optional[ModelConfig] = None,
    save_path: Optional[str] = None,
) -> Tuple[DelayPredictor, Dict[str, float]]:
    """
    Train delay prediction model.
    
    Args:
        train_df: Training DataFrame.
        val_df: Validation DataFrame.
        feature_cols: List of feature column names.
        target_col: Name of target column.
        config: ModelConfig instance. If None, uses defaults.
        save_path: Optional path to save trained model.
    
    Returns:
        Tuple of (trained model, metrics dictionary).
    """
    if config is None:
        config = ModelConfig()
    
    # Prepare data
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_val = val_df[feature_cols]
    y_val = val_df[target_col]
    
    # Train model
    model = DelayPredictor(config=config)
    metrics = model.fit(X_train, y_train, X_val, y_val)
    
    # Save if path provided
    if save_path:
        model.save(save_path)
    
    # Log feature importance
    importance_df = model.get_feature_importance()
    logger.info("Top 10 features by importance:")
    logger.info(importance_df.head(10).to_string())
    
    return model, metrics

