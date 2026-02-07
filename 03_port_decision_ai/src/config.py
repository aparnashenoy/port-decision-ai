"""
Configuration module for Port Decision AI system.

This module centralizes all configuration parameters for data generation,
feature engineering, model training, optimization, and serving.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class DataConfig:
    """Configuration for synthetic data generation."""
    
    n_samples: int = 10000
    start_date: str = "2024-01-01"
    freq_minutes: int = 15
    noise_level: float = 0.1
    trend_strength: float = 0.05
    seasonality_periods: List[int] = None
    spike_probability: float = 0.02
    spike_magnitude: float = 3.0
    
    def __post_init__(self) -> None:
        """Initialize default seasonality periods if not provided."""
        if self.seasonality_periods is None:
            self.seasonality_periods = [24, 168]  # Daily and weekly


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    
    rolling_windows_hours: List[float] = None
    lag_features: List[int] = None
    target_col: str = "delay_minutes"
    timestamp_col: str = "timestamp"
    
    def __post_init__(self) -> None:
        """Initialize default windows and lags if not provided."""
        if self.rolling_windows_hours is None:
            self.rolling_windows_hours = [1.0, 3.0, 6.0]
        if self.lag_features is None:
            self.lag_features = [1, 2, 4, 8]


@dataclass
class ModelConfig:
    """Configuration for ML model training."""
    
    model_type: str = "lightgbm"
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    random_state: int = 42
    
    # LightGBM hyperparameters
    lgbm_params: Dict = None
    n_estimators: int = 200
    learning_rate: float = 0.05
    max_depth: int = 8
    num_leaves: int = 31
    min_child_samples: int = 20
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    
    # Uncertainty estimation
    quantile_levels: List[float] = None
    use_conformal: bool = True
    conformal_alpha: float = 0.1
    
    model_save_path: str = "models/delay_predictor.pkl"
    
    def __post_init__(self) -> None:
        """Initialize default parameters if not provided."""
        if self.lgbm_params is None:
            self.lgbm_params = {
                "objective": "regression",
                "metric": "rmse",
                "boosting_type": "gbdt",
                "num_leaves": self.num_leaves,
                "learning_rate": self.learning_rate,
                "feature_fraction": self.colsample_bytree,
                "bagging_fraction": self.subsample,
                "bagging_freq": 5,
                "verbose": -1,
                "random_state": self.random_state,
            }
        if self.quantile_levels is None:
            self.quantile_levels = [0.1, 0.5, 0.9]


@dataclass
class OptimizationConfig:
    """Configuration for optimization engine."""
    
    solver: str = "SCIP"
    time_limit_seconds: int = 30
    lambda_cost: float = 0.1  # Trade-off between delay and action cost
    resource_limit_per_hour: float = 100.0
    min_allocation: float = 10.0
    max_allocation: float = 50.0
    switching_penalty: float = 5.0
    horizon_hours: int = 4


@dataclass
class ServingConfig:
    """Configuration for API serving."""
    
    host: str = "0.0.0.0"
    port: int = 8050
    reload: bool = False
    log_level: str = "INFO"
    api_version: str = "v1"
    model_path: str = "models/delay_predictor.pkl"


@dataclass
class ProjectConfig:
    """Main project configuration aggregating all sub-configs."""
    
    project_root: Path = Path(__file__).parent.parent
    data_dir: Path = None
    models_dir: Path = None
    logs_dir: Path = None
    
    data: DataConfig = None
    features: FeatureConfig = None
    model: ModelConfig = None
    optimization: OptimizationConfig = None
    serving: ServingConfig = None
    
    def __post_init__(self) -> None:
        """Initialize default configs and directories."""
        if self.data_dir is None:
            self.data_dir = self.project_root / "data"
        if self.models_dir is None:
            self.models_dir = self.project_root / "models"
        if self.logs_dir is None:
            self.logs_dir = self.project_root / "logs"
        
        # Create directories
        self.data_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        
        # Initialize sub-configs if not provided
        if self.data is None:
            self.data = DataConfig()
        if self.features is None:
            self.features = FeatureConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.optimization is None:
            self.optimization = OptimizationConfig()
        if self.serving is None:
            self.serving = ServingConfig()


def get_config() -> ProjectConfig:
    """
    Get the default project configuration.
    
    Returns:
        ProjectConfig: Configured project settings.
    """
    return ProjectConfig()

