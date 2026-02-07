"""
Example usage script for Port Decision AI.

Demonstrates how to use the system for prediction and decision making.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from src.config import get_config
from src.data.make_synthetic import generate_synthetic_data
from src.data.preprocess import prepare_data_for_training
from src.data.features import engineer_features, get_feature_columns
from src.models.train_delay_model import DelayPredictor
from src.decision.recommend import DecisionRecommender
from src.decision.constraints import OptimizationConstraints


def example_prediction():
    """Example: Load model and make predictions."""
    print("=" * 60)
    print("Example: Making Predictions")
    print("=" * 60)
    
    # Load model (assumes model is trained)
    model_path = "models/delay_predictor.pkl"
    
    try:
        model = DelayPredictor.load(model_path)
        print(f"✓ Model loaded from {model_path}")
    except FileNotFoundError:
        print(f"✗ Model not found at {model_path}")
        print("  Please run 'python train.py' first to train a model.")
        return
    
    # Generate sample data
    config = get_config()
    df = generate_synthetic_data(DataConfig(n_samples=100))
    df_clean = prepare_data_for_training(df, config.features)
    df_features = engineer_features(df_clean, config.features)
    
    # Get features
    feature_cols = get_feature_columns(
        df_features,
        exclude_cols=[config.features.timestamp_col],
    )
    
    # Make predictions
    X = df_features[feature_cols].iloc[:5]
    predictions = model.predict(X, return_uncertainty=True)
    
    print(f"\nPredictions for {len(X)} samples:")
    print("-" * 60)
    for i in range(len(X)):
        print(f"Sample {i+1}:")
        print(f"  Predicted delay: {predictions['prediction'][i]:.2f} minutes")
        print(f"  Confidence interval: [{predictions['lower'][i]:.2f}, {predictions['upper'][i]:.2f}]")
        print(f"  True delay: {df_features['delay_minutes'].iloc[i]:.2f} minutes")
        print()


def example_decision():
    """Example: Get decision recommendations."""
    print("=" * 60)
    print("Example: Decision Recommendations")
    print("=" * 60)
    
    # Load model
    model_path = "models/delay_predictor.pkl"
    
    try:
        model = DelayPredictor.load(model_path)
        print(f"✓ Model loaded from {model_path}")
    except FileNotFoundError:
        print(f"✗ Model not found at {model_path}")
        print("  Please run 'python train.py' first to train a model.")
        return
    
    # Create recommender
    constraints = OptimizationConstraints()
    recommender = DecisionRecommender(model, constraints=constraints)
    print("✓ Decision recommender initialized")
    
    # Generate sample state
    config = get_config()
    df = generate_synthetic_data(DataConfig(n_samples=50))
    df_clean = prepare_data_for_training(df, config.features)
    df_features = engineer_features(df_clean, config.features)
    
    feature_cols = get_feature_columns(
        df_features,
        exclude_cols=[config.features.timestamp_col],
    )
    
    # Get recommendation for current state
    current_state = df_features[feature_cols].iloc[[0]]
    current_allocation = 40.0
    
    recommendation = recommender.recommend(
        current_state=current_state,
        current_allocation=current_allocation,
        horizon_hours=1,
    )
    
    print("\nDecision Recommendation:")
    print("-" * 60)
    print(f"Baseline delay: {recommendation['baseline_delay']:.2f} minutes")
    print(f"  Confidence: [{recommendation['baseline_delay_lower']:.2f}, {recommendation['baseline_delay_upper']:.2f}]")
    print(f"Risk level: {recommendation['risk_level']}")
    print(f"\nCurrent allocation: {recommendation['current_allocation']:.2f}")
    print(f"Recommended allocation: {recommendation['recommended_allocation']:.2f}")
    print(f"Expected improvement: {recommendation['expected_improvement']:.2f} minutes")
    print(f"Expected delay after action: {recommendation['expected_delay_after']:.2f} minutes")
    print(f"Optimization status: {recommendation['optimization_status']}")
    print()


if __name__ == "__main__":
    from src.config import DataConfig
    
    print("\n" + "=" * 60)
    print("Port Decision AI - Example Usage")
    print("=" * 60 + "\n")
    
    # Run examples
    example_prediction()
    print()
    example_decision()
    
    print("=" * 60)
    print("Examples completed!")
    print("=" * 60)

