"""
Training pipeline for Port Decision AI.

End-to-end training script that:
1. Generates synthetic data
2. Engineers features
3. Trains ML model
4. Evaluates performance
5. Saves model artifact
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from src.config import get_config
from src.data.make_synthetic import generate_synthetic_data, save_synthetic_data
from src.data.preprocess import prepare_data_for_training, train_test_split_temporal
from src.data.features import engineer_features, get_feature_columns
from src.models.train_delay_model import train_model
from src.models.evaluate import evaluate_model, create_evaluation_report
from src.models.interpret import create_comprehensive_interpretation, print_interpretation_report
from src.utils.logging import setup_logging, get_logger


def main():
    """Main training pipeline."""
    # Load configuration
    config = get_config()
    
    # Setup logging
    logger = setup_logging(config.logs_dir, log_level="INFO")
    logger.info("Starting training pipeline")
    
    # Step 1: Generate synthetic data
    logger.info("Generating synthetic data...")
    df = generate_synthetic_data(config.data)
    
    # Save raw data
    data_path = config.data_dir / "synthetic_data.parquet"
    save_synthetic_data(df, str(data_path), format="parquet")
    logger.info(f"Saved {len(df)} samples to {data_path}")
    
    # Step 2: Preprocess data
    logger.info("Preprocessing data...")
    df_clean = prepare_data_for_training(
        df,
        config.features,
        remove_outliers_cols=["delay_minutes"],
    )
    logger.info(f"Cleaned data: {len(df_clean)} samples")
    
    # Step 3: Engineer features
    logger.info("Engineering features...")
    df_features = engineer_features(df_clean, config.features)
    logger.info(f"Feature engineering complete. Shape: {df_features.shape}")
    
    # Step 4: Split data
    logger.info("Splitting data temporally...")
    train_df, val_df, test_df = train_test_split_temporal(
        df_features,
        train_split=config.model.train_split,
        val_split=config.model.val_split,
        test_split=config.model.test_split,
        timestamp_col=config.features.timestamp_col,
    )
    logger.info(
        f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}"
    )
    
    # Step 5: Get feature columns
    feature_cols = get_feature_columns(
        train_df,
        exclude_cols=[config.features.timestamp_col],
    )
    logger.info(f"Using {len(feature_cols)} features")
    
    # Step 6: Train model
    logger.info("Training model...")
    model_path = config.models_dir / "delay_predictor.pkl"
    model, train_metrics = train_model(
        train_df,
        val_df,
        feature_cols,
        config.features.target_col,
        config.model,
        save_path=str(model_path),
    )
    
    logger.info("Training metrics:")
    for key, value in train_metrics.items():
        logger.info(f"  {key}: {value:.4f}")
    
    # Step 7: Evaluate on test set
    logger.info("Evaluating on test set...")
    X_test = test_df[feature_cols]
    y_test = test_df[config.features.target_col]
    test_metrics = evaluate_model(model, X_test, y_test, dataset_name="test")
    
    logger.info("Test metrics:")
    for key, value in test_metrics.items():
        logger.info(f"  {key}: {value:.4f}")
    
    # Step 8: Create evaluation report
    logger.info("Creating evaluation report...")
    report = create_evaluation_report(
        model,
        X_test,
        y_test,
        timestamps=test_df[config.features.timestamp_col],
    )
    
    # Save report
    import json
    report_path = config.models_dir / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Evaluation report saved to {report_path}")
    
    # Step 9: Create comprehensive interpretation
    logger.info("Generating model interpretation...")
    interpretation = create_comprehensive_interpretation(
        model,
        test_metrics,
        X_test,
        y_test,
    )
    
    # Save interpretation
    interpretation_path = config.models_dir / "interpretation_report.json"
    with open(interpretation_path, "w") as f:
        json.dump(interpretation, f, indent=2, default=str)
    logger.info(f"Interpretation report saved to {interpretation_path}")
    
    # Print interpretation to console
    print_interpretation_report(interpretation)
    
    logger.info("Training pipeline completed successfully!")
    logger.info(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()

