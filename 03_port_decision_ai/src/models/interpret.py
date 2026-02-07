"""
Model interpretation and insights module for Port Decision AI.

Provides meaningful interpretation of model results, feature importance,
and business insights.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional

from src.models.train_delay_model import DelayPredictor
from src.utils.logging import get_logger

logger = get_logger("interpret")


def interpret_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 10,
) -> Dict[str, any]:
    """
    Interpret feature importance with business context.
    
    Args:
        importance_df: DataFrame with feature names and importance scores.
        top_n: Number of top features to analyze.
    
    Returns:
        Dictionary with interpretations and insights.
    """
    top_features = importance_df.head(top_n)
    
    interpretations = {
        "summary": f"Top {top_n} features account for {top_features['importance'].sum() / importance_df['importance'].sum() * 100:.1f}% of total importance",
        "top_features": top_features.to_dict("records"),
        "insights": [],
    }
    
    # Categorize features
    lag_features = [f for f in top_features["feature"] if "_lag_" in f]
    rolling_features = [f for f in top_features["feature"] if "_rolling_" in f]
    trend_features = [f for f in top_features["feature"] if "_trend_" in f]
    time_features = [f for f in top_features["feature"] if f in ["hour", "day_of_week", "is_weekend"]]
    
    insights = []
    
    if lag_features:
        insights.append(
            f"**Temporal Dependencies**: {len(lag_features)} lag features in top {top_n}, "
            "indicating strong historical patterns. Recent workload and resource allocation "
            "are key predictors of delays."
        )
    
    if rolling_features:
        insights.append(
            f"**Rolling Window Patterns**: {len(rolling_features)} rolling window features "
            "show that short-term averages (1-6 hours) capture important operational dynamics."
        )
    
    if trend_features:
        insights.append(
            "**Trend Indicators**: Trend slope features suggest that the direction of change "
            "in operational metrics is predictive of delays."
        )
    
    if time_features:
        insights.append(
            "**Time-based Patterns**: Time features indicate predictable daily/weekly cycles "
            "in port operations that affect delays."
        )
    
    # Specific feature insights
    if any("workload" in f for f in top_features["feature"]):
        insights.append(
            "**Workload Impact**: Workload features are highly predictive, confirming that "
            "operational volume directly correlates with delay risk."
        )
    
    if any("resource_allocation" in f for f in top_features["feature"]):
        insights.append(
            "**Resource Management**: Resource allocation features indicate that proper "
            "resource planning is critical for minimizing delays."
        )
    
    if any("congestion" in f for f in top_features["feature"]):
        insights.append(
            "**Congestion Signals**: Congestion metrics capture operational bottlenecks "
            "that lead to delays."
        )
    
    interpretations["insights"] = insights
    
    return interpretations


def interpret_model_metrics(
    metrics: Dict[str, float],
    dataset_name: str = "test",
) -> Dict[str, any]:
    """
    Interpret model performance metrics with business context.
    
    Args:
        metrics: Dictionary of metric names and values.
        dataset_name: Name of the dataset (train/val/test).
    
    Returns:
        Dictionary with metric interpretations.
    """
    mae = metrics.get(f"{dataset_name}_mae", metrics.get("mae", 0))
    rmse = metrics.get(f"{dataset_name}_rmse", metrics.get("rmse", 0))
    mape = metrics.get(f"{dataset_name}_mape", metrics.get("mape", 0))
    coverage = metrics.get(f"{dataset_name}_interval_coverage", None)
    interval_width = metrics.get(f"{dataset_name}_mean_interval_width", None)
    
    interpretations = {
        "performance_summary": "",
        "mae_interpretation": "",
        "rmse_interpretation": "",
        "mape_interpretation": "",
        "uncertainty_interpretation": "",
        "business_impact": [],
    }
    
    # MAE interpretation
    if mae < 2:
        mae_level = "excellent"
        mae_desc = "The model predicts delays with very high accuracy."
    elif mae < 5:
        mae_level = "good"
        mae_desc = "The model provides reliable delay predictions suitable for operational decisions."
    elif mae < 10:
        mae_level = "moderate"
        mae_desc = "The model provides reasonable predictions but may benefit from more features or data."
    else:
        mae_level = "needs_improvement"
        mae_desc = "The model accuracy could be improved with additional features or model tuning."
    
    interpretations["mae_interpretation"] = (
        f"Mean Absolute Error: {mae:.2f} minutes ({mae_level}). "
        f"{mae_desc} On average, predictions are within {mae:.1f} minutes of actual delays."
    )
    
    # RMSE interpretation
    if rmse < 3:
        rmse_level = "excellent"
    elif rmse < 6:
        rmse_level = "good"
    elif rmse < 12:
        rmse_level = "moderate"
    else:
        rmse_level = "needs_improvement"
    
    interpretations["rmse_interpretation"] = (
        f"Root Mean Squared Error: {rmse:.2f} minutes ({rmse_level}). "
        f"This metric penalizes large errors more heavily. A value of {rmse:.1f} minutes "
        f"indicates the model handles most cases well, with occasional larger errors."
    )
    
    # MAPE interpretation
    if mape < 10:
        mape_level = "excellent"
    elif mape < 20:
        mape_level = "good"
    elif mape < 30:
        mape_level = "moderate"
    else:
        mape_level = "needs_improvement"
    
    interpretations["mape_interpretation"] = (
        f"Mean Absolute Percentage Error: {mape:.1f}% ({mape_level}). "
        f"The model's predictions are on average {mape:.1f}% off from actual delays."
    )
    
    # Uncertainty interpretation
    if coverage is not None and interval_width is not None:
        if coverage >= 90:
            coverage_level = "excellent"
            coverage_desc = "The uncertainty estimates are well-calibrated and reliable."
        elif coverage >= 80:
            coverage_level = "good"
            coverage_desc = "The uncertainty estimates are reasonably well-calibrated."
        elif coverage >= 70:
            coverage_level = "moderate"
            coverage_desc = "The uncertainty estimates may need improvement."
        else:
            coverage_level = "needs_improvement"
            coverage_desc = "The uncertainty estimates are not well-calibrated and should be refined."
        
        interpretations["uncertainty_interpretation"] = (
            f"Prediction Interval Coverage: {coverage:.1f}% ({coverage_level}). "
            f"{coverage_desc} "
            f"Mean interval width: {interval_width:.2f} minutes, indicating "
            f"{'high' if interval_width > 10 else 'moderate' if interval_width > 5 else 'low'} "
            f"prediction uncertainty."
        )
    
    # Overall performance summary
    if mae < 5 and rmse < 6:
        summary = "The model demonstrates strong predictive performance suitable for production use."
    elif mae < 10 and rmse < 12:
        summary = "The model shows good predictive performance with room for improvement."
    else:
        summary = "The model performance needs improvement before production deployment."
    
    interpretations["performance_summary"] = summary
    
    # Business impact
    business_impacts = []
    
    if mae < 5:
        business_impacts.append(
            f"**Operational Planning**: With {mae:.1f} minute average error, the model enables "
            "precise resource allocation and schedule optimization."
        )
    
    if coverage and coverage >= 80:
        business_impacts.append(
            "**Risk Management**: Well-calibrated uncertainty estimates allow for informed "
            "risk assessment and contingency planning."
        )
    
    if rmse < 6:
        business_impacts.append(
            "**Decision Support**: Low RMSE indicates the model reliably supports operational "
            "decisions without frequent large prediction errors."
        )
    
    interpretations["business_impact"] = business_impacts
    
    return interpretations


def interpret_prediction(
    prediction: float,
    lower: float,
    upper: float,
    risk_level: str,
) -> Dict[str, str]:
    """
    Interpret a single prediction with business context.
    
    Args:
        prediction: Predicted delay in minutes.
        lower: Lower confidence bound.
        upper: Upper confidence bound.
        risk_level: Risk level (Low/Medium/High/Very High).
    
    Returns:
        Dictionary with prediction interpretation.
    """
    interval_width = upper - lower
    
    interpretation = {
        "prediction_summary": "",
        "risk_assessment": "",
        "recommended_action": "",
    }
    
    # Prediction summary
    if prediction < 5:
        severity = "minimal"
        desc = "No significant delay expected."
    elif prediction < 15:
        severity = "low"
        desc = "Minor delay expected, manageable with standard operations."
    elif prediction < 30:
        severity = "moderate"
        desc = "Moderate delay expected, may require resource adjustments."
    elif prediction < 60:
        severity = "high"
        desc = "Significant delay expected, proactive intervention recommended."
    else:
        severity = "critical"
        desc = "Major delay expected, immediate action required."
    
    interpretation["prediction_summary"] = (
        f"Predicted delay: {prediction:.1f} minutes ({severity} severity). {desc} "
        f"Confidence interval: [{lower:.1f}, {upper:.1f}] minutes."
    )
    
    # Risk assessment
    risk_descriptions = {
        "Low": "Low uncertainty - prediction is reliable for decision-making.",
        "Medium": "Moderate uncertainty - consider additional context when making decisions.",
        "High": "High uncertainty - prediction should be used with caution.",
        "Very High": "Very high uncertainty - prediction may not be reliable, gather more information.",
    }
    
    interpretation["risk_assessment"] = (
        f"Risk Level: {risk_level}. {risk_descriptions.get(risk_level, 'Unknown risk level.')} "
        f"Uncertainty range: {interval_width:.1f} minutes."
    )
    
    # Recommended action
    if prediction < 10 and risk_level in ["Low", "Medium"]:
        action = "No immediate action required. Monitor situation."
    elif prediction < 20:
        action = "Consider minor resource adjustments to mitigate potential delays."
    elif prediction < 40:
        action = "Increase resource allocation or adjust schedules proactively."
    else:
        action = "Immediate intervention required. Allocate additional resources and review operational plan."
    
    interpretation["recommended_action"] = action
    
    return interpretation


def create_comprehensive_interpretation(
    model: DelayPredictor,
    metrics: Dict[str, float],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, any]:
    """
    Create comprehensive interpretation of model results.
    
    Args:
        model: Trained DelayPredictor instance.
        metrics: Dictionary of evaluation metrics.
        X_test: Test feature DataFrame.
        y_test: Test target values.
    
    Returns:
        Dictionary with comprehensive interpretations.
    """
    logger.info("Creating comprehensive model interpretation...")
    
    # Feature importance interpretation
    importance_df = model.get_feature_importance()
    feature_interpretation = interpret_feature_importance(importance_df, top_n=10)
    
    # Metrics interpretation
    metrics_interpretation = interpret_model_metrics(metrics, dataset_name="test")
    
    # Prediction distribution analysis
    predictions = model.predict(X_test, return_uncertainty=True)
    pred_values = predictions["prediction"]
    
    interpretation = {
        "model_performance": metrics_interpretation,
        "feature_insights": feature_interpretation,
        "prediction_statistics": {
            "mean_predicted_delay": float(np.mean(pred_values)),
            "median_predicted_delay": float(np.median(pred_values)),
            "std_predicted_delay": float(np.std(pred_values)),
            "min_predicted_delay": float(np.min(pred_values)),
            "max_predicted_delay": float(np.max(pred_values)),
            "prediction_range": f"{np.min(pred_values):.1f} - {np.max(pred_values):.1f} minutes",
        },
        "recommendations": [],
    }
    
    # Generate recommendations
    recommendations = []
    
    if metrics.get("test_mae", 0) > 10:
        recommendations.append(
            "Consider adding more features or collecting additional training data to improve accuracy."
        )
    
    if metrics.get("test_interval_coverage", 0) < 80:
        recommendations.append(
            "Uncertainty estimation needs improvement. Consider using conformal prediction or "
            "refining quantile models."
        )
    
    if len([f for f in feature_interpretation["top_features"] if "lag" in f["feature"]]) < 3:
        recommendations.append(
            "Few lag features in top importance - consider exploring longer temporal dependencies."
        )
    
    if metrics.get("test_mae", 0) < 5:
        recommendations.append(
            "Model performance is strong. Ready for production deployment with monitoring."
        )
    
    interpretation["recommendations"] = recommendations
    
    return interpretation


def print_interpretation_report(interpretation: Dict[str, any]) -> None:
    """
    Print a formatted interpretation report.
    
    Args:
        interpretation: Dictionary with interpretation results.
    """
    print("\n" + "=" * 80)
    print("MODEL INTERPRETATION REPORT")
    print("=" * 80)
    
    # Performance Summary
    print("\n📊 PERFORMANCE SUMMARY")
    print("-" * 80)
    print(interpretation["model_performance"]["performance_summary"])
    print(f"\n{interpretation['model_performance']['mae_interpretation']}")
    print(f"{interpretation['model_performance']['rmse_interpretation']}")
    print(f"{interpretation['model_performance']['mape_interpretation']}")
    if interpretation["model_performance"]["uncertainty_interpretation"]:
        print(f"{interpretation['model_performance']['uncertainty_interpretation']}")
    
    # Feature Insights
    print("\n🔍 FEATURE INSIGHTS")
    print("-" * 80)
    print(interpretation["feature_insights"]["summary"])
    print("\nKey Insights:")
    for insight in interpretation["feature_insights"]["insights"]:
        print(f"  • {insight}")
    
    # Prediction Statistics
    print("\n📈 PREDICTION STATISTICS")
    print("-" * 80)
    stats = interpretation["prediction_statistics"]
    print(f"Mean predicted delay: {stats['mean_predicted_delay']:.2f} minutes")
    print(f"Median predicted delay: {stats['median_predicted_delay']:.2f} minutes")
    print(f"Standard deviation: {stats['std_predicted_delay']:.2f} minutes")
    print(f"Range: {stats['prediction_range']} minutes")
    
    # Business Impact
    if interpretation["model_performance"]["business_impact"]:
        print("\n💼 BUSINESS IMPACT")
        print("-" * 80)
        for impact in interpretation["model_performance"]["business_impact"]:
            print(f"  • {impact}")
    
    # Recommendations
    if interpretation["recommendations"]:
        print("\n💡 RECOMMENDATIONS")
        print("-" * 80)
        for i, rec in enumerate(interpretation["recommendations"], 1):
            print(f"  {i}. {rec}")
    
    print("\n" + "=" * 80 + "\n")

