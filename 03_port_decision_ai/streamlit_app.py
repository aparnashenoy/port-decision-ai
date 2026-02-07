"""
Streamlit web application for Port Decision AI.

Provides interactive UI for viewing model results, making predictions,
and exploring insights.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import json
import pandas as pd
import numpy as np
import streamlit as st
from typing import Optional

from src.config import get_config
from src.models.train_delay_model import DelayPredictor
from src.decision.recommend import DecisionRecommender
from src.decision.constraints import OptimizationConstraints
from src.models.interpret import interpret_prediction

# Page config
st.set_page_config(
    page_title="Port Decision AI",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False
if "model" not in st.session_state:
    st.session_state.model = None
if "recommender" not in st.session_state:
    st.session_state.recommender = None


@st.cache_resource
def load_model_and_recommender():
    """Load model and recommender with caching."""
    try:
        config = get_config()
        model_path = config.models_dir / "delay_predictor.pkl"
        
        if not model_path.exists():
            return None, None, "Model file not found. Please train the model first."
        
        model = DelayPredictor.load(str(model_path))
        constraints = OptimizationConstraints()
        recommender = DecisionRecommender(model, constraints=constraints)
        
        return model, recommender, None
    except Exception as e:
        return None, None, f"Error loading model: {str(e)}"


def load_reports():
    """Load evaluation and interpretation reports."""
    config = get_config()
    evaluation_path = config.models_dir / "evaluation_report.json"
    interpretation_path = config.models_dir / "interpretation_report.json"
    
    evaluation = None
    interpretation = None
    
    if evaluation_path.exists():
        try:
            with open(evaluation_path, "r") as f:
                evaluation = json.load(f)
        except Exception as e:
            st.error(f"Error loading evaluation report: {e}")
    
    if interpretation_path.exists():
        try:
            with open(interpretation_path, "r") as f:
                interpretation = json.load(f)
        except Exception as e:
            st.error(f"Error loading interpretation report: {e}")
    
    return evaluation, interpretation


def main():
    """Main Streamlit application."""
    st.title("🚢 Port Decision AI")
    st.markdown("**Delay Prediction and Optimization Recommendations**")
    
    # Sidebar
    with st.sidebar:
        st.header("Navigation")
        page = st.radio(
            "Select Page",
            ["Dashboard", "Model Performance", "Feature Insights", "Make Prediction", "About"],
        )
        
        st.divider()
        
        # Model status
        st.header("Model Status")
        model, recommender, error = load_model_and_recommender()
        
        if error:
            st.error(f"❌ {error}")
            st.session_state.model_loaded = False
        elif model is not None:
            st.success("✅ Model Loaded")
            st.session_state.model = model
            st.session_state.recommender = recommender
            st.session_state.model_loaded = True
        else:
            st.warning("⚠️ Model Not Available")
            st.session_state.model_loaded = False
    
    # Main content based on selected page
    if page == "Dashboard":
        show_dashboard()
    elif page == "Model Performance":
        show_model_performance()
    elif page == "Feature Insights":
        show_feature_insights()
    elif page == "Make Prediction":
        show_prediction_interface()
    elif page == "About":
        show_about()


def show_dashboard():
    """Display main dashboard with overview."""
    st.header("📊 Dashboard")
    
    evaluation, interpretation = load_reports()
    
    if not evaluation and not interpretation:
        st.warning("No reports found. Please train the model first using `python train.py`")
        return
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    if evaluation and "overall_metrics" in evaluation:
        metrics = evaluation["overall_metrics"]
        test_mae = metrics.get("test_mae", 0)
        test_rmse = metrics.get("test_rmse", 0)
        test_mape = metrics.get("test_mape", 0)
        coverage = metrics.get("test_interval_coverage", 0)
        
        with col1:
            st.metric("Mean Absolute Error", f"{test_mae:.2f} min")
        with col2:
            st.metric("Root Mean Squared Error", f"{test_rmse:.2f} min")
        with col3:
            st.metric("Mean Absolute % Error", f"{test_mape:.1f}%")
        with col4:
            st.metric("Interval Coverage", f"{coverage:.1f}%")
    
    st.divider()
    
    # Performance summary
    if interpretation and "model_performance" in interpretation:
        perf = interpretation["model_performance"]
        
        st.subheader("Performance Summary")
        st.info(perf.get("performance_summary", "No summary available"))
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**MAE Interpretation**")
            st.write(perf.get("mae_interpretation", ""))
        
        with col2:
            st.markdown("**RMSE Interpretation**")
            st.write(perf.get("rmse_interpretation", ""))
    
    # Business impact
    if interpretation and "model_performance" in interpretation:
        impacts = interpretation["model_performance"].get("business_impact", [])
        if impacts:
            st.subheader("💼 Business Impact")
            for impact in impacts:
                st.markdown(f"• {impact}")


def show_model_performance():
    """Display detailed model performance metrics."""
    st.header("📈 Model Performance")
    
    evaluation, interpretation = load_reports()
    
    if not evaluation:
        st.warning("Evaluation report not found. Please train the model first.")
        return
    
    # Overall metrics
    if "overall_metrics" in evaluation:
        st.subheader("Overall Metrics")
        metrics = evaluation["overall_metrics"]
        
        metrics_df = pd.DataFrame([
            {"Metric": "Mean Absolute Error (MAE)", "Value": f"{metrics.get('test_mae', 0):.2f} minutes"},
            {"Metric": "Root Mean Squared Error (RMSE)", "Value": f"{metrics.get('test_rmse', 0):.2f} minutes"},
            {"Metric": "Mean Absolute Percentage Error (MAPE)", "Value": f"{metrics.get('test_mape', 0):.2f}%"},
            {"Metric": "Interval Coverage", "Value": f"{metrics.get('test_interval_coverage', 0):.2f}%"},
            {"Metric": "Mean Interval Width", "Value": f"{metrics.get('test_mean_interval_width', 0):.2f} minutes"},
        ])
        
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    # Feature importance
    if "feature_importance" in evaluation:
        st.subheader("Top 20 Feature Importance")
        importance_data = evaluation["feature_importance"]
        importance_df = pd.DataFrame(importance_data).head(20)
        
        # Create bar chart
        st.bar_chart(
            importance_df.set_index("feature")["importance"],
            height=400,
        )
        
        # Table view
        with st.expander("View Full Feature Importance Table"):
            st.dataframe(importance_df, use_container_width=True)
    
    # Hourly metrics if available
    if "hourly_metrics" in evaluation:
        st.subheader("Performance by Hour")
        hourly_df = pd.DataFrame(evaluation["hourly_metrics"])
        
        if not hourly_df.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                st.line_chart(hourly_df.set_index("period")["mae"], height=300)
                st.caption("MAE by Hour")
            
            with col2:
                st.line_chart(hourly_df.set_index("period")["rmse"], height=300)
                st.caption("RMSE by Hour")
            
            with st.expander("View Hourly Metrics Table"):
                st.dataframe(hourly_df, use_container_width=True)


def show_feature_insights():
    """Display feature insights and interpretation."""
    st.header("🔍 Feature Insights")
    
    evaluation, interpretation = load_reports()
    
    if not interpretation:
        st.warning("Interpretation report not found. Please train the model first.")
        return
    
    if "feature_insights" not in interpretation:
        st.warning("Feature insights not available.")
        return
    
    insights = interpretation["feature_insights"]
    
    # Summary
    st.subheader("Summary")
    st.info(insights.get("summary", "No summary available"))
    
    # Key insights
    st.subheader("Key Insights")
    for insight in insights.get("insights", []):
        st.markdown(f"• {insight}")
    
    # Top features
    if "top_features" in insights:
        st.subheader("Top Features")
        top_features_df = pd.DataFrame(insights["top_features"])
        st.dataframe(top_features_df, use_container_width=True, hide_index=True)
    
    # Recommendations
    if "recommendations" in interpretation:
        st.subheader("💡 Recommendations")
        for i, rec in enumerate(interpretation["recommendations"], 1):
            st.markdown(f"{i}. {rec}")


def show_prediction_interface():
    """Interactive prediction interface."""
    st.header("🔮 Make Prediction")
    
    if not st.session_state.model_loaded:
        st.error("Model not loaded. Please ensure the model file exists.")
        return
    
    # Input form
    with st.form("prediction_form"):
        st.subheader("Current State")
        
        col1, col2 = st.columns(2)
        
        with col1:
            workload = st.number_input("Workload", min_value=0.0, value=50.0, step=1.0)
            resource_allocation = st.number_input("Resource Allocation", min_value=0.0, value=40.0, step=1.0)
            congestion = st.slider("Congestion Level", 0.0, 1.0, 0.5, 0.01)
        
        with col2:
            planned_schedule = st.number_input("Planned Schedule (minutes)", min_value=0.0, value=30.0, step=1.0)
            current_allocation = st.number_input("Current Allocation", min_value=0.0, value=40.0, step=1.0)
            horizon_hours = st.number_input("Horizon (hours)", min_value=1, max_value=24, value=1)
        
        submitted = st.form_submit_button("Get Prediction", use_container_width=True)
    
    if submitted:
        try:
            # Create feature dict
            feature_dict = {
                "workload": workload,
                "resource_allocation": resource_allocation,
                "congestion": congestion,
                "planned_schedule": planned_schedule,
            }
            
            # Create DataFrame (simplified - in production would use full feature engineering)
            df = pd.DataFrame([feature_dict])
            df["timestamp"] = pd.Timestamp.now()
            
            # Get recommendation
            if st.session_state.recommender:
                # Create state DataFrame with model's expected features
                if st.session_state.model.feature_names:
                    state_df = pd.DataFrame(columns=st.session_state.model.feature_names)
                    for col in st.session_state.model.feature_names:
                        if col in df.columns:
                            state_df[col] = df[col].values
                        else:
                            state_df[col] = 0.0
                    state_df = state_df.fillna(0)
                else:
                    state_df = df
                
                recommendation = st.session_state.recommender.recommend(
                    current_state=state_df,
                    current_allocation=current_allocation,
                    horizon_hours=horizon_hours,
                )
                
                # Display results
                st.success("Prediction Generated!")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Predicted Delay", f"{recommendation['baseline_delay']:.1f} min")
                    st.caption(f"Range: [{recommendation['baseline_delay_lower']:.1f}, {recommendation['baseline_delay_upper']:.1f}]")
                
                with col2:
                    st.metric("Expected Improvement", f"{recommendation['expected_improvement']:.1f} min")
                    st.caption(f"After action: {recommendation['expected_delay_after']:.1f} min")
                
                with col3:
                    st.metric("Risk Level", recommendation['risk_level'])
                    st.caption(f"Uncertainty: {recommendation['confidence_band']['width']:.1f} min")
                
                # Interpretation
                if "interpretation" in recommendation:
                    st.subheader("📝 Interpretation")
                    interp = recommendation["interpretation"]
                    
                    st.info(interp.get("prediction_summary", ""))
                    st.warning(interp.get("risk_assessment", ""))
                    st.success(interp.get("recommended_action", ""))
                
                # Recommendation details
                st.subheader("💡 Recommendation")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**Current Allocation:** {recommendation['current_allocation']:.1f}")
                    st.markdown(f"**Recommended Allocation:** {recommendation['recommended_allocation']:.1f}")
                    st.markdown(f"**Change:** {recommendation['recommended_allocation'] - recommendation['current_allocation']:.1f}")
                
                with col2:
                    st.markdown(f"**Optimization Status:** {recommendation['optimization_status']}")
                    st.markdown(f"**Action Cost:** {recommendation['action_cost']:.2f}")
        
        except Exception as e:
            st.error(f"Error generating prediction: {str(e)}")
            st.exception(e)


def show_about():
    """Display about page."""
    st.header("ℹ️ About Port Decision AI")
    
    st.markdown("""
    ### Overview
    Port Decision AI is a production-quality machine learning system for port operations 
    decision intelligence. It combines time-series forecasting, uncertainty quantification, 
    and constrained optimization to provide actionable recommendations for minimizing 
    operational delays.
    
    ### Features
    - **ML-based Delay Prediction**: LightGBM regression with uncertainty estimation
    - **Constrained Optimization**: OR-Tools for resource allocation decisions
    - **Real-time Decision Support**: REST API and interactive web interface
    - **Comprehensive Interpretation**: Business insights and recommendations
    
    ### Model Architecture
    - **Feature Engineering**: Rolling windows, lag features, trend analysis
    - **Uncertainty Estimation**: Quantile regression and residual-based intervals
    - **Optimization**: Linear programming with resource constraints
    
    ### Usage
    1. Train the model: `python train.py`
    2. Start API server: `python -m src.serving.api`
    3. Use this Streamlit app: `streamlit run streamlit_app.py`
    
    ### API Endpoints
    - `GET /health` - Health check
    - `POST /decision` - Get decision recommendation
    - `POST /predict` - Predict delays
    - `GET /results` - Get evaluation and interpretation reports
    - `GET /evaluation` - Get evaluation metrics
    - `GET /interpretation` - Get interpretation insights
    """)


if __name__ == "__main__":
    main()

