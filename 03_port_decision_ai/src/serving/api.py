"""
FastAPI serving layer for Port Decision AI.

Exposes REST API endpoints for delay prediction and decision recommendations.
"""

import os
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.config import ProjectConfig, ServingConfig, get_config
from src.decision.constraints import OptimizationConstraints
from src.data.features import engineer_features, get_feature_columns
from src.models.interpret import interpret_prediction
from src.utils.logging import setup_logging, get_logger
from src.serving.schema import (
    DecisionRequest,
    DecisionResponse,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
    ResultsResponse,
    StateFeature,
)

# Lazy import to handle missing dependencies gracefully
try:
    from src.models.train_delay_model import DelayPredictor
    from src.decision.recommend import DecisionRecommender
    LIGHTGBM_AVAILABLE = True
except (ImportError, OSError) as e:
    LIGHTGBM_AVAILABLE = False
    DelayPredictor = None
    DecisionRecommender = None
    import_error = str(e)

# Initialize configuration
config = get_config()
serving_config = config.serving

# Setup logging
logger = setup_logging(config.logs_dir, serving_config.log_level)

# Initialize FastAPI app
app = FastAPI(
    title="Port Decision AI",
    description="Delay Prediction and Optimization Recommendations API",
    version=serving_config.api_version,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model and recommender
model: Optional[DelayPredictor] = None
recommender: Optional[DecisionRecommender] = None


def load_model(model_path: Optional[str] = None) -> DelayPredictor:
    """
    Load trained model from file.
    
    Args:
        model_path: Path to model file. If None, uses config default.
    
    Returns:
        Loaded DelayPredictor instance.
    
    Raises:
        ImportError: If LightGBM dependencies are not available.
        FileNotFoundError: If model file doesn't exist.
    """
    if not LIGHTGBM_AVAILABLE:
        raise ImportError(
            f"LightGBM is not available. Error: {import_error}\n"
            "Please install libomp: brew install libomp"
        )
    
    if model_path is None:
        model_path = serving_config.model_path
    
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    logger.info(f"Loading model from {model_path}")
    return DelayPredictor.load(str(model_path))


def initialize_recommender() -> None:
    """Initialize model and recommender on startup."""
    global model, recommender
    
    try:
        if not LIGHTGBM_AVAILABLE:
            logger.warning(
                f"LightGBM not available: {import_error}\n"
                "API will start but model endpoints will return errors.\n"
                "To fix: brew install libomp"
            )
            model = None
            recommender = None
            return
        
        model = load_model()
        constraints = OptimizationConstraints()
        recommender = DecisionRecommender(model, constraints=constraints)
        logger.info("Model and recommender initialized successfully")
    except FileNotFoundError as e:
        logger.warning(f"Model file not found: {e}. API will start but model endpoints will return errors.")
        model = None
        recommender = None
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        model = None
        recommender = None


@app.on_event("startup")
async def startup_event() -> None:
    """Initialize services on startup."""
    initialize_recommender()


@app.get("/")
async def root() -> dict:
    """
    Root endpoint providing API information.
    
    Returns:
        Dictionary with API information and available endpoints.
    """
    return {
        "name": "Port Decision AI",
        "version": serving_config.api_version,
        "description": "Delay Prediction and Optimization Recommendations API",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "health": "GET /health",
            "decision": "POST /decision",
            "predict": "POST /predict",
            "results": "GET /results",
            "evaluation": "GET /evaluation",
            "interpretation": "GET /interpretation",
        },
        "model_loaded": model is not None and LIGHTGBM_AVAILABLE,
    }


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.
    
    Returns:
        Health status and service information.
    """
    return HealthResponse(
        status="healthy" if (model is not None and LIGHTGBM_AVAILABLE) else "degraded",
        version=serving_config.api_version,
        model_loaded=model is not None and LIGHTGBM_AVAILABLE,
    )


@app.post("/decision", response_model=DecisionResponse)
async def get_decision(request: DecisionRequest) -> DecisionResponse:
    """
    Get decision recommendation based on current state.
    
    This endpoint:
    1. Predicts baseline delay using ML model
    2. Optimizes resource allocation to minimize delay
    3. Returns actionable recommendation with uncertainty estimates
    
    Args:
        request: DecisionRequest with current state features.
    
    Returns:
        DecisionResponse with recommendation and metrics.
    
    Raises:
        HTTPException: If model not loaded or prediction fails.
    """
    if model is None or recommender is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert request to DataFrame
        feature_dict = request.features.dict()
        
        # Create minimal DataFrame with required features
        # In production, you'd engineer full features here
        df = pd.DataFrame([feature_dict])
        
        # Add timestamp if not present (required for feature engineering)
        if "timestamp" not in df.columns:
            df["timestamp"] = pd.Timestamp.now()
        
        # Engineer features (simplified - in production use full pipeline)
        # For now, use basic features directly
        feature_cols = get_feature_columns(df, exclude_cols=["timestamp"])
        
        # If we have the model's expected features, use them
        if model.feature_names:
            # Create DataFrame with model's expected features
            state_df = pd.DataFrame(columns=model.feature_names)
            
            # Map available features
            for col in model.feature_names:
                if col in df.columns:
                    state_df[col] = df[col].values
                else:
                    # Fill missing with 0 (in production, use proper imputation)
                    state_df[col] = 0.0
            
            state_df = state_df.fillna(0)
        else:
            state_df = df[feature_cols] if feature_cols else df
        
        # Get recommendation
        recommendation = recommender.recommend(
            current_state=state_df,
            current_allocation=request.current_allocation,
            horizon_hours=request.horizon_hours or 1,
        )
        
        # Convert to response
        return DecisionResponse(**recommendation)
    
    except Exception as e:
        logger.error(f"Error in decision endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict", response_model=PredictionResponse)
async def predict_delay(request: PredictionRequest) -> PredictionResponse:
    """
    Predict delays for given state features (without optimization).
    
    Args:
        request: PredictionRequest with state features.
    
    Returns:
        PredictionResponse with predictions and uncertainty bounds.
    
    Raises:
        HTTPException: If model not loaded or prediction fails.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert requests to DataFrame
        feature_dicts = [f.dict() for f in request.features]
        df = pd.DataFrame(feature_dicts)
        
        # Add timestamp if not present
        if "timestamp" not in df.columns:
            df["timestamp"] = pd.Timestamp.now()
        
        # Prepare features (simplified)
        if model.feature_names:
            state_df = pd.DataFrame(columns=model.feature_names)
            for col in model.feature_names:
                if col in df.columns:
                    state_df[col] = df[col].values
                else:
                    state_df[col] = 0.0
            state_df = state_df.fillna(0)
        else:
            feature_cols = get_feature_columns(df, exclude_cols=["timestamp"])
            state_df = df[feature_cols] if feature_cols else df
        
        # Predict
        predictions = model.predict(state_df, return_uncertainty=True)
        
        return PredictionResponse(
            predictions=predictions["prediction"].tolist(),
            lower_bounds=predictions.get("lower", predictions["prediction"]).tolist(),
            upper_bounds=predictions.get("upper", predictions["prediction"]).tolist(),
        )
    
    except Exception as e:
        logger.error(f"Error in predict endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/evaluation", response_model=Dict)
async def get_evaluation_report() -> Dict:
    """
    Get model evaluation report.
    
    Returns:
        Dictionary with evaluation metrics, feature importance, and time-based analysis.
    
    Raises:
        HTTPException: If report file not found.
    """
    report_path = config.models_dir / "evaluation_report.json"
    
    if not report_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Evaluation report not found at {report_path}. Please train the model first."
        )
    
    try:
        import json
        with open(report_path, "r") as f:
            report = json.load(f)
        return report
    except Exception as e:
        logger.error(f"Error reading evaluation report: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error loading report: {str(e)}")


@app.get("/interpretation", response_model=Dict)
async def get_interpretation_report() -> Dict:
    """
    Get model interpretation report.
    
    Returns:
        Dictionary with model interpretation, insights, and recommendations.
    
    Raises:
        HTTPException: If report file not found.
    """
    report_path = config.models_dir / "interpretation_report.json"
    
    if not report_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Interpretation report not found at {report_path}. Please train the model first."
        )
    
    try:
        import json
        with open(report_path, "r") as f:
            report = json.load(f)
        return report
    except Exception as e:
        logger.error(f"Error reading interpretation report: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error loading report: {str(e)}")


@app.get("/results", response_model=ResultsResponse)
async def get_results() -> ResultsResponse:
    """
    Get comprehensive results including evaluation and interpretation.
    
    Returns:
        ResultsResponse with both evaluation and interpretation reports.
    """
    evaluation_path = config.models_dir / "evaluation_report.json"
    interpretation_path = config.models_dir / "interpretation_report.json"
    
    evaluation_report = None
    interpretation_report = None
    
    if evaluation_path.exists():
        try:
            import json
            with open(evaluation_path, "r") as f:
                evaluation_report = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load evaluation report: {e}")
    
    if interpretation_path.exists():
        try:
            import json
            with open(interpretation_path, "r") as f:
                interpretation_report = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load interpretation report: {e}")
    
    model_info = {
        "model_loaded": model is not None and LIGHTGBM_AVAILABLE,
        "model_path": str(config.models_dir / "delay_predictor.pkl"),
        "model_exists": (config.models_dir / "delay_predictor.pkl").exists(),
        "evaluation_available": evaluation_report is not None,
        "interpretation_available": interpretation_report is not None,
    }
    
    return ResultsResponse(
        evaluation_report=evaluation_report,
        interpretation_report=interpretation_report,
        model_info=model_info,
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.serving.api:app",
        host=serving_config.host,
        port=serving_config.port,
        reload=serving_config.reload,
        log_level=serving_config.log_level.lower(),
    )

