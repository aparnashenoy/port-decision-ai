"""
API schema definitions for Port Decision AI.

Defines Pydantic models for request/response validation.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class StateFeature(BaseModel):
    """Single state feature for prediction."""
    
    workload: float = Field(..., description="Current workload")
    resource_allocation: float = Field(..., description="Current resource allocation")
    congestion: float = Field(..., ge=0, le=1, description="Congestion level (0-1)")
    planned_schedule: float = Field(..., description="Planned schedule time")
    utilization_ratio: Optional[float] = Field(None, description="Utilization ratio")
    resource_gap: Optional[float] = Field(None, description="Resource gap")


class DecisionRequest(BaseModel):
    """Request schema for decision endpoint."""
    
    features: StateFeature = Field(..., description="Current state features")
    current_allocation: Optional[float] = Field(
        None, description="Current resource allocation"
    )
    horizon_hours: Optional[int] = Field(
        1, ge=1, le=24, description="Optimization horizon in hours"
    )


class ConfidenceBand(BaseModel):
    """Confidence interval for predictions."""
    
    lower: float = Field(..., description="Lower bound")
    upper: float = Field(..., description="Upper bound")
    width: float = Field(..., description="Interval width")


class DecisionResponse(BaseModel):
    """Response schema for decision endpoint."""
    
    baseline_delay: float = Field(..., description="Predicted delay without action")
    baseline_delay_lower: float = Field(..., description="Lower bound of baseline delay")
    baseline_delay_upper: float = Field(..., description="Upper bound of baseline delay")
    recommended_allocation: float = Field(..., description="Recommended resource allocation")
    current_allocation: float = Field(..., description="Current resource allocation")
    expected_improvement: float = Field(..., description="Expected delay reduction (minutes)")
    expected_delay_after: float = Field(..., description="Expected delay after action")
    risk_level: str = Field(..., description="Risk level (Low/Medium/High/Very High)")
    confidence_band: ConfidenceBand = Field(..., description="Confidence interval")
    optimization_status: str = Field(..., description="Optimization solver status")
    action_cost: float = Field(..., description="Cost of recommended action")
    interpretation: Optional[Dict[str, str]] = Field(
        None,
        description="Human-readable interpretation of the prediction and recommendation",
    )


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    model_loaded: bool = Field(..., description="Whether model is loaded")


class PredictionRequest(BaseModel):
    """Request schema for prediction-only endpoint."""
    
    features: List[StateFeature] = Field(..., description="State features for prediction")


class PredictionResponse(BaseModel):
    """Response schema for prediction endpoint."""
    
    predictions: List[float] = Field(..., description="Predicted delays")
    lower_bounds: List[float] = Field(..., description="Lower confidence bounds")
    upper_bounds: List[float] = Field(..., description="Upper confidence bounds")


class ResultsResponse(BaseModel):
    """Response schema for results endpoint."""
    
    evaluation_report: Optional[Dict] = Field(None, description="Model evaluation metrics and analysis")
    interpretation_report: Optional[Dict] = Field(None, description="Model interpretation and insights")
    model_info: Dict = Field(..., description="Model metadata and status")

