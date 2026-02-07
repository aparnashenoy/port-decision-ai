"""
Decision recommendation module for Port Decision AI.

Combines ML predictions with optimization to generate actionable recommendations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from src.models.train_delay_model import DelayPredictor
from src.models.interpret import interpret_prediction
from src.decision.optimizer import DelayOptimizer
from src.decision.constraints import OptimizationConstraints
from src.config import OptimizationConfig
from src.utils.logging import get_logger

logger = get_logger("recommend")


class DecisionRecommender:
    """
    Recommends actions based on ML predictions and optimization.
    """
    
    def __init__(
        self,
        model: DelayPredictor,
        optimizer: Optional[DelayOptimizer] = None,
        constraints: Optional[OptimizationConstraints] = None,
    ):
        """
        Initialize decision recommender.
        
        Args:
            model: Trained DelayPredictor instance.
            optimizer: DelayOptimizer instance. If None, creates default.
            constraints: OptimizationConstraints instance. If None, uses defaults.
        """
        self.model = model
        self.constraints = constraints or OptimizationConstraints()
        self.optimizer = optimizer or DelayOptimizer(constraints=self.constraints)
    
    def recommend(
        self,
        current_state: pd.DataFrame,
        current_allocation: Optional[float] = None,
        horizon_hours: int = 4,
    ) -> Dict:
        """
        Generate decision recommendation based on current state.
        
        Args:
            current_state: DataFrame with current state features (single row or multiple).
            current_allocation: Current resource allocation.
            horizon_hours: Number of hours to optimize for.
        
        Returns:
            Dictionary with recommendation including:
                - baseline_delay: Predicted delay without action
                - recommended_allocation: Optimal resource allocation
                - expected_improvement: Expected delay reduction
                - risk_level: Risk assessment
                - confidence_band: Uncertainty bounds
        """
        # Get predictions
        predictions = self.model.predict(current_state, return_uncertainty=True)
        baseline_delay = float(predictions["prediction"][0])
        lower_bound = float(predictions.get("lower", [baseline_delay])[0])
        upper_bound = float(predictions.get("upper", [baseline_delay])[0])
        
        # Determine risk level based on uncertainty
        uncertainty_width = upper_bound - lower_bound
        if uncertainty_width < 5:
            risk_level = "Low"
        elif uncertainty_width < 10:
            risk_level = "Medium"
        elif uncertainty_width < 20:
            risk_level = "High"
        else:
            risk_level = "Very High"
        
        # Optimize for single step (can be extended to multi-step)
        opt_result = self.optimizer.optimize_single_step(
            expected_delay=baseline_delay,
            action_cost=0.0,  # Can be parameterized
            current_allocation=current_allocation or self.constraints.current_allocation,
        )
        
        recommended_allocation = opt_result["allocations"][0]
        
        # Calculate expected improvement
        # Simple model: delay reduction = sensitivity * allocation
        delay_sensitivity = 0.1
        expected_improvement = min(
            baseline_delay,
            delay_sensitivity * (recommended_allocation - (current_allocation or 0)),
        )
        
        # Get interpretation
        interpretation = interpret_prediction(
            baseline_delay,
            lower_bound,
            upper_bound,
            risk_level,
        )
        
        # Build recommendation
        recommendation = {
            "baseline_delay": baseline_delay,
            "baseline_delay_lower": lower_bound,
            "baseline_delay_upper": upper_bound,
            "recommended_allocation": recommended_allocation,
            "current_allocation": current_allocation or 0.0,
            "expected_improvement": max(0, expected_improvement),
            "expected_delay_after": max(0, baseline_delay - expected_improvement),
            "risk_level": risk_level,
            "confidence_band": {
                "lower": lower_bound,
                "upper": upper_bound,
                "width": uncertainty_width,
            },
            "optimization_status": opt_result["status"],
            "action_cost": opt_result["total_action_cost"],
            "interpretation": interpretation,
        }
        
        return recommendation
    
    def recommend_multi_horizon(
        self,
        current_state: pd.DataFrame,
        future_states: Optional[pd.DataFrame] = None,
        current_allocation: Optional[float] = None,
        horizon_hours: int = 4,
    ) -> Dict:
        """
        Generate multi-horizon recommendations.
        
        Args:
            current_state: Current state features.
            future_states: Optional future state features (if available).
            current_allocation: Current resource allocation.
            horizon_hours: Optimization horizon.
        
        Returns:
            Dictionary with multi-horizon recommendations.
        """
        # Predict for current and future states
        if future_states is not None and len(future_states) >= horizon_hours:
            states = pd.concat([current_state, future_states.iloc[:horizon_hours]])
        else:
            # Use current state for all horizons (simplified)
            states = pd.concat([current_state] * horizon_hours)
        
        predictions = self.model.predict(states, return_uncertainty=True)
        expected_delays = predictions["prediction"][:horizon_hours]
        
        # Optimize
        opt_result = self.optimizer.optimize(
            expected_delays=expected_delays,
            current_allocation=current_allocation,
        )
        
        # Build recommendation
        recommendation = {
            "horizon_hours": horizon_hours,
            "allocations": opt_result["allocations"],
            "expected_delays": expected_delays.tolist(),
            "total_expected_delay": opt_result["total_expected_delay"],
            "total_action_cost": opt_result["total_action_cost"],
            "optimization_status": opt_result["status"],
            "recommended_first_allocation": opt_result["allocations"][0],
        }
        
        return recommendation

