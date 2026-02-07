"""
Optimization engine for Port Decision AI.

Uses OR-Tools to solve constrained optimization problem for resource allocation
decisions that minimize expected delay while considering action costs.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

from ortools.linear_solver import pywraplp

from src.config import OptimizationConfig
from src.decision.constraints import OptimizationConstraints
from src.utils.logging import get_logger

logger = get_logger("optimizer")


class DelayOptimizer:
    """
    Constrained optimization engine for delay minimization.
    
    Solves:
        minimize: expected_delay + lambda * action_cost
        subject to: resource constraints, allocation bounds, switching penalties
    """
    
    def __init__(
        self,
        config: Optional[OptimizationConfig] = None,
        constraints: Optional[OptimizationConstraints] = None,
    ):
        """
        Initialize optimizer.
        
        Args:
            config: OptimizationConfig instance. If None, uses defaults.
            constraints: OptimizationConstraints instance. If None, uses defaults.
        """
        self.config = config or OptimizationConfig()
        self.constraints = constraints or OptimizationConstraints()
        self.solver = None
    
    def optimize(
        self,
        expected_delays: np.ndarray,
        action_costs: Optional[np.ndarray] = None,
        current_allocation: Optional[float] = None,
    ) -> Dict:
        """
        Solve optimization problem for resource allocation.
        
        Args:
            expected_delays: Expected delay for each hour (shape: [horizon_hours]).
            action_costs: Cost of actions per hour (shape: [horizon_hours]).
                If None, uses zero costs.
            current_allocation: Current resource allocation (for switching penalty).
                If None, uses constraints.current_allocation.
        
        Returns:
            Dictionary with:
                - allocations: Optimal resource allocation per hour
                - total_expected_delay: Total expected delay
                - total_action_cost: Total action cost
                - objective_value: Final objective value
                - status: Solver status
        """
        horizon = len(expected_delays)
        
        if action_costs is None:
            action_costs = np.zeros(horizon)
        
        if current_allocation is None:
            current_allocation = self.constraints.current_allocation or 0.0
        
        # Create solver
        solver = pywraplp.Solver.CreateSolver(self.config.solver)
        if not solver:
            raise ValueError(f"Solver {self.config.solver} not available")
        
        solver.SetTimeLimit(self.config.time_limit_seconds * 1000)  # Convert to milliseconds
        
        # Decision variables: allocation per hour
        allocations = [
            solver.NumVar(
                self.constraints.resource.min_allocation,
                self.constraints.resource.max_allocation,
                f"allocation_{h}",
            )
            for h in range(horizon)
        ]
        
        # Decision variables: switching cost (absolute change)
        switching_vars = [
            solver.NumVar(0, solver.infinity(), f"switching_{h}")
            for h in range(horizon)
        ]
        
        # Constraints: Resource limit per hour
        for h in range(horizon):
            solver.Add(allocations[h] <= self.constraints.resource.resource_limit_per_hour)
        
        # Constraints: Switching penalty (absolute change)
        # First hour: compare with current_allocation
        solver.Add(
            switching_vars[0] >= allocations[0] - current_allocation
        )
        solver.Add(
            switching_vars[0] >= current_allocation - allocations[0]
        )
        
        # Subsequent hours: compare with previous hour
        for h in range(1, horizon):
            solver.Add(switching_vars[h] >= allocations[h] - allocations[h - 1])
            solver.Add(switching_vars[h] >= allocations[h - 1] - allocations[h])
        
        # Objective: Minimize expected delay + action cost + switching penalty
        objective = solver.Objective()
        
        # Delay component (assume linear relationship: delay decreases with allocation)
        # delay_reduction = allocation * delay_sensitivity
        # We'll use a simple model: delay = base_delay - sensitivity * allocation
        delay_sensitivity = 0.1  # Minutes of delay reduction per unit of allocation
        
        for h in range(horizon):
            # Expected delay after allocation
            delay_after = expected_delays[h] - delay_sensitivity * allocations[h]
            objective.SetCoefficient(allocations[h], -delay_sensitivity)
            
            # Action cost
            objective.SetCoefficient(allocations[h], self.config.lambda_cost * action_costs[h])
            
            # Switching penalty
            objective.SetCoefficient(
                switching_vars[h],
                self.config.lambda_cost * self.constraints.resource.switching_penalty,
            )
        
        # Add constant term (expected delays)
        objective.SetOffset(sum(expected_delays))
        
        objective.SetMinimization()
        
        # Solve
        status = solver.Solve()
        
        if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
            optimal_allocations = [allocations[h].solution_value() for h in range(horizon)]
            total_switching = sum(switching_vars[h].solution_value() for h in range(horizon))
            
            # Calculate metrics
            total_expected_delay = sum(
                max(0, expected_delays[h] - delay_sensitivity * optimal_allocations[h])
                for h in range(horizon)
            )
            total_action_cost = sum(
                action_costs[h] * optimal_allocations[h] for h in range(horizon)
            )
            total_switching_cost = (
                self.constraints.resource.switching_penalty * total_switching
            )
            objective_value = (
                total_expected_delay
                + self.config.lambda_cost * total_action_cost
                + self.config.lambda_cost * total_switching_cost
            )
            
            result = {
                "allocations": optimal_allocations,
                "total_expected_delay": total_expected_delay,
                "total_action_cost": total_action_cost,
                "total_switching_cost": total_switching_cost,
                "objective_value": objective_value,
                "status": "optimal" if status == pywraplp.Solver.OPTIMAL else "feasible",
            }
        else:
            logger.warning(f"Optimization failed with status: {status}")
            # Return default allocation
            default_allocation = (
                self.constraints.resource.min_allocation
                + self.constraints.resource.max_allocation
            ) / 2.0
            result = {
                "allocations": [default_allocation] * horizon,
                "total_expected_delay": sum(expected_delays),
                "total_action_cost": 0.0,
                "total_switching_cost": 0.0,
                "objective_value": sum(expected_delays),
                "status": "failed",
            }
        
        return result
    
    def optimize_single_step(
        self,
        expected_delay: float,
        action_cost: float = 0.0,
        current_allocation: Optional[float] = None,
    ) -> Dict:
        """
        Optimize for a single time step.
        
        Args:
            expected_delay: Expected delay for current hour.
            action_cost: Cost of action per unit allocation.
            current_allocation: Current resource allocation.
        
        Returns:
            Dictionary with optimal allocation and metrics.
        """
        return self.optimize(
            expected_delays=np.array([expected_delay]),
            action_costs=np.array([action_cost]),
            current_allocation=current_allocation,
        )

