"""
Constraint definitions for optimization engine.

Defines resource limits, allocation bounds, and operational constraints.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ResourceConstraints:
    """
    Resource allocation constraints for optimization.
    
    Attributes:
        min_allocation: Minimum resource allocation per hour.
        max_allocation: Maximum resource allocation per hour.
        resource_limit_per_hour: Total resource limit per hour.
        switching_penalty: Penalty for changing allocation between periods.
    """
    
    min_allocation: float = 10.0
    max_allocation: float = 50.0
    resource_limit_per_hour: float = 100.0
    switching_penalty: float = 5.0
    
    def validate(self) -> bool:
        """
        Validate constraint parameters.
        
        Returns:
            True if valid, raises ValueError otherwise.
        
        Raises:
            ValueError: If constraints are invalid.
        """
        if self.min_allocation < 0:
            raise ValueError("min_allocation must be non-negative")
        if self.max_allocation < self.min_allocation:
            raise ValueError("max_allocation must be >= min_allocation")
        if self.resource_limit_per_hour < self.max_allocation:
            raise ValueError("resource_limit_per_hour must be >= max_allocation")
        if self.switching_penalty < 0:
            raise ValueError("switching_penalty must be non-negative")
        return True


@dataclass
class OptimizationConstraints:
    """
    Complete optimization constraints including resource and operational limits.
    
    Attributes:
        resource: Resource allocation constraints.
        horizon_hours: Optimization horizon in hours.
        current_allocation: Current resource allocation (for switching penalty).
    """
    
    resource: ResourceConstraints = None
    horizon_hours: int = 4
    current_allocation: Optional[float] = None
    
    def __post_init__(self) -> None:
        """Initialize default resource constraints if not provided."""
        if self.resource is None:
            self.resource = ResourceConstraints()
        self.resource.validate()

