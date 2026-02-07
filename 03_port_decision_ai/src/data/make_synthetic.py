"""
Synthetic data generation for Port Decision AI.

Generates realistic time-series operational data with noise, trends,
seasonality, and occasional spikes to simulate port operations.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

from src.config import DataConfig


def generate_synthetic_data(config: Optional[DataConfig] = None) -> pd.DataFrame:
    """
    Generate synthetic operational time-series data.
    
    Creates realistic port operation data with:
    - Timestamps
    - Workload (number of vessels/containers)
    - Resource allocation (cranes, workers, etc.)
    - Congestion signal (0-1 scale)
    - Planned schedule adherence
    - True delay minutes (target variable)
    
    Args:
        config: DataConfig instance. If None, uses defaults.
    
    Returns:
        DataFrame with synthetic operational data.
    """
    if config is None:
        config = DataConfig()
    
    # Generate timestamps
    start = datetime.strptime(config.start_date, "%Y-%m-%d")
    timestamps = pd.date_range(
        start=start,
        periods=config.n_samples,
        freq=f"{config.freq_minutes}min",
    )
    
    # Base workload with trend and seasonality
    t = np.arange(config.n_samples)
    trend = config.trend_strength * t
    seasonality = np.zeros(config.n_samples)
    
    for period in config.seasonality_periods:
        period_samples = period * (60 // config.freq_minutes)
        seasonality += 0.3 * np.sin(2 * np.pi * t / period_samples)
        seasonality += 0.2 * np.cos(4 * np.pi * t / period_samples)
    
    base_workload = 50 + trend + seasonality
    
    # Add noise
    noise = np.random.normal(0, config.noise_level * 10, config.n_samples)
    workload = np.maximum(10, base_workload + noise)
    
    # Resource allocation (correlated with workload but with lag)
    resource_base = 0.6 * workload + 0.4 * np.roll(workload, -2)
    resource_noise = np.random.normal(0, 5, config.n_samples)
    resource_allocation = np.maximum(10, resource_base + resource_noise)
    
    # Congestion signal (inverse relationship with resource/workload ratio)
    utilization = workload / (resource_allocation + 1e-6)
    congestion = np.clip(utilization / 2.0, 0, 1)
    congestion += np.random.normal(0, 0.05, config.n_samples)
    congestion = np.clip(congestion, 0, 1)
    
    # Planned schedule (baseline with some variance)
    planned_schedule = 30 + 0.1 * workload + np.random.normal(0, 5, config.n_samples)
    planned_schedule = np.maximum(10, planned_schedule)
    
    # Generate spikes (occasional disruptions)
    spike_mask = np.random.random(config.n_samples) < config.spike_probability
    spike_effect = np.where(
        spike_mask,
        np.random.exponential(config.spike_magnitude, config.n_samples),
        0,
    )
    
    # True delay calculation
    # Delay increases with congestion, workload, and spikes
    delay_base = (
        5 * congestion
        + 0.1 * (workload - resource_allocation)
        + 0.05 * np.maximum(0, workload - 60)
    )
    delay_spikes = spike_effect * 10
    delay_noise = np.random.exponential(2, config.n_samples)
    
    delay_minutes = np.maximum(0, delay_base + delay_spikes + delay_noise)
    
    # Create DataFrame
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "workload": workload,
            "resource_allocation": resource_allocation,
            "congestion": congestion,
            "planned_schedule": planned_schedule,
            "delay_minutes": delay_minutes,
        }
    )
    
    # Add derived features
    df["utilization_ratio"] = df["workload"] / (df["resource_allocation"] + 1e-6)
    df["resource_gap"] = df["workload"] - df["resource_allocation"]
    
    return df


def save_synthetic_data(
    df: pd.DataFrame,
    filepath: str,
    format: str = "parquet",
) -> None:
    """
    Save synthetic data to file.
    
    Args:
        df: DataFrame to save.
        filepath: Path to save file.
        format: File format ('parquet', 'csv', or 'json').
    """
    if format == "parquet":
        df.to_parquet(filepath, index=False)
    elif format == "csv":
        df.to_csv(filepath, index=False)
    elif format == "json":
        df.to_json(filepath, orient="records", date_format="iso")
    else:
        raise ValueError(f"Unsupported format: {format}")


if __name__ == "__main__":
    # Generate and save example data
    config = DataConfig(n_samples=10000)
    df = generate_synthetic_data(config)
    print(f"Generated {len(df)} samples")
    print(df.head())
    print(f"\nDelay statistics:")
    print(df["delay_minutes"].describe())

