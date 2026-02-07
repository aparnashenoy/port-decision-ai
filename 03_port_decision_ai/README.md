# Port Decision AI — Delay Prediction and Optimization Recommendations

A production-quality machine learning system for port operations decision intelligence. This system combines time-series forecasting, uncertainty quantification, and constrained optimization to provide actionable recommendations for minimizing operational delays.

## Business Problem

Port operations face constant challenges in resource allocation and schedule management. Delays can cascade through the supply chain, causing significant economic impact. This system addresses the core question:

> **Given current operational state, what resource allocation decisions will minimize expected delays while respecting operational constraints?**

The solution integrates:
- **ML-based delay prediction** with uncertainty estimates
- **Constrained optimization** for resource allocation
- **Real-time decision support** via REST API

## Architecture

### System Design

```
┌─────────────────┐
│  Synthetic Data │
│   Generation    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Feature       │
│  Engineering    │
│  (Rolling Win)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌─────────────────┐
│  LightGBM       │─────▶│  Uncertainty    │
│  Regression     │      │  Quantification │
└────────┬────────┘      └─────────────────┘
         │
         ▼
┌─────────────────┐
│  OR-Tools       │
│  Optimization   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  FastAPI        │
│  Decision Pack  │
└─────────────────┘
```

### Key Components

#### 1. Data Layer (`src/data/`)
- **`make_synthetic.py`**: Generates realistic time-series operational data with noise, trends, seasonality, and spikes
- **`preprocess.py`**: Data cleaning, validation, and temporal splitting
- **`features.py`**: Rolling window features, lag features, trend analysis, utilization metrics

#### 2. Model Layer (`src/models/`)
- **`train_delay_model.py`**: LightGBM regression with quantile-based uncertainty
- **`predict.py`**: Batch and single prediction interfaces
- **`evaluate.py`**: Comprehensive evaluation metrics and analysis

#### 3. Decision Layer (`src/decision/`)
- **`optimizer.py`**: OR-Tools linear optimization for resource allocation
- **`recommend.py`**: Combines ML predictions with optimization
- **`constraints.py`**: Resource limits, allocation bounds, switching penalties

#### 4. Serving Layer (`src/serving/`)
- **`api.py`**: FastAPI REST endpoints
- **`schema.py`**: Pydantic models for request/response validation

#### 5. Utilities (`src/utils/`)
- **`logging.py`**: Structured logging configuration
- **`metrics.py`**: Regression metrics and uncertainty measures

## ML Design

### Feature Engineering

**Rolling Window Features:**
- Mean, sum, and standard deviation over 1h, 3h, 6h windows
- Captures short-term trends and volatility

**Lag Features:**
- 1, 2, 4, 8 period lags for key operational variables
- Captures temporal dependencies

**Derived Features:**
- Utilization ratio: `workload / resource_allocation`
- Resource gap: `workload - resource_allocation`
- Trend slope: Linear regression over rolling window
- Time-based: Hour, day-of-week, cyclical encodings

### Model Architecture

**LightGBM Regression:**
- Gradient boosting with tree-based learners
- Handles non-linear relationships and feature interactions
- Fast training and inference

**Uncertainty Estimation:**
- **Quantile Regression**: Separate models for 10th, 50th, 90th percentiles
- **Residual-based Intervals**: 95% confidence intervals from training residuals
- Provides both point predictions and uncertainty bounds

### Optimization Design

**Objective Function:**
```
minimize: expected_delay + λ * (action_cost + switching_penalty)
```

**Constraints:**
- Resource allocation bounds: `min_allocation ≤ allocation ≤ max_allocation`
- Resource limits: `allocation ≤ resource_limit_per_hour`
- Switching penalty: Penalizes changes from current allocation

**Solver:**
- OR-Tools linear programming
- Handles constraints efficiently
- Configurable time limits for real-time decisions

## Installation

### Prerequisites

- Python 3.9+
- pip or conda
- **macOS users**: Homebrew (for installing libomp)

### Setup

```bash
# Clone or navigate to project directory
cd 03_port_decision_ai

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# macOS: Install OpenMP library (required for LightGBM)
# If you get "Library not loaded: @rpath/libomp.dylib" error, run:
brew install libomp

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Generate Data and Train Model

```bash
# Run training pipeline
python train.py
```

This will:
- Generate 10,000 synthetic samples
- Engineer features
- Train LightGBM model
- Evaluate on test set
- Save model to `models/delay_predictor.pkl`

### 2. Start API Server

```bash
# Start FastAPI server
python -m src.serving.api

# Or using uvicorn directly
uvicorn src.serving.api:app --host 0.0.0.0 --port 8050
```

The API will be available at `http://localhost:8050`

### 3. Start Streamlit Web App (Alternative UI)

```bash
# Start Streamlit application
streamlit run streamlit_app.py
```

The web app will be available at `http://localhost:8501`

The Streamlit app provides:
- Interactive dashboard with key metrics
- Model performance visualization
- Feature insights and interpretation
- Interactive prediction interface
- Real-time decision recommendations

### 3. API Endpoints

#### Health Check
```bash
curl http://localhost:8050/health
```

#### Get Decision Recommendation
```bash
curl -X POST http://localhost:8050/decision \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "workload": 65.0,
      "resource_allocation": 45.0,
      "congestion": 0.7,
      "planned_schedule": 30.0
    },
    "current_allocation": 40.0,
    "horizon_hours": 1
  }'
```

#### Predict Delay Only
```bash
curl -X POST http://localhost:8050/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [{
      "workload": 65.0,
      "resource_allocation": 45.0,
      "congestion": 0.7,
      "planned_schedule": 30.0
    }]
  }'
```

### Example API Response

```json
{
  "baseline_delay": 12.5,
  "baseline_delay_lower": 8.2,
  "baseline_delay_upper": 16.8,
  "recommended_allocation": 48.3,
  "current_allocation": 40.0,
  "expected_improvement": 2.1,
  "expected_delay_after": 10.4,
  "risk_level": "Medium",
  "confidence_band": {
    "lower": 8.2,
    "upper": 16.8,
    "width": 8.6
  },
  "optimization_status": "optimal",
  "action_cost": 0.0
}
```

## Project Structure

```
03_port_decision_ai/
├── src/
│   ├── config.py                 # Configuration management
│   ├── data/
│   │   ├── make_synthetic.py     # Synthetic data generation
│   │   ├── preprocess.py          # Data cleaning
│   │   └── features.py            # Feature engineering
│   ├── models/
│   │   ├── train_delay_model.py  # Model training
│   │   ├── predict.py             # Prediction interface
│   │   └── evaluate.py            # Model evaluation
│   ├── decision/
│   │   ├── optimizer.py           # OR-Tools optimization
│   │   ├── recommend.py           # Decision recommendation
│   │   └── constraints.py         # Constraint definitions
│   ├── serving/
│   │   ├── api.py                 # FastAPI endpoints
│   │   └── schema.py              # Request/response schemas
│   └── utils/
│       ├── logging.py              # Logging utilities
│       └── metrics.py              # Evaluation metrics
├── tests/
│   ├── test_features.py           # Feature engineering tests
│   └── test_metrics.py             # Metrics tests
├── train.py                       # Training pipeline
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Testing

```bash
# Run tests
pytest tests/

# With coverage
pytest tests/ --cov=src --cov-report=html
```

## Configuration

Configuration is centralized in `src/config.py`. Key parameters:

- **Data**: Sample size, noise level, seasonality
- **Features**: Rolling windows, lag periods
- **Model**: LightGBM hyperparameters, uncertainty method
- **Optimization**: Resource limits, cost trade-offs
- **Serving**: API host, port, model path

## Future Production Improvements

### MLOps Enhancements
- **Model Versioning**: MLflow or DVC for experiment tracking
- **Continuous Training**: Automated retraining pipelines
- **A/B Testing**: Framework for model comparison
- **Monitoring**: Drift detection, prediction monitoring

### Model Improvements
- **Deep Learning**: LSTM/Transformer for complex temporal patterns
- **Ensemble Methods**: Combine multiple models for robustness
- **Online Learning**: Incremental updates from new data
- **Causal Inference**: Understand intervention effects

### System Scalability
- **Distributed Training**: Scale to large datasets
- **Model Serving**: TensorFlow Serving or TorchServe
- **Caching**: Redis for frequent predictions
- **Async Processing**: Celery for batch optimization

### Data Quality
- **Data Validation**: Great Expectations or Pydantic validators
- **Feature Store**: Feast or Tecton for feature management
- **Data Lineage**: Track data transformations
- **Real-time Features**: Streaming feature computation

### Operational Excellence
- **Containerization**: Docker for deployment
- **Orchestration**: Kubernetes for scaling
- **CI/CD**: Automated testing and deployment
- **Documentation**: API docs with Swagger/OpenAPI

## Technical Notes

### Deterministic Training
- Random seeds configured in `ModelConfig`
- Temporal splitting ensures no data leakage
- Reproducible feature engineering

### Type Safety
- Type hints throughout codebase
- Pydantic validation for API inputs
- Mypy-compatible code structure

### Logging
- Structured logging with file and console handlers
- Log levels configurable per module
- Production-ready error handling

## License

This project is provided as a demonstration of production ML architecture. Adapt as needed for your use case.

## Contact

For questions or contributions, please refer to the project repository.

---

**Built with:** Python, LightGBM, OR-Tools, FastAPI  
**Architecture:** Modular, type-safe, production-ready  
**Quality:** Senior Data Scientist / MLOps standards

