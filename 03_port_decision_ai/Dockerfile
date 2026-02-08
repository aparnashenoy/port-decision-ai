# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
# libgomp1 is required for LightGBM on Linux
# gcc and g++ are needed for some Python packages that compile C extensions
RUN apt-get update && apt-get install -y \
    libgomp1 \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Create necessary directories if they don't exist
RUN mkdir -p models data logs

# Expose ports
# 8050 for FastAPI
# 8501 for Streamlit
EXPOSE 8050 8501

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Default command: run FastAPI server
# To run Streamlit instead, override with:
# docker run ... streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0
CMD ["python", "-m", "src.serving.api"]
