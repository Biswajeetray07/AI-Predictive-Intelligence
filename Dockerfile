FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    duckdb \
    && rm -rf /var/lib/apt/lists/*

# Add non-root user for security
RUN useradd -m -s /bin/bash modeluser

# Copy requirements in a separate layer to cache dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir fastapi uvicorn pydantic pydantic-settings streamlit passlib

# Copy remaining project files
COPY . .

# Install project in editable mode
RUN pip install -e .

# Create necessary directories for persistence and set permissions
RUN mkdir -p data/raw data/processed data/features data/processed/model_inputs saved_models \
    && chown -R modeluser:modeluser /app

# Switch to non-root user
USER modeluser

# Expose API and Dashboard ports
EXPOSE 8000 8501

# Entrypoint script (overridden in docker-compose)
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
