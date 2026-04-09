"""
FastAPI Application for AI Predictive Intelligence
REST API for multi-modal market forecasting with monitoring and drift detection
"""

import logging
from datetime import datetime
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import os

from fastapi import FastAPI, HTTPException, Security, status, Depends, Request
from fastapi.security.api_key import APIKeyHeader
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from src.api.schemas import PredictionRequest, PredictionResponse, NLPAnalysis
from src.monitoring import (
    MetricsCollector,
    setup_logging,
    get_logger,
    registry,
)

# ============================================================================
# Configuration
# ============================================================================

API_KEY = os.getenv("API_KEY", "predictive_intel_dev_key_2026")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Initialize logging
setup_logging(log_level=os.getenv("LOG_LEVEL", "INFO"))
logger = get_logger("api")

# Global instances cache
_predictor_instance = None
_metadata_cache = None
_features_cache = None

# ============================================================================
# FastAPI App Setup
# ============================================================================

app = FastAPI(
    title="AI Predictive Intelligence API",
    description="REST API for multi-modal market forecasting",
    version="0.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Authentication
# ============================================================================

def get_api_key(api_key: str = Security(api_key_header)) -> str:
    """Validate API key from request header"""
    if api_key == API_KEY:
        return api_key
    
    logger.warning(f"Failed authentication attempt")
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing X-API-Key header",
    )

# ============================================================================
# Lazy Loading Functions
# ============================================================================

def get_predictor():
    """Lazy load the predictor to avoid startup delay"""
    global _predictor_instance
    if _predictor_instance is None:
        logger.info("Initializing multi-modal Predictor model...")
        from src.pipelines.inference_pipeline import Predictor
        try:
            _predictor_instance = Predictor()
            logger.info("✅ Predictor initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Predictor: {e}")
            raise RuntimeError(f"Model initialization failed: {e}")
    return _predictor_instance


def load_data_caches():
    """Load metadata and feature caches.
    
    Resolution order:
    1. Local filesystem (fastest — mmap support)
    2. S3 streaming (cloud/container deployment fallback)
    
    Both paths are optional; API endpoints that need live inference
    can use the RealTimeFeatureBuilder (Phase 2) instead.
    """
    global _metadata_cache, _features_cache
    if _metadata_cache is not None and _features_cache is not None:
        return  # Already loaded

    try:
        from src.utils.pipeline_utils import get_project_root
        root = get_project_root()
    except ImportError:
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    metadata_path = os.path.join(root, 'data', 'processed', 'model_inputs', 'metadata_test.csv')
    x_test_path = os.path.join(root, 'data', 'processed', 'model_inputs', 'X_test.npy')

    # Strategy 1: Local filesystem
    if os.path.exists(metadata_path) and os.path.exists(x_test_path):
        _metadata_cache = pd.read_csv(metadata_path)
        _features_cache = np.load(x_test_path, mmap_mode='r')
        logger.info(f"✅ Loaded test data cache (local): {_metadata_cache.shape[0]} sequences")
        return

    # Strategy 2: S3 streaming fallback
    try:
        from src.cloud_storage.aws_storage import SimpleStorageService
        s3_bucket = os.getenv("MODEL_BUCKET_NAME", "my-model-mlopsproj012")
        use_s3 = os.getenv("USE_S3", "False").lower() in ("true", "1", "yes")
        if use_s3:
            s3 = SimpleStorageService()
            _metadata_cache = s3.read_csv('data/processed/model_inputs/metadata_test.csv', s3_bucket)
            _features_cache = s3.read_numpy('data/processed/model_inputs/X_test.npy', s3_bucket)
            if _metadata_cache is not None and _features_cache is not None:
                logger.info(f"✅ Loaded test data cache (S3): {_metadata_cache.shape[0]} sequences")
                return
    except Exception as e:
        logger.warning(f"S3 fallback for test data caches failed: {e}")

    # Neither source available — log warning but don't crash
    # Endpoints can still work via RealTimeFeatureBuilder (Phase 2)
    logger.warning("⚠️  Test data caches not available (local or S3). "
                    "Prediction endpoints will use live feature builder or return 404.")

# ============================================================================
# Startup & Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info("🚀 API starting up...")
    try:
        get_predictor()
        load_data_caches()
        logger.info("✅ All systems initialized")
    except Exception as e:
        logger.error(f"⚠️  Startup warning: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🛑 API shutting down...")

# ============================================================================
# Endpoints: Health & System
# ============================================================================

@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """Health check endpoint"""
    model_loaded = _predictor_instance is not None
    data_loaded = _metadata_cache is not None
    
    return {
        "status": "healthy" if (model_loaded and data_loaded) else "degraded",
        "model_loaded": model_loaded,
        "data_loaded": data_loaded,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(
        content=generate_latest(registry),
        media_type=CONTENT_TYPE_LATEST
    )


@app.get("/version")
async def version():
    """API version endpoint"""
    return {
        "api_version": "0.2.0",
        "python_version": "3.10+",
        "timestamp": datetime.now().isoformat()
    }

# ============================================================================
# Endpoints: Prediction
# ============================================================================

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    api_key: str = Depends(get_api_key)
):
    """
    Generate a market prediction for a given ticker and date.
    
    Args:
        request: PredictionRequest with ticker and date
        api_key: X-API-Key header validation
    
    Returns:
        PredictionResponse with multi-modal predictions
    """
    with MetricsCollector.track_prediction(request.ticker, horizon="multi"):
        try:
            load_data_caches()
            predictor = get_predictor()
        except Exception as e:
            logger.error(f"❌ Server initialization failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Server not ready: {str(e)}")
        
        sample_x = None
        
        # Strategy 1: Test data cache lookup
        if _metadata_cache is not None and _features_cache is not None:
            matches = _metadata_cache[
                (_metadata_cache['ticker'] == request.ticker) & 
                (_metadata_cache['date'] == request.date)
            ]
            if not matches.empty:
                idx = matches.index[0]
                sample_x = _features_cache[idx]

        # Strategy 2: Live feature builder (decoupled from test data)
        if sample_x is None:
            try:
                from src.pipelines.feature_builder import RealTimeFeatureBuilder
                builder = RealTimeFeatureBuilder()
                sample_x = builder.build_sequence(request.ticker, as_of_date=request.date)
            except ImportError:
                logger.debug("RealTimeFeatureBuilder not available")
            except Exception as e:
                logger.warning(f"Live feature builder failed for {request.ticker}: {e}")

        if sample_x is None:
            raise HTTPException(
                status_code=404,
                detail=f"No data for {request.ticker} on {request.date}. "
                       f"Neither test cache nor live features available."
            )
        
        try:
            # Track inference latency
            with MetricsCollector.track_inference("fusion_ensemble"):
                results = predictor.predict(sample_x)
            
            # Parse NLP signals
            nlp_data = None
            if results.get('nlp_signals'):
                sent_array = results['nlp_signals'].get('sentiment', [[0,0,0]])[0]
                evt_array = results['nlp_signals'].get('events', [[0]*8])[0]
                
                sent_names = ["Negative", "Neutral", "Positive"]
                evt_names = ["Policy", "Tech", "Supply", "Crash", "Launch", "Reg", "Economic", "None"]
                
                nlp_data = NLPAnalysis(
                    sentiment={name: float(val) for name, val in zip(sent_names, sent_array)},
                    events={name: float(val) for name, val in zip(evt_names, evt_array)}
                )
            
            # Build response
            response = PredictionResponse(
                ticker=request.ticker,
                date=request.date,
                multi_horizon_predictions=results['multi_horizon_predictions'],
                confidence=float(results['confidence']),
                ts_ensemble_breakdown={k: float(v) for k, v in results['ts_ensemble'].items()},
                ensemble_weights={k: float(v) for k, v in results['ensemble_weights'].items()},
                nlp_analysis=nlp_data,
                timestamp=datetime.now().isoformat()
            )
            
            logger.info(f"✅ Prediction generated for {request.ticker}")
            return response
            
        except Exception as e:
            logger.error(f"❌ Inference error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@app.get("/predictions/batch")
async def batch_predictions(
    tickers: str,
    start_date: str,
    end_date: str,
    api_key: str = Depends(get_api_key)
):
    """
    Batch prediction endpoint
    
    Args:
        tickers: Comma-separated ticker list (e.g., "AAPL,MSFT,GOOGL")
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
    
    Returns:
        Dict with predictions for each ticker-date pair
    """
    try:
        load_data_caches()
        ticker_list = [t.strip().upper() for t in tickers.split(',')]
        
        predictions = {}
        for ticker in ticker_list:
            try:
                # Filter metadata for date range
                mask = (
                    (_metadata_cache['ticker'] == ticker) &
                    (_metadata_cache['date'] >= start_date) &
                    (_metadata_cache['date'] <= end_date)
                )
                matching_rows = _metadata_cache[mask]
                
                if matching_rows.empty:
                    predictions[ticker] = {"status": "no_data"}
                    continue
                
                ticker_preds = []
                for idx in matching_rows.index[:5]:  # Limit to 5 predictions per ticker
                    sample_x = _features_cache[idx]
                    results = get_predictor().predict(sample_x)
                    ticker_preds.append({
                        "date": _metadata_cache.loc[idx, 'date'],
                        "prediction": results['multi_horizon_predictions']['1d'],
                        "confidence": float(results['confidence'])
                    })
                
                predictions[ticker] = {"predictions": ticker_preds}
                
            except Exception as e:
                predictions[ticker] = {"error": str(e)}
        
        return {"batch_predictions": predictions}
        
    except Exception as e:
        logger.error(f"❌ Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/predictions/report")
async def prediction_report(api_key: str = Depends(get_api_key)):
    """Get accuracy report from logged predictions"""
    try:
        from src.evaluation.prediction_tracker import get_tracker
        tracker = get_tracker()
        return tracker.get_accuracy_report()
    except Exception as e:
        logger.error(f"❌ Report generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")

# ============================================================================
# Endpoints: Info
# ============================================================================

@app.get("/info/models")
async def info_models(api_key: str = Depends(get_api_key)):
    """Get information about loaded models"""
    return {
        "models": {
            "timeseries_ensemble": ["lstm", "gru", "transformer", "tft"],
            "nlp_engine": "deberta-v3-base",
            "fusion_model": "multi_horizon_fusion",
            "regime_detector": "hmm_5state"
        },
        "features": 49,
        "sequence_length": 60,
        "horizons": ["1d", "7d", "30d"]
    }


@app.get("/info/data-sources")
async def info_data_sources(api_key: str = Depends(get_api_key)):
    """Get information about data sources"""
    return {
        "sources": 27,
        "domains": {
            "finance": ["Yahoo Finance", "Alpha Vantage", "CoinGecko"],
            "news": ["NewsAPI", "GDELT"],
            "social": ["Reddit", "GitHub", "HackerNews", "Mastodon", "YouTube"],
            "economy": ["FRED"],
            "other": ["OpenSky", "NASA", "Blockchain.com", "WorldBank"]
        },
        "coverage": "503 stocks, 6 years (2020-2026)",
        "update_frequency": "Daily"
    }

# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler"""
    logger.error(f"HTTP Exception: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "timestamp": datetime.now().isoformat()}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "timestamp": datetime.now().isoformat()
        }
    )

# ============================================================================
# Root Endpoint
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint with API documentation"""
    return {
        "service": "AI Predictive Intelligence API",
        "version": "0.2.0",
        "status": "operational",
        "documentation": "/docs",
        "metrics": "/metrics",
        "endpoints": {
            "health": "GET /health",
            "predict": "POST /predict",
            "batch_predict": "GET /predictions/batch",
            "metrics": "GET /metrics",
            "models_info": "GET /info/models",
            "sources_info": "GET /info/data-sources"
        }
    }

# ============================================================================
# Run Instructions
# ============================================================================
"""
Development:
    uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000

Production:
    gunicorn src.api.app:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

Docker:
    docker build -t ai-predictive-intelligence .
    docker run -p 8000:8000 ai-predictive-intelligence

Test:
    curl -H "X-API-Key: predictive_intel_dev_key_2026" http://localhost:8000/health
"""
