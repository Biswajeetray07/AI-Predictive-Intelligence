import logging
from datetime import datetime
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import os

from fastapi import FastAPI, HTTPException, Security, status, Depends, Request
from fastapi.security.api_key import APIKeyHeader
from fastapi.responses import JSONResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from src.api.schemas import PredictionRequest, PredictionResponse, NLPAnalysis
from src.monitoring import (
    MetricsCollector,
    setup_logging,
    get_logger,
    registry,
    DriftDetector,
)

API_KEY = os.getenv("API_KEY", "predictive_intel_dev_key_2026")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Initialize logging
setup_logging(log_level=os.getenv("LOG_LEVEL", "INFO"))
logger = get_logger("api")

# Global instances cache to prevent reloading the model on every request
_predictor_instance = None
_metadata_cache = None
_features_cache = None

def get_predictor():
    """Lazy load the predictor to avoid startup delay if API is just starting."""
    global _predictor_instance
    if _predictor_instance is None:
        logger.info("Initializing multi-modal Predictor model...")
        # Local import to prevent circular or path issues before package is fixed
        from src.pipelines.inference_pipeline import Predictor
        try:
            _predictor_instance = Predictor()
        except Exception as e:
            logger.error(f"Failed to initialize Predictor: {e}")
            raise RuntimeError(f"Model initialization failed: {e}")
    return _predictor_instance

def load_data_caches():
    """Load metadata and features purely to map requests to the pre-computed test set for demo purposes."""
    global _metadata_cache, _features_cache
    if _metadata_cache is None or _features_cache is None:
        from src.utils.pipeline_utils import get_project_root
        root = get_project_root()
        metadata_path = os.path.join(root, 'data', 'processed', 'model_inputs', 'metadata_test.csv')
        x_test_path = os.path.join(root, 'data', 'processed', 'model_inputs', 'X_test.npy')
        
        if not os.path.exists(metadata_path) or not os.path.exists(x_test_path):
            raise FileNotFoundError("Test data not found. Run the data pipeline first.")
            
        _metadata_cache = pd.read_csv(metadata_path)
        _features_cache = np.load(x_test_path, mmap_mode='r')
        logger.info(f"Loaded test data cache: {_metadata_cache.shape[0]} rows")

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up API...")
    # Load model and caches in background to be ready for requests
    try:
        get_predictor()
        load_data_caches()
    except Exception as e:
        logger.error(f"Startup warning: {e}")

@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """Basic health check endpoint."""
    model_loaded = _predictor_instance is not None
    data_loaded = _metadata_cache is not None
    return {
        "status": "online",
        "model_loaded": model_loaded,
        "data_loaded": data_loaded,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest, api_key: str = Depends(get_api_key)):
    """
    Generate a market prediction for a given ticker and date.
    Note: Currently maps to the pre-processed test set data.
    """
    try:
        load_data_caches()
        predictor = get_predictor()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server not ready: {str(e)}")
    
    # Ensure caches are loaded
    if _metadata_cache is None or _features_cache is None:
        raise HTTPException(status_code=500, detail="Data caches not loaded")
        
    # Find the specific row index in the test set metadata
    matches = _metadata_cache[
        (_metadata_cache['ticker'] == request.ticker) & 
        (_metadata_cache['date'] == request.date)
    ]
    
    if matches.empty:
        raise HTTPException(
            status_code=404, 
            detail=f"No data available for ticker '{request.ticker}' on date '{request.date}'. "
                   f"Please choose a date from the test set range."
        )
        
    idx = matches.index[0]
    sample_x = _features_cache[idx]
    
    try:
        # Run deep inference
        results = predictor.predict(sample_x)
        
        # Parse NLP signals into the Pydantic schema
        nlp_data = None
        if results.get('nlp_signals'):
            sent_array = results['nlp_signals'].get('sentiment', [[0,0,0]])[0]
            evt_array = results['nlp_signals'].get('events', [[0]*8])[0]
            
            # Map indices to human-readable names based on model design
            sent_names = ["Negative", "Neutral", "Positive"]
            evt_names = ["Policy", "Tech", "Supply", "Crash", "Launch", "Reg", "Economic", "None"]
            
            nlp_data = NLPAnalysis(
                sentiment={name: float(val) for name, val in zip(sent_names, sent_array)},
                events={name: float(val) for name, val in zip(evt_names, evt_array)}
            )
            
        # Construct response
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
        
        # Log prediction for tracking
        try:
            from src.evaluation.prediction_tracker import get_tracker
            tracker = get_tracker()
            tracker.log_prediction(
                ticker=request.ticker,
                predictions=results['multi_horizon_predictions'],
                confidence=float(results['confidence']),
                ensemble_weights=results.get('ensemble_weights', {}),
            )
        except Exception as log_err:
            logger.warning(f"Prediction logging failed: {log_err}")
        
        return response
        
    except Exception as e:
        logger.error(f"Inference error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@app.get("/predictions/report")
async def prediction_report(api_key: str = Depends(get_api_key)):
    """Get accuracy report from logged predictions."""
    try:
        from src.evaluation.prediction_tracker import get_tracker
        tracker = get_tracker()
        return tracker.get_accuracy_report()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")

# To run: uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
