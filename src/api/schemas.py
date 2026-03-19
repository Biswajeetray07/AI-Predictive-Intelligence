import re
from pydantic import BaseModel, Field, validator
from typing import Dict, Any, Optional

class PredictionRequest(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol (e.g., AAPL)")
    date: str = Field(..., description="Target prediction date in YYYY-MM-DD format")

    @validator('ticker')
    def validate_ticker(cls, v):
        if not re.match(r"^[A-Z]{1,5}$", v):
            raise ValueError("Ticker must be 1-5 uppercase letters (e.g., AAPL, TSLA)")
        return v
        
    @validator('date')
    def validate_date(cls, v):
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", v):
            raise ValueError("Date must be in YYYY-MM-DD format (e.g., 2026-03-12)")
        return v

class NLPAnalysis(BaseModel):
    sentiment: Dict[str, float] = Field(..., description="Probabilities for Negative, Neutral, Positive")
    events: Dict[str, float] = Field(..., description="Probabilities for various event types")

class PredictionResponse(BaseModel):
    ticker: str
    date: str
    multi_horizon_predictions: Dict[str, float] = Field(..., description="Predicted percentage change for 1d, 5d, 30d horizons")
    confidence: float = Field(..., description="Model confidence score (0-1)")
    ts_ensemble_breakdown: Dict[str, float] = Field(..., description="Individual TS model predictions")
    ensemble_weights: Dict[str, float] = Field(..., description="Dynamic regime-based ensemble weights applied")
    nlp_analysis: Optional[NLPAnalysis] = Field(None, description="NLP signals if text data was available")
    timestamp: str = Field(..., description="Prediction generation timestamp")
