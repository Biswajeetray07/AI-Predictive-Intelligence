"""
Prometheus Metrics Module
Exports metrics for monitoring model performance and system health
"""

from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
import time
import logging

logger = logging.getLogger(__name__)

# Create a registry for metrics
registry = CollectorRegistry()

# ============================================================================
# Prediction Metrics
# ============================================================================

predictions_total = Counter(
    'predictions_total',
    'Total number of predictions made',
    ['ticker', 'horizon'],
    registry=registry
)

prediction_latency = Histogram(
    'prediction_latency_seconds',
    'Time taken to generate predictions',
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0),
    registry=registry
)

prediction_errors = Counter(
    'prediction_errors_total',
    'Total number of prediction errors',
    ['error_type'],
    registry=registry
)

# ============================================================================
# Model Performance Metrics
# ============================================================================

model_accuracy = Gauge(
    'model_accuracy',
    'Model accuracy on test set',
    ['model_name'],
    registry=registry
)

model_loss = Gauge(
    'model_loss',
    'Model loss on validation set',
    ['model_name'],
    registry=registry
)

model_inference_latency = Histogram(
    'model_inference_latency_seconds',
    'Time for model inference',
    ['model_name'],
    buckets=(0.01, 0.05, 0.1, 0.5, 1.0),
    registry=registry
)

# ============================================================================
# Data Quality Metrics
# ============================================================================

data_collection_errors = Counter(
    'data_collection_errors_total',
    'Total data collection errors',
    ['source'],
    registry=registry
)

data_staleness_seconds = Gauge(
    'data_staleness_seconds',
    'How stale the data is (seconds since last update)',
    ['source'],
    registry=registry
)

feature_missing_rate = Gauge(
    'feature_missing_rate',
    'Percentage of missing features',
    registry=registry
)

# ============================================================================
# Drift Detection Metrics
# ============================================================================

feature_drift_detected = Counter(
    'feature_drift_detected_total',
    'Number of times feature drift was detected',
    ['feature_name'],
    registry=registry
)

feature_psi = Gauge(
    'feature_psi',
    'Population Stability Index for features',
    ['feature_name'],
    registry=registry
)

prediction_drift_detected = Counter(
    'prediction_drift_detected_total',
    'Number of times prediction drift was detected',
    registry=registry
)

# ============================================================================
# System Health Metrics
# ============================================================================

api_requests_total = Counter(
    'api_requests_total',
    'Total API requests',
    ['endpoint', 'method', 'status_code'],
    registry=registry
)

api_latency = Histogram(
    'api_latency_seconds',
    'API endpoint latency',
    ['endpoint'],
    buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 5.0),
    registry=registry
)

cache_hits = Counter(
    'cache_hits_total',
    'Total cache hits',
    registry=registry
)

cache_misses = Counter(
    'cache_misses_total',
    'Total cache misses',
    registry=registry
)

# ============================================================================
# Context Managers for Metrics Collection
# ============================================================================

class MetricsCollector:
    """Helper class for collecting metrics in with statements"""
    
    @staticmethod
    def track_prediction(ticker: str, horizon: int):
        """Context manager for tracking prediction metrics"""
        class PredictionTracker:
            def __enter__(self):
                self.start_time = time.time()
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                elapsed = time.time() - self.start_time
                
                if exc_type is None:
                    predictions_total.labels(ticker=ticker, horizon=horizon).inc()
                    prediction_latency.observe(elapsed)
                else:
                    prediction_errors.labels(error_type=exc_type.__name__).inc()
                
                logger.debug(f"Prediction for {ticker} completed in {elapsed:.3f}s")
        
        return PredictionTracker()
    
    @staticmethod
    def track_inference(model_name: str):
        """Context manager for tracking model inference time"""
        class InferenceTracker:
            def __enter__(self):
                self.start_time = time.time()
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                elapsed = time.time() - self.start_time
                model_inference_latency.labels(model_name=model_name).observe(elapsed)
        
        return InferenceTracker()
    
    @staticmethod
    def track_api_call(endpoint: str, method: str):
        """Context manager for tracking API calls"""
        class APITracker:
            def __init__(self):
                self.status_code = 500
            
            def __enter__(self):
                self.start_time = time.time()
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                elapsed = time.time() - self.start_time
                api_latency.labels(endpoint=endpoint).observe(elapsed)
                api_requests_total.labels(
                    endpoint=endpoint,
                    method=method,
                    status_code=self.status_code
                ).inc()
        
        return APITracker()


def set_model_metrics(model_name: str, accuracy: float, loss: float):
    """Set model performance metrics"""
    model_accuracy.labels(model_name=model_name).set(accuracy)
    model_loss.labels(model_name=model_name).set(loss)
    logger.info(f"Updated metrics for {model_name}: accuracy={accuracy:.4f}, loss={loss:.4f}")


def record_drift_detection(feature_name: str, psi_value: float, is_drift: bool):
    """Record drift detection results"""
    if is_drift:
        feature_drift_detected.labels(feature_name=feature_name).inc()
    
    feature_psi.labels(feature_name=feature_name).set(psi_value)
    logger.info(f"Drift metric recorded for {feature_name}: PSI={psi_value:.4f}, drift={is_drift}")


def record_data_source_error(source: str):
    """Record data collection error"""
    data_collection_errors.labels(source=source).inc()
    logger.warning(f"Data collection error recorded for {source}")


def set_data_staleness(source: str, staleness_seconds: float):
    """Record how stale data is"""
    data_staleness_seconds.labels(source=source).set(staleness_seconds)


def set_feature_missing_rate(missing_rate: float):
    """Record feature missing rate"""
    feature_missing_rate.set(missing_rate)
