"""
Evaluation Module — Public API
===============================
Provides all evaluation, analysis, and monitoring tools.
"""

from src.evaluation.metrics import (
    calculate_regression_metrics,
    calculate_financial_metrics,
    calculate_statistical_significance,
    calculate_per_ticker_metrics,
)

from src.evaluation.cross_validation import (
    TimeSeriesCrossValidator,
    run_ts_cross_validation,
)

from src.evaluation.ensemble_ablation import (
    EnsembleAblation,
    run_ensemble_ablation,
)

from src.evaluation.feature_analysis import (
    FeatureAnalyzer,
    run_feature_analysis,
)

from src.evaluation.prediction_tracker import (
    PredictionTracker,
    get_tracker,
)

__all__ = [
    # Metrics
    'calculate_regression_metrics',
    'calculate_financial_metrics',
    'calculate_statistical_significance',
    'calculate_per_ticker_metrics',
    # Cross-validation
    'TimeSeriesCrossValidator',
    'run_ts_cross_validation',
    # Ablation
    'EnsembleAblation',
    'run_ensemble_ablation',
    # Feature analysis
    'FeatureAnalyzer',
    'run_feature_analysis',
    # Prediction tracking
    'PredictionTracker',
    'get_tracker',
]
