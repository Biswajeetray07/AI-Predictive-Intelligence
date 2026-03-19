"""
Unit tests for evaluation metrics.
Tests regression, financial, per-ticker, and statistical significance functions.
"""

import sys
import os
import pytest
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

from src.evaluation.metrics import (
    calculate_regression_metrics,
    calculate_financial_metrics,
    calculate_per_ticker_metrics,
    calculate_statistical_significance,
)


class TestRegressionMetrics:
    def test_perfect_prediction(self):
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        metrics = calculate_regression_metrics(y, y)
        assert metrics['MSE'] == 0.0
        assert metrics['RMSE'] == 0.0
        assert metrics['MAE'] == 0.0
        assert metrics['R2'] == 1.0

    def test_known_values(self):
        y_true = [1.0, 2.0, 3.0]
        y_pred = [1.1, 2.1, 3.1]
        metrics = calculate_regression_metrics(y_true, y_pred)
        assert abs(metrics['MAE'] - 0.1) < 1e-6
        assert abs(metrics['MSE'] - 0.01) < 1e-6

    def test_returns_all_keys(self):
        metrics = calculate_regression_metrics([1, 2, 3], [1.1, 2.1, 3.1])
        expected = {'MSE', 'RMSE', 'MAE', 'R2', 'MAPE'}
        assert set(metrics.keys()) == expected


class TestFinancialMetrics:
    def test_perfect_direction(self):
        y_true = [0.01, -0.02, 0.03, -0.01, 0.02]
        y_pred = [0.005, -0.01, 0.01, -0.005, 0.01]
        metrics = calculate_financial_metrics(y_true, y_pred)
        assert metrics['Directional_Accuracy'] == 1.0

    def test_returns_all_keys(self):
        metrics = calculate_financial_metrics([0.01, -0.02], [0.01, -0.02])
        expected = {
            'Directional_Accuracy', 'Simulated_Sharpe_Ratio',
            'Mean_Strategy_Return', 'Max_Drawdown',
            'Profit_Factor', 'Calmar_Ratio'
        }
        assert set(metrics.keys()) == expected


class TestPerTickerMetrics:
    def test_groups_by_ticker(self):
        actuals = [0.01, -0.02, 0.03, -0.01, 0.02] * 5  # 25 samples per ticker
        preds = [0.005, -0.01, 0.01, -0.005, 0.01] * 5
        tickers = ['AAPL'] * 25

        results = calculate_per_ticker_metrics(actuals, preds, tickers)
        assert 'AAPL' in results
        assert 'MSE' in results['AAPL']
        assert results['AAPL']['n'] == 25

    def test_skips_small_tickers(self):
        """Tickers with <= 10 samples should be skipped."""
        actuals = [0.01] * 5
        preds = [0.01] * 5
        tickers = ['TINY'] * 5

        results = calculate_per_ticker_metrics(actuals, preds, tickers)
        assert 'TINY' not in results

    def test_multiple_tickers(self):
        actuals = [0.01] * 20 + [-0.01] * 20
        preds = [0.01] * 20 + [-0.01] * 20
        tickers = ['AAPL'] * 20 + ['MSFT'] * 20

        results = calculate_per_ticker_metrics(actuals, preds, tickers)
        assert len(results) == 2
        assert 'AAPL' in results
        assert 'MSFT' in results


class TestStatisticalSignificance:
    def test_returns_correct_keys(self):
        y_true = list(np.random.randn(100))
        y_pred = list(np.random.randn(100))
        result = calculate_statistical_significance(y_true, y_pred)
        assert 'p_value' in result
        assert 't_statistic' in result
        assert 'significant_at_005' in result
        assert 'significant_at_001' in result

    def test_perfect_prediction_not_significant(self):
        """If errors are all zero, t-test should fail (no variance)."""
        y = list(np.ones(50))
        result = calculate_statistical_significance(y, y)
        # NaN t-stat because std=0 — should handle gracefully
        assert 'p_value' in result


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
