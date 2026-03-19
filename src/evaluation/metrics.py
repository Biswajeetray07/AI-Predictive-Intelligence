import numpy as np
from scipy import stats
from typing import cast, Any

def calculate_regression_metrics(y_true, y_pred):
    """
    Calculate standard regression error metrics.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    
    # R-squared
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    
    # MAPE (Mean Absolute Percentage Error) - add small epsilon to avoid div by zero
    epsilon = 1e-8
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100
    
    return {
        'MSE': float(mse),
        'RMSE': float(rmse),
        'MAE': float(mae),
        'R2': float(r2),
        'MAPE': float(mape)
    }

def calculate_financial_metrics(y_true, y_pred, previous_prices=None):
    """
    Calculate metrics relevant for trading/forecasting.
    y_true: actual price change (or target value)
    y_pred: predicted price change
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Directional Accuracy
    correct_direction = np.sum(np.sign(y_true) == np.sign(y_pred))
    directional_accuracy = correct_direction / len(y_true)
    
    # Strategy returns: go long if pred > 0, short if pred < 0
    strategy_returns = np.sign(y_pred) * y_true
    mean_return = np.mean(strategy_returns)
    std_return = np.std(strategy_returns)
    
    # Annualized Sharpe Ratio (252 trading days)
    sharpe_ratio = 0.0
    if std_return > 0:
        sharpe_ratio = (mean_return / std_return) * np.sqrt(252)

    # Maximum Drawdown
    cumulative_returns = np.cumprod(1 + strategy_returns)
    rolling_max = np.maximum.accumulate(cumulative_returns)
    drawdowns = (cumulative_returns - rolling_max) / rolling_max
    max_drawdown = np.min(drawdowns)

    # Profit Factor (Phase 9 — new)
    gross_profit = np.sum(strategy_returns[strategy_returns > 0])
    gross_loss = np.abs(np.sum(strategy_returns[strategy_returns < 0]))
    profit_factor = gross_profit / (gross_loss + 1e-10)
    
    # Calmar Ratio (Phase 9 — new): annualized return / |max drawdown|
    annualized_return = mean_return * 252
    calmar_ratio = annualized_return / (abs(max_drawdown) + 1e-10)

    return {
        'Directional_Accuracy': float(directional_accuracy),
        'Simulated_Sharpe_Ratio': float(sharpe_ratio),
        'Mean_Strategy_Return': float(mean_return),
        'Max_Drawdown': float(max_drawdown),
        'Profit_Factor': float(profit_factor),
        'Calmar_Ratio': float(calmar_ratio)
    }


def calculate_statistical_significance(y_true, y_pred, baseline_pred=None) -> dict:
    """
    Statistical significance testing on prediction errors.
    
    If baseline_pred is provided, performs paired t-test comparing model errors
    to baseline errors. Otherwise, tests if prediction errors are significantly
    different from zero (model is better than predicting zero).
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    errors = y_true - y_pred
    
    if baseline_pred is not None:
        baseline_errors = y_true - np.array(baseline_pred)
        # Diebold-Mariano style: compare squared errors
        d = baseline_errors**2 - errors**2
        result = cast(Any, stats.ttest_1samp(d, 0.0))
        t_stat_val = float(result[0])
        p_value_val = float(result[1])
    else:
        # Test if errors are significantly different from a naive zero-prediction
        result = cast(Any, stats.ttest_1samp(errors, 0.0))
        t_stat_val = float(result[0])
        p_value_val = float(result[1])
    
    return {
        't_statistic': t_stat_val,
        'p_value': p_value_val,
        'significant_at_005': p_value_val < 0.05,
        'significant_at_001': p_value_val < 0.01
    }


def calculate_per_ticker_metrics(actuals, predictions, tickers) -> dict:
    """
    Per-ticker metric breakdown from flat lists.
    
    Args:
        actuals: list of actual values
        predictions: list of predicted values
        tickers: list of ticker symbols (same length as actuals/predictions)
    
    Returns:
        Dict of {ticker: {regression + financial metrics + n_samples}}
    """
    # Group by ticker
    from collections import defaultdict
    ticker_actuals = defaultdict(list)
    ticker_preds = defaultdict(list)
    
    for a, p, t in zip(actuals, predictions, tickers):
        ticker_actuals[t].append(a)
        ticker_preds[t].append(p)
    
    results = {}
    for ticker in ticker_actuals:
        y_true = ticker_actuals[ticker]
        y_pred = ticker_preds[ticker]
        if len(y_true) > 10:  # Need minimum samples
            reg = calculate_regression_metrics(y_true, y_pred)
            fin = calculate_financial_metrics(y_true, y_pred)
            results[ticker] = {**reg, **fin, 'n': len(y_true)}
    return results

