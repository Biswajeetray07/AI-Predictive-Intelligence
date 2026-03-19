import pytest
import pandas as pd
import numpy as np
from src.feature_engineering.feature_selection import (
    compute_mutual_information,
    compute_variance_importance,
    compute_correlation_redundancy,
    select_features
)

@pytest.fixture
def sample_feature_data():
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=n),
        'ticker': 'TEST',
        'close': np.random.randn(n).cumsum(),
        'feature_good': np.random.randn(n),
        'feature_noise': np.random.randn(n) * 0.001,  # Low variance
        'feature_constant': np.ones(n),  # Zero variance
    })
    # Make good feature correlated with target
    df['feature_good'] = df['close'] + np.random.randn(n) * 0.1
    # Make a highly correlated redundant feature
    df['feature_redundant'] = df['feature_good'] + np.random.randn(n) * 0.01
    return df

def test_variance_importance(sample_feature_data):
    X = sample_feature_data.drop(columns=['date', 'ticker', 'close'])
    var_scores = compute_variance_importance(X)
    
    # Check that scores are returned and ordered
    assert len(var_scores) == 4
    assert list(var_scores.index) == ['feature_good', 'feature_redundant', 'feature_noise', 'feature_constant']
    # Constant feature should have 0 variance
    assert np.isclose(var_scores['feature_constant'], 0)

def test_correlation_redundancy(sample_feature_data):
    X = sample_feature_data.drop(columns=['date', 'ticker', 'close'])
    
    # 0.85 threshold should catch feature_redundant
    redundant = compute_correlation_redundancy(X, threshold=0.85)
    assert len(redundant) == 1
    assert 'feature_redundant' in redundant
    
    # 0.999 threshold might miss it (if noise is high enough) or catch it
    # Just verify the function runs correctly
    high_threshold = compute_correlation_redundancy(X, threshold=0.9999)
    assert len(high_threshold) <= 1

def test_select_features_pipeline(sample_feature_data):
    # Test the end-to-end selection logic
    selected, report = select_features(
        sample_feature_data, 
        target_col='close',
        correlation_threshold=0.85,
        variance_threshold=0.01,
        min_features=1,  # Avoid forcing keeping features for the test
        output_path="tests/__mock_selected_features.yaml"
    )
    
    # Expected behavior:
    # 1. feature_constant dropped (zero variance)
    # 2. feature_redundant dropped (correlated > 0.85 with feature_good)
    # 3. feature_noise might be dropped depending on MI score, but let's check redundancy/variance handled it
    
    assert 'feature_constant' in report['removed'].get('low_variance', [])
    assert 'feature_redundant' in report['removed'].get('correlated', [])
    assert 'feature_good' in selected
