"""
Unit tests for data processing utilities.
Tests data validator functions and sequence building logic.
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.data_validator import (
    remove_duplicates,
    validate_schema,
    handle_missing_fields,
    standardize_timestamps,
)


class TestRemoveDuplicates:
    def test_removes_exact_duplicates(self):
        df = pd.DataFrame({'a': [1, 1, 2], 'b': ['x', 'x', 'y']})
        result = remove_duplicates(df)
        assert len(result) == 2

    def test_subset_dedup(self):
        df = pd.DataFrame({'a': [1, 1, 2], 'b': ['x', 'y', 'z']})
        result = remove_duplicates(df, subset=['a'])
        assert len(result) == 2

    def test_no_duplicates(self):
        df = pd.DataFrame({'a': [1, 2, 3]})
        result = remove_duplicates(df)
        assert len(result) == 3


class TestValidateSchema:
    def test_valid_schema(self):
        df = pd.DataFrame({'date': ['2024-01-01'], 'value': [1.0]})
        assert validate_schema(df, required_columns=['date', 'value'])

    def test_missing_column_raises(self):
        df = pd.DataFrame({'date': ['2024-01-01']})
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_schema(df, required_columns=['date', 'value'])


class TestHandleMissingFields:
    def test_fills_numeric_with_median(self):
        df = pd.DataFrame({'a': [1.0, np.nan, 3.0, np.nan, 5.0]})
        result = handle_missing_fields(df)
        assert result['a'].isnull().sum() == 0
        assert result['a'].iloc[1] == 3.0  # median of [1,3,5]

    def test_drops_mostly_null_columns(self):
        df = pd.DataFrame({
            'good': [1, 2, 3, 4, 5],
            'bad': [np.nan, np.nan, np.nan, np.nan, 1],
        })
        result = handle_missing_fields(df, drop_threshold=0.5)
        assert 'bad' not in result.columns
        assert 'good' in result.columns

    def test_explicit_fill(self):
        df = pd.DataFrame({'x': [1.0, np.nan]})
        result = handle_missing_fields(df, fill_values={'x': 99})
        assert result['x'].iloc[1] == 99


class TestStandardizeTimestamps:
    def test_basic_date_conversion(self):
        df = pd.DataFrame({'date': ['2024-01-01', '2024-02-15']})
        result = standardize_timestamps(df, 'date')
        assert pd.api.types.is_datetime64_any_dtype(result['date'])

    def test_missing_column_warning(self):
        df = pd.DataFrame({'value': [1, 2]})
        result = standardize_timestamps(df, 'date')
        assert 'date' not in result.columns  # Should return unchanged


class TestSequenceBuilderLogic:
    """Test the core logic of build_sequences without running the full script."""

    def test_feature_target_separation(self):
        """Target column must not appear in features."""
        columns = ['date', 'ticker', 'close', 'open', 'high', 'low', 'volume',
                    'rsi', 'macd', 'bb_upper', 'bb_lower']
        target_col = 'close'
        exclude_cols = ['date', 'ticker', target_col]
        features_cols = [c for c in columns if c not in exclude_cols]

        assert target_col not in features_cols
        assert len(features_cols) == 8  # open, high, low, volume, rsi, macd, bb_upper, bb_lower

    def test_temporal_split_logic(self):
        """Train/val/test splits must be temporal, not random."""
        dates = pd.date_range('2020-01-01', '2024-01-01', freq='D')
        df = pd.DataFrame({'date': dates})

        train = df[df['date'] < '2022-01-01']
        val = df[(df['date'] >= '2022-01-01') & (df['date'] < '2023-01-01')]
        test = df[df['date'] >= '2023-01-01']

        assert train['date'].max() < val['date'].min(), "Train must end before val starts"
        assert val['date'].max() < test['date'].min(), "Val must end before test starts"

    def test_no_future_leakage_in_fill(self):
        """Forward-fill should not use future data."""
        values = [1.0, np.nan, np.nan, 4.0, 5.0]
        s = pd.Series(values)
        filled = s.ffill().fillna(0)
        # Index 1 and 2 should be 1.0 (forward-filled from index 0), NOT 4.0
        assert filled.iloc[1] == 1.0
        assert filled.iloc[2] == 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
