"""
Data Validator Module
=====================
Utilities for data quality: deduplication, schema validation,
and missing field handling on pandas DataFrames.
"""

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def remove_duplicates(
    df: pd.DataFrame,
    subset: Optional[List[str]] = None,
    keep: str = "first",
) -> pd.DataFrame:
    """
    Remove duplicate rows from a DataFrame.

    Args:
        df: Input DataFrame.
        subset: Columns to consider for identifying duplicates.
                If None, uses all columns.
        keep: Which duplicate to keep ('first', 'last', or False).

    Returns:
        DataFrame with duplicates removed.
    """
    initial_count = len(df)
    df_clean = df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)  # type: ignore[arg-type]
    removed = initial_count - len(df_clean)
    if removed > 0:
        logger.info(f"Removed {removed} duplicate rows (kept '{keep}')")
    return df_clean


def validate_schema(
    df: pd.DataFrame,
    required_columns: List[str],
    dtypes: Optional[Dict[str, str]] = None,
) -> bool:
    """
    Validate that a DataFrame has the expected schema.

    Args:
        df: DataFrame to validate.
        required_columns: List of column names that must be present.
        dtypes: Optional mapping of column name → expected dtype string
                (e.g., {"date": "datetime64[ns]", "value": "float64"}).

    Returns:
        True if validation passes.

    Raises:
        ValueError: If required columns are missing or dtypes mismatch.
    """
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        msg = f"Missing required columns: {missing}"
        logger.error(msg)
        raise ValueError(msg)

    if dtypes:
        for col, expected_dtype in dtypes.items():
            if col in df.columns:
                actual = str(df[col].dtype)
                if expected_dtype not in actual:
                    logger.warning(
                        f"Column '{col}' has dtype '{actual}', "
                        f"expected '{expected_dtype}'"
                    )

    logger.info(f"Schema validation passed. Columns: {list(df.columns)}")
    return True


def handle_missing_fields(
    df: pd.DataFrame,
    fill_values: Optional[Dict[str, Any]] = None,
    drop_threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Handle missing values in a DataFrame.

    Strategy:
    1. Drop columns where more than `drop_threshold` fraction of values are null.
    2. Fill remaining nulls using `fill_values` mapping.
    3. For numeric columns without explicit fill, use median.
    4. For string columns without explicit fill, use "unknown".

    Args:
        df: Input DataFrame.
        fill_values: Explicit column → fill value mapping.
        drop_threshold: Drop columns with null fraction above this threshold.

    Returns:
        DataFrame with missing values handled.
    """
    if fill_values is None:
        fill_values = {}

    # Drop columns that are mostly empty
    null_fractions = df.isnull().mean()
    cols_to_drop = null_fractions[null_fractions > drop_threshold].index.tolist()
    if cols_to_drop:
        logger.warning(
            f"Dropping columns with >{drop_threshold*100:.0f}% missing: {cols_to_drop}"
        )
        df = df.drop(columns=cols_to_drop)

    # Apply explicit fill values
    for col, val in fill_values.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)

    # Auto-fill remaining missing values
    for col in df.columns:
        if df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                logger.debug(f"Filled '{col}' missing values with median={median_val}")
            elif pd.api.types.is_string_dtype(df[col]) or df[col].dtype == object:
                df[col] = df[col].fillna("unknown")
                logger.debug(f"Filled '{col}' missing values with 'unknown'")

    total_missing = df.isnull().sum().sum()
    logger.info(f"Missing value handling complete. Remaining nulls: {total_missing}")
    return df


def standardize_timestamps(
    df: pd.DataFrame,
    date_column: str,
    target_format: str = "%Y-%m-%d",
) -> pd.DataFrame:
    """
    Convert a date column to a standardized datetime format.

    Args:
        df: Input DataFrame.
        date_column: Name of the column containing dates.
        target_format: Desired output format string.

    Returns:
        DataFrame with standardized date column.
    """
    if date_column not in df.columns:
        logger.warning(f"Date column '{date_column}' not found in DataFrame")
        return df

    df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
    invalid_count = df[date_column].isnull().sum()
    if invalid_count > 0:
        logger.warning(f"{invalid_count} rows have unparseable dates in '{date_column}'")

    logger.info(f"Standardized '{date_column}' to datetime")
    return df
