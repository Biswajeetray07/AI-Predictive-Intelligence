"""
Data Validator
===============
Validates collected datasets for schema integrity, data quality, and duplicates.
"""

import logging
import pandas as pd
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)


class DataValidator:
    """
    Validates collected DataFrames for quality, schema compliance, and duplicates.

    Usage:
        validator = DataValidator()
        df = validator.validate(df, required_columns=['id', 'title', 'date'])
    """

    def __init__(self, drop_duplicates: bool = True, drop_empty_rows: bool = True):
        self.drop_duplicates = drop_duplicates
        self.drop_empty_rows = drop_empty_rows

    def validate(
        self,
        df: pd.DataFrame,
        required_columns: Optional[List[str]] = None,
        unique_key: Optional[str] = None,
        min_rows: int = 0,
        name: str = "dataset",
    ) -> pd.DataFrame:
        """
        Validate and clean a DataFrame.

        Args:
            df: Input DataFrame.
            required_columns: List of column names that must be present.
            unique_key: Column to deduplicate on.
            min_rows: Minimum expected row count (logs a warning if below).
            name: Name of the dataset for logging.

        Returns:
            Cleaned and validated DataFrame.
        """
        original_count = len(df)
        logger.info(f"Validating '{name}': {original_count} rows, {len(df.columns)} columns")

        # Check required columns
        if required_columns:
            missing = [col for col in required_columns if col not in df.columns]
            if missing:
                logger.error(f"'{name}' missing required columns: {missing}")
                raise ValueError(f"Missing required columns: {missing}")

        # Drop duplicates
        if self.drop_duplicates:
            if unique_key and unique_key in df.columns:
                before = len(df)
                df = df.drop_duplicates(subset=[unique_key])
                dropped = before - len(df)
                if dropped > 0:
                    logger.info(f"  Dropped {dropped} duplicate rows (key: {unique_key})")
            else:
                before = len(df)
                df = df.drop_duplicates()
                dropped = before - len(df)
                if dropped > 0:
                    logger.info(f"  Dropped {dropped} exact duplicate rows")

        # Drop fully empty rows
        if self.drop_empty_rows:
            before = len(df)
            df = df.dropna(how="all")
            dropped = before - len(df)
            if dropped > 0:
                logger.info(f"  Dropped {dropped} fully empty rows")

        # Minimum row check
        if len(df) < min_rows:
            logger.warning(
                f"'{name}' has {len(df)} rows (expected minimum {min_rows})"
            )

        # Null report
        null_counts = df.isnull().sum()
        cols_with_nulls = null_counts[null_counts > 0]
        if not cols_with_nulls.empty:
            logger.info(f"  Columns with nulls: {dict(cols_with_nulls)}")

        final_count = len(df)
        logger.info(
            f"  Validation complete: {final_count} rows "
            f"({original_count - final_count} removed)"
        )
        return df

    def validate_schema(
        self,
        df: pd.DataFrame,
        expected_dtypes: Dict[str, str],
        name: str = "dataset",
    ) -> bool:
        """
        Check that DataFrame columns match expected data types.

        Args:
            df: DataFrame to validate.
            expected_dtypes: Dict mapping column name → expected dtype string.
            name: Dataset name for logging.

        Returns:
            True if schema is valid.
        """
        issues = []
        for col, expected in expected_dtypes.items():
            if col not in df.columns:
                issues.append(f"Missing column: {col}")
            elif str(df[col].dtype) != expected:
                issues.append(f"{col}: expected {expected}, got {df[col].dtype}")

        if issues:
            logger.warning(f"Schema issues in '{name}': {issues}")
            return False

        logger.info(f"Schema validation passed for '{name}'")
        return True
