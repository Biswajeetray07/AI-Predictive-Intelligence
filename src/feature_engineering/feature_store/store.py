"""
Feature Store — Centralized Parquet-Based Feature Storage with Versioning.

Provides a FeatureStore class that manages versioned feature datasets stored
as Parquet files. Supports saving, loading, listing versions, and fast
DuckDB-powered queries across feature tables.

Directory structure:
    data/feature_store/
        {feature_name}/
            v1/
                {feature_name}.parquet
                metadata.json
            v2/
                ...
"""

import os
import json
import logging
from datetime import datetime
from typing import Optional, List, Dict

import pandas as pd
import numpy as np

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FeatureStore")


class FeatureStore:
    """
    Centralized feature storage with Parquet backend and optional DuckDB queries.

    Usage:
        store = FeatureStore("/path/to/project")
        store.save_features("market_features", df, description="Daily OHLCV + technicals")
        df = store.load_features("market_features")
        df = store.load_features("market_features", version=1)
    """

    def __init__(self, project_root: str):
        self.project_root = os.path.abspath(project_root)
        self.store_dir = os.path.join(self.project_root, "data", "feature_store")
        os.makedirs(self.store_dir, exist_ok=True)
        logger.info(f"FeatureStore initialized at {self.store_dir}")

    # ── Save ─────────────────────────────────────────────────────────────────

    def save_features(
        self,
        name: str,
        df: pd.DataFrame,
        description: str = "",
        tags: Optional[List[str]] = None,
    ) -> int:
        """
        Save a feature DataFrame as a versioned Parquet file.

        Args:
            name: Feature set name (e.g., 'market_features').
            df: DataFrame to save.
            description: Human-readable description.
            tags: Optional tags for categorization.

        Returns:
            The version number assigned.
        """
        feature_dir = os.path.join(self.store_dir, name)
        os.makedirs(feature_dir, exist_ok=True)

        # Determine next version
        existing = self.list_versions(name)
        version = max(existing) + 1 if existing else 1

        version_dir = os.path.join(feature_dir, f"v{version}")
        os.makedirs(version_dir, exist_ok=True)

        # Save Parquet
        parquet_path = os.path.join(version_dir, f"{name}.parquet")
        df.to_parquet(parquet_path, index=False, engine="pyarrow")

        # Save metadata
        metadata = {
            "name": name,
            "version": version,
            "description": description,
            "tags": tags or [],
            "created_at": datetime.now().isoformat(),
            "num_rows": len(df),
            "num_columns": len(df.columns),
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "memory_mb": round(df.memory_usage(deep=True).sum() / (1024 ** 2), 2),
        }
        meta_path = os.path.join(version_dir, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(
            f"Saved '{name}' v{version}: {len(df)} rows × {len(df.columns)} cols "
            f"({metadata['memory_mb']} MB) → {parquet_path}"
        )
        return version

    # ── Load ─────────────────────────────────────────────────────────────────

    def load_features(
        self, name: str, version: Optional[int] = None
    ) -> Optional[pd.DataFrame]:
        """
        Load a feature set. Defaults to latest version.

        Args:
            name: Feature set name.
            version: Specific version (default: latest).

        Returns:
            DataFrame or None if not found.
        """
        if version is None:
            versions = self.list_versions(name)
            if not versions:
                logger.warning(f"No versions found for feature '{name}'")
                return None
            version = max(versions)

        parquet_path = os.path.join(
            self.store_dir, name, f"v{version}", f"{name}.parquet"
        )

        if not os.path.exists(parquet_path):
            logger.warning(f"Feature '{name}' v{version} not found at {parquet_path}")
            return None

        df = pd.read_parquet(parquet_path)
        logger.info(f"Loaded '{name}' v{version}: {df.shape}")
        return df

    # ── Metadata ─────────────────────────────────────────────────────────────

    def get_metadata(self, name: str, version: Optional[int] = None) -> Optional[Dict]:
        """Get metadata for a feature set version."""
        if version is None:
            versions = self.list_versions(name)
            if not versions:
                return None
            version = max(versions)

        meta_path = os.path.join(
            self.store_dir, name, f"v{version}", "metadata.json"
        )
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                return json.load(f)
        return None

    def list_versions(self, name: str) -> List[int]:
        """List all available versions for a feature set."""
        feature_dir = os.path.join(self.store_dir, name)
        if not os.path.exists(feature_dir):
            return []
        versions = []
        for d in os.listdir(feature_dir):
            if d.startswith("v") and os.path.isdir(os.path.join(feature_dir, d)):
                try:
                    versions.append(int(d[1:]))
                except ValueError:
                    pass
        return sorted(versions)

    def list_feature_sets(self) -> List[Dict]:
        """List all feature sets with their latest metadata."""
        results = []
        if not os.path.exists(self.store_dir):
            return results
        for name in os.listdir(self.store_dir):
            name_dir = os.path.join(self.store_dir, name)
            if os.path.isdir(name_dir):
                meta = self.get_metadata(name)
                results.append({
                    "name": name,
                    "versions": self.list_versions(name),
                    "latest_metadata": meta,
                })
        return results

    # ── DuckDB Queries ───────────────────────────────────────────────────────

    def query(self, sql: str) -> Optional[pd.DataFrame]:
        """
        Run a SQL query across feature store Parquet files using DuckDB.

        Table references in the query should use the format: '{feature_name}'
        They will be resolved to the latest version's Parquet file.

        Example:
            store.query("SELECT date, close FROM 'market_features' WHERE close > 100")
        """
        if not DUCKDB_AVAILABLE:
            logger.error("DuckDB not installed. Install with: pip install duckdb")
            return None

        try:
            import duckdb
            con = duckdb.connect()
            # Register all feature sets as views
            for name in os.listdir(self.store_dir):
                versions = self.list_versions(name)
                if versions:
                    latest = max(versions)
                    path = os.path.join(
                        self.store_dir, name, f"v{latest}", f"{name}.parquet"
                    )
                    if os.path.exists(path):
                        con.execute(
                            f"CREATE VIEW \"{name}\" AS SELECT * FROM read_parquet('{path}')"
                        )
            result = con.execute(sql).fetchdf()
            con.close()
            return result
        except Exception as e:
            logger.error(f"DuckDB query failed: {e}")
            return None

    # ── Comparison ───────────────────────────────────────────────────────────

    def compare_versions(
        self, name: str, v1: int, v2: int
    ) -> Optional[Dict]:
        """Compare two versions of a feature set."""
        m1 = self.get_metadata(name, v1)
        m2 = self.get_metadata(name, v2)

        if m1 is None or m2 is None:
            logger.warning(f"Cannot compare: one or both versions not found")
            return None

        added_cols = set(m2["columns"]) - set(m1["columns"])
        removed_cols = set(m1["columns"]) - set(m2["columns"])

        return {
            "v1": v1,
            "v2": v2,
            "rows_diff": m2["num_rows"] - m1["num_rows"],
            "cols_diff": m2["num_columns"] - m1["num_columns"],
            "added_columns": list(added_cols),
            "removed_columns": list(removed_cols),
            "memory_diff_mb": round(m2["memory_mb"] - m1["memory_mb"], 2),
        }


if __name__ == "__main__":
    # Quick smoke test
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        store = FeatureStore(tmpdir)

        # Create sample features
        df = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=100, freq="B"),
            "close": np.random.randn(100).cumsum() + 100,
            "volume": np.random.randint(1000, 10000, 100),
            "rsi_14": np.random.uniform(20, 80, 100),
        })

        v1 = store.save_features("market_features", df, description="OHLCV + RSI")
        print(f"Saved v{v1}")

        # Add a column and save v2
        df["macd"] = np.random.randn(100)
        v2 = store.save_features("market_features", df, description="+ MACD")
        print(f"Saved v{v2}")

        # Load latest
        loaded = store.load_features("market_features")
        if loaded is not None:
            print(f"Loaded: {loaded.shape}")

        # List
        print(f"Feature sets: {store.list_feature_sets()}")

        # Compare
        diff = store.compare_versions("market_features", 1, 2)
        print(f"Version diff: {diff}")

        print("\n✅ Feature Store smoke test passed!")
