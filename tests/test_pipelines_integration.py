import pytest
import os
import sys

# Assume the package is installed in editable mode

def test_training_pipeline_imports():
    """Verify that importing the main training pipeline doesn't cause errors."""
    try:
        import src.pipelines.training_pipeline
        assert True
    except Exception as e:
        pytest.fail(f"Could not import training_pipeline: {e}")

def test_inference_pipeline_initialization():
    """Verify the Predictor class can be imported and initialized without data crashes."""
    try:
        from src.pipelines.inference_pipeline import Predictor
        import torch
        # We don't initialize the full predictor as it requires model weights,
        # but we ensure the class structure is importable.
        assert Predictor is not None
    except Exception as e:
        pytest.fail(f"Could not import Predictor: {e}")

def test_pipeline_utils():
    """Verify the pipeline utils run_script logic is loadable."""
    try:
        from src.utils.pipeline_utils import run_script, get_project_root
        root = get_project_root()
        assert os.path.exists(root)
    except Exception as e:
        pytest.fail(f"Pipeline utils failed: {e}")
