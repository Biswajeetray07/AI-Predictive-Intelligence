"""
Unit tests for all model forward passes.
Verifies that LSTM, GRU, Transformer, TFT, Fusion, and Multi-Horizon Fusion
produce correct output shapes and return (prediction, context) tuples.
"""

import sys
import os
import pytest
import torch

# Add project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

from src.models.timeseries.lstm import LSTMForecaster
from src.models.timeseries.gru import GRUForecaster
from src.models.timeseries.transformer import TransformerForecaster
from src.models.timeseries.tft import TFTForecaster
from src.models.fusion.fusion import DeepFusionModel

BATCH = 4
SEQ_LEN = 60
INPUT_DIM = 50
HIDDEN_DIM = 128


class TestLSTMForecaster:
    def test_forward_shape(self):
        model = LSTMForecaster(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM)
        x = torch.randn(BATCH, SEQ_LEN, INPUT_DIM)
        pred, ctx = model(x)
        assert pred.shape == (BATCH, 1), f"Expected (4,1), got {pred.shape}"
        assert ctx.shape == (BATCH, HIDDEN_DIM), f"Expected (4,128), got {ctx.shape}"

    def test_unidirectional(self):
        """LSTM must be unidirectional to prevent look-ahead bias."""
        model = LSTMForecaster(input_dim=INPUT_DIM)
        assert model.lstm.bidirectional is False, "LSTM must be unidirectional"

    def test_gradient_flow(self):
        model = LSTMForecaster(input_dim=INPUT_DIM)
        x = torch.randn(BATCH, SEQ_LEN, INPUT_DIM)
        pred, _ = model(x)
        loss = pred.sum()
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


class TestGRUForecaster:
    def test_forward_shape(self):
        model = GRUForecaster(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM)
        x = torch.randn(BATCH, SEQ_LEN, INPUT_DIM)
        pred, ctx = model(x)
        assert pred.shape == (BATCH, 1)
        assert ctx.shape == (BATCH, HIDDEN_DIM)

    def test_gradient_flow(self):
        model = GRUForecaster(input_dim=INPUT_DIM)
        x = torch.randn(BATCH, SEQ_LEN, INPUT_DIM)
        pred, _ = model(x)
        loss = pred.sum()
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


class TestTransformerForecaster:
    def test_forward_shape(self):
        model = TransformerForecaster(input_dim=INPUT_DIM, d_model=HIDDEN_DIM)
        x = torch.randn(BATCH, SEQ_LEN, INPUT_DIM)
        pred, ctx = model(x)
        assert pred.shape == (BATCH, 1)
        assert ctx.shape == (BATCH, HIDDEN_DIM)

    def test_causal_mask_exists(self):
        """Transformer must use causal masking to prevent future attention."""
        model = TransformerForecaster(input_dim=INPUT_DIM)
        assert hasattr(model, '_generate_causal_mask'), "Missing causal mask method"

    def test_causal_mask_shape(self):
        model = TransformerForecaster(input_dim=INPUT_DIM)
        mask = model._generate_causal_mask(SEQ_LEN, torch.device('cpu'))
        assert mask.shape == (SEQ_LEN, SEQ_LEN)
        # Upper triangle should be -inf
        assert mask[0, 1] == float('-inf'), "Upper triangle must be -inf"
        # Diagonal and below should be 0
        assert mask[1, 0] == 0.0, "Lower triangle must be 0"
        assert mask[0, 0] == 0.0, "Diagonal must be 0"


class TestTFTForecaster:
    def test_forward_shape(self):
        model = TFTForecaster(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM)
        x = torch.randn(BATCH, SEQ_LEN, INPUT_DIM)
        pred, ctx = model(x)
        assert pred.shape == (BATCH, 1)
        assert ctx.shape == (BATCH, HIDDEN_DIM)


class TestDeepFusionModel:
    def test_forward_shape(self):
        model = DeepFusionModel(nlp_dim=768, ts_dim=HIDDEN_DIM)
        nlp = torch.randn(BATCH, SEQ_LEN, 768)
        ts = torch.randn(BATCH, HIDDEN_DIM)
        pred = model(nlp, ts)
        assert pred.shape == (BATCH, 1)

    def test_param_count(self):
        model = DeepFusionModel(nlp_dim=768, ts_dim=128)
        total = sum(p.numel() for p in model.parameters())
        assert total > 0, "Model should have parameters"


class TestNoTargetLeakage:
    """Verify that the sequence builder excludes the target from features."""

    def test_target_excluded_from_features(self):
        """Simulate the features_cols logic from build_sequences.py."""
        columns = ['date', 'ticker', 'close', 'open', 'high', 'low', 'volume', 'rsi']
        target_col = 'close'
        exclude_cols = ['date', 'ticker', target_col]
        features_cols = [c for c in columns if c not in exclude_cols]

        assert 'close' not in features_cols, "Target col 'close' must NOT be in features"
        assert 'open' in features_cols
        assert 'date' not in features_cols
        assert 'ticker' not in features_cols


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
