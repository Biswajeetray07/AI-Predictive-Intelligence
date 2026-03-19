import os
import sys
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.pipelines.inference_pipeline import Predictor

def test_multi_horizon():
    print("Testing Multi-Horizon Inference Pipeline...")
    predictor = Predictor(device='cpu')
    
    # Mock TS data: 60 seq length, 128 features
    # Adjust feature size to what TS ensemble expects (using input_dim=128 by default in pipeline)
    dummy_ts = np.random.randn(60, 128).astype(np.float32)
    
    # Run prediction
    res = predictor.predict(dummy_ts)
    
    print("\nInference Results:")
    for k, v in res.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for sub_k, sub_v in v.items():
                print(f"    {sub_k}: {sub_v}")
        else:
            print(f"  {k}: {v}")
            
    assert 'multi_horizon_predictions' in res
    assert '1d' in res['multi_horizon_predictions']
    assert '5d' in res['multi_horizon_predictions']
    assert '30d' in res['multi_horizon_predictions']
    assert 'ensemble_weights' in res
    print("\nTest passed!")

if __name__ == "__main__":
    test_multi_horizon()
