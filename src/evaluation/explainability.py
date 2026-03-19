"""
Explainability utility module for the AI-Predictive-Intelligence platform.

Provides SHAP integration for feature importance and Attention extraction 
to understand which historical time steps / NLP events drove the predictions.
"""

import numpy as np
import pandas as pd
import logging
import torch
import torch.nn as nn
from typing import Optional, Dict, List

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP library not installed. Install with `pip install shap` for feature importance.")

logger = logging.getLogger("Explainability")


def _require_shap():
    """Guard function to ensure SHAP is available."""
    if not SHAP_AVAILABLE:
        raise ImportError("shap library is required")
    import shap  # Local import after availability check
    return shap


def calculate_shap_values(model, background_data, test_data, feature_names=None):
    """
    Calculate SHAP values for a given PyTorch model.
    Uses DeepExplainer for neural networks.
    
    Args:
        model: PyTorch model (must be in eval mode).
        background_data: Tensor of background samples for computing SHAP baseline.
        test_data: Tensor of test samples to explain.
        feature_names: Optional list of feature names for labeling.
        
    Returns:
        shap_values: SHAP values array, or None if SHAP is not available.
    """
    if not SHAP_AVAILABLE:
        logger.error("Cannot calculate SHAP: library missing. Install with `pip install shap`.")
        return None
        
    model.eval()
    
    if not SHAP_AVAILABLE:
        logger.error("Cannot calculate SHAP: library missing. Install with `pip install shap`.")
        return None
    
    try:
        shap = _require_shap()
        explainer = shap.DeepExplainer(model, background_data)
        shap_values = explainer.shap_values(test_data)
        logger.info(f"SHAP values computed: shape={np.array(shap_values).shape}")
        return shap_values
    except Exception as e:
        logger.error(f"SHAP extraction failed: {str(e)}")
        return None


def extract_attention_weights(model, x, model_type: str = "auto") -> Optional[Dict[str, np.ndarray]]:
    """
    Extract attention weights from Transformer/TFT/Fusion models for visualization.
    
    Registers forward hooks on attention layers to capture the attention weight
    matrices produced during the forward pass.
    
    Args:
        model: PyTorch model with attention layers.
        x: Input tensor(s) — for TS models: [batch, seq, features];
           for fusion models: tuple of (nlp_seq, ts_features).
        model_type: One of 'transformer', 'tft', 'fusion', or 'auto' to detect.
        
    Returns:
        Dict mapping layer names to attention weight arrays [batch, heads, seq, seq],
        or None if no attention layers found.
    """
    model.eval()
    attention_maps = {}
    hooks = []
    
    def _hook_fn(name):
        def hook(module, input, output):
            # nn.MultiheadAttention returns (attn_output, attn_weights)
            if isinstance(output, tuple) and len(output) >= 2:
                attn_weights = output[1]
                if attn_weights is not None:
                    attention_maps[name] = attn_weights.detach().cpu().numpy()
        return hook
    
    # Register hooks on all MultiheadAttention modules
    for name, module in model.named_modules():
        if isinstance(module, nn.MultiheadAttention):
            # Temporarily enable weight output
            original_need_weights = getattr(module, '_qkv_same_embed_dim', True)
            hook = module.register_forward_hook(_hook_fn(name))
            hooks.append(hook)
    
    if not hooks:
        logger.warning("No MultiheadAttention layers found in the model.")
        return None
    
    try:
        with torch.no_grad():
            if isinstance(x, (tuple, list)):
                model(*x)
            else:
                model(x)
    except Exception as e:
        logger.error(f"Forward pass failed during attention extraction: {e}")
        # Clean up hooks
        for h in hooks:
            h.remove()
        return None
    
    # Clean up hooks
    for h in hooks:
        h.remove()
    
    if not attention_maps:
        logger.warning("Forward pass succeeded but no attention weights were captured. "
                       "The model may use a custom attention mechanism.")
        return None
    
    logger.info(f"Extracted attention weights from {len(attention_maps)} layers: "
                f"{list(attention_maps.keys())}")
    return attention_maps


def plot_feature_importance(
    shap_values, 
    feature_names: List[str],
    max_display: int = 20,
    plot_type: str = "bar",
    save_path: Optional[str] = None,
):
    """
    Generate and optionally save a summary plot of top features using SHAP.
    
    Args:
        shap_values: SHAP values (from calculate_shap_values).
        feature_names: List of feature names.
        max_display: Maximum number of features to display.
        plot_type: One of 'bar', 'dot', 'violin'.
        save_path: Optional path to save the plot image.
    """
    if not SHAP_AVAILABLE:
        logger.error("Cannot plot: SHAP not installed.")
        return
    
    if shap_values is None:
        logger.warning("No SHAP values provided — skipping plot.")
        return
        
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        
        shap = _require_shap()
        shap.summary_plot(
            shap_values, 
            feature_names=feature_names, 
            max_display=max_display,
            plot_type=plot_type,
            show=False,
        )
        
        if save_path:
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to: {save_path}")
        
        plt.close()
        
    except Exception as e:
        logger.error(f"Plotting failed: {str(e)}")


def get_top_features(
    shap_values: np.ndarray,
    feature_names: List[str],
    top_k: int = 10,
) -> pd.DataFrame:
    """
    Return a DataFrame of the top-k most important features by mean |SHAP|.
    
    Args:
        shap_values: SHAP values array [n_samples, n_features].
        feature_names: List of feature names.
        top_k: Number of top features to return.
        
    Returns:
        DataFrame with columns ['feature', 'mean_abs_shap', 'rank'].
    """
    if shap_values is None:
        return pd.DataFrame(columns=['feature', 'mean_abs_shap', 'rank'])
    
    vals = np.array(shap_values)
    if vals.ndim == 3:
        vals = vals.reshape(-1, vals.shape[-1])
    
    mean_abs = np.mean(np.abs(vals), axis=0)
    
    indices = np.argsort(mean_abs)[::-1][:top_k]
    
    results = []
    for rank, idx in enumerate(indices, 1):
        results.append({
            'feature': feature_names[idx] if idx < len(feature_names) else f"feature_{idx}",
            'mean_abs_shap': float(mean_abs[idx]),
            'rank': rank,
        })
    
    return pd.DataFrame(results)
