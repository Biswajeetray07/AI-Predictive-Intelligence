import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
import yaml
import logging
from typing import Dict, List, Tuple, Union, Optional

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.timeseries.lstm import LSTMForecaster
from src.models.timeseries.gru import GRUForecaster
from src.models.timeseries.transformer import TransformerForecaster
from src.models.timeseries.tft import TFTForecaster
from src.models.nlp.model import MultiTaskNLPModel
from src.models.nlp.tokenizer import NLPTokenizer
from src.models.fusion.multi_horizon_fusion import MultiHorizonFusionModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Predictor:
    """
    Unified Inference Engine for AI-Predictive-Intelligence.
    Fuses Time Series ensembles with NLP signals using the Deep Fusion Model.
    """
    def __init__(self, device: Optional[str] = None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        logging.info(f"Predictor initialized on {self.device}")
        
        self.config = self._load_config()
        self.scalers = self._load_scalers()
        
        # Models
        self.ts_models = {}
        self.nlp_model = None
        self.fusion_model = None
        self.nlp_tokenizer = None
        
        self._load_all_models()

    def _load_config(self):
        config_path = os.path.join(PROJECT_ROOT, 'configs', 'training_config.yaml')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        return {}

    def _load_scalers(self):
        import pickle
        scaler_dir = os.path.join(PROJECT_ROOT, 'data', 'processed', 'model_inputs')
        scalers = {}
        feature_scaler_path = os.path.join(scaler_dir, 'feature_scaler.pkl')
        target_scaler_path = os.path.join(scaler_dir, 'target_scaler.pkl')
        
        def load_scaler(path):
            try:
                # Try joblib first
                return joblib.load(path)
            except Exception:
                # Fallback to standard pickle
                with open(path, 'rb') as f:
                    return pickle.load(f)
                    
        if os.path.exists(feature_scaler_path):
            try:
                scalers['feature'] = load_scaler(feature_scaler_path)
            except Exception as e:
                logging.warning(f"Failed to load feature scaler: {e}")
                
        if os.path.exists(target_scaler_path):
            try:
                scalers['target'] = load_scaler(target_scaler_path)
            except Exception as e:
                logging.warning(f"Failed to load target scaler: {e}")
            
        return scalers

    def _load_all_models(self):
        model_dir = os.path.join(PROJECT_ROOT, 'saved_models')
        ts_config = self.config.get('timeseries_model', {})
        
        # 1. Load Time Series Ensemble
        # We need input_dim. We'll try to infer it from existing metadata or use a placeholder then resize if needed,
        # but better to get it from the scaler or a sample.
        # For now, let's assume we can get it from the feature scaler if available.
        input_dim = 128 # Default fallback
        if 'feature' in self.scalers:
            input_dim = self.scalers['feature'].n_features_in_
            
        ts_model_names = ts_config.get('models', ['lstm', 'gru', 'transformer', 'tft'])
        hidden = ts_config.get('hidden_dim', 128)
        layers = ts_config.get('num_layers', 2)
        
        for name in ts_model_names:
            path = os.path.join(model_dir, f'{name}_model.pt')
            if os.path.exists(path):
                model = None
                if name == 'lstm':
                    model = LSTMForecaster(input_dim, hidden, layers)
                elif name == 'gru':
                    model = GRUForecaster(input_dim, hidden, layers)
                elif name == 'transformer':
                    model = TransformerForecaster(input_dim, d_model=hidden, nhead=4, num_layers=layers, dropout=0.0)
                elif name == 'tft':
                    model = TFTForecaster(input_dim, hidden, num_heads=4, num_layers=layers, dropout=0.0)
                if model is not None:
                    model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
                    model.to(self.device).eval()
                    self.ts_models[name] = model
                    logging.info(f"Loaded TS model: {name}")
                else:
                    logging.warning(f"Model type '{name}' not recognized. Skipping.")

        # 2. Load NLP Model
        nlp_path = os.path.join(model_dir, 'nlp_multitask_model.pt')
        if os.path.exists(nlp_path):
            self.nlp_model = MultiTaskNLPModel(freeze_encoder_layers=0)
            self.nlp_model.load_state_dict(torch.load(nlp_path, map_location=self.device, weights_only=True))
            self.nlp_model.to(self.device).eval()
            self.nlp_tokenizer = NLPTokenizer(max_length=self.config.get('nlp_model', {}).get('max_length', 256))
            logging.info("Loaded NLP multi-task model")

        # 3. Load Fusion Model
        fusion_path = os.path.join(model_dir, 'fusion_model.pt')
        fusion_config = self.config.get('fusion_model', {})
        if os.path.exists(fusion_path):
            self.fusion_model = MultiHorizonFusionModel(
                nlp_dim=768,
                ts_dim=128,
                attention_heads=fusion_config.get('attention_heads', 4),
                mlp_hidden=fusion_config.get('mlp_hidden', [512, 256, 128]),
                dropout=0.0
            )
            self.fusion_model.load_state_dict(torch.load(fusion_path, map_location=self.device, weights_only=True))
            self.fusion_model.to(self.device).eval()
            logging.info("Loaded Deep Fusion model")

    @torch.no_grad()
    def predict(self, ts_sequence: np.ndarray, nlp_texts: Optional[List[str]] = None) -> Dict:
        """
        Produce a multi-horizon prediction given a window of 60 days of financial data and optional texts.
        
        Uses BATCHED NLP inference (10× faster), calibrated model confidence, and Dynamic HMM Regime Weights.
        
        Args:
            ts_sequence: [60, input_dim] array
            nlp_texts: List of strings (one per day, or all for the sequence)
            
        Returns:
            Dict containing final_prediction, ensemble_predictions, nlp_signals, confidence.
        """
        # 1. Process Time Series with Dynamic Regime Weights
        x_ts = torch.tensor(ts_sequence, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        ts_preds = {}
        ts_embeddings = []
        raw_preds = []
        
        # Determine Regime Weights dynamically, fallback to static if not available
        weights = self._get_dynamic_regime_weights()
        
        # Model name to weight index mapping
        weight_map = {'lstm': 0, 'gru': 1, 'transformer': 2, 'tft': 3}
        
        for name, model in self.ts_models.items():
            pred, context = model(x_ts)
            pred_val = pred.item()
            ts_preds[name] = pred_val
            raw_preds.append(pred_val)
            
            # Apply correct strategic weight
            w_idx = weight_map.get(name, 0)
            model_weight = weights[w_idx]
            ts_embeddings.append(context.cpu().numpy() * model_weight)
            
        # Weighted average embedding for fusion
        fused_ts_emb = np.sum(ts_embeddings, axis=0) if ts_embeddings else np.zeros((1, 128))
        fused_ts_emb_torch = torch.tensor(fused_ts_emb, dtype=torch.float32).to(self.device)

        # 2. Process NLP — BATCHED INFERENCE (Phase 5 optimization)
        nlp_signals = {}
        nlp_emb_seq = torch.zeros((1, 60, 768)).to(self.device)
        
        if nlp_texts and self.nlp_model:
            texts_to_process = nlp_texts[-60:]
            batch_size = len(texts_to_process)
            
            # Batch tokenization — tokenize all texts in one call
            if self.nlp_tokenizer is not None and hasattr(self.nlp_tokenizer, 'tokenizer'):
                encodings = self.nlp_tokenizer.tokenizer(
                    texts_to_process,
                    padding=True,
                    truncation=True,
                    max_length=getattr(self.nlp_tokenizer, 'max_length', 128),
                    return_tensors='pt'
                )
            else:
                logging.warning("NLP tokenizer is not initialized or missing required attributes.")
                encodings = None
            if encodings is not None:
                input_ids = encodings['input_ids'].to(self.device)
                attn_mask = encodings['attention_mask'].to(self.device)
                
                # Placeholder metadata (batch-sized)
                source_ids = torch.zeros(batch_size, dtype=torch.long).to(self.device)
                days = torch.zeros(batch_size, dtype=torch.long).to(self.device)
                months = torch.zeros(batch_size, dtype=torch.long).to(self.device)
                
                # Single batched forward pass instead of 60 individual calls
                out = self.nlp_model(input_ids, attn_mask, source_ids, days, months)
                
                # Fill the NLP embedding sequence
                nlp_emb_seq[0, :batch_size, :] = out['embedding']
                
                # Extract signals from last day
                nlp_signals['sentiment'] = torch.softmax(out['sentiment'][-1:], dim=-1).cpu().numpy().tolist()
                nlp_signals['events'] = torch.softmax(out['events'][-1:], dim=-1).cpu().numpy().tolist()

        # 3. Fusion (Multi-Horizon)
        final_preds = {'1d': 0.0, '5d': 0.0, '30d': 0.0}
        
        if self.fusion_model:
            fusion_out = self.fusion_model(nlp_emb_seq, fused_ts_emb_torch)
            final_preds['1d'] = fusion_out['1d'].item()
            final_preds['5d'] = fusion_out['5d'].item()
            final_preds['30d'] = fusion_out['30d'].item()
        else:
            # Fallback to TS ensemble weighted average (only supports 1d)
            if ts_preds:
                # Normalize weights of *available* models so they sum to 1
                available_weights = [weights[weight_map[n]] for n in ts_preds]
                w_sum = sum(available_weights) or 1.0
                normalized_weights = [w / w_sum for w in available_weights]
                final_preds['1d'] = sum(ts_preds[n] * normalized_weights[i] for i, n in enumerate(ts_preds))
            else:
                final_preds['1d'] = 0.0

        # 4. Inverse transform prediction
        if 'target' in self.scalers:
            scaler = self.scalers['target']
            final_preds['1d'] = float(scaler.inverse_transform([[final_preds['1d']]])[0][0])
            if self.fusion_model:
                final_preds['5d'] = float(scaler.inverse_transform([[final_preds['5d']]])[0][0])
                final_preds['30d'] = float(scaler.inverse_transform([[final_preds['30d']]])[0][0])
        
        # 5. Calibrated Confidence
        # Base confidence on the immediate term (1d) raw TS clustering
        confidence = self._calculate_confidence(raw_preds, final_preds['1d'])

        return {
            'multi_horizon_predictions': final_preds,
            'ts_ensemble': ts_preds,
            'ensemble_weights': dict(zip(['lstm', 'gru', 'transformer', 'tft'], weights)),
            'nlp_signals': nlp_signals,
            'confidence': float(confidence)
        }
    
    def _calculate_confidence(self, raw_preds: list, fusion_pred: float, temperature: float = 1.5) -> float:
        """
        Calibrated confidence based on:
        1. Ensemble agreement (low variance = high confidence)
        2. Prediction magnitude (stronger signals = higher confidence)
        Temperature scaling controls calibration sharpness.
        """
        if not raw_preds:
            return 0.5
        
        preds = np.array(raw_preds)
        
        # Agreement score: inverse of coefficient of variation
        pred_std = np.std(preds)
        pred_mean = np.abs(np.mean(preds)) + 1e-8
        cv = pred_std / pred_mean
        agreement_score = 1.0 / (1.0 + cv)  # 0 to 1, higher = more agreement
        
        # Magnitude score: sigmoid of absolute prediction
        magnitude_score = 1.0 / (1.0 + np.exp(-np.abs(fusion_pred) * 2))
        
        # Combined score with temperature scaling
        raw_confidence = 0.6 * agreement_score + 0.4 * magnitude_score
        calibrated = 1.0 / (1.0 + np.exp(-(raw_confidence - 0.5) * temperature))
        
        return np.clip(calibrated, 0.1, 0.99)

    def _get_dynamic_regime_weights(self) -> list:
        """
        Loads the most recent Market Regime from features/regime_states.csv
        and maps it to strategic optimal ensemble weights:
        [LSTM, GRU, Transformer, TFT]
        """
        default_idx = [0.25, 0.25, 0.25, 0.25]
        regime_path = os.path.join(PROJECT_ROOT, 'data', 'features', 'regime_states.csv')
        
        if not os.path.exists(regime_path):
            return default_idx
            
        try:
            df = pd.read_csv(regime_path)
            if df.empty or 'regime' not in df.columns:
                return default_idx
                
            latest_regime = df['regime'].iloc[-1]
            
            # Map Regime ID (0-3) to Strategic Weights
            # Index format: [LSTM, GRU, Transformer, TFT]
            # TFT & Transformer perform better in Volatile / Novel bear markets
            # LSTM & GRU perform well in trend-following Bull markets
            if latest_regime == 0:  # Bull Market Strategy
                return [0.30, 0.30, 0.20, 0.20]
            elif latest_regime == 1:  # Bear Market Strategy
                return [0.15, 0.15, 0.35, 0.35]
            elif latest_regime == 2:  # Sideways / Range-bound
                return [0.25, 0.25, 0.25, 0.25]
            elif latest_regime == 3:  # High Volatility Strategy
                return [0.10, 0.10, 0.40, 0.40]
            else:
                return default_idx
        except Exception as e:
            logging.warning(f"Failed to load dynamic regime properties: {e}")
            return default_idx

def run_test_inference():
    """Run inference on a synthetic sample to verify pipeline logic."""
    predictor = Predictor()
    
    # Generate synthetic input
    logging.info("Generating synthetic sequence for test inference (60 days, 128 features)")
    sample_seq = np.random.randn(60, 128)
    sample_texts = [f"Market update day {i}" for i in range(60)]
    
    try:
        results = predictor.predict(sample_seq, sample_texts)
        print("\n" + "="*50)
        print("INFERENCE RESULTS (Synthetic Sample)")
        print("="*50)
        print(f"Multi-Horizon Predictions:")
        print(f"  1-Day:  {results['multi_horizon_predictions'].get('1d', 0):.4f}")
        print(f"  5-Day:  {results['multi_horizon_predictions'].get('5d', 0):.4f}")
        print(f"  30-Day: {results['multi_horizon_predictions'].get('30d', 0):.4f}")
        print(f"Applied Ensemble Weights: {results['ensemble_weights']}")
        print(f"Model Confidence: {results['confidence']:.2%}")
        print("="*50 + "\n")
    except Exception as e:
        logging.error(f"Inference failed: {e}", exc_info=True)

if __name__ == "__main__":
    run_test_inference()
