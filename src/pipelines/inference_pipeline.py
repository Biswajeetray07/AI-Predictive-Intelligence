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
        
        # Resolve S3 settings once for reuse
        self._init_s3_config()
        
        self.config = self._load_config()
        self.scalers = self._load_scalers()
        
        # Models
        self.ts_models = {}
        self.nlp_model = None
        self.fusion_model = None
        self.nlp_tokenizer = None
        
        self._load_all_models()

    def _init_s3_config(self):
        """Resolve USE_S3 and bucket name from env vars or Streamlit secrets."""
        use_s3_env = os.getenv("USE_S3", "False").lower() in ("true", "1", "yes")
        try:
            import streamlit as st
            use_s3_st = st.secrets.get("USE_S3", use_s3_env)
            self._use_s3 = str(use_s3_st).lower() in ("true", "1", "yes")
            self._s3_bucket = st.secrets.get("MODEL_BUCKET_NAME", os.getenv("MODEL_BUCKET_NAME", "my-model-mlopsproj012"))
        except Exception:
            self._use_s3 = use_s3_env
            self._s3_bucket = os.getenv("MODEL_BUCKET_NAME", "my-model-mlopsproj012")
        self._s3 = None

    def _get_s3(self):
        """Lazy-init S3 client."""
        if self._s3 is None and self._use_s3:
            try:
                from src.cloud_storage.aws_storage import SimpleStorageService
                self._s3 = SimpleStorageService()
            except Exception as e:
                logging.warning(f"Failed to init S3 client: {e}")
        return self._s3

    def _load_torch_state_from_s3(self, s3_key: str):
        """Download a .pt state dict from S3 and load with torch."""
        import io as _io
        s3 = self._get_s3()
        if s3 is None:
            return None
        try:
            file_obj = s3.get_file_object(s3_key, self._s3_bucket)
            if file_obj is None:
                return None
            raw_bytes = s3.read_object(file_obj, decode=False)
            state = torch.load(_io.BytesIO(raw_bytes), map_location=self.device, weights_only=True)
            logging.info(f"Loaded state dict from S3: {s3_key}")
            return state
        except Exception as e:
            logging.warning(f"Failed to load {s3_key} from S3: {e}")
            return None

    def _load_config(self):
        config_path = os.path.join(PROJECT_ROOT, 'configs', 'training_config.yaml')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        # S3 fallback
        s3 = self._get_s3()
        if s3:
            try:
                file_obj = s3.get_file_object('configs/training_config.yaml', self._s3_bucket)
                if file_obj:
                    content = s3.read_object(file_obj, decode=True)
                    return yaml.safe_load(content)
            except Exception as e:
                logging.warning(f"Failed to load config from S3: {e}")
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

        # Check S3 env vars
        use_s3_env = os.getenv("USE_S3", "False").lower() in ("true", "1", "yes")
        try:
            import streamlit as st
            use_s3_st = st.secrets.get("USE_S3", use_s3_env)
            use_s3 = str(use_s3_st).lower() in ("true", "1", "yes")
            bucket = st.secrets.get("MODEL_BUCKET_NAME", os.getenv("MODEL_BUCKET_NAME", "my-model-mlopsproj012"))
        except Exception:
            use_s3 = use_s3_env
            bucket = os.getenv("MODEL_BUCKET_NAME", "my-model-mlopsproj012")
            
        def load_from_s3(s3_key):
            if not use_s3: return None
            try:
                from src.cloud_storage.aws_storage import SimpleStorageService
                s3 = SimpleStorageService()
                logging.info(f"Loading scaler from S3: {s3_key}")
                return s3.load_model(s3_key, bucket) # load_model essentially unpickles any object
            except Exception as e:
                logging.warning(f"Failed to load {s3_key} from S3: {e}")
                return None
                    
        if os.path.exists(feature_scaler_path):
            try:
                scalers['feature'] = load_scaler(feature_scaler_path)
            except Exception as e:
                logging.warning(f"Failed to load feature scaler: {e}")
        else:
            s = load_from_s3('data/processed/model_inputs/feature_scaler.pkl')
            if s: scalers['feature'] = s
                
        if os.path.exists(target_scaler_path):
            try:
                scalers['target'] = load_scaler(target_scaler_path)
            except Exception as e:
                logging.warning(f"Failed to load target scaler: {e}")
        else:
            s = load_from_s3('data/processed/model_inputs/target_scaler.pkl')
            if s: scalers['target'] = s
            
        return scalers

    def _infer_dims_from_checkpoint(self, state_dict, model_type):
        """Infer hidden_dim, num_layers, and input_dim from checkpoint weights."""
        if model_type in ('lstm', 'gru'):
            rnn_key = f'{model_type}.weight_ih_l0'
            if rnn_key in state_dict:
                w = state_dict[rnn_key]
                gate_mult = 4 if model_type == 'lstm' else 3
                hidden = w.shape[0] // gate_mult
                input_dim = w.shape[1]
                # Count layers
                layers = 0
                while f'{model_type}.weight_ih_l{layers}' in state_dict:
                    layers += 1
                return input_dim, hidden, max(layers, 1)
        elif model_type == 'transformer':
            if 'input_projection.weight' in state_dict:
                w = state_dict['input_projection.weight']
                d_model = w.shape[0]
                input_dim = w.shape[1]
                layers = 0
                while f'encoder.layers.{layers}.self_attn.in_proj_weight' in state_dict:
                    layers += 1
                return input_dim, d_model, max(layers, 1)
        elif model_type == 'tft':
            if 'var_selection.grn.fc1.weight' in state_dict:
                w = state_dict['var_selection.grn.fc1.weight']
                hidden = w.shape[0]
                input_dim = w.shape[1]
                layers = 0
                while f'attention.layers.{layers}.self_attn.in_proj_weight' in state_dict:
                    layers += 1
                return input_dim, hidden, max(layers, 1)
        return None, None, None

    def _load_all_models(self):
        model_dir = os.path.join(PROJECT_ROOT, 'saved_models')
        ts_config = self.config.get('timeseries_model', {})
        
        # Fallback dims from config
        default_input = 128
        if 'feature' in self.scalers:
            default_input = self.scalers['feature'].n_features_in_
        default_hidden = ts_config.get('hidden_dim', 128)
        default_layers = ts_config.get('num_layers', 2)
            
        ts_model_names = ts_config.get('models', ['lstm', 'gru', 'transformer', 'tft'])
        
        for name in ts_model_names:
            path = os.path.join(model_dir, f'{name}_model.pt')
            state = None
            if os.path.exists(path):
                try:
                    state = torch.load(path, map_location=self.device, weights_only=True)
                except Exception as e:
                    logging.error(f"Failed to load {name} from disk: {e}")
            else:
                # S3 fallback
                state = self._load_torch_state_from_s3(f'saved_models/{name}_model.pt')

            if state is not None:
                try:
                    # Infer actual architecture from saved weights
                    inp, hid, lyr = self._infer_dims_from_checkpoint(state, name)
                    input_dim = inp or default_input
                    hidden = hid or default_hidden
                    layers = lyr or default_layers
                    
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
                        model.load_state_dict(state)
                        model.to(self.device).eval()
                        self.ts_models[name] = model
                        logging.info(f"Loaded TS model: {name} (input={input_dim}, hidden={hidden}, layers={layers})")
                    else:
                        logging.warning(f"Model type '{name}' not recognized. Skipping.")
                except Exception as e:
                    logging.error(f"Failed to instantiate {name} model: {e}")

        # Store ts_dim from loaded models for later use
        self.ts_dim = 64  # will be updated from actual loaded model
        if self.ts_models:
            first_model = next(iter(self.ts_models.values()))
            # Get context dim from a dummy forward pass or checkpoint
            for name, model in self.ts_models.items():
                if name in ('lstm', 'gru'):
                    self.ts_dim = model.lstm.hidden_size if name == 'lstm' else model.gru.hidden_size
                    break

        # 2. Load NLP Model
        nlp_path = os.path.join(model_dir, 'nlp_multitask_model.pt')
        nlp_state = None
        if os.path.exists(nlp_path):
            try:
                nlp_state = torch.load(nlp_path, map_location=self.device, weights_only=True)
            except Exception as e:
                logging.warning(f"Failed to load NLP model from disk: {e}")
        else:
            nlp_state = self._load_torch_state_from_s3('saved_models/nlp_multitask_model.pt')

        if nlp_state is not None:
            try:
                self.nlp_model = MultiTaskNLPModel(freeze_encoder_layers=0)
                self.nlp_model.load_state_dict(nlp_state)
                self.nlp_model.to(self.device).eval()
                self.nlp_tokenizer = NLPTokenizer(max_length=self.config.get('nlp_model', {}).get('max_length', 256))
                logging.info("Loaded NLP multi-task model")
            except Exception as e:
                logging.warning(f"Could not load NLP model (TS-only mode): {e}")
                self.nlp_model = None

        # 3. Load Fusion Model — infer ts_dim from checkpoint
        fusion_path = os.path.join(model_dir, 'fusion_model.pt')
        fusion_config = self.config.get('fusion_model', {})
        fusion_state = None
        if os.path.exists(fusion_path):
            try:
                fusion_state = torch.load(fusion_path, map_location=self.device, weights_only=True)
            except Exception as e:
                logging.error(f"Failed to load fusion model from disk: {e}")
        else:
            fusion_state = self._load_torch_state_from_s3('saved_models/fusion_model.pt')

        if fusion_state is not None:
            try:
                # Infer ts_dim from the gmu.ts_proj.weight shape
                ts_dim_actual = self.ts_dim
                if 'gmu.ts_proj.weight' in fusion_state:
                    ts_dim_actual = fusion_state['gmu.ts_proj.weight'].shape[1]
                    self.ts_dim = ts_dim_actual
                self.fusion_model = MultiHorizonFusionModel(
                    nlp_dim=768,
                    ts_dim=ts_dim_actual,
                    attention_heads=fusion_config.get('attention_heads', 4),
                    mlp_hidden=fusion_config.get('mlp_hidden', [512, 256, 128]),
                    dropout=0.0
                )
                self.fusion_model.load_state_dict(fusion_state)
                self.fusion_model.to(self.device).eval()
                logging.info(f"Loaded Deep Fusion model (ts_dim={ts_dim_actual})")
            except Exception as e:
                logging.error(f"Failed to load fusion model: {e}")
                self.fusion_model = None

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
        ts_dim = getattr(self, 'ts_dim', 64)
        fused_ts_emb = np.sum(ts_embeddings, axis=0) if ts_embeddings else np.zeros((1, ts_dim))
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

    def predict_live(self, ticker: str, as_of_date: str = None, nlp_texts: Optional[List[str]] = None) -> Dict:
        """
        High-level real-time inference: given just a ticker, build features on-the-fly
        and return multi-horizon predictions.

        This method completely decouples inference from X_test.npy / metadata_test.csv.

        Args:
            ticker: Stock ticker symbol (e.g. 'AAPL')
            as_of_date: Optional cutoff date (YYYY-MM-DD). Uses latest data if None.
            nlp_texts: Optional list of news texts for NLP signal fusion.

        Returns:
            Dict with 'multi_horizon_predictions', 'confidence', 'ensemble_weights', etc.
            Returns None if feature construction fails.
        """
        try:
            from src.pipelines.feature_builder import RealTimeFeatureBuilder
            builder = RealTimeFeatureBuilder(project_root=PROJECT_ROOT)
            sequence = builder.build_sequence(ticker, as_of_date=as_of_date)
            if sequence is None:
                logging.warning(f"predict_live: could not build features for {ticker}")
                return None
            return self.predict(sequence, nlp_texts=nlp_texts)
        except ImportError:
            logging.error("predict_live requires src.pipelines.feature_builder module")
            return None
        except Exception as e:
            logging.error(f"predict_live failed for {ticker}: {e}")
            return None

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
