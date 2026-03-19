"""Phase 1-9: Full Pipeline Validation Script (no training, no downloads)."""
import importlib
import sys
import os
import traceback
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
results = {"passed": [], "failed": [], "warnings": []}

def test_import(mod_name):
    try:
        importlib.import_module(mod_name)
        results["passed"].append(mod_name)
        return True
    except Exception as e:
        results["failed"].append((mod_name, f"{type(e).__name__}: {e}"))
        return False

# ═══════════════════════════════════════════════════════════════════
# PHASE 1: IMPORT VALIDATION (core modules only — skip collectors)
# ═══════════════════════════════════════════════════════════════════
print("=" * 70)
print("PHASE 1: IMPORT VALIDATION")
print("=" * 70)

CORE_MODULES = [
    "src",
    "src.data_collection",
    "src.data_processing",
    "src.feature_engineering",
    "src.models",
    "src.training",
    "src.pipelines",
    "src.evaluation",
    "src.utils",
    "src.utils.pipeline_utils",
    "src.utils.logging_utils",
    "src.models.timeseries.lstm",
    "src.models.timeseries.gru",
    "src.models.timeseries.transformer",
    "src.models.timeseries.tft",
    "src.models.nlp.model",
    "src.models.nlp.tokenizer",
    "src.models.fusion.fusion",
    "src.training.timeseries.train",
    "src.training.nlp.train",
    "src.training.fusion.train",
    "src.training.optimization.optuna_ts",
    "src.training.optimization.optuna_nlp",
    "src.training.optimization.optuna_fusion",
    "src.training.optimization.run_hyperopt",
    "src.training.optimization.walk_forward",
    "src.evaluation.backtest",
    "src.evaluation.generate_report",
    "src.evaluation.monitoring.retraining_trigger",
    "src.pipelines.training_pipeline",
    "src.pipelines.train_optimized_pipeline",
    "src.pipelines.inference_pipeline",
    "src.pipelines.run_data_pipeline",
    "src.data_processing.financial_processing",
    "src.data_processing.news_processing",
    "src.data_processing.social_processing",
    "src.data_processing.weather_processing",
    "src.data_processing.macro_processing",
    "src.data_processing.external_processing",
    "src.data_processing.merge_datasets",
    "src.data_processing.build_sequences",
    "src.data_processing.process_kaggle_data",
    "src.feature_engineering.feature_generator",
    "src.feature_engineering.feature_selection",
    "src.feature_engineering.build_features",
    "src.feature_engineering.regime_detection.regime_features",
    "src.feature_engineering.regime_detection.regime_detector",
]

for mod in CORE_MODULES:
    ok = test_import(mod)
    status = "OK" if ok else "FAIL"
    print(f"  [{status}]  {mod}")

print(f"\n  Phase 1 Result: {len(results['passed'])} passed, {len(results['failed'])} failed")
if results["failed"]:
    print("\n  FAILED:")
    for mod, err in results["failed"]:
        print(f"    {mod}: {err}")

# ═══════════════════════════════════════════════════════════════════
# PHASE 3: CONFIG VALIDATION
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("PHASE 3: CONFIGURATION VALIDATION")
print("=" * 70)

import yaml

config_path = os.path.join(PROJECT_ROOT, "configs", "training_config.yaml")
config = None
if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print(f"  [OK]  Loaded {config_path}")
    
    required_sections = ["nlp_model", "timeseries_model", "fusion_model", "paths"]
    for sec in required_sections:
        if sec in config:
            print(f"  [OK]  Config section '{sec}' present")
        else:
            print(f"  [FAIL] Config section '{sec}' MISSING")
            results["failed"].append((f"config:{sec}", "Missing section"))
    
    # Validate paths
    if "paths" in config:
        for key, path in config["paths"].items():
            abs_path = os.path.join(PROJECT_ROOT, path)
            exists = os.path.exists(abs_path)
            status = "OK" if exists else "WARN"
            print(f"  [{status}]  paths.{key}: {path}")
            if not exists:
                results["warnings"].append(f"Path doesn't exist: {path}")
else:
    print(f"  [FAIL] Config file not found: {config_path}")
    results["failed"].append(("training_config.yaml", "File not found"))

# ═══════════════════════════════════════════════════════════════════
# PHASE 5: MODEL FORWARD PASS TEST
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("PHASE 5: MODEL FORWARD PASS TEST")
print("=" * 70)

import torch

INPUT_DIM = 128
HIDDEN_DIM = 128
SEQ_LEN = 60
BATCH = 2
NUM_LAYERS = 2
DROPOUT = 0.2

# Test Time-Series Models
ts_models_to_test = {}
try:
    from src.models.timeseries.lstm import LSTMForecaster
    ts_models_to_test["LSTM"] = LSTMForecaster(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
except Exception as e:
    print(f"  [FAIL] LSTM instantiation: {e}")
    results["failed"].append(("LSTM", str(e)))

try:
    from src.models.timeseries.gru import GRUForecaster
    ts_models_to_test["GRU"] = GRUForecaster(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
except Exception as e:
    print(f"  [FAIL] GRU instantiation: {e}")
    results["failed"].append(("GRU", str(e)))

try:
    from src.models.timeseries.transformer import TransformerForecaster
    ts_models_to_test["Transformer"] = TransformerForecaster(INPUT_DIM, d_model=HIDDEN_DIM, nhead=4, num_layers=NUM_LAYERS, dropout=DROPOUT)
except Exception as e:
    print(f"  [FAIL] Transformer instantiation: {e}")
    results["failed"].append(("Transformer", str(e)))

try:
    from src.models.timeseries.tft import TFTForecaster
    ts_models_to_test["TFT"] = TFTForecaster(INPUT_DIM, HIDDEN_DIM, num_heads=4, num_layers=NUM_LAYERS, dropout=DROPOUT)
except Exception as e:
    print(f"  [FAIL] TFT instantiation: {e}")
    results["failed"].append(("TFT", str(e)))

x_ts = torch.randn(BATCH, SEQ_LEN, INPUT_DIM)
ts_outputs = {}

for name, model in ts_models_to_test.items():
    try:
        model.eval()
        with torch.no_grad():
            pred, context = model(x_ts)
        print(f"  [OK]  {name}: pred={list(pred.shape)}, context={list(context.shape)}")
        ts_outputs[name] = {"pred_shape": list(pred.shape), "context_shape": list(context.shape)}
        results["passed"].append(f"forward:{name}")
    except Exception as e:
        print(f"  [FAIL] {name} forward pass: {e}")
        results["failed"].append((f"forward:{name}", str(e)))

# Test NLP Model
print()
try:
    from src.models.nlp.model import MultiTaskNLPModel
    nlp_model = MultiTaskNLPModel(freeze_encoder_layers=6)
    nlp_model.eval()
    
    input_ids = torch.randint(0, 1000, (BATCH, 64))
    attention_mask = torch.ones(BATCH, 64, dtype=torch.long)
    source_ids = torch.zeros(BATCH, dtype=torch.long)
    days = torch.zeros(BATCH, dtype=torch.long)
    months = torch.zeros(BATCH, dtype=torch.long)
    
    with torch.no_grad():
        out = nlp_model(input_ids, attention_mask, source_ids, days, months)
    
    print(f"  [OK]  NLP Model: embedding={list(out['embedding'].shape)}, sentiment={list(out['sentiment'].shape)}")
    results["passed"].append("forward:NLP")
except Exception as e:
    print(f"  [FAIL] NLP forward pass: {e}")
    results["failed"].append(("forward:NLP", str(e)))

# ═══════════════════════════════════════════════════════════════════
# PHASE 6: ENSEMBLE CONNECTIVITY CHECK
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("PHASE 6: ENSEMBLE CONNECTIVITY CHECK")
print("=" * 70)

if ts_outputs:
    context_dims = {name: out["context_shape"][-1] for name, out in ts_outputs.items()}
    unique_dims = set(context_dims.values())
    
    if len(unique_dims) == 1:
        dim = list(unique_dims)[0]
        print(f"  [OK]  All TS models output context dim = {dim}")
        results["passed"].append("ensemble:dim_check")
    else:
        print(f"  [FAIL] Dimension mismatch: {context_dims}")
        results["failed"].append(("ensemble:dim_check", f"Mismatch: {context_dims}"))

# ═══════════════════════════════════════════════════════════════════
# PHASE 7: FUSION MODEL VALIDATION
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("PHASE 7: FUSION MODEL VALIDATION")
print("=" * 70)

try:
    from src.models.fusion.fusion import DeepFusionModel
    
    fusion_config = config.get("fusion_model", {}) if config else {}
    fusion_model = DeepFusionModel(
        nlp_dim=768,
        ts_dim=HIDDEN_DIM,
        attention_heads=fusion_config.get("attention_heads", 4),
        mlp_hidden=fusion_config.get("mlp_hidden", [512, 256, 128]),
        dropout=fusion_config.get("dropout", 0.2)
    )
    fusion_model.eval()
    
    nlp_emb = torch.randn(BATCH, SEQ_LEN, 768)
    ts_emb = torch.randn(BATCH, HIDDEN_DIM)
    
    with torch.no_grad():
        fusion_out = fusion_model(nlp_emb, ts_emb)
    
    print(f"  [OK]  Fusion Model: input_nlp={list(nlp_emb.shape)}, input_ts={list(ts_emb.shape)} -> output={list(fusion_out.shape)}")
    results["passed"].append("forward:Fusion")
except Exception as e:
    print(f"  [FAIL] Fusion forward pass: {e}")
    traceback.print_exc()
    results["failed"].append(("forward:Fusion", str(e)))

# ═══════════════════════════════════════════════════════════════════
# PHASE 9: FILE PATH AND ASSET CHECK
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("PHASE 9: FILE PATH AND ASSET CHECK")
print("=" * 70)

dirs_to_check = [
    "data/raw",
    "data/processed",
    "data/features",
    "data/processed/model_inputs",
    "saved_models",
    "configs",
    "scripts",
    "src/models",
    "src/training",
    "src/pipelines",
]

for d in dirs_to_check:
    abs_d = os.path.join(PROJECT_ROOT, d)
    exists = os.path.exists(abs_d)
    status = "OK" if exists else "MISSING"
    print(f"  [{status}]  {d}")
    if not exists:
        results["warnings"].append(f"Directory missing: {d}")

# ═══════════════════════════════════════════════════════════════════
# PHASE 10: SYSTEM HEALTH REPORT
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("PHASE 10: SYSTEM HEALTH REPORT")
print("=" * 70)

total_tests = len(results["passed"]) + len(results["failed"])
pass_rate = len(results["passed"]) / total_tests * 100 if total_tests > 0 else 0

print(f"\n  Total Tests:    {total_tests}")
print(f"  Passed:         {len(results['passed'])}")
print(f"  Failed:         {len(results['failed'])}")
print(f"  Warnings:       {len(results['warnings'])}")
print(f"  Pass Rate:      {pass_rate:.1f}%")

if results["failed"]:
    print(f"\n  FAILURES:")
    for item in results["failed"]:
        if isinstance(item, tuple):
            print(f"    {item[0]}: {item[1]}")
        else:
            print(f"    {item}")

if results["warnings"]:
    print(f"\n  WARNINGS:")
    for w in results["warnings"]:
        print(f"    {w}")

health_score = int(pass_rate)
print(f"\n  PIPELINE HEALTH SCORE: {health_score}/100")
if health_score >= 90:
    print("  STATUS: READY FOR TRAINING")
elif health_score >= 70:
    print("  STATUS: CONDITIONAL PASS — fix failures before training")
else:
    print("  STATUS: NOT READY — critical failures detected")
