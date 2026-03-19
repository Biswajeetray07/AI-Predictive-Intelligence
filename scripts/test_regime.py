import traceback, sys, os
try:
    from src.feature_engineering.regime_detection.regime_detector import RegimeDetector
    print("detector OK")
except Exception:
    traceback.print_exc()
try:
    from src.feature_engineering.regime_detection.regime_features import generate_regime_features
    print("features OK")
except Exception:
    traceback.print_exc()
