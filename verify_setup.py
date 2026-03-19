#!/usr/bin/env python3
"""
AI Predictive Intelligence - System Verification & Setup Script
==============================================================

This script verifies that all fixes have been applied and checks
whether the system is ready to use real data.

Usage:
    python verify_setup.py

Output:
    - ✓ Green checkmarks for fixed components
    - ✗ Red X for missing components
    - Yellow warnings for non-critical issues
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def check_imports():
    """Verify critical imports work"""
    print("\n" + "="*70)
    print("CHECKING IMPORTS")
    print("="*70)
    
    imports_to_check = [
        ("src.training.fusion.train", "load_fusion_data"),
        ("src.training.nlp.train", "main"),
        ("src.training.timeseries.train", "main"),
        ("src.pipelines.inference_pipeline", "Predictor"),
        ("src.api.app", "app"),
        ("src.data_collection.async_fetcher", "AsyncFetcher"),
    ]
    
    all_ok = True
    for module_name, item_name in imports_to_check:
        try:
            module = __import__(module_name, fromlist=[item_name])
            getattr(module, item_name)
            print(f"✓ {module_name}.{item_name}")
        except Exception as e:
            print(f"✗ {module_name}.{item_name}: {str(e)[:50]}")
            all_ok = False
    
    return all_ok

def check_directories():
    """Verify critical directories exist"""
    print("\n" + "="*70)
    print("CHECKING DIRECTORY STRUCTURE")
    print("="*70)
    
    dirs_to_check = [
        "data/raw",
        "data/processed",
        "data/features",
        "data/processed/model_inputs",
        "saved_models",
        "configs",
        "logs",
    ]
    
    all_ok = True
    for dir_path in dirs_to_check:
        full_path = PROJECT_ROOT / dir_path
        if full_path.exists():
            print(f"✓ {dir_path}/")
        else:
            print(f"✗ {dir_path}/ (will be created automatically)")
            os.makedirs(full_path, exist_ok=True)
            print(f"  → Created {dir_path}/")
    
    return all_ok

def check_embeddings():
    """Check if real embeddings exist"""
    print("\n" + "="*70)
    print("CHECKING FOR REAL EMBEDDINGS")
    print("="*70)
    
    embeddings_to_check = [
        ("data/features/nlp_embeddings.npy", "NLP Embeddings"),
        ("data/features/nlp_embedding_dates.csv", "NLP Metadata"),
        ("data/features/ts_embeddings.npy", "TS Embeddings"),
        ("data/processed/model_inputs/metadata_val.csv", "Validation Metadata"),
        ("data/processed/model_inputs/y_multi_val.npy", "Validation Targets"),
    ]
    
    real_data_ready = True
    for file_path, description in embeddings_to_check:
        full_path = PROJECT_ROOT / file_path
        if full_path.exists():
            size_mb = full_path.stat().st_size / (1024*1024)
            print(f"✓ {description}: {size_mb:.1f} MB")
        else:
            print(f"⚠ {description}: NOT FOUND")
            real_data_ready = False
    
    if not real_data_ready:
        print("\n⚠  Real embeddings not found. System will use synthetic data.")
        print("   To generate real embeddings:")
        print("   1. Run: python scripts/run_training_only.py")
        print("   2. Or:  python scripts/run_full_pipeline.py")
    else:
        print("\n✓ All real embeddings found! System will use real market data.")
    
    return real_data_ready

def check_configurations():
    """Verify configuration files exist"""
    print("\n" + "="*70)
    print("CHECKING CONFIGURATIONS")
    print("="*70)
    
    configs_to_check = [
        "configs/training_config.yaml",
        "configs/best_params.yaml",
        "configs/test_config.yaml",
    ]
    
    for config_path in configs_to_check:
        full_path = PROJECT_ROOT / config_path
        if full_path.exists():
            print(f"✓ {config_path}")
        else:
            print(f"⚠ {config_path} (not critical, defaults will be used)")
    
    return True

def test_real_data_loading():
    """Test if the real data loading function works"""
    print("\n" + "="*70)
    print("TESTING REAL DATA LOADING")
    print("="*70)
    
    try:
        from src.training.fusion.train import load_fusion_data
        
        print("Loading fusion data...")
        nlp, ts, targets = load_fusion_data(str(PROJECT_ROOT))
        
        print(f"✓ Successfully loaded data:")
        print(f"  - NLP embeddings: {nlp.shape}")
        print(f"  - TS features: {ts.shape}")
        print(f"  - Targets: {targets.shape}")
        
        # Check if it's real or synthetic
        import numpy as np
        if np.all(np.isfinite(nlp)):
            # This is a simple heuristic - real embeddings from DeBERTa
            # typically have specific statistical properties
            nlp_mean = np.mean(np.abs(nlp))
            nlp_std = np.std(nlp)
            print(f"\n  Data Statistics:")
            print(f"    - NLP embedding mean: {nlp_mean:.6f}")
            print(f"    - NLP embedding std: {nlp_std:.6f}")
            
            # Real embeddings from transformers typically have mean ~0 and std ~1-2
            if 0.5 < nlp_std < 3.0 and nlp_mean < 0.5:
                print("  → Appears to be REAL embedding data ✓")
            else:
                print("  → Appears to be SYNTHETIC data (using fallback)")
        
        return True
    except Exception as e:
        print(f"⚠ Error loading data: {str(e)}")
        return False

def generate_report():
    """Generate a summary report"""
    print("\n" + "="*70)
    print("SETUP VERIFICATION REPORT")
    print("="*70)
    
    timestamp = datetime.now().isoformat()
    
    report = {
        "timestamp": timestamp,
        "project_root": str(PROJECT_ROOT),
        "python_version": sys.version,
        "checks_performed": {
            "imports": check_imports(),
            "directories": check_directories(),
            "configurations": check_configurations(),
            "embeddings": check_embeddings(),
            "data_loading": test_real_data_loading(),
        }
    }
    
    all_checks = list(report["checks_performed"].values())
    passed = sum(all_checks)
    total = len(all_checks)
    
    print(f"\n{'='*70}")
    print(f"RESULTS: {passed}/{total} checks passed")
    print(f"{'='*70}\n")
    
    if passed == total:
        print("✓ System is READY for training with real or synthetic data")
        print("\nNext steps:")
        print("  1. Run: python scripts/run_training_only.py")
        print("  2. Monitor the training progress")
        print("  3. Check saved_models/ for trained models")
    else:
        print("⚠ Some checks failed. System is partially ready")
        if not report["checks_performed"]["embeddings"]:
            print("\nTo generate real embeddings:")
            print("  python scripts/run_training_only.py")
    
    # Save report
    report_path = PROJECT_ROOT / "verification_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to: {report_path}")
    
    return all_checks

def main():
    print("\n" + "╔" + "="*68 + "╗")
    print("║  AI PREDICTIVE INTELLIGENCE - SYSTEM VERIFICATION             ║")
    print("║" + " "*68 + "║")
    print(f"║  Project Root: {str(PROJECT_ROOT):<47}║")
    print("╚" + "="*68 + "╝")
    
    try:
        results = generate_report()
        
        if not all(results):
            print("\n⚠ Please fix the issues above and run this script again.")
            sys.exit(1)
        else:
            print("\n✓ All systems ready!")
            sys.exit(0)
    except Exception as e:
        print(f"\n✗ Fatal error during verification: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
