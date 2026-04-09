import os
import sys
import glob
import json
import datetime
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv('.env')

from src.cloud_storage.aws_storage import SimpleStorageService

def _scan_directory(path, name):
    files = glob.glob(os.path.join(path, '**', '*.*'), recursive=True)
    data_files = [f for f in files if f.endswith(('.csv', '.parquet', '.json'))]
    if not data_files:
        return None
    total_size = sum(os.path.getsize(f) for f in data_files if os.path.exists(f))
    total_records = 0
    latest_modified = None
    for f in data_files:
        try:
            mtime = datetime.datetime.fromtimestamp(os.path.getmtime(f))
            if latest_modified is None or mtime > latest_modified:
                latest_modified = mtime
        except Exception:
            pass
        try:
            if f.endswith('.csv'):
                with open(f, 'r') as fh:
                    total_records += sum(1 for _ in fh) - 1
            elif f.endswith('.parquet'):
                total_records += len(pd.read_parquet(f))
        except Exception:
            pass
    return {
        'name': name,
        'directory': os.path.basename(path),
        'files': len(data_files),
        'records': total_records,
        'size_mb': round(total_size / (1024 * 1024), 2),
        'last_updated': latest_modified.strftime('%Y-%m-%d %H:%M') if latest_modified else 'N/A',
        'status': 'Active' if data_files else 'No Data',
    }

def generate_manifest():
    print("Generating comprehensive data manifest...")
    manifest = {'sources': [], 'kpis': {}}
    raw_dir = os.path.join(PROJECT_ROOT, 'data', 'raw')
    if os.path.exists(raw_dir):
        for domain in sorted(os.listdir(raw_dir)):
            domain_path = os.path.join(raw_dir, domain)
            if not os.path.isdir(domain_path): continue
            subdirs = [s for s in os.listdir(domain_path) if os.path.isdir(os.path.join(domain_path, s))]
            if subdirs:
                for sub in subdirs:
                    info = _scan_directory(os.path.join(domain_path, sub), f"{domain.replace('_', ' ').title()} / {sub.replace('_', ' ').title()}")
                    if info: manifest['sources'].append(info)
            else:
                info = _scan_directory(domain_path, domain.replace('_', ' ').title())
                if info: manifest['sources'].append(info)
                
    manifest['kpis']['total_records'] = sum(s['records'] for s in manifest['sources'])
    manifest['kpis']['data_sources'] = len(manifest['sources'])
    
    # Estimate feature files
    feat_dir = os.path.join(PROJECT_ROOT, 'data', 'features')
    manifest['kpis']['features_generated'] = len(glob.glob(os.path.join(feat_dir, '*.parquet'))) + len(glob.glob(os.path.join(feat_dir, '*.csv'))) if os.path.exists(feat_dir) else 0
    
    # Estimate models
    model_dir = os.path.join(PROJECT_ROOT, 'saved_models')
    manifest['kpis']['models_saved'] = len(glob.glob(os.path.join(model_dir, '**', '*.pt'), recursive=True)) if os.path.exists(model_dir) else 0
    
    # Active APIs
    env_path = os.path.join(PROJECT_ROOT, '.env')
    apis = 0
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and ('KEY' in line or 'TOKEN' in line or 'API' in line) and '=' in line:
                    apis += 1
    manifest['kpis']['active_apis'] = apis
    
    manifest_path = os.path.join(PROJECT_ROOT, 'data_manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=4)
    print(f"Manifest created at {manifest_path}")
    return manifest_path

def main():
    s3 = SimpleStorageService()
    bucket_name = os.getenv("MODEL_BUCKET_NAME", "my-model-mlopsproj012")
    
    manifest_path = generate_manifest()
    
    print(f"\nStarting comprehensive bulk upload to S3 Bucket: {bucket_name}...\n")
    
    # Define directories critical for Real-Time Inference & App functionality + raw
    sync_targets = [
        "data_manifest.json",
        "data/raw/**/*.parquet",
        "data/raw/**/*.csv",
        "data/raw/**/*.json",
        "data/processed/**/*.parquet",
        "data/processed/**/*.csv",
        "data/processed/**/feature_scaler.pkl",  # Critical for RealTimeFeatureBuilder
        "data/processed/**/target_scaler.pkl",   # Critical for inverse-scaling predictions
        "data/features/**/*.parquet",
        "data/features/**/*.csv",
        "data/features/**/*.npy",
        "data/processed/**/X_test.npy",      # Only the test tensor (889 MB)
        "data/processed/**/y_test.npy",      # Test labels
        "data/processed/**/y_multi_test.npy", # Multi-horizon test labels
        "data/processed/**/metadata_test.csv", # Feature mappings
        "saved_models/**/*.pt",
        "saved_models/**/*.json",      # e.g., nlp_label_quality.json
        "saved_models/**/*.pkl",
    ]
    
    total_files_uploaded = 0
    for target_pattern in sync_targets:
        search_path = os.path.join(PROJECT_ROOT, target_pattern) if not target_pattern.startswith("data_manifest") else manifest_path
        files = glob.glob(search_path, recursive=True) if target_pattern != "data_manifest.json" else [manifest_path]
        if not files:
            continue
            
        print(f"--- Syncing target: {target_pattern} ({len(files)} files) ---")
        for i, file_path in enumerate(files):
            # Standardize s3_key format identically to how utils.py requests it
            s3_key = file_path.replace("\\", "/").split(PROJECT_ROOT.replace("\\", "/") + "/")[-1]
            
            # Print occasionally to avoid spam
            if len(files) > 20 and i % max(1, (len(files) // 10)) == 0:
                print(f"   [{i}/{len(files)}] Uploading {s3_key}...")
            elif len(files) <= 20:
                print(f"   Uploading {s3_key}...")
                
            try:
                # Basic check to avoid re-uploading if file exists unless it's manifest
                if "data_manifest" in s3_key or not s3.s3_key_path_available(bucket_name, s3_key):
                    s3.upload_file(file_path, s3_key, bucket_name)
                    total_files_uploaded += 1
            except Exception as e:
                print(f"   ❌ Failed to upload {file_path}: {e}")

    print(f"\n✅ Sync Completed! Uploaded {total_files_uploaded} items to S3.")

if __name__ == "__main__":
    main()
