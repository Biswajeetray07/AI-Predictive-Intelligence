import os
import sys
import glob

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv('.env')

from src.cloud_storage.aws_storage import SimpleStorageService

def main():
    s3 = SimpleStorageService()
    bucket_name = os.getenv("MODEL_BUCKET_NAME", "my-model-mlopsproj012")
    
    print(f"Starting comprehensive bulk upload to S3 Bucket: {bucket_name}...\n")
    
    # Define directories critical for Real-Time Inference & App functionality
    sync_targets = [
        "data/processed/**/*.parquet",
        "data/processed/**/*.csv",
        "data/features/**/*.parquet",
        "data/features/**/*.csv",
        "data/features/**/*.npy",
        "data/processed/**/X_test.npy",      # Only the test tensor (889 MB)
        "data/processed/**/metadata_test.csv", # Feature mappings
        "saved_models/**/*.pt",
        "saved_models/**/*.json",      # e.g., nlp_label_quality.json
        "saved_models/**/*.pkl",
    ]
    
    total_files_uploaded = 0
    for target_pattern in sync_targets:
        files = glob.glob(target_pattern, recursive=True)
        if not files:
            continue
            
        print(f"--- Syncing target: {target_pattern} ({len(files)} files) ---")
        for i, file_path in enumerate(files):
            # Standardize s3_key format identically to how utils.py requests it
            s3_key = file_path.replace("\\", "/")
            
            # Print occasionally to avoid spam
            if len(files) > 20 and i % max(1, (len(files) // 10)) == 0:
                print(f"   [{i}/{len(files)}] Uploading {s3_key}...")
            elif len(files) <= 20:
                print(f"   Uploading {s3_key}...")
                
            try:
                # Basic check to avoid re-uploading if file exists to save you time.
                if not s3.s3_key_path_available(bucket_name, s3_key):
                    s3.upload_file(file_path, s3_key, bucket_name)
                    total_files_uploaded += 1
            except Exception as e:
                print(f"   ❌ Failed to upload {file_path}: {e}")

    print(f"\n✅ Sync Completed! Uploaded {total_files_uploaded} missing components to S3.")
    print("Zero raw uncleaned files were uploaded to save AWS costs.")

if __name__ == "__main__":
    main()
