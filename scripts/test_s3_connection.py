import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.cloud_storage.aws_storage import SimpleStorageService

def main():
    print("="*50)
    print(" AWS S3 Connection Test ".center(50))
    print("="*50)

    bucket_name = os.getenv("MODEL_BUCKET_NAME", "my-model-mlopsproj012")
    print(f"Target Bucket: {bucket_name}")
    print(f"USE_S3 Flag in Env: {os.getenv('USE_S3', 'Not Set')}\n")

    try:
        s3 = SimpleStorageService()
        
        # Test 1: Can we get the bucket?
        print("Testing bucket access...")
        bucket = s3.get_bucket(bucket_name)
        print(f"✅ Successfully accessed bucket '{bucket.name}'")
        
        # Test 2: List top-level files/folders
        print("\nListing root files (max 5):")
        files = s3.list_files("", bucket_name)[:5]
        if files:
            for f in files:
                print(f" - {f}")
        else:
            print(" (Bucket is empty or no files at root)")
            
        print("\n✅ All S3 connectivity tests passed!")
        
    except Exception as e:
        print(f"\n❌ S3 Connection Failed!")
        print(f"Error: {e}")
        print("\nPlease verify you have set the following environment variables:")
        print(" - AWS_ACCESS_KEY_ID")
        print(" - AWS_SECRET_ACCESS_KEY")
        print(" - AWS_REGION (default: us-east-1)")
        print("\nOr configured your ~/.aws/credentials file.")

if __name__ == "__main__":
    main()
