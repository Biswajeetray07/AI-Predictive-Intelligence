import numpy as np
import pandas as pd
import os
import glob

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
FEATURES_DIR = os.path.join(PROJECT_ROOT, 'data', 'features')
os.makedirs(FEATURES_DIR, exist_ok=True)

def main():
    print("❌ CRITICAL: Mock NLP generation is strictly disabled to enforce REAL data usage.")
    print("Please fix the root cause in the NLP training pipeline to generate actual embeddings.")
    raise NotImplementedError("Mock data disabled as per user request.")

if __name__ == "__main__":
    main()
