import boto3
from src.cloud_storage.aws_connection import S3Client
from io import StringIO
from typing import Union, List, Optional
import os, sys
import logging
from botocore.exceptions import ClientError
from pandas import DataFrame, read_csv, read_parquet
import pandas as pd
import pickle
import io

logger = logging.getLogger(__name__)

class SimpleStorageService:
    """
    A class for interacting with AWS S3 storage, providing methods for file management, 
    data uploads, and data retrieval in S3 buckets.
    """

    def __init__(self):
        """
        Initializes the SimpleStorageService instance with S3 resource and client
        from the S3Client class.
        """
        s3_client = S3Client()
        self.s3_resource = s3_client.s3_resource
        self.s3_client = s3_client.s3_client

    def s3_key_path_available(self, bucket_name, s3_key) -> bool:
        """
        Checks if a specified S3 key path (file path) is available in the specified bucket.
        """
        try:
            bucket = self.get_bucket(bucket_name)
            file_objects = [file_object for file_object in bucket.objects.filter(Prefix=s3_key)]
            return len(file_objects) > 0
        except Exception as e:
            logger.warning(f"Error checking key path in S3: {e}")
            return False

    @staticmethod
    def read_object(object_name: object, decode: bool = True, make_readable: bool = False) -> Union[StringIO, str, bytes]:
        """
        Reads the specified S3 object with optional decoding and formatting.
        """
        try:
            # Check if this is a binary read (non decoded)
            if not decode:
                return object_name.get()["Body"].read()

            func = lambda: object_name.get()["Body"].read().decode()
            conv_func = lambda: StringIO(func()) if make_readable else func()
            return conv_func()
        except Exception as e:
            logger.error(f"Error reading object from S3: {e}")
            raise

    def get_bucket(self, bucket_name: str):
        """
        Retrieves the S3 bucket object based on the provided bucket name.
        """
        try:
            bucket = self.s3_resource.Bucket(bucket_name)
            return bucket
        except Exception as e:
            logger.error(f"Error getting bucket {bucket_name}: {e}")
            raise

    def get_file_object(self, filename: str, bucket_name: str) -> Union[List[object], object, None]:
        """
        Retrieves the file object(s) from the specified bucket based on the filename.
        """
        try:
            bucket = self.get_bucket(bucket_name)
            file_objects = [file_object for file_object in bucket.objects.filter(Prefix=filename)]
            
            if not file_objects:
                return None
                
            func = lambda x: x[0] if len(x) == 1 else x
            file_objs = func(file_objects)
            return file_objs
        except Exception as e:
            logger.error(f"Error retrieving file {filename}: {e}")
            return None

    def load_model(self, model_name: str, bucket_name: str, model_dir: str = None) -> object:
        """
        Loads a serialized model from the specified S3 bucket.
        """
        try:
            model_file = model_dir + "/" + model_name if model_dir else model_name
            file_object = self.get_file_object(model_file, bucket_name)
            if not file_object:
                return None
                
            model_obj = self.read_object(file_object, decode=False)
            import joblib
            model = joblib.load(io.BytesIO(model_obj))
            return model
        except Exception as e:
            logger.error(f"Failed to load model {model_name} from S3: {e}")
            return None

    def upload_file(self, from_filename: str, to_filename: str, bucket_name: str, remove: bool = False):
        """
        Uploads a local file to the specified S3 bucket.
        """
        try:
            logger.info(f"Uploading {from_filename} to {to_filename} in {bucket_name}")
            self.s3_resource.meta.client.upload_file(from_filename, bucket_name, to_filename)

            if remove:
                os.remove(from_filename)
        except Exception as e:
            logger.error(f"Failed to upload file to S3: {e}")
            raise

    def read_csv(self, filename: str, bucket_name: str) -> Optional[DataFrame]:
        """
        Reads a CSV file from the specified S3 bucket and converts it to a DataFrame.
        """
        try:
            csv_obj = self.get_file_object(filename, bucket_name)
            if not csv_obj:
                return None
            content = self.read_object(csv_obj, make_readable=True)
            df = read_csv(content, na_values="na")
            return df
        except Exception as e:
            logger.warning(f"Failed to read CSV {filename} from S3: {e}")
            return None

    def read_parquet(self, filename: str, bucket_name: str) -> Optional[DataFrame]:
        """
        Reads a Parquet file from the specified S3 bucket and converts it to a DataFrame.
        Note: Needs fastparquet or pyarrow installed.
        """
        try:
            pq_obj = self.get_file_object(filename, bucket_name)
            if not pq_obj:
                return None
            content = self.read_object(pq_obj, decode=False)
            df = read_parquet(io.BytesIO(content))
            return df
        except Exception as e:
            logger.warning(f"Failed to read Parquet {filename} from S3: {e}")
            return None

    def read_numpy(self, filename: str, bucket_name: str):
        """
        Reads a Numpy file (.npy) from the specified S3 bucket and converts it to a numpy array.
        """
        import numpy as np
        try:
            np_obj = self.get_file_object(filename, bucket_name)
            if not np_obj:
                return None
            content = self.read_object(np_obj, decode=False)
            arr = np.load(io.BytesIO(content))
            return arr
        except Exception as e:
            logger.warning(f"Failed to read Numpy {filename} from S3: {e}")
            return None

    def list_files(self, prefix: str, bucket_name: str) -> List[str]:
        """
        List all file keys under a specific prefix in the bucket.
        """
        try:
            bucket = self.get_bucket(bucket_name)
            file_objects = [obj.key for obj in bucket.objects.filter(Prefix=prefix) if not obj.key.endswith('/')]
            return file_objects
        except Exception as e:
            logger.warning(f"Failed to list files with prefix {prefix}: {e}")
            return []

    def get_file_metadata(self, key: str, bucket_name: str) -> Optional[dict]:
        """
        Get size and last modified date for a specific object.
        """
        try:
            obj = self.s3_resource.Object(bucket_name, key)
            return {
                'size': obj.content_length,
                'last_modified': obj.last_modified
            }
        except Exception as e:
            return None
