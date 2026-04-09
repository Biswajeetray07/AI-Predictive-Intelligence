import boto3
import os

AWS_ACCESS_KEY_ID_ENV_KEY = "AWS_ACCESS_KEY_ID"
AWS_SECRET_ACCESS_KEY_ENV_KEY = "AWS_SECRET_ACCESS_KEY"
AWS_REGION_ENV_KEY = "AWS_REGION"
DEFAULT_REGION = "us-east-1"

class S3Client:
    """
    Singleton connection to AWS S3.
    It reads credentials from environment variables. If they aren't provided, 
    boto3 will attempt to use default AWS configuration files or IAM roles.
    """
    s3_client = None
    s3_resource = None

    def __init__(self, region_name=None):
        if region_name is None:
            region_name = os.getenv(AWS_REGION_ENV_KEY, DEFAULT_REGION)

        if S3Client.s3_resource is None or S3Client.s3_client is None:
            __access_key_id = os.getenv(AWS_ACCESS_KEY_ID_ENV_KEY)
            __secret_access_key = os.getenv(AWS_SECRET_ACCESS_KEY_ENV_KEY)
            
            if not __access_key_id or not __secret_access_key:
                try:
                    import streamlit as st
                    if AWS_ACCESS_KEY_ID_ENV_KEY in st.secrets:
                        __access_key_id = st.secrets[AWS_ACCESS_KEY_ID_ENV_KEY]
                        __secret_access_key = st.secrets.get(AWS_SECRET_ACCESS_KEY_ENV_KEY, "")
                except Exception:
                    pass
            
            if __access_key_id and __secret_access_key:
                S3Client.s3_resource = boto3.resource('s3',
                                                aws_access_key_id=__access_key_id,
                                                aws_secret_access_key=__secret_access_key,
                                                region_name=region_name
                                                )
                S3Client.s3_client = boto3.client('s3',
                                            aws_access_key_id=__access_key_id,
                                            aws_secret_access_key=__secret_access_key,
                                            region_name=region_name
                                            )
            else:
                # Fallback to local configs (~/.aws/credentials) or IAM contexts
                S3Client.s3_resource = boto3.resource('s3', region_name=region_name)
                S3Client.s3_client = boto3.client('s3', region_name=region_name)
                
        self.s3_resource = S3Client.s3_resource
        self.s3_client = S3Client.s3_client
