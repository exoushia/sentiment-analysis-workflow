import boto3
import yaml
import os

class Connections:
    def __init__(self, credentials_path='config/credentials.yaml'):
        with open(credentials_path, 'r') as f:
            creds = yaml.safe_load(f)
        aws_creds = creds.get('aws', {})
        self.s3 = boto3.client(
            's3',
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
        )

    def download_from_s3(self, bucket, key, local_path):
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        self.s3.download_file(bucket, key, local_path) 