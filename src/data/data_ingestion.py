import os
import yaml
from connections import Connections
from scripts.utils import get_logger

logger = get_logger()

def load_config():
    with open('config/config.yaml', 'r') as f:
        return yaml.safe_load(f)

def load_params():
    with open('params.yaml', 'r') as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    params = load_params()
    data_cache = params.get('data_cache', True)
    local_path = config['data']['raw']
    
    if data_cache and os.path.exists(local_path):
        logger.info(f"Using cached data at {local_path}")
    else:
        logger.info("Downloading data from S3...")
        creds_path = 'config/credentials.yaml'
        s3_bucket = 'sentiment-raw'
        s3_key = 'sentiment_data.csv'
        conn = Connections(credentials_path=creds_path)
        conn.download_from_s3(s3_bucket, s3_key, local_path)
        print(f"Downloaded data to {local_path}")

if __name__ == '__main__':
    main() 