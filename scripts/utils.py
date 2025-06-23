import os
import yaml
import logging
from typing import Any, Dict

def load_config(config_path: str = 'config/config.yaml') -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    
def load_params(params_path: str = 'params.yaml') -> dict:
    """Load parameters for grid search on baseline models from a YAML file."""
    with open(params_path, 'r') as f:
        return yaml.safe_load(f)

def load_credentials(creds_path: str = 'config/credentials.yaml') -> dict:
    """Load credentials for AWS/Dagshub/Git from a YAML file."""
    if os.path.exists(creds_path):
        with open(creds_path, 'r') as f:
            return yaml.safe_load(f)
    return {}

def get_logger(name: str = "baseline", log_file: str = "baseline.log") -> logging.Logger:
    """Set up and return a logger with the specified format."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    log_formatter = logging.Formatter('%(asctime)s | baseline | %(pathname)s.%(funcName)s | %(message)s')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(log_formatter)
    if not logger.hasHandlers():
        logger.addHandler(file_handler)
    return logger 


