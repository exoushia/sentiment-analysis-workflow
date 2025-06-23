import nltk
import os
import yaml

def load_config(config_path: str = 'config/config.yaml') -> dict:
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# Load configuration
config = load_config()
NLTK_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', config['data']['nltk_data']))
os.makedirs(NLTK_DATA_DIR, exist_ok=True)

# Add the local nltk_data directory as the first search path
nltk.data.path.insert(0, NLTK_DATA_DIR)

# Check if each resource exists before downloading
resources = [
    ('punkt', 'tokenizers/punkt'),  # Download punkt first
    ('stopwords', 'corpora/stopwords'),
    ('wordnet', 'corpora/wordnet')
]

for resource_name, resource_path in resources:
    try:
        nltk.data.find(resource_path)
        print(f"{resource_name} already present.")
    except LookupError:
        print(f"Downloading {resource_name}...")
        nltk.download(resource_name, download_dir=NLTK_DATA_DIR, quiet=True)

print(f"NLTK data ensured at {NLTK_DATA_DIR}") 