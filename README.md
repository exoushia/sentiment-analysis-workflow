# Sentiment Analysis Workflow

This repository contains a modular, reproducible pipeline for sentiment analysis experiments using DVC, MLflow, and modern ML best practices.

## Structure
- `src/` - Source code (data processing, models, training, evaluation)
- `config/` - Configuration files
- `scripts/` - Utility scripts
- `dvc.yaml`, `dvc.lock` - DVC pipeline definitions
- `params.yaml` - Experiment parameters
- `requirements.txt` - Python dependencies

## Getting Started
1. Create and activate a virtual environment:
   ```sh
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
2. Configure DVC and MLflow as needed.
3. Run experiments and track results.

## Notes
- Data and experiment artifacts are tracked with DVC and not stored in Git.
- Credentials and local environment files are ignored for security. 