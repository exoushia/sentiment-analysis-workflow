import os
import mlflow
import yaml

def load_config(config_path: str = 'config/config.yaml') -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def log_eda_artifacts(eda_dir: str, experiment_name: str = "eda") -> None:
    os.makedirs(eda_dir, exist_ok=True)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name="eda-artifacts"):
        for fname in os.listdir(eda_dir):
            if fname.endswith('.png'):
                mlflow.log_artifact(os.path.join(eda_dir, fname), artifact_path="eda")

if __name__ == "__main__":
    config = load_config()
    eda_dir = config['data']['eda_dir']
    log_eda_artifacts(eda_dir) 