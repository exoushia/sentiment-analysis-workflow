import mlflow

# Set your tracking URI if needed
# mlflow.set_tracking_uri("your_tracking_uri")

# 1. Find the experiment by name
exp = mlflow.get_experiment_by_name("baseline")
if exp is not None:
    exp_id = exp.experiment_id
    print(f"Found experiment 'baseline' with ID: {exp_id}")
    # 2. Mark the experiment as deleted
    mlflow.delete_experiment(exp_id)
    print("Experiment marked as deleted.")
    # 3. Permanently delete the experiment (for file-based tracking)
    try:
        store = mlflow.tracking.MlflowClient()._tracking_client.store
        store.delete_experiment(exp_id)
        print("Experiment permanently deleted.")
    except Exception as e:
        print(f"Permanent deletion may not be supported for your backend: {e}")
else:
    print("Experiment 'baseline' not found or already deleted.") 