import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import mlflow
import mlflow.sklearn
import nltk
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from scripts.utils import load_config, get_logger, load_credentials, load_params

# Set up logger
logger = get_logger("baseline", "baseline.log")


def plot_confusion_matrix(cm: np.ndarray, classes: list[str], out_path: str) -> None:
    """Plot and save a confusion matrix as a PNG file."""
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def run_baseline() -> None:
    """Run the baseline model pipeline with MLflow logging and grid search."""
    # Load configuration and data
    config = load_config()
    params = load_params()
    processed_data_path = config['data']['processed']
    text_col = config['data']['columns']['text']
    sentiment_col = config['data']['columns']['sentiment']
    sentiment_labels = params['sentiment_labels']
    baseline_params = params['baseline']
    
    # Load and prepare data
    train_df = pd.read_csv(os.path.join(processed_data_path, 'train.csv'))
    test_df = pd.read_csv(os.path.join(processed_data_path, 'test.csv'))
    X_train, y_train = train_df[text_col], train_df[sentiment_col]
    X_test, y_test = test_df[text_col], test_df[sentiment_col]
    
    # Log the number of samples and experiment name
    logger.info(f"Experiment: {baseline_params.get('model', 'unknown')}")
    logger.info(f"Number of training samples: {len(X_train)}")
    logger.info(f"Number of test samples: {len(X_test)}")

    # Fill NaN values with empty strings
    X_train = X_train.fillna("")
    X_test = X_test.fillna("")
    
    # Vectorize text
    vectorizer = TfidfVectorizer(max_features=baseline_params['tfidf_max_features'])
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Define models
    models = {
        "bernoulinaivebayes": {
            "model": BernoulliNB(),
            "param_grid": baseline_params['bernoulinaivebayes']['param_grid'] if baseline_params['grid_search'] else {}
        },
        "logisticregression": {
            "model": OneVsRestClassifier(LogisticRegression(max_iter=baseline_params['logisticregression']['max_iter'])),
            "param_grid": baseline_params['logisticregression']['param_grid'] if baseline_params['grid_search'] else {}
        },
        "randomforest": {
            "model": RandomForestClassifier(),
            "param_grid": baseline_params['randomforest']['param_grid'] if baseline_params['grid_search'] else {}
        }
    }
    
    selected_model = baseline_params['model']
    model_info = models[selected_model]

    # Parent MLflow run for grid search
    with mlflow.start_run(run_name=f"baseline_{selected_model}") as parent_run:
        if 'tags' in baseline_params:
            mlflow.set_tags(baseline_params['tags'])
        mlflow.log_param("model_type", selected_model)
        mlflow.log_param("tfidf_max_features", baseline_params['tfidf_max_features'])

        if baseline_params['grid_search'] and model_info['param_grid']:
            grid_search = GridSearchCV(
                model_info['model'],
                model_info['param_grid'],
                cv=5,
                scoring='f1_weighted',
                n_jobs=-1,
                return_train_score=True
            )
            grid_search.fit(X_train_vec, y_train)
            results = grid_search.cv_results_
            best_idx = grid_search.best_index_
            best_params = grid_search.best_params_
            best_score = grid_search.best_score_
            # Log each grid search result as a child run
            for i, params in enumerate(results['params']):
                with mlflow.start_run(run_name=f"child_{i}", nested=True):
                    mlflow.log_params(params)
                    mlflow.log_metric("mean_test_f1", results['mean_test_score'][i])
                    # Optionally: retrain and log metrics/artifacts for this config
                    model = model_info['model'].set_params(**params)
                    model.fit(X_train_vec, y_train)
                    preds = model.predict(X_test_vec)
                    acc = accuracy_score(y_test, preds)
                    f1 = f1_score(y_test, preds, average='weighted')
                    cm = confusion_matrix(y_test, preds)
                    report = classification_report(
                        y_test, 
                        preds, 
                        target_names=[sentiment_labels[i] for i in sorted(sentiment_labels.keys())],
                        output_dict=True
                    )
                    mlflow.log_metric("accuracy", acc)
                    mlflow.log_metric("f1_score", f1)
                    mlflow.log_dict(report, "classification_report.json")
                    cm_path = f"confusion_matrix_{selected_model}_child_{i}.png"
                    plot_confusion_matrix(
                        cm, 
                        classes=[sentiment_labels[i] for i in sorted(sentiment_labels.keys())],
                        out_path=cm_path
                    )
                    mlflow.log_artifact(cm_path)
                    os.remove(cm_path)
                    try:
                        mlflow.sklearn.log_model(model, artifact_path="model")
                    except Exception as e:
                        print(f"Model logging failed: {e}")
                    with open("tfidf_vectorizer.pkl", "wb") as f:
                        pickle.dump(vectorizer, f)
                    mlflow.log_artifact("tfidf_vectorizer.pkl")
                    os.remove("tfidf_vectorizer.pkl")
                    # Log per-class metrics
                    for class_idx, class_name in sorted(sentiment_labels.items()):
                        if class_name in report:
                            
                            mlflow.log_metric(f"precision_{class_name}", report[class_name]["precision"])
                            mlflow.log_metric(f"recall_{class_name}", report[class_name]["recall"])
                            mlflow.log_metric(f"f1_{class_name}", report[class_name]["f1-score"])
                    logger.info(f"Grid search child run {i} params: {params}")
                    logger.info(f"Grid search child run {i} metrics: accuracy={acc}, f1={f1}, mean_test_f1={results['mean_test_score'][i]}")
                    logger.info(f"TF-IDF max_features (vector size): {baseline_params['tfidf_max_features']}")
            # Log the best model and parameters to the parent run
            mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
            mlflow.log_metric("best_cv_f1_score", best_score)
            # Refit best model and log as parent run artifact
            best_model = grid_search.best_estimator_
            best_model.fit(X_train_vec, y_train)
            preds = best_model.predict(X_test_vec)
            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average='weighted')
            cm = confusion_matrix(y_test, preds)
            report = classification_report(
                y_test, 
                preds, 
                target_names=[sentiment_labels[i] for i in sorted(sentiment_labels.keys())],
                output_dict=True
            )
            mlflow.log_metric("best_accuracy", acc)
            mlflow.log_metric("best_f1_score", f1)
            mlflow.log_dict(report, "best_classification_report.json")
            cm_path = f"confusion_matrix_{selected_model}_best.png"
            plot_confusion_matrix(
                cm, 
                classes=[sentiment_labels[i] for i in sorted(sentiment_labels.keys())],
                out_path=cm_path
            )
            mlflow.log_artifact(cm_path)
            os.remove(cm_path)
            try:
                mlflow.sklearn.log_model(best_model, artifact_path="best_model")
            except Exception as e:
                print(f"Best model logging failed: {e}")
            with open("tfidf_vectorizer.pkl", "wb") as f:
                pickle.dump(vectorizer, f)
            mlflow.log_artifact("tfidf_vectorizer.pkl")
            os.remove("tfidf_vectorizer.pkl")
            logger.info(f"Best model params: {best_params}")
            logger.info(f"Best model metrics: best_cv_f1_score={best_score}, best_accuracy={acc}, best_f1_score={f1}")
            logger.info(f"Best model TF-IDF max_features (vector size): {baseline_params['tfidf_max_features']}")
        else:
            # Train without grid search
            model = model_info['model']
            model.fit(X_train_vec, y_train)
            preds = model.predict(X_test_vec)
            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average='weighted')
            cm = confusion_matrix(y_test, preds)
            report = classification_report(
                y_test, 
                preds, 
                target_names=[sentiment_labels[i] for i in sorted(sentiment_labels.keys())],
                output_dict=True
            )
            logger.info(f"No grid search. Model: {selected_model}, accuracy={acc}, f1={f1}")
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_dict(report, "classification_report.json")
            cm_path = f"confusion_matrix_{selected_model}.png"
            plot_confusion_matrix(
                cm, 
                classes=[sentiment_labels[i] for i in sorted(sentiment_labels.keys())],
                out_path=cm_path
            )
            mlflow.log_artifact(cm_path)
            os.remove(cm_path)
            try:
                mlflow.sklearn.log_model(model, artifact_path="model")
            except Exception as e:
                print(f"Model logging failed: {e}")
            with open("tfidf_vectorizer.pkl", "wb") as f:
                pickle.dump(vectorizer, f)
            mlflow.log_artifact("tfidf_vectorizer.pkl")
            os.remove("tfidf_vectorizer.pkl")

if __name__ == '__main__':
    # Check run mode
    params = load_params()
    config = load_config()
    creds = load_credentials()
    run_mode = params.get('run_mode', 'both')

    # Set MLflow tracking URI from config
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    # Set MLflow authentication from credentials if available
    if 'mlflow' in creds:
        os.environ['MLFLOW_TRACKING_USERNAME'] = creds['mlflow'].get('username', '')
        os.environ['MLFLOW_TRACKING_PASSWORD'] = creds['mlflow'].get('password', '')
    mlflow.set_experiment(config['mlflow']['experiments']['baseline'])

    if run_mode in ['both', 'baseline']:
        # Add custom nltk_data path if it exists
        nltk_data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../', config['data']['nltk_data']))
        if os.path.exists(nltk_data_path):
            nltk.data.path.insert(0, nltk_data_path)
        run_baseline() 