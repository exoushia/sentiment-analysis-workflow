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

def load_config(config_path='config/config.yaml'):
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_params(params_path='params.yaml'):
    import yaml
    with open(params_path, 'r') as f:
        return yaml.safe_load(f)

def plot_confusion_matrix(cm, classes, out_path):
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def run_baseline():
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
            "model": LogisticRegression(max_iter=baseline_params['logisticregression']['max_iter']),
            "param_grid": baseline_params['logisticregression']['param_grid'] if baseline_params['grid_search'] else {}
        }
    }
    
    # Set up MLflow experiment
    mlflow.set_experiment("baseline")
    
    # Train and evaluate models
    selected_model = baseline_params['model']
    model_info = models[selected_model]
    
    with mlflow.start_run(run_name=selected_model):
        # Set tags if provided
        if 'tags' in baseline_params:
            mlflow.set_tags(baseline_params['tags'])
        
        # Log parameters
        mlflow.log_param("model_type", selected_model)
        mlflow.log_param("tfidf_max_features", baseline_params['tfidf_max_features'])
        
        if baseline_params['grid_search'] and model_info['param_grid']:
            # Perform grid search
            grid_search = GridSearchCV(
                model_info['model'],
                model_info['param_grid'],
                cv=5,
                scoring='f1_weighted',
                n_jobs=-1
            )
            grid_search.fit(X_train_vec, y_train)
            model = grid_search.best_estimator_
            
            # Log best parameters
            mlflow.log_params(grid_search.best_params_)
            mlflow.log_metric("cv_f1_score", grid_search.best_score_)
        else:
            # Train without grid search
            model = model_info['model']
            model.fit(X_train_vec, y_train)
        
        # Make predictions and evaluate
        preds = model.predict(X_test_vec)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average='weighted')
        cm = confusion_matrix(y_test, preds)
        
        # Get classification report with proper label names
        report = classification_report(
            y_test, 
            preds, 
            target_names=[sentiment_labels[str(i)] for i in sorted(sentiment_labels.keys())],
            output_dict=True
        )
        
        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_dict(report, "classification_report.json")
        
        # Save and log confusion matrix plot
        cm_path = f"confusion_matrix_{selected_model}.png"
        plot_confusion_matrix(
            cm, 
            classes=[sentiment_labels[str(i)] for i in sorted(sentiment_labels.keys())],
            out_path=cm_path
        )
        mlflow.log_artifact(cm_path)
        os.remove(cm_path)
        
        # Log model and vectorizer
        mlflow.sklearn.log_model(model, "model")
        with open("tfidf_vectorizer.pkl", "wb") as f:
            pickle.dump(vectorizer, f)
        mlflow.log_artifact("tfidf_vectorizer.pkl")
        os.remove("tfidf_vectorizer.pkl")

if __name__ == '__main__':
    # Check run mode
    params = load_params()
    config = load_config()
    run_mode = params.get('run_mode', 'both')
    
    if run_mode in ['both', 'baseline']:
        # Add custom nltk_data path if it exists
        nltk_data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../', config['data']['nltk_data']))
        if os.path.exists(nltk_data_path):
            nltk.data.path.insert(0, nltk_data_path)
        
        # Set MLflow experiment from config
        mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
        mlflow.set_experiment(config['mlflow']['experiments']['baseline'])
        
        run_baseline() 