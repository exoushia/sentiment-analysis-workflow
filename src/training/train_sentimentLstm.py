import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import os
import yaml
import mlflow
import mlflow.pytorch
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
import pickle
import sys
from typing import List, Dict, Any, Tuple, Optional

sys.path.append('.')

from src.models.model_sentimentLstm import SentimentLSTM

def load_config(config_path: str = 'config/config.yaml') -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_params(params_path: str = 'params.yaml') -> Dict[str, Any]:
    with open(params_path, 'r') as f:
        return yaml.safe_load(f)

def build_vocab(texts: pd.Series, min_freq: int = 2) -> Dict[str, int]:
    counter = Counter()
    for text in texts:
        counter.update(str(text).split())
    vocab = {'<unk>': 0, '<pad>': 1}
    for word, count in counter.items():
        if count >= min_freq:
            vocab[word] = len(vocab)
    return vocab

def text_to_indices(texts: pd.Series, vocab: Dict[str, int]) -> List[List[int]]:
    return [[vocab.get(word, vocab['<unk>']) for word in str(text).split()] for text in texts]

def train(config: Dict[str, Any], params: Dict[str, Any]) -> None:
    processed_data_path = config['data']['processed']
    text_col = config['data']['columns']['text']
    sentiment_col = config['data']['columns']['sentiment']
    train_df = pd.read_csv(os.path.join(processed_data_path, 'train.csv'))
    test_df = pd.read_csv(os.path.join(processed_data_path, 'test.csv'))
    # Sampling logic for fast experimentation
    if params.get('fast_run', False):
        train_sample_size = params.get('train_sample_size')
        val_sample_size = params.get('val_sample_size')
        if train_sample_size:
            train_df = train_df.sample(n=train_sample_size, random_state=42)
        if val_sample_size:
            test_df = test_df.sample(n=val_sample_size, random_state=42)
    # Build vocabulary
    vocab = build_vocab(train_df[text_col])
    os.makedirs('models', exist_ok=True)
    with open('models/vocab_sentimentLstm.pkl', 'wb') as f:
        pickle.dump(vocab, f)
    # Convert text to indices
    train_indices = text_to_indices(train_df[text_col], vocab)
    test_indices = text_to_indices(test_df[text_col], vocab)
    # Filter out zero-length sequences
    train_data_filtered = [(x, y) for x, y in zip(train_indices, train_df[sentiment_col].values) if len(x) > 0]
    test_data_filtered = [(x, y) for x, y in zip(test_indices, test_df[sentiment_col].values) if len(x) > 0]
    train_indices, y_train_values = zip(*train_data_filtered)
    test_indices, y_test_values = zip(*test_data_filtered)
    # Create tensors
    X_train = [torch.tensor(x) for x in train_indices]
    y_train = torch.tensor(y_train_values).float()
    X_test = [torch.tensor(x) for x in test_indices]
    y_test = torch.tensor(y_test_values).float()
    # Get text lengths
    train_lengths = torch.tensor([len(x) for x in X_train])
    test_lengths = torch.tensor([len(x) for x in X_test])
    # Pad sequences
    X_train_padded = pad_sequence(X_train, batch_first=True, padding_value=vocab['<pad>'])
    X_test_padded = pad_sequence(X_test, batch_first=True, padding_value=vocab['<pad>'])
    # Create datasets and dataloaders
    batch_size = params['batch_size']
    train_data = TensorDataset(X_train_padded, y_train, train_lengths)
    test_data = TensorDataset(X_test_padded, y_test, test_lengths)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    # Model parameters
    model_params = params
    input_dim = len(vocab)
    output_dim = 1  # For binary sentiment
    pad_idx = vocab['<pad>']
    # Initialize model
    model = SentimentLSTM(input_dim, model_params['embedding_dim'], model_params['hidden_dim'], 
                          output_dim, model_params['n_layers'], model_params['bidirectional'], 
                          model_params['dropout'], pad_idx)
    # Optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
    criterion = nn.BCEWithLogitsLoss()
    # Move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = criterion.to(device)
    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_params(model_params)
        mlflow.log_artifact('models/vocab_sentimentLstm.pkl')
        epochs = params['epochs']
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            for batch in train_loader:
                text, labels, text_lengths = batch
                text, labels = text.to(device), labels.to(device)
                optimizer.zero_grad()
                predictions = model(text, text_lengths).squeeze(1)
                loss = criterion(predictions, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            # Evaluate the model
            model.eval()
            y_pred, y_true = [], []
            with torch.no_grad():
                for batch in test_loader:
                    text, labels, text_lengths = batch
                    text, labels = text.to(device), labels.to(device)
                    predictions = model(text, text_lengths).squeeze(1)
                    rounded_preds = torch.round(torch.sigmoid(predictions))
                    y_pred.extend(rounded_preds.cpu().numpy())
                    y_true.extend(labels.cpu().numpy())
            accuracy = accuracy_score(y_true, y_pred)
            print(f'Epoch {epoch+1:02} | Loss: {epoch_loss/len(train_loader):.3f} | Accuracy: {accuracy*100:.2f}%')
            mlflow.log_metric('train_loss', epoch_loss / len(train_loader), step=epoch)
            mlflow.log_metric('test_accuracy', accuracy, step=epoch)
        mlflow.pytorch.log_model(model, "model")

def main() -> None:
    config = load_config()
    params = load_params()['lstm']
    train(config, params)

if __name__ == '__main__':
    main()
