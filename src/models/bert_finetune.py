import os
import yaml
import shutil
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, classification_report
import mlflow
import mlflow.pytorch
from typing import List, Dict, Any, Tuple, Optional

def load_config(config_path: str = 'config/config.yaml') -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    
def load_params(params_path: str = 'params.yaml') -> Dict[str, Any]:
    with open(params_path, 'r') as f:
        return yaml.safe_load(f)
    
class SentimentDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer: BertTokenizer, max_len: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self) -> int:
        return len(self.texts)
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class BertWithNNHead(nn.Module):
    def __init__(self, model_name: str, nn_head_hidden_dim: int, num_labels: int):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.nn_head = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, nn_head_hidden_dim),
            nn.ReLU(),
            nn.Linear(nn_head_hidden_dim, num_labels)
        )
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return self.nn_head(pooled_output)

def train_epoch(model: nn.Module, data_loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device, scheduler: Optional[Any] = None) -> Tuple[float, float, float]:
    model.train()
    losses = []
    all_preds, all_labels = [], []
    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        if isinstance(outputs, tuple) or isinstance(outputs, list):
            logits = outputs[0]
        else:
            logits = outputs
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
        losses.append(loss.item())
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.detach().cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return np.mean(losses), acc, f1

def eval_epoch(model: nn.Module, data_loader: DataLoader, device: torch.device) -> Tuple[float, float, float, Dict[str, Any]]:
    model.eval()
    losses = []
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            if isinstance(outputs, tuple) or isinstance(outputs, list):
                logits = outputs[0]
            else:
                logits = outputs
            loss = F.cross_entropy(logits, labels)
            losses.append(loss.item())
            preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.detach().cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    report = classification_report(all_labels, all_preds, output_dict=True)
    return np.mean(losses), acc, f1, report

def run_bert_finetuning():
    # Load configuration
    config = load_config()
    params = load_params()
    bert_params = params['bert']
    sentiment_labels = params['sentiment_labels']
    
    # Set up MLflow
    mlflow.set_experiment("bert_finetuning")
    with mlflow.start_run():
        # Set tags if provided
        if 'tags' in bert_params:
            mlflow.set_tags(bert_params['tags'])
        
        # Log parameters
        mlflow.log_params({
            "model_name": bert_params['model_name'],
            "use_nn_head": bert_params['use_nn_head'],
            "nn_head_hidden_dim": bert_params['nn_head_hidden_dim'],
            "epochs": bert_params['epochs'],
            "batch_size": bert_params['batch_size'],
            "learning_rate": bert_params['learning_rate'],
            "max_len": bert_params['max_len'],
            "fast_run": bert_params['fast_run'],
            "train_sample_size": bert_params['train_sample_size'],
            "val_sample_size": bert_params['val_sample_size']
        })
        
        # Rest of your existing BERT training code...
        # Make sure to use sentiment_labels for proper label names in metrics

if __name__ == "__main__":
    params = load_params()
    run_mode = params.get('run_mode', 'both')
    
    if run_mode in ['both', 'bert']:
        run_bert_finetuning() 