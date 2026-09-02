"""Baseline classifiers: Logistic Regression, Random Forest, Neural Network."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import torch
import torch.nn as nn


class BaselineModels:
    """Train and evaluate baseline models on PIMA source data."""

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = {}

    def build_models(self):
        self.models["LogReg"] = LogisticRegression(C=1.0, class_weight="balanced", 
                                                     max_iter=1000, random_state=self.random_state)
        self.models["RandomForest"] = RandomForestClassifier(n_estimators=200, max_depth=10,
                                                               class_weight="balanced", 
                                                               random_state=self.random_state)
        self.models["NeuralNet"] = MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu",
                                                   solver="adam", alpha=0.001, batch_size=32,
                                                   learning_rate_init=1e-3, max_iter=200,
                                                   early_stopping=True, validation_fraction=0.1,
                                                   random_state=self.random_state)
        return self.models

    def train(self, X, y):
        if not self.models:
            self.build_models()
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X, y)
        return self.models

    def evaluate(self, X, y, dataset_name="Test"):
        metrics = []
        for name, model in self.models.items():
            y_pred = model.predict(X)
            y_prob = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None

            row = {
                "Model": name,
                "Dataset": dataset_name,
                "Accuracy": accuracy_score(y, y_pred),
                "Precision": precision_score(y, y_pred, zero_division=0),
                "Recall": recall_score(y, y_pred, zero_division=0),
                "F1": f1_score(y, y_pred, zero_division=0),
                "AUC": roc_auc_score(y, y_prob) if y_prob is not None else np.nan
            }
            metrics.append(row)
        return pd.DataFrame(metrics)


class SimpleNN(nn.Module):
    """PyTorch neural network for DANN and baselines."""
    def __init__(self, input_dim, hidden_dims=[64, 32], dropout=0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return torch.sigmoid(self.net(x))
