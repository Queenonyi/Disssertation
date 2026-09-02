"""Fairness constraints: Demographic Parity, Equalized Odds, Adversarial Debiasing."""
import numpy as np
import pandas as pd
from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression


class FairnessAwareModel(BaseEstimator, ClassifierMixin):
    """Wrapper for fairness-aware training using Fairlearn reductions."""
    def __init__(self, estimator, constraint="demographic_parity", eps=0.01):
        self.estimator = estimator
        self.constraint_name = constraint
        self.eps = eps
        self.model = None
        self.classes_ = np.array([0, 1])
        if constraint == "demographic_parity":
            self.constraint = DemographicParity()
        elif constraint == "equalized_odds":
            self.constraint = EqualizedOdds()
        else:
            raise ValueError("Unknown constraint. Use 'demographic_parity' or 'equalized_odds'")

    def fit(self, X, y, sensitive_features):
        # Strip pandas indices so Fairlearn cannot misalign inputs
        X = np.asarray(X)
        y = np.asarray(y).ravel().astype(int)
        sensitive_features = np.asarray(sensitive_features).ravel()

        if len(np.unique(y)) < 2:
            raise ValueError("y must contain at least two unique classes.")
        if len(np.unique(sensitive_features)) < 2:
            raise ValueError("sensitive_features must contain at least two unique groups.")

        self.model = ExponentiatedGradient(
            estimator=self.estimator,
            constraints=self.constraint,
            eps=self.eps,
        )
        self.model.fit(X, y, sensitive_features=sensitive_features)
        return self

    def predict(self, X):
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        return self.model.predict(X)

    def predict_proba(self, X):
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        if hasattr(self.estimator, "predict_proba"):
            return self.estimator.predict_proba(X)
        preds = self.predict(X)
        return np.column_stack([1 - preds, preds])


class AdaptedModelWrapper(BaseEstimator, ClassifierMixin):
    """Wraps a domain-adapted model (DANN, CORAL, MMD) so Fairlearn can use it.
    Assumes the adapted model has predict_proba and predict methods."""
    def __init__(self, adapted_model, threshold=0.5):
        self.adapted_model = adapted_model
        self.threshold = threshold
        self.classes_ = np.array([0, 1])

    def fit(self, X, y, sample_weight=None, **kwargs):
        # The adapted model is already trained; no-op.
        return self

    def predict(self, X):
        # CRITICAL FIX: ravel to 1D so Fairlearn never broadcasts (n,1) into (n,n)
        preds = self.adapted_model.predict(X)
        return np.asarray(preds).ravel()

    def predict_proba(self, X):
        probs = self.adapted_model.predict_proba(X)
        # Handle torch tensors and squash (n,1) or (n,) to 1-D positive-class probs
        if hasattr(probs, "cpu"):
            probs = probs.cpu().numpy()
        probs = np.asarray(probs)
        if probs.ndim > 1:
            probs = probs.ravel()
        # Fairlearn / sklearn expect (n_samples, n_classes) for binary
        return np.column_stack([1 - probs, probs])


def compute_fairness_metrics(y_true, y_pred, sensitive):
    """Compute demographic parity and equalized odds differences."""
    dp = demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive)
    eo = equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive)
    return {"Demographic_Parity_Diff": dp, "Equalized_Odds_Diff": eo}