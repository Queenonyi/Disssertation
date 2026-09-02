"""Evaluation metrics for subgroup performance and fairness."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, brier_score_loss,
)
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference


def compute_subgroup_metrics(y_true, y_pred, y_prob=None, subgroup_name=""):
    """
    Compute predictive metrics for a single subgroup.
    
    Returns dict with: Subgroup, N, Accuracy, Precision, Recall, F1,
    AUC, Brier, Prevalence.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    metrics = {
        "Subgroup": subgroup_name,
        "N": len(y_true),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Prevalence": float(np.mean(y_true)),
    }

    # AUC: needs both classes and probability scores
    if y_prob is not None and len(np.unique(y_true)) > 1:
        y_prob = np.asarray(y_prob).ravel()
        mask = ~np.isnan(y_prob)
        if mask.sum() > 0 and len(np.unique(y_true[mask])) > 1:
            metrics["AUC"] = roc_auc_score(y_true[mask], y_prob[mask])
        else:
            metrics["AUC"] = np.nan
    else:
        metrics["AUC"] = np.nan

    # Brier score
    if y_prob is not None:
        y_prob = np.asarray(y_prob).ravel()
        mask = ~np.isnan(y_prob)
        if mask.sum() > 0:
            metrics["Brier"] = brier_score_loss(y_true[mask], y_prob[mask])
        else:
            metrics["Brier"] = np.nan
    else:
        metrics["Brier"] = np.nan

    return metrics


def evaluate_by_subgroup(eval_df, y_true_col="y_true", y_pred_col="y_pred",
                         y_prob_col="y_prob", subgroup_col="subgroup",
                         protected_col="protected"):
    """
    Evaluate performance by subgroup and attach ONE dataset-level fairness score.
    
    Returns DataFrame with one row per subgroup.
    """
    rows = []

    for subgroup, gdf in eval_df.groupby(subgroup_col):
        yt = gdf[y_true_col].values
        yp = gdf[y_pred_col].values
        ypr = gdf[y_prob_col].values if y_prob_col in gdf.columns else None

        row = compute_subgroup_metrics(yt, yp, ypr, subgroup_name=subgroup)
        rows.append(row)

    # Dataset-level fairness: computed ONCE across all protected groups
    if protected_col in eval_df.columns and eval_df[protected_col].nunique() >= 2:
        fm = compute_fairness_metrics(
            eval_df[y_true_col].values,
            eval_df[y_pred_col].values,
            eval_df[protected_col].values,
        )
    else:
        fm = {
            "demographic_parity_difference": np.nan,
            "equalized_odds_difference": np.nan,
        }

    for row in rows:
        row["Demographic_Parity_Diff"] = fm["demographic_parity_difference"]
        row["Equalized_Odds_Diff"] = fm["equalized_odds_difference"]

    return pd.DataFrame(rows)


def compute_calibration_error(y_true, y_prob, n_bins=10):
    """
    Expected Calibration Error (ECE) with equal-width bins.
    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()

    mask = ~np.isnan(y_prob)
    y_true = y_true[mask]
    y_prob = y_prob[mask]

    if len(y_true) == 0:
        return np.nan

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        if i == 0:
            in_bin = (y_prob >= bin_boundaries[i]) & (y_prob <= bin_boundaries[i + 1])
        else:
            in_bin = (y_prob > bin_boundaries[i]) & (y_prob <= bin_boundaries[i + 1])

        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            avg_confidence = y_prob[in_bin].mean()
            avg_accuracy = y_true[in_bin].mean()
            ece += np.abs(avg_accuracy - avg_confidence) * prop_in_bin

    return float(ece)


def compute_fairness_metrics(y_true, y_pred, sensitive):
    """
    Compute dataset-level fairness metrics using Fairlearn.
    
    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
    y_pred : array-like, shape (n_samples,)
    sensitive : array-like, shape (n_samples,)
        Protected attribute for each sample. Must contain >= 2 unique groups.
    
    Returns
    -------
    dict with keys:
        - demographic_parity_difference
        - equalized_odds_difference
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    sensitive = np.asarray(sensitive)

    # Drop rows with missing protected attributes
    mask = pd.notna(sensitive)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    sensitive = sensitive[mask]

    if len(np.unique(sensitive)) < 2:
        return {
            "Demographic_Parity_Diff": dp,
            "Equalized_Odds_Diff": eo,
        }

    try:
        dp = demographic_parity_difference(
            y_true, y_pred, sensitive_features=sensitive
        )
        eo = equalized_odds_difference(
            y_true, y_pred, sensitive_features=sensitive
        )
    except Exception:
        dp = np.nan
        eo = np.nan

    return {
        "demographic_parity_difference": dp,
        "equalized_odds_difference": eo,
    }


def build_bias_amplification_map(eval_df, model_name, dataset_name):
    """
    Build per-subgroup predictive metrics + ONE dataset-level fairness score.
    
    eval_df must contain columns: y_true, y_pred, y_prob (optional),
    protected, subgroup.
    """
    rows = []

    # A. Per-subgroup predictive metrics
    for subgroup, gdf in eval_df.groupby("subgroup"):
        if len(gdf) < 5:
            continue
        yt = gdf["y_true"].values
        yp = gdf["y_pred"].values
        ypr = gdf["y_prob"].values if "y_prob" in gdf.columns and gdf["y_prob"].notna().any() else None

        row = compute_subgroup_metrics(yt, yp, ypr, subgroup_name=subgroup)
        row["Model"] = model_name
        row["Dataset"] = dataset_name
        rows.append(row)

    # B. Dataset-level fairness: computed ONCE across entire dataset
    if "protected" in eval_df.columns and eval_df["protected"].nunique() >= 2:
        fm = compute_fairness_metrics(
            eval_df["y_true"].values,
            eval_df["y_pred"].values,
            eval_df["protected"].values,
        )
    else:
        fm = {
            "demographic_parity_difference": np.nan,
            "equalized_odds_difference": np.nan,
        }

    # C. Attach same fairness numbers to every subgroup row
    for row in rows:
        row["Demographic_Parity_Diff"] = fm["demographic_parity_difference"]
        row["Equalized_Odds_Diff"] = fm["equalized_odds_difference"]

    return pd.DataFrame(rows)