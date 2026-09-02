"""Statistical testing for model comparison."""
import numpy as np
from scipy.stats import norm, chi2
from sklearn.metrics import roc_auc_score


def delong_test(y_true, prob1, prob2):
    """
    DeLong test for comparing two correlated AUCs.
    Returns dict with auc1, auc2, p_value, and z_statistic.
    """
    y_true = np.asarray(y_true)
    prob1 = np.asarray(prob1)
    prob2 = np.asarray(prob2)

    auc1 = roc_auc_score(y_true, prob1)
    auc2 = roc_auc_score(y_true, prob2)

    # DeLong covariance estimation
    n = len(y_true)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    n_pos = len(pos_idx)
    n_neg = len(neg_idx)

    if n_pos == 0 or n_neg == 0:
        return {"auc1": auc1, "auc2": auc2, "p_value": 1.0, "z_statistic": 0.0}

    # Compute structural components
    v10_1 = np.zeros(n_pos)
    v10_2 = np.zeros(n_pos)
    for i, idx in enumerate(pos_idx):
        v10_1[i] = np.mean(prob1[idx] > prob1[neg_idx]) + 0.5 * np.mean(prob1[idx] == prob1[neg_idx])
        v10_2[i] = np.mean(prob2[idx] > prob2[neg_idx]) + 0.5 * np.mean(prob2[idx] == prob2[neg_idx])

    v01_1 = np.zeros(n_neg)
    v01_2 = np.zeros(n_neg)
    for j, idx in enumerate(neg_idx):
        v01_1[j] = np.mean(prob1[pos_idx] > prob1[idx]) + 0.5 * np.mean(prob1[pos_idx] == prob1[idx])
        v01_2[j] = np.mean(prob2[pos_idx] > prob2[idx]) + 0.5 * np.mean(prob2[pos_idx] == prob2[idx])

    # Covariances
    s01 = np.cov(v01_1, v01_2)[0, 1] / n_neg if n_neg > 1 else 0
    s10 = np.cov(v10_1, v10_2)[0, 1] / n_pos if n_pos > 1 else 0

    var_auc1 = np.var(v10_1) / n_pos + np.var(v01_1) / n_neg if n_pos > 1 and n_neg > 1 else 1e-8
    var_auc2 = np.var(v10_2) / n_pos + np.var(v01_2) / n_neg if n_pos > 1 and n_neg > 1 else 1e-8
    cov = s10 + s01

    se = np.sqrt(max(var_auc1 + var_auc2 - 2 * cov, 1e-12))
    z = (auc1 - auc2) / se if se > 0 else 0
    p_value = 2 * (1 - norm.cdf(abs(z)))

    return {
        "auc1": auc1,
        "auc2": auc2,
        "p_value": p_value,
        "z_statistic": z,
    }


def mcnemar_test(y_true, pred1, pred2):
    """McNemar's test for comparing two classifiers on same test set."""
    y_true = np.asarray(y_true)
    pred1 = np.asarray(pred1)
    pred2 = np.asarray(pred2)

    b = np.sum((pred1 == 0) & (pred2 == 1))  # Model2 correct, Model1 wrong
    c = np.sum((pred1 == 1) & (pred2 == 0))  # Model1 correct, Model2 wrong

    if b + c == 0:
        return {"statistic": 0.0, "p_value": 1.0, "b": b, "c": c}

    stat = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) > 0 else 0
    p_value = 1 - chi2.cdf(stat, 1)
    return {"statistic": stat, "p_value": p_value, "b": b, "c": c}