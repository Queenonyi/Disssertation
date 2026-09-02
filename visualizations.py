"""Visualization utilities for domain adaptation and fairness analysis."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.calibration import calibration_curve
import umap
from config import OUTPUT_FIG

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 10

def save_fig(name, fig_dir=OUTPUT_FIG):
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(os.path.join(fig_dir, name), dpi=300, bbox_inches="tight")
    print(f"Saved figure: {name}")
    plt.close()

def plot_bias_amplification_map(df_results, title="Bias Amplification Map", save_as=None):
    """Heatmap of AUC drops across subgroups. Returns pivot dataframe."""
    if df_results.empty or "AUC" not in df_results.columns:
        print("Warning: empty data for bias amplification map")
        return pd.DataFrame()
    pivot = df_results.pivot_table(values="AUC", index="Subgroup", columns="Model")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn", vmin=0.5, vmax=1.0, ax=ax)
    ax.set_title(title)
    if save_as:
        save_fig(save_as)
    return pivot.reset_index()

def plot_pareto_frontier(df, acc_col="AUC", fairness_col="Demographic_Parity_Diff",
                         label_col="Model", title="Accuracy-Fairness Trade-off", save_as=None):
    """Plot Pareto frontier for accuracy vs fairness. Returns plotted dataframe."""
    fig, ax = plt.subplots(figsize=(10, 7))
    models = df[label_col].unique()
    cmap = plt.get_cmap("tab10", len(models))
    for i, model in enumerate(models):
        sub = df[df[label_col] == model]
        ax.scatter(sub[acc_col], sub[fairness_col], label=model, s=120, alpha=0.8, color=cmap(i))
        for _, row in sub.iterrows():
            ax.annotate(row.get("Dataset", ""), (row[acc_col], row[fairness_col]), fontsize=7, alpha=0.7)
    ax.set_xlabel(acc_col)
    ax.set_ylabel(f"{fairness_col} (lower is better)")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    if save_as:
        save_fig(save_as)
    return df.copy()

def plot_feature_space(X, domains, protected, method_name="", save_as=None):
    """t-SNE/UMAP visualization of feature space colored by domain and protected attr.
    Returns dataframe with embedding coordinates and metadata."""
    n_samples = len(X)
    idx_full = np.arange(n_samples)
    if n_samples > 3000:
        idx = np.random.choice(n_samples, 3000, replace=False)
        X = X[idx]
        domains = domains[idx]
        idx_full = idx
        if protected is not None:
            protected = protected[idx]
    else:
        idx = None
    
    try:
        reducer = umap.UMAP(n_components=2, random_state=42)
        embedding = reducer.fit_transform(X)
    except Exception:
        embedding = TSNE(n_components=2, random_state=42, perplexity=min(30, n_samples - 1)).fit_transform(X)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for domain in np.unique(domains):
        mask = domains == domain
        axes[0].scatter(embedding[mask, 0], embedding[mask, 1], label=domain, alpha=0.5, s=10)
    axes[0].set_title(f"{method_name} - By Domain")
    axes[0].legend()
    
    if protected is not None and not pd.isna(protected).all():
        valid = ~pd.isna(protected)
        for val in np.unique(protected[valid]):
            mask = (protected == val) & valid
            axes[1].scatter(embedding[mask, 0], embedding[mask, 1], label=str(val), alpha=0.5, s=10)
        axes[1].set_title(f"{method_name} - By Protected Attribute")
        axes[1].legend()
    
    if save_as:
        save_fig(save_as)
    
    # Return numeric embedding data
    embed_df = pd.DataFrame({
        "umap_x": embedding[:, 0],
        "umap_y": embedding[:, 1],
        "domain": domains,
        "sample_idx": idx_full,
    })
    if protected is not None:
        embed_df["protected"] = protected
    embed_df["method"] = method_name
    return embed_df

def plot_calibration_curves(df, prob_col, true_col, subgroup_col, save_as=None):
    """Plot calibration curves by subgroup. Returns calibration dataframe."""
    fig, ax = plt.subplots(figsize=(10, 8))
    cal_data = []
    for subgroup, group_df in df.groupby(subgroup_col):
        if len(group_df) < 30:
            continue
        y_true = group_df[true_col].values
        y_prob = group_df[prob_col].values
        if len(np.unique(y_true)) < 2:
            continue
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
        ax.plot(prob_pred, prob_true, marker="o", label=str(subgroup), linewidth=2)
        for i in range(len(prob_true)):
            cal_data.append({
                "subgroup": subgroup,
                "bin_center": prob_pred[i],
                "fraction_positives": prob_true[i],
                "n_bins": len(prob_true),
            })
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curves by Subgroup")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    if save_as:
        save_fig(save_as)
    return pd.DataFrame(cal_data)

def plot_accuracy_fairness_tradeoff(results_df, save_as=None):
    """Scatter plot of accuracy vs demographic parity difference. Returns dataframe."""
    fig, ax = plt.subplots(figsize=(12, 7))
    markers = {"Baseline": "o", "DANN": "s", "CORAL": "^", "MMD": "D"}
    for model in results_df["Model"].unique():
        sub = results_df[results_df["Model"] == model]
        marker = markers.get(model.split("_")[0], "o")
        ax.scatter(sub["Accuracy"], sub["Demographic_Parity_Diff"], label=model, s=120, alpha=0.7, marker=marker)
    ax.set_xlabel("Accuracy")
    ax.set_ylabel("Demographic Parity Difference")
    ax.set_title("Accuracy vs Fairness Trade-off")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    if save_as:
        save_fig(save_as)
    return results_df.copy()

def plot_subgroup_auc_bars(df, save_as=None):
    """Bar plot of AUC by subgroup for each model. Returns pivot dataframe."""
    fig, ax = plt.subplots(figsize=(14, 7))
    pivot = df.pivot_table(values="AUC", index="Subgroup", columns="Model")
    pivot.plot(kind="bar", ax=ax, width=0.8)
    ax.set_ylabel("AUC-ROC")
    ax.set_title("AUC by Subgroup and Model")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_ylim(0.5, 1.0)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    if save_as:
        save_fig(save_as)
    return pivot.reset_index()