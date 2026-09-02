"""generate_simple_figures.py

Run this AFTER the main pipeline completes (Phase 5).
It reads the output CSV files and generates 4 simple, publication-ready
comparison figures that are easy to interpret for Chapters 4-7.

Usage:
    python generate_simple_figures.py

Outputs (saved to outputs/figures/):
    1. comparison_all_models.png       - Side-by-side AUC bars (NHANES vs BRFSS)
    2. fairness_improvement.png        - Fairness before/after constraints
    3. subgroup_heatmap_nhanes.png     - AUC heatmap by race and model
    4. training_curves_combined.png    - All 6 adaptation histories in one view
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import OUTPUT_TAB, OUTPUT_FIG


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def plot_comparison_all_models():
    """Figure 1: Simple grouped bar chart comparing mean AUC across all models."""
    path = os.path.join(OUTPUT_TAB, "subgroup_metrics_all_models.csv")
    if not os.path.exists(path):
        print("[SKIP] subgroup_metrics_all_models.csv not found")
        return

    df = pd.read_csv(path)
    summary = df.groupby(["Model", "Dataset"])["AUC"].mean().reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Model Performance Comparison by Dataset", fontsize=14, fontweight="bold")

    for ax, dataset in zip(axes, ["NHANES", "BRFSS"]):
        sub = summary[summary["Dataset"] == dataset]
        if sub.empty:
            ax.set_visible(False)
            continue

        colors = plt.cm.Set2(np.linspace(0, 1, len(sub)))
        bars = ax.bar(range(len(sub)), sub["AUC"], color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels(sub["Model"], rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Mean AUC", fontsize=11)
        ax.set_title(dataset, fontsize=12, fontweight="bold")
        ax.set_ylim(0.5, 1.0)
        ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.3, label="Random (AUC=0.5)")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)

        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.3f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(OUTPUT_FIG, "comparison_all_models.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved {out_path}")


def plot_fairness_improvement():
    """Figure 2: Grouped bars showing fairness metrics before vs after constraints."""
    path = os.path.join(OUTPUT_TAB, "subgroup_metrics_all_models.csv")
    if not os.path.exists(path):
        print("[SKIP] subgroup_metrics_all_models.csv not found")
        return

    df = pd.read_csv(path)
    nhanes = df[df["Dataset"] == "NHANES"].copy()
    if nhanes.empty:
        print("[SKIP] No NHANES data for fairness plot")
        return

    def classify_type(name):
        if any(t in str(name) for t in ["DemographicParity", "EqualizedOdds"]):
            return "Fairness-Aware"
        elif any(t in str(name) for t in ["DANN", "CORAL", "MMD"]):
            return "Adaptation-Only"
        return "Baseline"

    def base_method(name):
        name = str(name)
        for m in ["DANN", "CORAL", "MMD"]:
            if m in name:
                return m
        return "Baseline"

    nhanes["Type"] = nhanes["Model"].apply(classify_type)
    nhanes["BaseMethod"] = nhanes["Model"].apply(base_method)

    compare = nhanes[nhanes["Type"].isin(["Adaptation-Only", "Fairness-Aware"])]
    if compare.empty:
        print("[SKIP] No adaptation/fairness models to compare")
        return

    metrics = [
        ("Demographic_Parity_Diff", "Demographic Parity Difference (lower is better)"),
        ("Equalized_Odds_Diff", "Equalized Odds Difference (lower is better)")
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Fairness Metrics: Adaptation-Only vs Fairness-Aware (NHANES)",
                 fontsize=13, fontweight="bold")

    for ax, (col, title) in zip(axes, metrics):
        if col not in compare.columns:
            ax.set_visible(False)
            continue
        pivot = compare.groupby(["BaseMethod", "Type"])[col].mean().unstack()
        pivot = pivot.fillna(0)
        pivot.plot(kind="bar", ax=ax, color=["coral", "steelblue"], edgecolor="black", width=0.7)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("Mean Difference", fontsize=10)
        ax.set_xlabel("")
        ax.legend(title="", loc="upper right")
        ax.tick_params(axis="x", rotation=0)
        ax.grid(axis="y", alpha=0.3)
        ax.axhline(y=0.05, color="red", linestyle="--", alpha=0.4, label="Threshold = 0.05")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(OUTPUT_FIG, "fairness_improvement.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved {out_path}")


def plot_subgroup_heatmap():
    """Figure 3: Heatmap of AUC by race/ethnicity and model (NHANES only).

    NOTE: The subgroup_metrics CSV stores intersectional subgroups in the
    'Subgroup' column as "race|age|gender". We split on '|' to extract race.
    """
    path = os.path.join(OUTPUT_TAB, "subgroup_metrics_all_models.csv")
    if not os.path.exists(path):
        print("[SKIP] subgroup_metrics_all_models.csv not found")
        return

    df = pd.read_csv(path)
    nhanes = df[df["Dataset"] == "NHANES"].copy()
    if nhanes.empty or "Subgroup" not in nhanes.columns:
        print("[SKIP] No NHANES subgroup data for heatmap")
        return

    # Extract race from "race|age_group|gender" format
    nhanes["race"] = nhanes["Subgroup"].astype(str).str.split("|").str[0].str.strip()

    # Remove "Overall" or empty race entries
    nhanes = nhanes[~nhanes["race"].isin(["", "nan", "None", "Overall", "U"])]
    if nhanes.empty:
        print("[SKIP] No valid race entries after parsing Subgroup")
        return

    pivot = nhanes.pivot_table(index="race", columns="Model", values="AUC", aggfunc="mean")
    if pivot.empty:
        print("[SKIP] Pivot table empty")
        return

    fig, ax = plt.subplots(figsize=(max(10, len(pivot.columns) * 0.8), max(6, len(pivot) * 0.6)))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn", vmin=0.5, vmax=1.0,
                linewidths=0.5, ax=ax, cbar_kws={"label": "AUC"})
    ax.set_title("AUC by Race/Ethnicity and Model (NHANES)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Model", fontsize=11)
    ax.set_ylabel("Race / Ethnicity", fontsize=11)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_FIG, "subgroup_heatmap_nhanes.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved {out_path}")


def plot_training_curves_combined():
    """Figure 4: 2x3 grid showing training convergence for all 6 adaptation models."""
    histories = {}
    for key in ["history_dann_nhanes", "history_coral_nhanes", "history_mmd_nhanes",
                "history_dann_brfss", "history_coral_brfss", "history_mmd_brfss"]:
        path = os.path.join(OUTPUT_TAB, f"{key}.csv")
        if os.path.exists(path):
            histories[key] = pd.read_csv(path)

    if not histories:
        print("[SKIP] No training history files found")
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for ax, (name, hist) in zip(axes, histories.items()):
        epochs = hist["epoch"] if "epoch" in hist.columns else range(len(hist))

        label_col = [c for c in hist.columns if "label" in c or "total" in c]
        if label_col:
            ax.plot(epochs, hist[label_col[0]], label="Label Loss", linewidth=2, color="navy")

        align_col = [c for c in hist.columns if any(x in c for x in ["domain", "coral", "mmd"])]
        if align_col:
            ax.plot(epochs, hist[align_col[0]], label="Alignment Loss", linewidth=2,
                    color="darkorange", linestyle="--")

        ax.set_title(name.replace("history_", "").replace("_", " ").upper(),
                     fontsize=10, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel("Loss", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Domain Adaptation Training Convergence", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_FIG, "training_curves_combined.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved {out_path}")


def main():
    ensure_dir(OUTPUT_FIG)
    print("=" * 60)
    print("GENERATING SIMPLE COMPARISON FIGURES")
    print("=" * 60)
    plot_comparison_all_models()
    plot_fairness_improvement()
    plot_subgroup_heatmap()
    plot_training_curves_combined()
    print("=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()