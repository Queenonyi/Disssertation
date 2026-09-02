import pandas as pd
import os
from config import OUTPUT_TAB

path = os.path.join(OUTPUT_TAB, "subgroup_metrics_all_models.csv")
df = pd.read_csv(path)

print("=== Columns ===")
print(df.columns.tolist())
print("\n=== Unique Models ===")
print(df["Model"].unique())
print("\n=== Fairness metrics for NHANES ===")
nhanes = df[df["Dataset"] == "NHANES"]
print(nhanes.groupby("Model")[["Demographic_Parity_Diff", "Equalized_Odds_Diff"]].agg(["mean", "std", "min", "max", "count"]))
print("\n=== Sample rows (non-zero?) ===")
sample = nhanes[nhanes["Demographic_Parity_Diff"] != 0]
print(sample.head(10)[["Model", "Subgroup", "AUC", "Demographic_Parity_Diff", "Equalized_Odds_Diff"]])