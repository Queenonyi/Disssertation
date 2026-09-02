"""NHANES 2017-March 2020 processor."""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from config import NHANES_RACE_MAP, RANDOM_SEED


class NhanesProcessor:
    """Process NHANES prepandemic files: merge, label, stratify, weights."""

    def __init__(self):
        self.scaler = None
        self.demo_cols = ["SEQN", "RIDRETH3", "RIDAGEYR", "RIAGENDR", 
                          "SDMVPSU", "SDMVSTRA", "WTINTPRP", "WTMECPRP"]
        self.bmx_cols = ["SEQN", "BMXBMI"]
        self.glu_cols = ["SEQN", "LBXGLU", "WTSAFPRP"]
        self.ghb_cols = ["SEQN", "LBXGH"]
        self.diq_cols = ["SEQN", "DIQ010"]

    def merge_datasets(self, data_dict):
        """Merge NHANES components on SEQN."""
        dfs = []
        for key in ["demo", "bmx", "glu", "ghb", "diq"]:
            if data_dict.get(key) is not None:
                dfs.append(data_dict[key])
        if not dfs:
            raise ValueError("No NHANES data loaded")

        merged = dfs[0]
        for df in dfs[1:]:
            merged = merged.merge(df, on="SEQN", how="outer")
        return merged

    def build_diabetes_label(self, df):
        """Create diabetes label: DIQ010=1 OR LBXGLU>=126 OR LBXGH>=6.5."""
        cond1 = df["DIQ010"] == 1
        cond2 = df["LBXGLU"] >= 126
        cond3 = df["LBXGH"] >= 6.5
        df["Diabetes"] = (cond1 | cond2 | cond3).astype(int)
        return df

    def filter_adults(self, df, min_age=18):
        df = df[df["RIDAGEYR"] >= min_age].copy()
        return df

    def create_subgroups(self, df):
        """Create intersectional subgroups."""
        df = df.copy()
        df["Race_Ethnicity"] = df["RIDRETH3"].map(NHANES_RACE_MAP)
        df["Age_Group"] = pd.cut(df["RIDAGEYR"], bins=[18, 39, 59, 120], 
                                   labels=["18-39", "40-59", "60+"], include_lowest=True)
        df["Gender_Label"] = df["RIAGENDR"].map({1: "Male", 2: "Female"})
        return df

    def fit_transform(self, data_dict):
        df = self.merge_datasets(data_dict)
        df = self.build_diabetes_label(df)
        df = self.filter_adults(df)
        df = self.create_subgroups(df)

        # Select core variables
        core = ["SEQN", "RIDRETH3", "RIDAGEYR", "RIAGENDR", "BMXBMI",
                "LBXGLU", "LBXGH", "Diabetes", "Race_Ethnicity", "Age_Group", 
                "Gender_Label", "SDMVPSU", "SDMVSTRA", "WTINTPRP", "WTMECPRP"]
        if "WTSAFPRP" in df.columns:
            core.append("WTSAFPRP")

        df = df[[c for c in core if c in df.columns]].copy()
        df = df.dropna(subset=["Diabetes", "BMXBMI", "RIDAGEYR"])

        # Scale continuous variables
        cont_vars = ["RIDAGEYR", "BMXBMI", "LBXGLU", "LBXGH"]
        cont_vars = [c for c in cont_vars if c in df.columns]
        self.scaler = StandardScaler()
        df[[f"{c}_scaled" for c in cont_vars]] = self.scaler.fit_transform(df[cont_vars])

        return df
