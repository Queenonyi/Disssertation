"""Cross-dataset feature alignment for domain adaptation."""
import pandas as pd
import numpy as np


class FeatureAligner:
    """Align features across PIMA, NHANES, and BRFSS into common space."""

    def __init__(self):
        self.feature_map = {
            "age": {"pima": "Age", "nhanes": "RIDAGEYR", "brfss": "Age"},
            "bmi": {"pima": "BMI", "nhanes": "BMXBMI", "brfss": "BMI"},
            "glucose": {"pima": "Glucose", "nhanes": "LBXGLU", "brfss": None},
            "gender": {"pima": "Gender", "nhanes": "RIAGENDR", "brfss": "Sex"},
        }
        self.common_features = ["age", "bmi", "gender"]  # glucose missing in BRFSS

    def align_pima(self, df):
        df = df.copy()
        aligned = pd.DataFrame()
        aligned["age"] = df[self.feature_map["age"]["pima"]]
        aligned["bmi"] = df[self.feature_map["bmi"]["pima"]]
        aligned["glucose"] = df[self.feature_map["glucose"]["pima"]]
        aligned["gender"] = df[self.feature_map["gender"]["pima"]]
        aligned["bp"] = df.get("BloodPressure", np.nan)
        aligned["target"] = df["Outcome"]
        aligned["domain"] = "PIMA"
        aligned["race"] = "Pima Indian"
        aligned["age_group"] = pd.cut(aligned["age"], bins=[18, 39, 59, 120], 
                                      labels=["18-39", "40-59", "60+"], include_lowest=True)
        aligned["gender_label"] = "Female"
        return aligned

    def align_nhanes(self, df):
        df = df.copy()
        aligned = pd.DataFrame()
        aligned["age"] = df[self.feature_map["age"]["nhanes"]]
        aligned["bmi"] = df[self.feature_map["bmi"]["nhanes"]]
        aligned["glucose"] = df.get(self.feature_map["glucose"]["nhanes"], np.nan)
        aligned["gender"] = df[self.feature_map["gender"]["nhanes"]]
        aligned["bp"] = np.nan  # Not available in user's files
        aligned["target"] = df["Diabetes"]
        aligned["domain"] = "NHANES"
        aligned["race"] = df.get("Race_Ethnicity", np.nan)
        aligned["age_group"] = df.get("Age_Group", np.nan)
        aligned["gender_label"] = df.get("Gender_Label", np.nan)
        # Survey weights
        for w in ["WTINTPRP", "WTMECPRP", "WTSAFPRP", "SDMVPSU", "SDMVSTRA"]:
            if w in df.columns:
                aligned[w] = df[w]
        return aligned

    def align_brfss(self, df):
        df = df.copy()
        aligned = pd.DataFrame()
        aligned["age"] = df[self.feature_map["age"]["brfss"]]
        aligned["bmi"] = df[self.feature_map["bmi"]["brfss"]]
        aligned["glucose"] = np.nan  # Not available in BRFSS
        aligned["gender"] = df[self.feature_map["gender"]["brfss"]]
        aligned["bp"] = df.get("HighBP", np.nan)
        aligned["target"] = df["Diabetes_binary"]
        aligned["domain"] = "BRFSS"
        aligned["race"] = np.nan  # Not in cleaned BRFSS standard
        aligned["age_group"] = df.get("Age_Group", np.nan)
        aligned["gender_label"] = df.get("Gender_Label", np.nan)
        return aligned

    def combine(self, pima_df, nhanes_df, brfss_df):
        """Combine aligned datasets."""
        p = self.align_pima(pima_df)
        n = self.align_nhanes(nhanes_df)
        b = self.align_brfss(brfss_df)
        combined = pd.concat([p, n, b], ignore_index=True)
        return combined
