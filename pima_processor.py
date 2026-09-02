"""PIMA Indians Diabetes Dataset processor."""
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from config import PIMA_FEATURES, PIMA_TARGET, RANDOM_SEED


class PimaProcessor:
    """Process PIMA dataset: handle zero-valued missings, impute, scale."""

    ZERO_MISSING_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

    def __init__(self, strategy="median"):
        self.strategy = strategy
        self.imputer = None
        self.scaler = None

    def fit_transform(self, df_raw):
        df = df_raw.copy()
        # Replace biologically impossible zeros with NaN
        for col in self.ZERO_MISSING_COLS:
            if col in df.columns:
                df[col] = df[col].replace(0, np.nan)

        # Impute missing values
        self.imputer = SimpleImputer(strategy=self.strategy)
        df[PIMA_FEATURES] = self.imputer.fit_transform(df[PIMA_FEATURES])

        # Add gender constant (all female = 2 to match NHANES coding, or 0/1)
        df["Gender"] = 2  # NHANES: 1=Male, 2=Female

        # Scale features
        self.scaler = StandardScaler()
        df[[f"{c}_scaled" for c in PIMA_FEATURES]] = self.scaler.fit_transform(df[PIMA_FEATURES])

        return df

    def transform(self, df_raw):
        df = df_raw.copy()
        for col in self.ZERO_MISSING_COLS:
            if col in df.columns:
                df[col] = df[col].replace(0, np.nan)
        df[PIMA_FEATURES] = self.imputer.transform(df[PIMA_FEATURES])
        df["Gender"] = 2
        df[[f"{c}_scaled" for c in PIMA_FEATURES]] = self.scaler.transform(df[PIMA_FEATURES])
        return df
