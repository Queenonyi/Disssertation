"""BRFSS 2015 raw data processor."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


class BrfssProcessor:
    """Process raw BRFSS 2015 XPT/ASCII into cleaned ML-ready format."""

    # Raw variable names in BRFSS 2015
    RAW_VARS = {
        "diabetes": "DIABETE3",
        "sex": "SEX",
        "age": "_AGE80",  # or AGE
        "bmi": "_BMI5",   # calculated BMI * 100 usually
        "high_bp": "_RFHYPE5",
        "high_chol": "TOLDHI2",
        "chol_check": "CHOLCHK",
        "smoker": "SMOKE100",
        "stroke": "CVDSTRK3",
        "heart_disease": "CVDINFR4",
        "phys_activity": "EXERANY2",
        "fruits": "FRUIT1",
        "veggies": "VEGETAB1",
        "heavy_alc": "_RFDRHV5",
        "healthcare": "HLTHPLN1",
        "no_doc_cost": "MEDCOST",
        "gen_health": "GENHLTH",
        "mental_health": "MENTHLTH",
        "physical_health": "PHYSHLTH",
        "diff_walk": "DIFFWALK",
        "income": "INCOME2",
        "education": "EDUCA",
    }

    def __init__(self):
        self.scaler = None
        self.audit = []

    def _clean_diabetes(self, df):
        """Map DIABETE3 to binary: 1=prediabetes/diabetes, 0=no diabetes."""
        d = df[self.RAW_VARS["diabetes"]].copy()

        # BRFSS 2015:
        # 1 = Yes
        # 2 = Yes, pregnancy
        # 3 = No
        # 4 = No, prediabetes
        # 7 = Don't know
        # 9 = Refused
        #
        # Existing cleaning logic preserved:
        # 0 = no diabetes
        # 1 = prediabetes or diabetes
        mapping = {
            1: 1,
            2: 1,
            4: 1,
            3: 0
        }

        d = d.map(mapping)

        return d

    def _clean_bmi(self, df):
        """BMI in BRFSS is often _BMI5 (BMI*100) or BMI."""
        if "_BMI5" in df.columns:
            return df["_BMI5"] / 100.0

        elif "BMI" in df.columns:
            return df["BMI"]

        else:
            return np.nan

    def _clean_age(self, df):
        """Extract BRFSS age using _AGE80 or AGE."""
        if "_AGE80" in df.columns:
            return df["_AGE80"]

        elif "AGE" in df.columns:
            return df["AGE"]

        else:
            return np.nan

    def fit_transform(self, df_raw):
        """
        Clean and engineer BRFSS variables while recording an audit trail.

        The existing cleaning and feature-engineering logic is preserved.
        This method additionally records what happened at each stage.
        """

        # Reset audit for every processing run.
        self.audit = []

        def record(
            stage,
            operation,
            before,
            after,
            variables_before,
            variables_after
        ):
            """Record one preprocessing operation."""
            self.audit.append({
                "Dataset": "BRFSS",
                "Stage": stage,
                "Operation": operation,
                "Records_Before": before,
                "Records_After": after,
                "Records_Removed": before - after,
                "Variables_Before": variables_before,
                "Variables_After": variables_after,
                "Source": "CDC BRFSS 2015 LLCP2015.XPT"
            })

        # --------------------------------------------------------------
        # A. RAW INGESTION
        # --------------------------------------------------------------
        df = df_raw.copy()

        record(
            "Raw ingestion",
            "Full CDC XPT",
            len(df),
            len(df),
            len(df.columns),
            len(df.columns)
        )

        # --------------------------------------------------------------
        # B. VARIABLE EXTRACTION
        # --------------------------------------------------------------
        cleaned = pd.DataFrame(index=df.index)

        cleaned["Diabetes_binary"] = self._clean_diabetes(df)

        cleaned["Sex"] = df.get(
            self.RAW_VARS["sex"]
        )

        cleaned["Age"] = self._clean_age(df)

        cleaned["BMI"] = self._clean_bmi(df)

        cleaned["HighBP"] = df.get(
            self.RAW_VARS["high_bp"]
        )

        cleaned["HighChol"] = df.get(
            self.RAW_VARS["high_chol"]
        )

        cleaned["CholCheck"] = df.get(
            self.RAW_VARS["chol_check"]
        )

        cleaned["Smoker"] = df.get(
            self.RAW_VARS["smoker"]
        )

        cleaned["Stroke"] = df.get(
            self.RAW_VARS["stroke"]
        )

        cleaned["HeartDiseaseOrAttack"] = df.get(
            self.RAW_VARS["heart_disease"]
        )

        cleaned["PhysActivity"] = df.get(
            self.RAW_VARS["phys_activity"]
        )

        cleaned["Fruits"] = df.get(
            self.RAW_VARS["fruits"]
        )

        cleaned["Veggies"] = df.get(
            self.RAW_VARS["veggies"]
        )

        cleaned["HvyAlcoholConsump"] = df.get(
            self.RAW_VARS["heavy_alc"]
        )

        cleaned["AnyHealthcare"] = df.get(
            self.RAW_VARS["healthcare"]
        )

        cleaned["NoDocbcCost"] = df.get(
            self.RAW_VARS["no_doc_cost"]
        )

        cleaned["GenHlth"] = df.get(
            self.RAW_VARS["gen_health"]
        )

        cleaned["MentHlth"] = df.get(
            self.RAW_VARS["mental_health"]
        )

        cleaned["PhysHlth"] = df.get(
            self.RAW_VARS["physical_health"]
        )

        cleaned["DiffWalk"] = df.get(
            self.RAW_VARS["diff_walk"]
        )

        cleaned["Income"] = df.get(
            self.RAW_VARS["income"]
        )

        cleaned["Education"] = df.get(
            self.RAW_VARS["education"]
        )

        record(
            "Variable extraction",
            "Extract BRFSS analysis variables",
            len(df),
            len(cleaned),
            len(df.columns),
            len(cleaned.columns)
        )

        # --------------------------------------------------------------
        # C. TARGET CLEANING
        # --------------------------------------------------------------
        before = len(cleaned)

        cleaned = cleaned.dropna(
            subset=["Diabetes_binary"]
        ).copy()

        record(
            "Target cleaning",
            "Remove unknown/refused diabetes status",
            before,
            len(cleaned),
            len(cleaned.columns),
            len(cleaned.columns)
        )

        # --------------------------------------------------------------
        # D. FEATURE ENGINEERING
        # --------------------------------------------------------------
        before_cols = len(cleaned.columns)

        cleaned["Age_Group"] = pd.cut(
            cleaned["Age"],
            bins=[18, 39, 59, 120],
            labels=[
                "18-39",
                "40-59",
                "60+"
            ],
            include_lowest=True
        )

        cleaned["Gender_Label"] = cleaned["Sex"].map({
            1: "Male",
            2: "Female"
        })

        record(
            "Feature engineering",
            "Create age groups and gender labels",
            len(cleaned),
            len(cleaned),
            before_cols,
            len(cleaned.columns)
        )

        # --------------------------------------------------------------
        # E. CONTINUOUS-VARIABLE TRANSFORMATION
        # --------------------------------------------------------------
        cont = [
            "Age",
            "BMI",
            "MentHlth",
            "PhysHlth"
        ]

        cont = [
            c for c in cont
            if c in cleaned.columns
        ]

        self.scaler = StandardScaler()

        cleaned[
            [f"{c}_scaled" for c in cont]
        ] = self.scaler.fit_transform(
            cleaned[cont]
        )

        record(
            "Feature engineering",
            "Standardise continuous BRFSS variables",
            len(cleaned),
            len(cleaned),
            before_cols,
            len(cleaned.columns)
        )

        # --------------------------------------------------------------
        # F. FINAL PROCESSING RESULT
        # --------------------------------------------------------------
        record(
            "Final analytical BRFSS",
            "BRFSS preprocessing complete",
            len(cleaned),
            len(cleaned),
            len(cleaned.columns),
            len(cleaned.columns)
        )

        return cleaned