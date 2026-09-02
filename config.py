"""Project configuration and constants."""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW = os.path.join(BASE_DIR, "data", "raw")
DATA_PROC = os.path.join(BASE_DIR, "data", "processed")
OUTPUT_FIG = os.path.join(BASE_DIR, "outputs", "figures")
OUTPUT_TAB = os.path.join(BASE_DIR, "outputs", "tables")
OUTPUT_MOD = os.path.join(BASE_DIR, "outputs", "models")

# Random seed for reproducibility
RANDOM_SEED = 42

# PIMA
PIMA_PATH = os.path.join(DATA_RAW, "diabetes.csv")
PIMA_FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                 "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
PIMA_TARGET = "Outcome"

# NHANES files
NHANES_FILES = {
    "demo": os.path.join(DATA_RAW, "P_DEMO.xpt"),
    "bmx": os.path.join(DATA_RAW, "P_BMX.xpt"),
    "glu": os.path.join(DATA_RAW, "P_GLU.xpt"),
    "ghb": os.path.join(DATA_RAW, "P_GHB.xpt"),
    "diq": os.path.join(DATA_RAW, "P_DIQ.xpt"),
}

# BRFSS
BRFSS_PATH = os.path.join(DATA_RAW, "LLCP2015.XPT")
BRFSS_ASC = os.path.join(DATA_RAW, "LLCP2015.ASC")

# Feature alignment
COMMON_FEATURES = {
    "age": {"pima": "Age", "nhanes": "RIDAGEYR", "brfss": "Age"},
    "bmi": {"pima": "BMI", "nhanes": "BMXBMI", "brfss": "BMI"},
    "glucose": {"pima": "Glucose", "nhanes": "LBXGLU", "brfss": None},
    "bp": {"pima": "BloodPressure", "nhanes": None, "brfss": "HighBP"},
    "gender": {"pima": "Gender", "nhanes": "RIAGENDR", "brfss": "Sex"},
}

# Subgroup definitions
NHANES_RACE_MAP = {
    1: "Mexican American",
    2: "Other Hispanic",
    3: "Non-Hispanic White",
    4: "Non-Hispanic Black",
    6: "Non-Hispanic Asian",
    7: "Other Race"
}

NHANES_AGE_TERTILES = [18, 40, 60, 120]
BRFSS_AGE_CATS = [18, 40, 60, 120]

# Model hyperparameters
BASELINE_PARAMS = {
    "logreg": {"C": 1.0, "class_weight": "balanced", "max_iter": 1000, "random_state": 42},
    "rf": {"n_estimators": 200, "max_depth": 10, "class_weight": "balanced", "random_state": 42},
    "nn": {"hidden": [64, 32], "dropout": 0.3, "epochs": 100, "batch_size": 32, "lr": 1e-3}
}

DANN_PARAMS = {"lambda": 1.0, "lr": 1e-3, "epochs": 100, "batch_size": 32}
CORAL_PARAMS = {"alpha": 1.0, "lr": 1e-3, "epochs": 100, "batch_size": 32}
MMD_PARAMS = {"beta": 1.0, "kernel": "rbf", "lr": 1e-3, "epochs": 100, "batch_size": 32}
