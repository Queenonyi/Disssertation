"""Main experimental pipeline for dissertation."""
import os
import sys
import json
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import OUTPUT_TAB, OUTPUT_FIG, OUTPUT_MOD, DATA_PROC, RANDOM_SEED, COMMON_FEATURES
from data.loaders import load_pima, load_nhanes_xpt, load_brfss
from data.pima_processor import PimaProcessor
from data.nhanes_processor import NhanesProcessor
from data.brfss_processor import BrfssProcessor
from data.alignment import FeatureAligner
from models.baselines import BaselineModels
from models.dann import DANNTrainer
from models.coral import CORALTrainer
from models.mmd import MMDTrainer
from models.fairness import FairnessAwareModel, AdaptedModelWrapper, compute_fairness_metrics
from evaluation.metrics import (compute_subgroup_metrics, evaluate_by_subgroup,
                                compute_calibration_error, build_bias_amplification_map)
from evaluation.statistical_tests import delong_test, mcnemar_test
from evaluation.visualizations import (plot_bias_amplification_map, plot_pareto_frontier,
                                       plot_feature_space, plot_calibration_curves,
                                       plot_accuracy_fairness_tradeoff, plot_subgroup_auc_bars)
from consolidated_report import generate_consolidated_report
class DissertationPipeline:
    """Orchestrates the full five-phase experimental pipeline."""
    def __init__(self):
        self.pima_raw = None
        self.nhanes_raw = None
        self.brfss_raw = None
        self.preprocessing_audit = []
        self.pima_proc = None
        self.nhanes_proc = None
        self.brfss_proc = None
        self.pima_df = None
        self.nhanes_df = None
        self.brfss_df = None
        self.combined = None
        self.results = {}
        self.models = {}
        self.imputer = None
        self.scaler = None
        self.feature_cols = None

    def phase1_load_and_process(self):
        """Phase 1: Data Loading and Processing."""
        print("\n" + "=" * 60)
        print("PHASE 1: Data Loading and Processing")
        print("=" * 60)

        # --------------------------------------------------------------
        # INITIALIZE
        # --------------------------------------------------------------
        self.pima_df = None
        self.nhanes_df = None
        self.brfss_df = None
        self.brfss_raw = None
        self.combined = None

        if not hasattr(self, "preprocessing_audit"):
            self.preprocessing_audit = []

        # --------------------------------------------------------------
        # PIMA
        # --------------------------------------------------------------
        try:
            self.pima_raw = load_pima()

            proc = PimaProcessor(strategy="median")
            self.pima_df = proc.fit_transform(self.pima_raw)

            print(f"PIMA loaded: {self.pima_df.shape}")

        except Exception as e:
            print(f"PIMA load failed: {e}")
            self.pima_df = None

        # --------------------------------------------------------------
        # NHANES
        # --------------------------------------------------------------
        try:
            nhanes_dict = load_nhanes_xpt()

            nhanes_proc = NhanesProcessor()
            self.nhanes_df = nhanes_proc.fit_transform(nhanes_dict)

            print(f"NHANES loaded: {self.nhanes_df.shape}")

        except Exception as e:
            print(f"NHANES load failed: {e}")
            self.nhanes_df = None

        # --------------------------------------------------------------
        # BRFSS
        # --------------------------------------------------------------
        try:
            self.brfss_raw, brfss_load_audit = load_brfss(
                chunk_size=50000,
                return_audit=True
            )

            self.preprocessing_audit.extend(brfss_load_audit)

            # ----------------------------------------------------------
            # BRFSS RAW-DATA INTEGRITY CHECK
            # ----------------------------------------------------------
            expected_minimum = 441456

            if len(self.brfss_raw) < expected_minimum:
                raise RuntimeError(
                    f"BRFSS integrity check failed: "
                    f"expected at least {expected_minimum:,} raw records, "
                    f"but loaded {len(self.brfss_raw):,}."
                )

            # ----------------------------------------------------------
            # BRFSS PROCESSING
            # ----------------------------------------------------------
            brfss_proc = BrfssProcessor()

            self.brfss_df = brfss_proc.fit_transform(
                self.brfss_raw
            )

            self.preprocessing_audit.extend(
                brfss_proc.audit
            )

            print(
                f"BRFSS FULL FILE loaded and processed: "
                f"{self.brfss_df.shape}"
            )

        except Exception as e:
            print(f"BRFSS load failed: {e}")
            self.brfss_df = None

        # --------------------------------------------------------------
        # DETERMINE AVAILABLE DATASETS
        # --------------------------------------------------------------
        available = []

        if self.pima_df is not None:
            available.append("PIMA")

        if self.nhanes_df is not None:
            available.append("NHANES")

        if self.brfss_df is not None:
            available.append("BRFSS")

        # --------------------------------------------------------------
        # ALIGN DATASETS
        # --------------------------------------------------------------
        if len(available) >= 2:

            try:
                aligner = FeatureAligner()
                aligned_parts = []

                if self.pima_df is not None:
                    aligned_parts.append(
                        aligner.align_pima(self.pima_df)
                    )

                if self.nhanes_df is not None:
                    aligned_parts.append(
                        aligner.align_nhanes(self.nhanes_df)
                    )

                if self.brfss_df is not None:
                    aligned_parts.append(
                        aligner.align_brfss(self.brfss_df)
                    )

                self.combined = pd.concat(
                    aligned_parts,
                    ignore_index=True
                )

                print(
                    f"Combined aligned data: "
                    f"{self.combined.shape} "
                    f"(from {','.join(available)})"
                )

            except Exception as e:
                print(f"Dataset alignment failed: {e}")
                self.combined = None

        else:
            print(
                "WARNING: Fewer than 2 datasets loaded. "
                "Some phases will be limited."
            )
            self.combined = None

        # --------------------------------------------------------------
        # SAVE PROCESSED DATA
        # --------------------------------------------------------------
        os.makedirs(DATA_PROC, exist_ok=True)

        if self.pima_df is not None:
            self.pima_df.to_csv(
                os.path.join(
                    DATA_PROC,
                    "pima_processed.csv"
                ),
                index=False
            )

        if self.nhanes_df is not None:
            self.nhanes_df.to_csv(
                os.path.join(
                    DATA_PROC,
                    "nhanes_processed.csv"
                ),
                index=False
            )

        if self.brfss_df is not None:
            self.brfss_df.to_csv(
                os.path.join(
                    DATA_PROC,
                    "brfss_processed.csv"
                ),
                index=False
            )

        if self.combined is not None:
            self.combined.to_csv(
                os.path.join(
                    DATA_PROC,
                    "combined_aligned.csv"
                ),
                index=False
            )

        print(
            f"Processed data saved to {DATA_PROC}"
        )

        # --------------------------------------------------------------
        # SAVE PREPROCESSING AUDIT
        # --------------------------------------------------------------
        os.makedirs(OUTPUT_TAB, exist_ok=True)

        audit_df = pd.DataFrame(
            self.preprocessing_audit
        )

        if not audit_df.empty:

            audit_path = os.path.join(
                OUTPUT_TAB,
                "preprocessing_audit.csv"
            )

            audit_df.to_csv(
                audit_path,
                index=False
            )

            print(
                f"Preprocessing audit saved: "
                f"{audit_path}"
            )

        return self

    def _get_aligned_features(
        self,
        df,
        fit_imputer=False,
        imputer=None,
        scaler=None
    ):
        """
        Extract common modelling features.

        When fit_imputer=True:
            preprocessing parameters are fitted ONLY on the source
            dataset (PIMA).

        When fit_imputer=False:
            existing source-fitted preprocessing objects are reused.
        """

        feature_cols = [
            "age",
            "bmi",
            "glucose"
        ]

        df = df.copy()

        # --------------------------------------------------------------
        # ENSURE COMMON FEATURES EXIST
        # --------------------------------------------------------------
        for col in feature_cols:
            if col not in df.columns:
                df[col] = np.nan

        # --------------------------------------------------------------
        # FEATURES
        # --------------------------------------------------------------
        X = df[
            feature_cols
        ].values

        # --------------------------------------------------------------
        # TARGET
        # --------------------------------------------------------------
        y = (
            df["target"].values
            if "target" in df.columns
            else None
        )

        # --------------------------------------------------------------
        # SOURCE FIT
        # --------------------------------------------------------------
        if fit_imputer:

            imputer = SimpleImputer(
                strategy="median"
            )

            X = imputer.fit_transform(X)

            scaler = StandardScaler()

            X = scaler.fit_transform(X)

            # Persist source-fitted preprocessing objects
            self.imputer = imputer
            self.scaler = scaler
            self.feature_cols = feature_cols

            return (
                X,
                y,
                feature_cols,
                imputer,
                scaler
            )

        # --------------------------------------------------------------
        # TARGET TRANSFORMATION
        # --------------------------------------------------------------
        if imputer is None:
            raise ValueError(
                "Target preprocessing requires the "
                "source-fitted imputer."
            )

        if scaler is None:
            raise ValueError(
                "Target preprocessing requires the "
                "source-fitted scaler."
            )

        X = imputer.transform(X)

        X = scaler.transform(X)

        return (
            X,
            y,
            feature_cols
        )

    def phase2_baseline_bias(self):
        """Phase 2: Baseline Bias Quantification."""
        print("\n" + "=" * 60)
        print("PHASE 2: Baseline Bias Quantification")
        print("=" * 60)
        if self.pima_df is None or self.combined is None:
            print("Skipping Phase 2: missing data")
            return self
        
        pima_aligned = self.combined[self.combined["domain"] == "PIMA"].copy()
        if len(pima_aligned) == 0:
            print("No aligned PIMA data found.")
            return self
        
        X_source, y_source, self.feature_cols, self.imputer, self.scaler = self._get_aligned_features(
            pima_aligned, fit_imputer=True
        )
        print(f"Training baselines on aligned PIMA features: {self.feature_cols} ({len(X_source)} samples)")
        
        baselines = BaselineModels(random_state=RANDOM_SEED)
        baselines.train(X_source, y_source)
        self.models["baselines"] = baselines
        
        os.makedirs(OUTPUT_MOD, exist_ok=True)
        import joblib
        for name, model in baselines.models.items():
            joblib.dump(model, os.path.join(OUTPUT_MOD, f"baseline_{name}.pkl"))
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_source, y_source, test_size=0.2, random_state=RANDOM_SEED, stratify=y_source
        )
        baselines_test = BaselineModels(random_state=RANDOM_SEED)
        baselines_test.train(X_train, y_train)
        
        # PIMA test evaluation
        pima_eval = []
        for name, model in baselines_test.models.items():
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
            row = compute_subgroup_metrics(y_test, y_pred, y_prob, subgroup_name="PIMA_Overall")
            row["Model"] = name
            row["Dataset"] = "PIMA_Test"
            pima_eval.append(row)
            
            # Save raw predictions
            pred_df = pd.DataFrame({
                "y_true": y_test,
                "y_pred": y_pred,
                "y_prob": y_prob if y_prob is not None else np.nan,
                "model": name,
                "dataset": "PIMA_Test",
                "phase": "phase2"
            })
            pred_df.to_csv(os.path.join(OUTPUT_TAB, f"predictions_baseline_{name}_pima_test.csv"), index=False)
        
        pima_eval_df = pd.DataFrame(pima_eval)
        pima_eval_df.to_csv(os.path.join(OUTPUT_TAB, "pima_test_metrics.csv"), index=False)
        
        nhanes_bias_rows = []
        if self.nhanes_df is not None and "NHANES" in self.combined["domain"].values:
            nhanes_aligned = self.combined[self.combined["domain"] == "NHANES"].copy()
            X_nhanes, y_nhanes, _ = self._get_aligned_features(nhanes_aligned, imputer=self.imputer, scaler=self.scaler)
            if y_nhanes is not None and len(y_nhanes) > 0:
                eval_df = pd.DataFrame({"y_true": y_nhanes, "domain": "NHANES"})
                for col in ["race", "age_group", "gender_label"]:
                    eval_df[col] = nhanes_aligned[col].values if col in nhanes_aligned.columns else "Unknown"
                eval_df["subgroup"] = (eval_df["race"].astype(str) + " | " +
                                       eval_df["age_group"].astype(str) + " | " +
                                       eval_df["gender_label"].astype(str))
                for name, model in baselines_test.models.items():
                    y_pred = model.predict(X_nhanes)
                    y_prob = model.predict_proba(X_nhanes)[:, 1] if hasattr(model, "predict_proba") else None
                    eval_df["y_pred"] = y_pred
                    eval_df["y_prob"] = y_prob if y_prob is not None else np.nan
                    eval_df["protected"] = eval_df["race"].values
                    nhanes_bias_rows.append(build_bias_amplification_map(eval_df, name, "NHANES"))
                    
                    # Save raw predictions
                    pred_df = eval_df[["y_true", "y_pred", "y_prob", "protected", "subgroup"]].copy()
                    pred_df["model"] = name
                    pred_df["dataset"] = "NHANES"
                    pred_df.to_csv(os.path.join(OUTPUT_TAB, f"predictions_baseline_{name}_nhanes.csv"), index=False)
        
        brfss_bias_rows = []
        if self.brfss_df is not None and "BRFSS" in self.combined["domain"].values:
            brfss_aligned = self.combined[self.combined["domain"] == "BRFSS"].copy()
            X_brfss, y_brfss, _ = self._get_aligned_features(
                brfss_aligned,
                imputer=self.imputer,
                scaler=self.scaler
            )
            if y_brfss is not None and len(y_brfss) > 0:
                eval_df = pd.DataFrame({"y_true": y_brfss, "domain": "BRFSS"})
                for col in ["sex", "age_group", "income", "education"]:
                    eval_df[col] = brfss_aligned[col].values if col in brfss_aligned.columns else "Unknown"
                eval_df["subgroup"] = (eval_df["sex"].astype(str) + " | " +
                                       eval_df["age_group"].astype(str) + " | " +
                                       eval_df["income"].astype(str) + " | " +
                                       eval_df["education"].astype(str))
                for name, model in baselines_test.models.items():
                    y_pred = model.predict(X_brfss)
                    y_prob = model.predict_proba(X_brfss)[:, 1] if hasattr(model, "predict_proba") else None
                    eval_df["y_pred"] = y_pred
                    eval_df["y_prob"] = y_prob if y_prob is not None else np.nan
                    eval_df["protected"] = eval_df["sex"].values
                    brfss_bias_rows.append(build_bias_amplification_map(eval_df, name, "BRFSS"))
                    
                    # Save raw predictions
                    pred_df = eval_df[["y_true", "y_pred", "y_prob", "protected", "subgroup"]].copy()
                    pred_df["model"] = name
                    pred_df["dataset"] = "BRFSS"
                    pred_df.to_csv(os.path.join(OUTPUT_TAB, f"predictions_baseline_{name}_brfss.csv"), index=False)
        
        all_bias = [pima_eval_df]
        if nhanes_bias_rows:
            all_bias.extend(nhanes_bias_rows)
        if brfss_bias_rows:
            all_bias.extend(brfss_bias_rows)
        bias_map = pd.concat(all_bias, ignore_index=True) if all_bias else pd.DataFrame()
        
        if not bias_map.empty:
            bias_map.to_csv(os.path.join(OUTPUT_TAB, "bias_amplification_map.csv"), index=False)
            print(f"Saved bias amplification map: {len(bias_map)} rows")
            pivot_df = plot_bias_amplification_map(bias_map, save_as="bias_amplification_map.png")
            if pivot_df is not None and not pivot_df.empty:
                pivot_df.to_csv(os.path.join(OUTPUT_TAB, "bias_amplification_pivot.csv"), index=False)
        else:
            print("Warning: no bias amplification data generated.")
        
        self.results["phase2"] = bias_map
        print("Phase 2 complete.")
        return self

    def phase3_domain_adaptation(self):
        """Phase 3: Domain Adaptation (DANN, CORAL, MMD)."""
        print("\n" + "=" * 60)
        print("PHASE 3: Domain Adaptation")
        print("=" * 60)
        if self.combined is None or self.pima_df is None:
            print("Skipping Phase 3: missing data")
            return self
        
        pima_aligned = self.combined[self.combined["domain"] == "PIMA"].copy()
        X_source, y_source, _, _, _ = self._get_aligned_features(pima_aligned, fit_imputer=True)
        print(f"Source (PIMA) shape: {X_source.shape}, features: {self.feature_cols}")
        
        nhanes_mask = self.combined["domain"] == "NHANES"
        if nhanes_mask.sum() > 0:
            nhanes_aligned = self.combined.loc[nhanes_mask].copy()
            X_nhanes, _, _ = self._get_aligned_features(nhanes_aligned, imputer=self.imputer, scaler=self.scaler)
            print(f"Training DANN on PIMA -> NHANES ({len(X_source)} -> {len(X_nhanes)})")
            try:
                dann = DANNTrainer(input_dim=len(self.feature_cols), epochs=30)
                dann.fit(X_source, y_source, X_nhanes)
                self.models["DANN_NHANES"] = dann
                import torch
                torch.save(dann, os.path.join(OUTPUT_MOD, "dann_nhanes.pth"))
                hist_df = pd.DataFrame(dann.get_history())
                hist_df.to_csv(os.path.join(OUTPUT_TAB, "history_dann_nhanes.csv"), index=False)
                print("DANN trained.")
            except Exception as e:
                print(f"DANN failed: {e}")
            
            try:
                coral = CORALTrainer(input_dim=len(self.feature_cols), epochs=30)
                coral.fit(X_source, y_source, X_nhanes)
                self.models["CORAL_NHANES"] = coral
                import torch
                torch.save(coral, os.path.join(OUTPUT_MOD, "coral_nhanes.pth"))
                hist_df = pd.DataFrame(coral.get_history())
                hist_df.to_csv(os.path.join(OUTPUT_TAB, "history_coral_nhanes.csv"), index=False)
                print("CORAL trained.")
            except Exception as e:
                print(f"CORAL failed: {e}")
            
            try:
                print("Training MMD on PIMA -> NHANES...")
                mmd = MMDTrainer(input_dim=len(self.feature_cols), epochs=30, mmd_sample_size=2000)
                mmd.fit(X_source, y_source, X_nhanes)
                self.models["MMD_NHANES"] = mmd
                import torch
                torch.save(mmd, os.path.join(OUTPUT_MOD, "mmd_nhanes.pth"))
                hist_df = pd.DataFrame(mmd.get_history())
                hist_df.to_csv(os.path.join(OUTPUT_TAB, "history_mmd_nhanes.csv"), index=False)
                print("MMD trained.")
            except Exception as e:
                print(f"MMD failed: {e}")
        
        brfss_mask = self.combined["domain"] == "BRFSS"
        if brfss_mask.sum() > 0:
            brfss_aligned = self.combined.loc[brfss_mask].copy()
            X_brfss, _, _ = self._get_aligned_features(brfss_aligned, imputer=self.imputer, scaler=self.scaler)
            print(f"Training on PIMA -> BRFSS ({len(X_source)} -> {len(X_brfss)})")
            try:
                dann_b = DANNTrainer(input_dim=len(self.feature_cols), epochs=30)
                dann_b.fit(X_source, y_source, X_brfss)
                self.models["DANN_BRFSS"] = dann_b
                import torch
                torch.save(dann_b, os.path.join(OUTPUT_MOD, "dann_brfss.pth"))
                hist_df = pd.DataFrame(dann_b.get_history())
                hist_df.to_csv(os.path.join(OUTPUT_TAB, "history_dann_brfss.csv"), index=False)
                print("DANN (BRFSS) trained.")
            except Exception as e:
                print(f"DANN BRFSS failed: {e}")
            
            try:
                coral_b = CORALTrainer(input_dim=len(self.feature_cols), epochs=30)
                coral_b.fit(X_source, y_source, X_brfss)
                self.models["CORAL_BRFSS"] = coral_b
                import torch
                torch.save(coral_b, os.path.join(OUTPUT_MOD, "coral_brfss.pth"))
                hist_df = pd.DataFrame(coral_b.get_history())
                hist_df.to_csv(os.path.join(OUTPUT_TAB, "history_coral_brfss.csv"), index=False)
                print("CORAL (BRFSS) trained.")
            except Exception as e:
                print(f"CORAL BRFSS failed: {e}")
            
            try:
                mmd_b = MMDTrainer(input_dim=len(self.feature_cols), epochs=30, mmd_sample_size=2000)
                mmd_b.fit(X_source, y_source, X_brfss)
                self.models["MMD_BRFSS"] = mmd_b
                import torch
                torch.save(mmd_b, os.path.join(OUTPUT_MOD, "mmd_brfss.pth"))
                hist_df = pd.DataFrame(mmd_b.get_history())
                hist_df.to_csv(os.path.join(OUTPUT_TAB, "history_mmd_brfss.csv"), index=False)
                print("MMD (BRFSS) trained.")
            except Exception as e:
                print(f"MMD BRFSS failed: {e}")
        
        print("Phase 3 complete.")
        return self

    def phase4_fairness_aware(self):
        """Phase 4: Fairness-Aware Adaptation."""
        print("\n" + "="*60)
        print("PHASE 4: Fairness-Aware Adaptation")
        print("="*60)
        if self.combined is None:
            print("Skipping Phase 4: no combined data")
            return self
        
        nhanes_mask = self.combined["domain"] == "NHANES"
        if nhanes_mask.sum() == 0:
            print("No NHANES data for fairness adaptation.")
            return self
        
        nhanes_aligned = self.combined.loc[nhanes_mask].copy()
        X_nhanes, y_nhanes, _ = self._get_aligned_features(
            nhanes_aligned,
            imputer=self.imputer,
            scaler=self.scaler
        )
        if y_nhanes is None or len(np.unique(y_nhanes)) < 2:
            print("Insufficient NHANES labels for fairness training.")
            return self
        
        protected = nhanes_aligned.get(
            "race", pd.Series(["Unknown"] * len(nhanes_aligned))
        )
        protected = protected.fillna("Unknown").values.ravel()
        y_nhanes = np.asarray(y_nhanes).ravel().astype(int)
        X_nhanes = np.asarray(X_nhanes)
        
        print(f" DEBUG shapes X={X_nhanes.shape} y={y_nhanes.shape} s={protected.shape}")
        print(f" DEBUG y unique={np.unique(y_nhanes)} s unique={np.unique(protected)}")
        
        fairness_configs = [
            ("DemographicParity", "demographic_parity"),
            ("EqualizedOdds", "equalized_odds"),
        ]
        
        for method in ["DANN", "CORAL", "MMD"]:
            model_key = f"{method}_NHANES"
            if model_key not in self.models:
                continue
            adapted = self.models[model_key]
            wrapper = AdaptedModelWrapper(adapted)
            
            for fairness_label, constraint_name in fairness_configs:
                print(f" Applying {fairness_label} to {method}...")
                try:
                    fair_model = FairnessAwareModel(
                        estimator=wrapper,
                        constraint=constraint_name,
                        eps=0.01,
                    )
                    fair_model.fit(
                        X_nhanes, y_nhanes, sensitive_features=protected
                    )
                    fair_key = f"{method}_{fairness_label}_NHANES"
                    self.models[fair_key] = fair_model
                    print(f" -> Saved as {fair_key}")
                except Exception as e:
                    print(f" -> Failed: {e}")
                    import traceback
                    traceback.print_exc()
        
        print("Phase 4 complete.")
        return self

    def phase5_evaluation(self):
        """Phase 5: Evaluation and Synthesis."""
        print("\n" + "=" * 60)
        print("PHASE 5: Evaluation and Synthesis")
        print("=" * 60)
        os.makedirs(OUTPUT_FIG, exist_ok=True)
        os.makedirs(OUTPUT_TAB, exist_ok=True)
        os.makedirs(OUTPUT_MOD, exist_ok=True)

        if self.combined is None:
            print("No combined data for evaluation.")
            return self

        # ------------------------------------------------------------------
        # Helper: evaluate ONE model on ONE dataset
        # Returns: list of per-subgroup rows, plus ONE dataset-level fairness dict
        # ------------------------------------------------------------------
        def evaluate_model_on_dataset(
            model, model_name, dataset_name, df_aligned, protected_col, subgroup_cols
        ):
            X, y_true, _ = self._get_aligned_features(
                df_aligned, imputer=self.imputer, scaler=self.scaler
            )
            if y_true is None or len(y_true) == 0:
                return [], {}

            # --- Predictions (defensive shape handling) ---
            y_pred = model.predict(X)
            y_pred = np.asarray(y_pred).ravel()

            y_prob = None
            if hasattr(model, "predict_proba"):
                prob_raw = model.predict_proba(X)
                prob_raw = np.asarray(prob_raw)
                if prob_raw.ndim > 1 and prob_raw.shape[1] >= 2:
                    y_prob = prob_raw[:, 1].ravel()
                elif prob_raw.ndim > 1 and prob_raw.shape[1] == 1:
                    y_prob = prob_raw[:, 0].ravel()
                else:
                    y_prob = prob_raw.ravel()

            # --- Build evaluation dataframe ---
            eval_df = pd.DataFrame({
                "y_true": np.asarray(y_true).ravel(),
                "y_pred": y_pred,
                "y_prob": y_prob if y_prob is not None else np.nan,
            })

            if protected_col in df_aligned.columns:
                eval_df["protected"] = df_aligned[protected_col].values
            else:
                eval_df["protected"] = "Unknown"

            for col in subgroup_cols:
                eval_df[col] = df_aligned[col].values if col in df_aligned.columns else "Unknown"

            # Build subgroup label string
            if dataset_name == "NHANES":
                eval_df["subgroup"] = (
                    eval_df["race"].astype(str) + "|" +
                    eval_df["age_group"].astype(str) + "|" +
                    eval_df["gender_label"].astype(str)
                )
            elif dataset_name == "BRFSS":
                eval_df["subgroup"] = (
                    eval_df["sex"].astype(str) + "|" +
                    eval_df["age_group"].astype(str) + "|" +
                    eval_df["income"].astype(str) + "|" +
                    eval_df["education"].astype(str)
                )
            else:
                eval_df["subgroup"] = "Overall"

            # Save raw predictions
            pred_save = eval_df.copy()
            pred_save["model"] = model_name
            pred_save["dataset"] = dataset_name
            pred_save.to_csv(
                os.path.join(OUTPUT_TAB, f"predictions_{model_name}_{dataset_name}.csv"),
                index=False,
            )

            # --- A. Per-subgroup PREDICTIVE metrics (AUC, Accuracy, etc.) ---
            rows = []
            for subgroup, gdf in eval_df.groupby("subgroup"):
                if len(gdf) < 5:          # skip tiny cells
                    continue
                yt = gdf["y_true"].values
                yp = gdf["y_pred"].values
                ypr = gdf["y_prob"].values if gdf["y_prob"].notna().any() else None

                metrics = compute_subgroup_metrics(yt, yp, ypr, subgroup_name=subgroup)
                metrics["Model"] = model_name
                metrics["Dataset"] = dataset_name
                metrics["Subgroup"] = subgroup
                metrics["N"] = len(gdf)
                rows.append(metrics)

            # --- B. Dataset-level FAIRNESS metrics (computed ONCE across all groups) ---
            # This is the fix: compute fairness on the FULL eval_df, not per-subgroup
            fm = {"Demographic_Parity_Diff": np.nan,
                  "Equalized_Odds_Diff": np.nan}

            if eval_df["protected"].notna().any() and eval_df["protected"].nunique() >= 2:
                try:
                    fm = compute_fairness_metrics(
                        eval_df["y_true"].values,
                        eval_df["y_pred"].values,
                        eval_df["protected"].values,
                    )
                except Exception as e:
                    print(f"  Fairness metrics failed for {model_name} on {dataset_name}: {e}")

            # --- C. Attach the SAME fairness numbers to every subgroup row ---
            for row in rows:
                row["Demographic_Parity_Diff"] = fm["Demographic_Parity_Diff"]
                row["Equalized_Odds_Diff"] = fm["Equalized_Odds_Diff"]

            return rows, fm

        # ------------------------------------------------------------------
        # 1. Feature-space visualisations (unchanged from your original)
        # ------------------------------------------------------------------
        print("Generating feature space visualizations...")
        feature_cols_viz = ["age", "bmi"]
        if "glucose" in self.combined.columns and not self.combined["glucose"].isna().all():
            feature_cols_viz.append("glucose")
        X_viz = self.combined[feature_cols_viz].fillna(0).values
        domains = self.combined["domain"].values
        protected = self.combined.get(
            "race", pd.Series([np.nan] * len(self.combined))
        ).values

        try:
            embed_df = plot_feature_space(
                X_viz, domains, protected, method_name="Raw",
                save_as="feature_space_raw.png"
            )
            if embed_df is not None and not embed_df.empty:
                embed_df.to_csv(os.path.join(OUTPUT_TAB, "feature_space_raw.csv"), index=False)
        except Exception as e:
            print(f"Raw feature space plot failed: {e}")

        for method in ["DANN", "CORAL", "MMD"]:
            for target in ["NHANES", "BRFSS"]:
                key = f"{method}_{target}"
                if key not in self.models:
                    continue
                try:
                    model = self.models[key]
                    if hasattr(model, "feature_extractor"):
                        import torch
                        model.feature_extractor.eval()
                        with torch.no_grad():
                            X_t = torch.tensor(X_viz, dtype=torch.float32).to(model.device)
                            feats = model.feature_extractor(X_t).cpu().numpy()
                    elif hasattr(model, "model") and hasattr(model.model, "feature"):
                        import torch
                        model.model.eval()
                        with torch.no_grad():
                            X_t = torch.tensor(X_viz, dtype=torch.float32).to(model.device)
                            _, feats = model.model(X_t)
                            feats = feats.cpu().numpy()
                    else:
                        feats = X_viz

                    embed_df = plot_feature_space(
                        feats, domains, protected, method_name=key,
                        save_as=f"feature_space_{method.lower()}_{target.lower()}.png"
                    )
                    if embed_df is not None and not embed_df.empty:
                        embed_df.to_csv(
                            os.path.join(OUTPUT_TAB, f"feature_space_{method.lower()}_{target.lower()}.csv"),
                            index=False,
                        )
                except Exception as e:
                    print(f"Feature space plot for {key} failed: {e}")

        # ------------------------------------------------------------------
        # 2. Subgroup metrics for every model (FIXED fairness computation)
        # ------------------------------------------------------------------
        print("Computing subgroup metrics for all models...")
        all_eval_rows = []

        # --- Baselines ---
        if "baselines" in self.models:
            for name, model in self.models["baselines"].models.items():
                if self.nhanes_df is not None:
                    nhanes_aligned = self.combined[self.combined["domain"] == "NHANES"].copy()
                    rows, _ = evaluate_model_on_dataset(
                        model, f"Baseline_{name}", "NHANES", nhanes_aligned,
                        "race", ["race", "age_group", "gender_label"]
                    )
                    all_eval_rows.extend(rows)
                if self.brfss_df is not None:
                    brfss_aligned = self.combined[self.combined["domain"] == "BRFSS"].copy()
                    rows, _ = evaluate_model_on_dataset(
                        model, f"Baseline_{name}", "BRFSS", brfss_aligned,
                        "sex", ["sex", "age_group", "income", "education"]
                    )
                    all_eval_rows.extend(rows)

        # --- Domain-adapted models (Phase 3) ---
        for key in [
            "DANN_NHANES", "CORAL_NHANES", "MMD_NHANES",
            "DANN_BRFSS", "CORAL_BRFSS", "MMD_BRFSS",
        ]:
            if key not in self.models:
                continue
            model = self.models[key]
            target = key.split("_")[1]
            df_aligned = self.combined[self.combined["domain"] == target].copy()
            protected_col = "race" if target == "NHANES" else "sex"
            subgroup_cols = (
                ["race", "age_group", "gender_label"] if target == "NHANES"
                else ["sex", "age_group", "income", "education"]
            )
            rows, _ = evaluate_model_on_dataset(
                model, key, target, df_aligned, protected_col, subgroup_cols
            )
            all_eval_rows.extend(rows)

        # --- Fairness-aware models (Phase 4) ---
        for key in self.models:
            if "DemographicParity" in key or "EqualizedOdds" in key:
                target = key.split("_")[-1]
                df_aligned = self.combined[self.combined["domain"] == target].copy()
                protected_col = "race" if target == "NHANES" else "sex"
                subgroup_cols = (
                    ["race", "age_group", "gender_label"] if target == "NHANES"
                    else ["sex", "age_group", "income", "education"]
                )
                rows, _ = evaluate_model_on_dataset(
                    self.models[key], key, target, df_aligned, protected_col, subgroup_cols
                )
                all_eval_rows.extend(rows)

        # Save master subgroup table
        eval_df = pd.DataFrame(all_eval_rows)
        if not eval_df.empty:
            eval_df.to_csv(
                os.path.join(OUTPUT_TAB, "subgroup_metrics_all_models.csv"),
                index=False,
            )
            print(f"Saved subgroup metrics: {len(eval_df)} rows")

            auc_pivot = plot_subgroup_auc_bars(eval_df, save_as="subgroup_auc_bars.png")
            if auc_pivot is not None and not auc_pivot.empty:
                auc_pivot.to_csv(os.path.join(OUTPUT_TAB, "subgroup_auc_pivot.csv"), index=False)

            # --- Pareto frontier (group by Model, use the dataset-level fairness values) ---
            pareto_df = eval_df.groupby("Model").agg({
                "AUC": "mean",
                "Demographic_Parity_Diff": "first",   # same for all subgroups of this model
                "Equalized_Odds_Diff": "first",
            }).reset_index()

            if not pareto_df.empty:
                pareto_df.to_csv(os.path.join(OUTPUT_TAB, "pareto_frontier_data.csv"), index=False)

                pareto_out = plot_pareto_frontier(
                    pareto_df, acc_col="AUC", fairness_col="Demographic_Parity_Diff",
                    label_col="Model", title="AUC vs Demographic Parity",
                    save_as="pareto_frontier_auc_dp.png",
                )
                if pareto_out is not None:
                    pareto_out.to_csv(os.path.join(OUTPUT_TAB, "pareto_frontier_auc_dp.csv"), index=False)

                pareto_out2 = plot_pareto_frontier(
                    pareto_df, acc_col="AUC", fairness_col="Equalized_Odds_Diff",
                    label_col="Model", title="AUC vs Equalized Odds",
                    save_as="pareto_frontier_auc_eo.png",
                )
                if pareto_out2 is not None:
                    pareto_out2.to_csv(os.path.join(OUTPUT_TAB, "pareto_frontier_auc_eo.csv"), index=False)

        # ------------------------------------------------------------------
        # 3. Calibration curves (unchanged)
        # ------------------------------------------------------------------
        print("Generating calibration curves...")
        for key in ["Baseline_LogReg", "DANN_NHANES", "CORAL_NHANES", "MMD_NHANES"]:
            if key == "Baseline_LogReg" and "baselines" in self.models:
                model = self.models["baselines"].models.get("LogReg")
                if model is None:
                    continue
                target = "NHANES"
            elif key in self.models:
                model = self.models[key]
                target = key.split("_")[1]
            else:
                continue

            df_aligned = self.combined[self.combined["domain"] == target].copy()
            X, y_true, _ = self._get_aligned_features(
                df_aligned, imputer=self.imputer, scaler=self.scaler
            )
            if y_true is None:
                continue

            y_prob = None
            if hasattr(model, "predict_proba"):
                prob_raw = model.predict_proba(X)
                if prob_raw.ndim > 1 and prob_raw.shape[1] >= 2:
                    y_prob = np.asarray(prob_raw[:, 1]).ravel()
                else:
                    y_prob = np.asarray(prob_raw).ravel()
            if y_prob is None:
                continue

            protected_vals = df_aligned.get(
                "race" if target == "NHANES" else "sex",
                pd.Series(["Unknown"] * len(df_aligned)),
            ).values

            cal_df = pd.DataFrame({
                "y_true": np.asarray(y_true).ravel(),
                "y_prob": y_prob,
                "protected": protected_vals,
            })

            cal_data = plot_calibration_curves(
                cal_df, "y_prob", "y_true", "protected",
                save_as=f"calibration_{key.lower()}.png",
            )
            if cal_data is not None and not cal_data.empty:
                cal_data.to_csv(
                    os.path.join(OUTPUT_TAB, f"calibration_data_{key.lower()}.csv"),
                    index=False,
                )

        # ------------------------------------------------------------------
        # 4. Statistical tests (unchanged)
        # ------------------------------------------------------------------
        print("Running statistical tests...")
        test_results = []
        nhanes_aligned = self.combined[self.combined["domain"] == "NHANES"].copy()
        X_nhanes, y_nhanes, _ = self._get_aligned_features(
            nhanes_aligned, imputer=self.imputer, scaler=self.scaler
        )

        if y_nhanes is not None and len(y_nhanes) > 0:
            baseline_probs = {}
            if "baselines" in self.models:
                for name, model in self.models["baselines"].models.items():
                    if hasattr(model, "predict_proba"):
                        baseline_probs[name] = model.predict_proba(X_nhanes)[:, 1]

            adapted_probs = {}
            for key in ["DANN_NHANES", "CORAL_NHANES", "MMD_NHANES"]:
                if key in self.models and hasattr(self.models[key], "predict_proba"):
                    prob_raw = self.models[key].predict_proba(X_nhanes)
                    if prob_raw.ndim > 1 and prob_raw.shape[1] >= 2:
                        adapted_probs[key] = prob_raw[:, 1]
                    else:
                        adapted_probs[key] = np.asarray(prob_raw).ravel()

            for b_name, b_prob in baseline_probs.items():
                for a_name, a_prob in adapted_probs.items():
                    try:
                        res = delong_test(y_nhanes, b_prob, a_prob)
                        test_results.append({
                            "Test": "DeLong",
                            "Model1": f"Baseline_{b_name}",
                            "Model2": a_name,
                            "AUC1": res["auc1"],
                            "AUC2": res["auc2"],
                            "p_value": res["p_value"],
                            "z_statistic": res["z_statistic"],
                            "Dataset": "NHANES",
                        })
                    except Exception as e:
                        print(f"DeLong test failed for {b_name} vs {a_name}: {e}")

            for b_name, b_prob in baseline_probs.items():
                b_pred = (b_prob > 0.5).astype(int)
                for a_name, a_prob in adapted_probs.items():
                    a_pred = (a_prob > 0.5).astype(int)
                    try:
                        res = mcnemar_test(y_nhanes, b_pred, a_pred)
                        test_results.append({
                            "Test": "McNemar",
                            "Model1": f"Baseline_{b_name}",
                            "Model2": a_name,
                            "p_value": res["p_value"],
                            "statistic": res["statistic"],
                            "b": res["b"],
                            "c": res["c"],
                            "Dataset": "NHANES",
                        })
                    except Exception as e:
                        print(f"McNemar test failed for {b_name} vs {a_name}: {e}")

            if test_results:
                test_df = pd.DataFrame(test_results)
                test_df.to_csv(
                    os.path.join(OUTPUT_TAB, "statistical_tests.csv"), index=False
                )
                print(f"Saved statistical tests: {len(test_df)} rows")

        # ------------------------------------------------------------------
        # 5. Model-status summaries (unchanged)
        # ------------------------------------------------------------------
        model_list = [
            {"Model": k, "Status": "Trained"}
            for k in self.models.keys() if k != "baselines"
        ]
        if model_list:
            pd.DataFrame(model_list).to_csv(
                os.path.join(OUTPUT_TAB, "trained_models.csv"), index=False
            )

        if "baselines" in self.models:
            baseline_results = []
            for name, model in self.models["baselines"].models.items():
                baseline_results.append({
                    "Model": name,
                    "Phase": "Baseline_Trained_on_PIMA",
                })
            pd.DataFrame(baseline_results).to_csv(
                os.path.join(OUTPUT_TAB, "baseline_models_summary.csv"),
                index=False,
            )

        # ------------------------------------------------------------------
        # 6. Output manifest (unchanged)
        # ------------------------------------------------------------------
        summary = {
            "processed_data": ["pima_processed.csv", "nhanes_processed.csv", "brfss_processed.csv", "combined_aligned.csv"],
            "tables": [
                "pima_test_metrics.csv",
                "bias_amplification_map.csv",
                "bias_amplification_pivot.csv",
                "history_dann_nhanes.csv", "history_coral_nhanes.csv", "history_mmd_nhanes.csv",
                "history_dann_brfss.csv", "history_coral_brfss.csv", "history_mmd_brfss.csv",
                "subgroup_metrics_all_models.csv",
                "subgroup_auc_pivot.csv",
                "pareto_frontier_data.csv",
                "pareto_frontier_auc_dp.csv",
                "pareto_frontier_auc_eo.csv",
                "calibration_data_baseline_logreg.csv",
                "calibration_data_dann_nhanes.csv",
                "calibration_data_coral_nhanes.csv",
                "calibration_data_mmd_nhanes.csv",
                "statistical_tests.csv",
                "trained_models.csv",
                "baseline_models_summary.csv",
            ],
            "predictions": [
                "predictions_baseline_LogReg_pima_test.csv",
                "predictions_baseline_RandomForest_pima_test.csv",
                "predictions_baseline_NeuralNet_pima_test.csv",
                "predictions_baseline_LogReg_nhanes.csv",
                "predictions_baseline_RandomForest_nhanes.csv",
                "predictions_baseline_NeuralNet_nhanes.csv",
                "predictions_baseline_LogReg_brfss.csv",
                "predictions_baseline_RandomForest_brfss.csv",
                "predictions_baseline_NeuralNet_brfss.csv",
            ],
            "embeddings": [
                "feature_space_raw.csv",
                "feature_space_dann_nhanes.csv", "feature_space_dann_brfss.csv",
                "feature_space_coral_nhanes.csv", "feature_space_coral_brfss.csv",
                "feature_space_mmd_nhanes.csv", "feature_space_mmd_brfss.csv",
            ],
        }
        with open(os.path.join(OUTPUT_TAB, "output_manifest.json"), "w") as f:
            json.dump(summary, f, indent=2)

        print("Phase 5 complete.")
        return self

    def phase6_generate_consolidated_report(self):
        """Phase 6: Generate consolidated .txt report for easy reference."""
        print("\n" + "=" * 60)
        print("PHASE 6: Generating Consolidated Report")
        print("=" * 60)
        try:
            from consolidated_report import generate_consolidated_report
            report_path = generate_consolidated_report(
                output_dir=OUTPUT_TAB, 
                filename="consolidated_report.txt"
            )
            print(f"Consolidated report saved to: {report_path}")
        except Exception as e:
            print(f"Failed: {e}")
            import traceback
            traceback.print_exc()
        print("Phase 6 complete.")
        return self

    def run_all(self):
        self.phase1_load_and_process()
        self.phase2_baseline_bias()
        self.phase3_domain_adaptation()
        self.phase4_fairness_aware()
        self.phase5_evaluation()
        self.phase6_generate_consolidated_report()
        print("\n" + "=" * 60)
        print("FULL PIPELINE COMPLETE")
        print("=" * 60)
        return self
    
if __name__ == "__main__":
    pipe = DissertationPipeline()
    pipe.run_all()