"""Automated retraining pipeline for the LightGBM credit scoring model.

Detects data drift using Evidently AI, retrains if needed, validates the
candidate model, and creates a human-in-the-loop approval gate.

Usage:
    python pipeline/retrain.py [--production-data PATH]
"""

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"
DATA_DIR = BASE_DIR / "data"
PIPELINE_DIR = BASE_DIR / "pipeline"

REFERENCE_DATA_PATH = DATA_DIR / "reference_data.parquet"
PRODUCTION_DATA_PATH = DATA_DIR / "production_sample.parquet"
TEST_DATA_PATH = DATA_DIR / "test_data.parquet"
CHAMPION_MODEL_PATH = MODEL_DIR / "model.lgb"
CANDIDATE_MODEL_PATH = MODEL_DIR / "model_candidate.lgb"
VALIDATION_REPORT_PATH = PIPELINE_DIR / "validation_report.json"
PENDING_APPROVAL_PATH = PIPELINE_DIR / "pending_approval.json"
DATASET_REGISTRY_PATH = BASE_DIR / "data" / "dataset_registry.json"

# Path to original training data (optional fallback)
TRAINING_CSV_PATH = BASE_DIR / "data" / "application_train.csv"


# ---------------------------------------------------------------------------
# Step 1 — Drift detection
# ---------------------------------------------------------------------------
def detect_drift(
    reference_df: pd.DataFrame, production_df: pd.DataFrame
) -> tuple[bool, dict]:
    """Run Evidently DataDriftPreset and return (is_drifted, report_dict)."""
    print("[Step 1] Running data drift detection ...")

    # Use only numeric feature columns present in both datasets
    # Exclude SK_ID_CURR (ID) and TARGET (label) — they're not features
    exclude = {"SK_ID_CURR", "TARGET"}
    common_cols = sorted(
        (set(reference_df.select_dtypes(include="number").columns)
         & set(production_df.select_dtypes(include="number").columns))
        - exclude
    )
    if not common_cols:
        print("  WARNING: No common numeric columns found. Skipping drift check.")
        return False, {}

    ref = reference_df[common_cols]
    prod = production_df[common_cols]

    column_mapping = ColumnMapping()

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref, current_data=prod, column_mapping=column_mapping)

    result = report.as_dict()

    # Extract top-level drift flag
    dataset_drift = False
    try:
        for metric_result in result.get("metrics", []):
            metric_data = metric_result.get("result", {})
            if "dataset_drift" in metric_data:
                dataset_drift = bool(metric_data["dataset_drift"])
                break
    except (KeyError, TypeError):
        pass

    drift_summary = {
        "dataset_drift": dataset_drift,
        "n_features_checked": len(common_cols),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if dataset_drift:
        print("  DRIFT DETECTED across the dataset.")
    else:
        print("  No significant drift detected.")

    return dataset_drift, drift_summary


# ---------------------------------------------------------------------------
# Step 2 — Retrain model
# ---------------------------------------------------------------------------
def retrain_model(
    reference_df: pd.DataFrame, production_df: pd.DataFrame
) -> lgb.Booster:
    """Train a new LightGBM model on reference + production data."""
    print("[Step 2] Retraining model ...")

    # Combine reference data (historical) with production data (new population)
    # This simulates real production: we retrain on all available labeled data
    # including the new data that caused the drift.
    if "TARGET" in reference_df.columns and "TARGET" in production_df.columns:
        print("  Combining reference + production data for retraining ...")
        combined = pd.concat([reference_df, production_df], ignore_index=True)
        labels = combined["TARGET"]
        features = combined.drop(columns=["TARGET", "SK_ID_CURR"], errors="ignore")
        features = features.select_dtypes(include="number")
    elif "TARGET" in reference_df.columns:
        # Production data has no labels — use reference only + load labels for production via CSV
        print("  Production data has no TARGET — loading labels from training CSV ...")
        if TRAINING_CSV_PATH.exists() and "SK_ID_CURR" in production_df.columns:
            train_csv = pd.read_csv(TRAINING_CSV_PATH, usecols=["SK_ID_CURR", "TARGET"])
            prod_with_labels = production_df.merge(train_csv, on="SK_ID_CURR", how="left")
            combined = pd.concat([reference_df, prod_with_labels], ignore_index=True)
        else:
            print("  WARNING: Cannot match production labels. Using reference data only.")
            combined = reference_df
        labels = combined["TARGET"]
        features = combined.drop(columns=["TARGET", "SK_ID_CURR"], errors="ignore")
        features = features.select_dtypes(include="number")
    else:
        print(
            f"  WARNING: No TARGET column found. "
            "Simulating labels with random values for demonstration."
        )
        features = reference_df.select_dtypes(include="number")
        rng = np.random.RandomState(42)
        labels = pd.Series(rng.randint(0, 2, size=len(features)))

    # Drop rows where the label is missing
    valid_mask = labels.notna()
    features = features.loc[valid_mask].reset_index(drop=True)
    labels = labels.loc[valid_mask].reset_index(drop=True).astype(int)

    print(f"  Training set: {len(features)} rows, {features.shape[1]} features")

    # LightGBM parameters — reasonable defaults for binary credit scoring
    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "n_jobs": -1,
        "verbose": -1,
        "is_unbalance": True,
    }

    train_data = lgb.Dataset(features, label=labels, free_raw_data=False)
    booster = lgb.train(params, train_data, num_boost_round=200)

    # Save candidate model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(CANDIDATE_MODEL_PATH))
    print(f"  Candidate model saved to {CANDIDATE_MODEL_PATH}")

    return booster


# ---------------------------------------------------------------------------
# Step 3 — Validate candidate model
# ---------------------------------------------------------------------------
def validate_candidate(candidate: lgb.Booster) -> dict:
    """Compare candidate model against the champion on test data."""
    print("[Step 3] Validating candidate model ...")

    validation = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "candidate_path": str(CANDIDATE_MODEL_PATH),
        "champion_path": str(CHAMPION_MODEL_PATH),
    }

    # Load test data
    if TEST_DATA_PATH.exists():
        test_df = pd.read_parquet(TEST_DATA_PATH)
        print(f"  Test data loaded: {test_df.shape}")
    else:
        print(f"  WARNING: Test data not found at {TEST_DATA_PATH}. Using reference data.")
        test_df = pd.read_parquet(REFERENCE_DATA_PATH)

    test_numeric = test_df.select_dtypes(include="number")
    if "SK_ID_CURR" in test_numeric.columns:
        test_numeric = test_numeric.drop(columns=["SK_ID_CURR"])
    if "TARGET" in test_numeric.columns:
        test_labels = test_numeric["TARGET"].values
        test_numeric = test_numeric.drop(columns=["TARGET"])
        has_labels = True
    else:
        test_labels = None
        has_labels = False

    # Candidate predictions — align to candidate's feature set
    candidate_feature_names = candidate.feature_name()
    candidate_test = test_numeric.reindex(columns=candidate_feature_names)
    candidate_preds = candidate.predict(candidate_test)

    # Champion predictions — align to champion's 795-feature set
    if CHAMPION_MODEL_PATH.exists():
        champion = lgb.Booster(model_file=str(CHAMPION_MODEL_PATH))
        champion_feature_names = champion.feature_name()
        champion_test = test_numeric.reindex(columns=champion_feature_names)
        champion_preds = champion.predict(champion_test)
    else:
        print("  WARNING: No champion model found. Skipping comparison.")
        champion_preds = None

    # Compute metrics
    if has_labels:
        from sklearn.metrics import roc_auc_score

        candidate_auc = float(roc_auc_score(test_labels, candidate_preds))
        validation["candidate_auc"] = round(candidate_auc, 6)
        print(f"  Candidate AUC: {candidate_auc:.6f}")

        if champion_preds is not None:
            champion_auc = float(roc_auc_score(test_labels, champion_preds))
            validation["champion_auc"] = round(champion_auc, 6)
            validation["auc_diff"] = round(candidate_auc - champion_auc, 6)
            print(f"  Champion AUC:  {champion_auc:.6f}")
            print(f"  AUC diff:      {candidate_auc - champion_auc:+.6f}")
    else:
        # No labels — compare score distributions
        validation["candidate_mean_score"] = round(float(np.mean(candidate_preds)), 6)
        validation["candidate_std_score"] = round(float(np.std(candidate_preds)), 6)
        print(f"  Candidate mean score: {np.mean(candidate_preds):.6f}")
        print(f"  Candidate std score:  {np.std(candidate_preds):.6f}")

        if champion_preds is not None:
            validation["champion_mean_score"] = round(float(np.mean(champion_preds)), 6)
            validation["champion_std_score"] = round(float(np.std(champion_preds)), 6)
            validation["mean_score_diff"] = round(
                float(np.mean(candidate_preds) - np.mean(champion_preds)), 6
            )
            print(f"  Champion mean score:  {np.mean(champion_preds):.6f}")
            print(f"  Mean score diff:      {np.mean(candidate_preds) - np.mean(champion_preds):+.6f}")

    # Save validation report
    PIPELINE_DIR.mkdir(parents=True, exist_ok=True)
    with open(VALIDATION_REPORT_PATH, "w") as f:
        json.dump(validation, f, indent=2)
    print(f"  Validation report saved to {VALIDATION_REPORT_PATH}")

    return validation


# ---------------------------------------------------------------------------
# Step 4 — Human-in-the-loop gate
# ---------------------------------------------------------------------------
def create_approval_gate(validation: dict) -> None:
    """Write pending_approval.json so a human can review and approve."""
    print("[Step 4] Creating approval gate ...")

    approval = {
        "candidate_model_path": str(CANDIDATE_MODEL_PATH),
        "validation_metrics": validation,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "pending",
    }

    with open(PENDING_APPROVAL_PATH, "w") as f:
        json.dump(approval, f, indent=2)

    print(f"  Approval file written to {PENDING_APPROVAL_PATH}")
    print()
    print("=" * 60)
    print("  VALIDATION RESULTS")
    print("=" * 60)
    for key, value in validation.items():
        if key == "timestamp":
            continue
        print(f"    {key}: {value}")
    print("=" * 60)
    print()
    print("Model candidate ready for review. Approve with: python pipeline/approve.py")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Automated retraining pipeline")
    parser.add_argument(
        "--production-data",
        type=str,
        default=None,
        help="Path to production data parquet file (default: data/production_sample.parquet)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  CREDIT SCORING — AUTOMATED RETRAINING PIPELINE")
    print("=" * 60)
    print()

    # Load reference data
    if not REFERENCE_DATA_PATH.exists():
        print(f"ERROR: Reference data not found at {REFERENCE_DATA_PATH}")
        sys.exit(1)
    reference_df = pd.read_parquet(REFERENCE_DATA_PATH)
    print(f"Reference data loaded: {reference_df.shape}")

    # Load production data
    prod_path = Path(args.production_data) if args.production_data else PRODUCTION_DATA_PATH
    if not prod_path.exists():
        print(f"ERROR: Production data not found at {prod_path}")
        sys.exit(1)
    production_df = pd.read_parquet(prod_path)
    print(f"Production data loaded: {production_df.shape}")
    print()

    # Step 1: Drift detection
    is_drifted, drift_summary = detect_drift(reference_df, production_df)
    if not is_drifted:
        print()
        print("No drift detected. Retraining is not needed. Exiting.")
        sys.exit(0)
    print()

    # Step 2: Retrain on reference + production data
    candidate = retrain_model(reference_df, production_df)
    print()

    # Step 3: Validate
    validation = validate_candidate(candidate)
    validation["drift_summary"] = drift_summary
    print()

    # Step 4: Approval gate
    create_approval_gate(validation)


if __name__ == "__main__":
    main()
