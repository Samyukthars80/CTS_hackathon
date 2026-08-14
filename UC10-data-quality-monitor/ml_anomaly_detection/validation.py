from __future__ import annotations

import json
import pickle
import pandas as pd
import numpy as np

from ml_anomaly_detection.config import (
    DATASETS, MODELS_DIR, OUTPUTS_DIR
)


def validate_ml_stage():
    print("\n" + "="*50)
    print("RUNNING MODEL VALIDATION SUITE")
    print("="*50)
    
    validation_passed = True
    
    # Check 1: Record counts match and no duplicates
    print("\nCheck 1 & 2: Validating Record Counts & Duplicates...")
    model_name_map = {
        "MEDICAL_CLAIM": "medical",
        "PHARMACY_CLAIM": "pharmacy",
        "PRIOR_AUTH": "prior_auth"
    }
    
    for rtype, file_path in DATASETS.items():
        prefix = model_name_map[rtype]
        orig_df = pd.read_csv(file_path)
        pred_df = pd.read_csv(OUTPUTS_DIR / f"{prefix}_ml_predictions.csv")
        anom_df = pd.read_csv(OUTPUTS_DIR / f"{prefix}_ml_anomalies.csv")
        
        # Match count
        if len(orig_df) != len(pred_df):
            print(f"  [FAIL] Count mismatch for {rtype}: Orig={len(orig_df)}, Pred={len(pred_df)}")
            validation_passed = False
        else:
            print(f"  [PASS] Record counts match for {rtype} ({len(orig_df)} rows).")
            
        # No duplicates
        if pred_df["Record_ID"].duplicated().any():
            print(f"  [FAIL] Duplicated Record_ID found in predictions for {rtype}.")
            validation_passed = False
        else:
            print(f"  [PASS] No duplicate Record_IDs in predictions for {rtype}.")
            
        # Check if predictions contain both normal and anomalies
        anomaly_flags = pred_df["ML_Anomaly_Flag"].unique()
        if len(anomaly_flags) > 1 and 1 in anomaly_flags and 0 in anomaly_flags:
            print(f"  [PASS] Model output for {rtype} contains both normal and anomalous records.")
        else:
            print(f"  [FAIL] Model output for {rtype} does not contain both classes. Unique flags: {anomaly_flags}")
            validation_passed = False

    # Check 3: Check feature list doesn't contain leakage or identifiers
    print("\nCheck 3: Inspecting model feature lists...")
    leakage_terms = ["bene_id", "npi", "record_id", "ndc_code", "diagnosis_code", "procedure_code", "batch_id", "timestamp", "date", "flag", "breach", "anomaly"]
    for rtype, prefix in model_name_map.items():
        with open(MODELS_DIR / f"{prefix}_features.json") as f:
            features = json.load(f)
            
        print(f"  {rtype} Features used: {features}")
        
        # Check each feature
        bad_features = []
        for feat in features:
            for term in leakage_terms:
                if term in feat.lower():
                    # Allow Provider_Total_Records, Provider_Denial_Rate, Beneficiary_Record_Count, Urgency_Flag, Auth_Required_Flag
                    if "total_records" in feat.lower() or "denial_rate" in feat.lower() or "record_count" in feat.lower() or "urgency_flag" in feat.lower() or "auth_required_flag" in feat.lower():
                        continue
                    bad_features.append(feat)
                    
        if bad_features:
            print(f"  [FAIL] Potential identifier or leakage column used as feature in {rtype}: {bad_features}")
            validation_passed = False
        else:
            print(f"  [PASS] Feature list for {rtype} contains no identifiers or leakage columns.")

    # Check 4: Check for infinite or unexpected missing values in output
    print("\nCheck 4: Checking for infinite or NaN values in predictions...")
    for prefix in model_name_map.values():
        pred_df = pd.read_csv(OUTPUTS_DIR / f"{prefix}_ml_predictions.csv")
        
        has_nans = pred_df.isna().sum().sum() > 0
        has_infs = np.isinf(pred_df.select_dtypes(include=np.number)).sum().sum() > 0
        
        if has_nans:
            print(f"  [FAIL] Nan values found in predictions for {prefix}_ml_predictions.csv")
            validation_passed = False
        else:
            print(f"  [PASS] No NaN values found in predictions for {prefix}_ml_predictions.csv")
            
        if has_infs:
            print(f"  [FAIL] Infinite values found in predictions for {prefix}_ml_predictions.csv")
            validation_passed = False
        else:
            print(f"  [PASS] No infinite values found in predictions for {prefix}_ml_predictions.csv")

    # Check 5: Verify pre-existing stages were not modified
    # We will let git status run in command line, but we check if files exist
    print("\nCheck 5: Confirming models and preprocessors saved correctly...")
    for prefix in model_name_map.values():
        mod_exists = (MODELS_DIR / f"{prefix}_isolation_forest.pkl").exists()
        prep_exists = (MODELS_DIR / f"{prefix}_preprocessor.pkl").exists()
        
        if not mod_exists or not prep_exists:
            print(f"  [FAIL] Missing saved model or preprocessor for {prefix}.")
            validation_passed = False
        else:
            print(f"  [PASS] Model and preprocessor for {prefix} verified in models directory.")
            
    if validation_passed:
        print("\n" + "="*50)
        print("ALL VALIDATION CHECKS PASSED SUCCESSFULLY!")
        print("="*50)
    else:
        print("\n" + "="*50)
        print("VALIDATION SUITE ENCOUNTERED FAILURES. REVIEW LOGS.")
        print("="*50)


if __name__ == "__main__":
    validate_ml_stage()
