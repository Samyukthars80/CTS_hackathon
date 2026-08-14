from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

# Paths
RISK_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = RISK_DIR.parent
CLEANED_DIR = WORKSPACE_DIR / "outputs"
ML_DIR = WORKSPACE_DIR / "ml_anomaly_detection"
OUTPUTS_DIR = RISK_DIR / "outputs"


def run_self_validation():
    print("="*60)
    print("RUNNING RISK SCORING VALIDATION TESTS")
    print("="*60)
    
    validation_passed = True
    
    # 1. Load outputs
    scores_path = OUTPUTS_DIR / "record_risk_scores.csv"
    queue_path = OUTPUTS_DIR / "investigation_queue.csv"
    provider_path = OUTPUTS_DIR / "provider_risk_summary.csv"
    
    for p in [scores_path, queue_path, provider_path]:
        if not p.exists():
            print(f"[FAIL] Missing output file: {p}")
            validation_passed = False
            return
            
    df_scores = pd.read_csv(scores_path)
    df_queue = pd.read_csv(queue_path)
    df_prov = pd.read_csv(provider_path)
    
    # Check 1: Record ID counts match consensus (10,000 records)
    consensus_path = ML_DIR / "model_validation" / "cross_method_consensus.csv"
    df_consensus = pd.read_csv(consensus_path)
    
    if len(df_scores) != len(df_consensus):
        print(f"[FAIL] Consolidated scores row count ({len(df_scores)}) does not match consensus ({len(df_consensus)})")
        validation_passed = False
    else:
        print(f"[PASS] Consolidated score rows match consensus row count ({len(df_scores)}).")
        
    # Check 2: No duplicate Record_ID in scores or queue
    if df_scores["Record_ID"].duplicated().any():
        print("[FAIL] Duplicated Record_ID found in record_risk_scores.csv")
        validation_passed = False
    else:
        print("[PASS] No duplicate Record_IDs found in record_risk_scores.csv.")
        
    if df_queue["Record_ID"].duplicated().any():
        print("[FAIL] Duplicated Record_ID found in investigation_queue.csv")
        validation_passed = False
    else:
        print("[PASS] No duplicate Record_IDs found in investigation_queue.csv.")
        
    # Check 3: Risk scores between 0 and 100
    scores_min = df_scores["Risk_Score"].min()
    scores_max = df_scores["Risk_Score"].max()
    if scores_min < 0.0 or scores_max > 100.0:
        print(f"[FAIL] Risk scores outside [0, 100] bounds. Min={scores_min}, Max={scores_max}")
        validation_passed = False
    else:
        print(f"[PASS] All risk scores fall within [0, 100] limits. Min={scores_min:.2f}, Max={scores_max:.2f}")
        
    # Check 4: Risk levels correctly correspond to score ranges
    # 0-24 = LOW, 25-49 = MEDIUM, 50-74 = HIGH, 75-100 = CRITICAL
    invalid_levels = 0
    for _, row in df_scores.iterrows():
        s = row["Risk_Score"]
        lvl = row["Risk_Level"]
        if s < 25.0 and lvl != "LOW":
            invalid_levels += 1
        elif 25.0 <= s < 50.0 and lvl != "MEDIUM":
            invalid_levels += 1
        elif 50.0 <= s < 75.0 and lvl != "HIGH":
            invalid_levels += 1
        elif s >= 75.0 and lvl != "CRITICAL":
            invalid_levels += 1
            
    if invalid_levels > 0:
        print(f"[FAIL] Found {invalid_levels} records with mismatched Risk_Level bounds.")
        validation_passed = False
    else:
        print("[PASS] All Risk_Levels correspond to the correct score range thresholds.")
        
    # Check 5: Queue sorting check
    # CRITICAL=1, HIGH=2, MEDIUM=3, LOW=4
    priorities = df_queue["Priority"].tolist()
    is_priority_sorted = all(priorities[i] <= priorities[i+1] for i in range(len(priorities)-1))
    
    # Check if within each priority block, risk scores are descending
    is_score_sorted = True
    for p_val in [1, 2, 3, 4]:
        subset_scores = df_queue[df_queue["Priority"] == p_val]["Risk_Score"].tolist()
        if not all(subset_scores[i] >= subset_scores[i+1] for i in range(len(subset_scores)-1)):
            is_score_sorted = False
            
    if not is_priority_sorted or not is_score_sorted:
        print("[FAIL] Investigation queue is not correctly sorted by Priority ascending and Risk_Score descending.")
        validation_passed = False
    else:
        print("[PASS] Investigation queue sorting checks passed successfully.")
        
    # Check 6: Consensus values validity
    invalid_consensus = df_scores[~df_scores["Consensus_Level"].isin([0, 1, 2, 3])]
    if len(invalid_consensus) > 0:
        print(f"[FAIL] Invalid consensus levels found: {invalid_consensus['Consensus_Level'].unique()}")
        validation_passed = False
    else:
        print("[PASS] All Consensus_Levels are valid values (0, 1, 2, 3).")
        
    # Check 7: Provider aggregation checks
    if df_prov["Provider_NPI"].duplicated().any():
        print("[FAIL] Duplicate Provider_NPI records found in provider_risk_summary.csv")
        validation_passed = False
    else:
        print("[PASS] No duplicate NPIs found in provider_risk_summary.csv.")
        
    # Check 8: Unknown NPI handling check
    # Check if 'UNKNOWN_PROVIDER' exists
    has_unknown = (df_prov["Provider_NPI"] == "UNKNOWN_PROVIDER").any()
    if has_unknown:
        print("[PASS] Null/missing NPIs were correctly aggregated under 'UNKNOWN_PROVIDER'.")
    else:
        print("[NOTE] No 'UNKNOWN_PROVIDER' found in provider summary (all claims had valid NPIs).")
        
    # Summarize checks
    print("\n" + "="*60)
    if validation_passed:
        print("ALL SELF-VALIDATION TESTS PASSED SUCCESSFULLY!")
    else:
        print("VALIDATION SUITE ENCOUNTERED ERRORS. REVIEW ABOVE LOG.")
    print("="*60)


if __name__ == "__main__":
    run_self_validation()
