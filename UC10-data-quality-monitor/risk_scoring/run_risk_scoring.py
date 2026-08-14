from __future__ import annotations

import os
from pathlib import Path
import pandas as pd
import numpy as np

from risk_scoring.config import (
    RECORD_RISK_SCORES_PATH, INVESTIGATION_QUEUE_PATH,
    PROVIDER_RISK_SUMMARY_PATH, VALIDATION_REPORT_PATH,
    CLEANED_DIR, DATASETS, WORKSPACE_DIR
)
from risk_scoring.risk_scoring_engine import build_record_level_risk_scores


def main():
    print("="*60)
    print("RUNNING RISK SCORING AND INVESTIGATION PIPELINE")
    print("="*60)
    
    # 1. Run record level risk scoring engine
    df_scores = build_record_level_risk_scores()
    df_scores.to_csv(RECORD_RISK_SCORES_PATH, index=False)
    print(f"\nSaved record risk scores to: {RECORD_RISK_SCORES_PATH}")
    
    # 2. Generate Prioritized Investigation Queue
    print("\nOrchestrating prioritized investigation queue...")
    # Add severity numerical sort mapper
    sev_rank = {"High": 1, "Medium": 2, "Low": 3, "None": 4}
    df_scores["sev_num"] = df_scores["highest_rule_severity"].map(sev_rank).fillna(4)
    
    # Sort hierarchically:
    # 1. priority_num ascending (1 is P1 Critical, 4 is P4 Low)
    # 2. overall_risk_score descending
    # 3. sev_num ascending (High=1, None=4)
    # 4. consensus_score descending (100=Level 3, 0=Level 0)
    df_queue = df_scores.sort_values(
        by=["priority_num", "overall_risk_score", "sev_num", "consensus_score"],
        ascending=[True, False, True, False]
    ).reset_index(drop=True)
    
    # Drop temp sorting column
    df_queue = df_queue.drop(columns=["sev_num"])
    
    # Select columns for queue schema
    queue_cols = [
        "Record_ID", "Record_Type", "overall_risk_score", "risk_category",
        "rule_risk_score", "statistical_risk_score", "ml_risk_score", "consensus_score",
        "number_of_rule_violations", "statistical_flag", "ml_flag", "highest_rule_severity",
        "primary_risk_reason", "secondary_risk_reason", "recommended_priority",
        "risk_score_breakdown", "explanation"
    ]
    df_queue_final = df_queue[queue_cols]
    df_queue_final.to_csv(INVESTIGATION_QUEUE_PATH, index=False)
    print(f"Saved ranked investigation queue to: {INVESTIGATION_QUEUE_PATH}")
    
    # 3. Generate Provider-Level Risk Summary
    print("\nCompiling provider-level risk summary...")
    
    # Extract statuses from cleaned files to calculate empirical denial rate
    status_map = {}
    for rtype, path in DATASETS.items():
        clean_df = pd.read_csv(path)
        for _, row in clean_df.iterrows():
            rid = str(row["Record_ID"])
            status_map[rid] = str(row.get("Status")).upper() if pd.notna(row.get("Status")) else "UNKNOWN"
            
    df_scores["Status"] = df_scores["Record_ID"].map(status_map).fillna("UNKNOWN")
    
    # Group by Provider_NPI
    grouped = df_scores.groupby("Provider_NPI")
    provider_rows = []
    
    for npi, group in grouped:
        npi_str = str(npi)
        
        # predominant type
        rtypes = group["Record_Type"].value_counts()
        pred_type = rtypes.index[0] if len(rtypes) > 0 else "UNKNOWN"
        
        total_records = len(group)
        flagged_records = (group["overall_risk_score"] >= 25.0).sum()
        flagged_pct = (flagged_records / total_records) * 100
        
        ml_anom_cnt = (group["ml_flag"] == 1).sum()
        stat_anom_cnt = (group["statistical_flag"] == 1).sum()
        rule_viol_cnt = group["number_of_rule_violations"].sum()
        
        # High/Critical risk counts
        high_cnt = (group["risk_category"] == "HIGH").sum()
        crit_cnt = (group["risk_category"] == "CRITICAL").sum()
        
        # Consensus counts
        # Mapping consensus level back from consensus_score
        c2_cnt = (group["consensus_score"] == 60.0).sum()
        c3_cnt = (group["consensus_score"] == 100.0).sum()
        
        avg_score = group["overall_risk_score"].mean()
        max_score = group["overall_risk_score"].max()
        
        # Denial Rate
        denied_flags = ["DENIED", "REJECTED"]
        non_pend_group = group[group["Status"] != "PENDING"]
        total_non_pending = len(non_pend_group)
        denied_cnt = non_pend_group["Status"].isin(denied_flags).sum()
        denied_rate = (denied_cnt / total_non_pending) * 100 if total_non_pending > 0 else 0.0
        
        # Review categorization
        prov_status = "NORMAL"
        if crit_cnt > 0 or high_cnt > 0 or flagged_pct > 10.0 or avg_score > 35.0:
            prov_status = "PROVIDER REQUIRING REVIEW"
            
        reasons = []
        if crit_cnt > 0:
            reasons.append(f"{crit_cnt} critical-risk records")
        if high_cnt > 0:
            reasons.append(f"{high_cnt} high-risk records")
        if flagged_pct > 10.0:
            reasons.append(f"{flagged_pct:.1f}% flagged records")
        if avg_score > 35.0:
            reasons.append(f"Average claim risk score = {avg_score:.1f}")
        if denied_rate > 20.0:
            reasons.append(f"Empirical claim denial rate = {denied_rate:.1f}%")
            
        primary_reasons = "; ".join(reasons) if reasons else "No suspicious patterns identified"
        
        provider_rows.append({
            "Provider_NPI": npi_str,
            "Predominant_Record_Type": pred_type,
            "Total_Records": int(total_records),
            "Flagged_Records": int(flagged_records),
            "Flagged_Record_Percentage": round(flagged_pct, 2),
            "ML_Anomaly_Count": int(ml_anom_cnt),
            "Statistical_Anomaly_Count": int(stat_anom_cnt),
            "Rule_Violation_Count": int(rule_viol_cnt),
            "Consensus_Level_2_Count": int(c2_cnt),
            "Consensus_Level_3_Count": int(c3_cnt),
            "High_Risk_Record_Count": int(high_cnt),
            "Critical_Risk_Record_Count": int(crit_cnt),
            "Average_Risk_Score": round(float(avg_score), 2),
            "Maximum_Risk_Score": round(float(max_score), 2),
            "Denial_Rate_Percentage": round(denied_rate, 2),
            "Provider_Risk_Level": "HIGH-RISK PROVIDER PATTERN" if prov_status == "PROVIDER REQUIRING REVIEW" else "LOW-RISK PROVIDER PATTERN",
            "Provider_Status": prov_status,
            "Primary_Risk_Factors": primary_reasons
        })
        
    df_provider = pd.DataFrame(provider_rows)
    df_provider = df_provider.sort_values(
        by=["Critical_Risk_Record_Count", "High_Risk_Record_Count", "Average_Risk_Score", "Total_Records"],
        ascending=False
    ).reset_index(drop=True)
    df_provider.to_csv(PROVIDER_RISK_SUMMARY_PATH, index=False)
    print(f"Saved provider risk summary to: {PROVIDER_RISK_SUMMARY_PATH}")
    
    # 4. Perform validation checks
    print("\nRunning self-validation constraints check...")
    validation_checks = []
    
    def add_check(name, status, details):
        validation_checks.append({
            "Check_Name": name,
            "Status": "PASS" if status else "FAIL",
            "Details": details
        })
        
    # Check 1: No duplicates
    no_dups = not df_queue_final["Record_ID"].duplicated().any()
    add_check("No Duplicates", no_dups, "No duplicate Record_IDs exist in investigation_queue.csv")
    
    # Check 2: Row counts match inputs
    ids_consensus = set(df_scores["Record_ID"].tolist())
    clean_ids = set()
    for rtype, path in DATASETS.items():
        c_df = pd.read_csv(path)
        clean_ids.update(c_df["Record_ID"].astype(str).tolist())
    all_belong = ids_consensus.issubset(clean_ids)
    add_check("Input Record Belonging", all_belong, "All scored Record_IDs belong to cleaned inputs")
    
    # Check 3: Score bounds
    scores_min = df_scores["overall_risk_score"].min()
    scores_max = df_scores["overall_risk_score"].max()
    bounds_ok = (scores_min >= 0.0) and (scores_max <= 100.0)
    add_check("Score Bounds [0, 100]", bounds_ok, f"Scores are between 0 and 100. Min={scores_min}, Max={scores_max}")
    
    # Check 4: No missing Record_ID
    no_missing_id = df_scores["Record_ID"].notna().all()
    add_check("No Missing IDs", no_missing_id, "No Record_IDs are missing or NaN")
    
    # Check 5: Risk category matching
    category_ok = True
    for _, row in df_scores.iterrows():
        s = row["overall_risk_score"]
        cat = row["risk_category"]
        if s < 25.0 and cat != "LOW":
            category_ok = False
        elif 25.0 <= s < 50.0 and cat != "MEDIUM":
            category_ok = False
        elif 50.0 <= s < 75.0 and cat != "HIGH":
            category_ok = False
        elif s >= 75.0 and cat != "CRITICAL":
            category_ok = False
    add_check("Category Matching", category_ok, "Risk categories strictly align with overall score thresholds")
    
    # Check 6: Priority mapping check
    priority_ok = True
    for _, row in df_queue.iterrows():
        cat = row["risk_category"]
        pri = row["recommended_priority"]
        if cat == "CRITICAL" and not pri.startswith("P1"):
            priority_ok = False
        elif cat == "HIGH" and not pri.startswith("P2"):
            priority_ok = False
        elif cat == "MEDIUM" and not pri.startswith("P3"):
            priority_ok = False
        elif cat == "LOW" and not pri.startswith("P4"):
            priority_ok = False
    add_check("Priority Mapping", priority_ok, "Recommended priorities map correctly to risk categories")
    
    # Check 7: Provider duplicates check
    no_prov_dups = not df_provider["Provider_NPI"].duplicated().any()
    add_check("No Duplicate Providers", no_prov_dups, "No duplicate NPI groups exist in provider_risk_summary.csv")
    
    # Check 8: Consensus weight boost check
    # Check that Consensus Level 3 claims are higher overall risk score than consensus Level 1 claims
    avg_l3 = df_scores[df_scores["consensus_score"] == 100.0]["overall_risk_score"].mean()
    avg_l1 = df_scores[df_scores["consensus_score"] == 20.0]["overall_risk_score"].mean()
    consensus_boost_ok = avg_l3 > avg_l1
    add_check("Consensus Boost Verification", consensus_boost_ok, f"Level 3 claims score higher on average than Level 1. L3={avg_l3:.2f}, L1={avg_l1:.2f}")
    
    # Save validation report
    df_val = pd.DataFrame(validation_checks)
    df_val.to_csv(VALIDATION_REPORT_PATH, index=False)
    print(f"Saved validation report to: {VALIDATION_REPORT_PATH}")
    print(df_val.to_string(index=False))


if __name__ == "__main__":
    main()
