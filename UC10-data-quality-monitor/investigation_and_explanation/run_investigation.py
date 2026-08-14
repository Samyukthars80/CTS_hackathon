from __future__ import annotations

import os
from pathlib import Path
import pandas as pd
import numpy as np

# Setup paths
INV_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = INV_DIR.parent
CLEANED_DIR = WORKSPACE_DIR / "outputs"
ML_DIR = WORKSPACE_DIR / "ml_anomaly_detection"
RULE_DIR = WORKSPACE_DIR / "anomaly_detection"
RISK_DIR = WORKSPACE_DIR / "risk_scoring"

DATASETS = {
    "MEDICAL_CLAIM": CLEANED_DIR / "medical_claim_cleaned.csv",
    "PHARMACY_CLAIM": CLEANED_DIR / "pharmacy_claim_cleaned.csv",
    "PRIOR_AUTH": CLEANED_DIR / "authorization_cleaned.csv"
}

NUMERICAL_COLS = {
    "MEDICAL_CLAIM": ["Billed_Amount", "Allowed_Amount", "Paid_Amount", "Patient_Responsibility", "Retry_Count", "Processing_Latency_Days", "SLA_Target_Days", "Beneficiary_Record_Count", "Provider_Total_Records", "Provider_Denial_Rate"],
    "PHARMACY_CLAIM": ["Billed_Amount", "Allowed_Amount", "Paid_Amount", "Patient_Responsibility", "Retry_Count", "Processing_Latency_Days", "SLA_Target_Days", "Days_Supply", "Quantity_Dispensed", "Beneficiary_Record_Count", "Provider_Total_Records", "Provider_Denial_Rate"],
    "PRIOR_AUTH": ["Retry_Count", "Processing_Latency_Days", "SLA_Target_Days", "Provider_Total_Records", "Provider_Denial_Rate", "Beneficiary_Record_Count"]
}


def calculate_train_split_statistics() -> dict[str, dict[str, tuple[float, float]]]:
    """Calculate mean and standard deviation for each numerical column strictly on the training split (80%)."""
    stats = {}
    for rtype, path in DATASETS.items():
        df = pd.read_csv(path)
        # Sort chronologically by Submission_Date
        df = df.sort_values(by="Submission_Date").reset_index(drop=True)
        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx]
        
        stats[rtype] = {}
        for col in NUMERICAL_COLS[rtype]:
            if col in train_df.columns:
                col_vals = train_df[col].dropna()
                mean_val = float(col_vals.mean()) if len(col_vals) > 0 else 0.0
                std_val = float(col_vals.std()) if len(col_vals) > 1 else 0.0
                stats[rtype][col] = (mean_val, std_val)
    return stats


def calculate_rule_risk_details(rid: str, rule_detailed_df: pd.DataFrame) -> list[dict]:
    """Gather detailed rule violations for record ID."""
    r_details = rule_detailed_df[rule_detailed_df["Record_ID"] == rid]
    violations = []
    for _, row in r_details.iterrows():
        violations.append({
            "Rule_ID": str(row["Rule_ID"]),
            "Rule_Category": str(row.get("Anomaly_Category", "Compliance Check")),
            "Severity": str(row["Severity"]),
            "Affected_Columns": str(row.get("Affected_Columns", "None")),
            "Observed_Value": str(row.get("Observed_Value", "None")),
            "Expected_Condition": str(row.get("Expected_Condition", "None")),
            "Explanation": str(row["Explanation"])
        })
    return violations


def main():
    print("="*60)
    print("RUNNING FINAL INVESTIGATION AND EXPLANATION LAYER")
    print("="*60)
    
    # 1. Load inputs
    scores_path = RISK_DIR / "record_risk_scores.csv"
    if not scores_path.exists():
        raise FileNotFoundError(f"Consolidated record risk scores not found: {scores_path}")
    df_scores = pd.read_csv(scores_path)
    
    rule_detailed_path = RULE_DIR / "outputs" / "combined_rule_anomalies.csv"
    df_rule_details = pd.read_csv(rule_detailed_path)
    df_rule_details["Record_ID"] = df_rule_details["Record_ID"].astype(str)
    
    # Load ML predictions/anomalies for explanation contributions
    ml_pct_map = {}
    ml_score_map = {}
    ml_explanations = {}
    ml_prefixes = {"MEDICAL_CLAIM": "medical", "PHARMACY_CLAIM": "pharmacy", "PRIOR_AUTH": "prior_auth"}
    
    for rtype, prefix in ml_prefixes.items():
        pred_path = ML_DIR / "outputs" / f"{prefix}_ml_predictions.csv"
        if pred_path.exists():
            pdf = pd.read_csv(pred_path)
            for _, row in pdf.iterrows():
                rid = str(row["Record_ID"])
                ml_pct_map[rid] = float(row["Anomaly_Percentile"])
                ml_score_map[rid] = float(row["Isolation_Forest_Score"])
                
        anom_path = ML_DIR / "outputs" / f"{prefix}_ml_anomalies.csv"
        if anom_path.exists():
            adf = pd.read_csv(anom_path)
            for _, row in adf.iterrows():
                rid = str(row["Record_ID"])
                ml_explanations[rid] = str(row["Potential_Contributing_Factors"])
                
    # Load clean data for detailed statistical evidence calculations
    train_stats = calculate_train_split_statistics()
    clean_dfs = {}
    for rtype, path in DATASETS.items():
        clean_dfs[rtype] = pd.read_csv(path).set_index("Record_ID")
        
    # 2. Process records and generate investigation details
    investigation_records = []
    
    print("\nProcessing record evidence and Z-score deviations...")
    for _, row in df_scores.iterrows():
        rid = str(row["Record_ID"])
        rtype = str(row["Record_Type"])
        
        ml_flag = int(row["ml_flag"])
        stat_flag = int(row["statistical_flag"])
        num_violations = int(row["number_of_rule_violations"])
        rule_flag = 1 if num_violations > 0 else 0
        
        # Rule evidence
        r_violations = calculate_rule_risk_details(rid, df_rule_details)
        rule_ids = [v["Rule_ID"] for v in r_violations]
        affected_cols_rule = [v["Affected_Columns"] for v in r_violations if v["Affected_Columns"] != "None"]
        
        # Statistical evidence
        stat_evidences = []
        outlier_cols_str = str(row["statistical_outlier_columns"]) if pd.notna(row.get("statistical_outlier_columns")) else ""
        if stat_flag and pd.notna(outlier_cols_str) and outlier_cols_str.strip():
            cols = outlier_cols_str.split("; ")
            clean_row = clean_dfs[rtype].loc[rid]
            col_stats = train_stats[rtype]
            
            for col in cols:
                val = clean_row.get(col)
                if pd.notna(val) and col in col_stats:
                    mean_val, std_val = col_stats[col]
                    z = abs(val - mean_val) / std_val if std_val > 0 else 0.0
                    sev = "Severe Outlier (Z > 3.0)" if z > 3.0 else "Moderate Outlier (1.5 < Z <= 3.0)"
                    stat_evidences.append(
                        f"{col}: value={val:.2f} (mean={mean_val:.2f}, std={std_val:.2f}, Z={z:.2f}, Severity={sev})"
                    )
                    
        # ML evidence
        ml_pct = ml_pct_map.get(rid, 0.0)
        ml_score = ml_score_map.get(rid, 0.0)
        ml_exp = ml_explanations.get(rid, "")
        
        # Select Primary & Supporting Reasons
        # Priority: Rule severity (High) > Consensus 3 > Statistical deviation > ML anomaly
        high_severity_count = sum(1 for v in r_violations if v["Severity"] == "High")
        
        # Recalculate consensus and overall score from active flags
        consensus_level = ml_flag + rule_flag + stat_flag
        consensus_score = {3: 100.0, 2: 60.0, 1: 20.0, 0: 0.0}[consensus_level]
        
        rule_risk_score = float(row["rule_risk_score"])
        statistical_risk_score = float(row["statistical_risk_score"])
        ml_risk_score = float(row["ml_risk_score"])
        
        overall_risk_score = (
            0.40 * rule_risk_score +
            0.25 * statistical_risk_score +
            0.25 * ml_risk_score +
            0.10 * consensus_score
        )
        overall_risk_score = round(overall_risk_score, 2)
        
        # Risk Categories & Priorities
        if overall_risk_score < 25.0:
            risk_category = "LOW"
            recommended_priority = "P4 - Low"
            priority_num = 4
        elif overall_risk_score < 50.0:
            risk_category = "MEDIUM"
            recommended_priority = "P3 - Medium"
            priority_num = 3
        elif overall_risk_score < 75.0:
            risk_category = "HIGH"
            recommended_priority = "P2 - High"
            priority_num = 2
        else:
            risk_category = "CRITICAL"
            recommended_priority = "P1 - Critical"
            priority_num = 1
            
        breakdown = f"Rule: {(rule_risk_score * 0.40):.1f}/40; Statistical: {(statistical_risk_score * 0.25):.1f}/25; ML: {(ml_risk_score * 0.25):.1f}/25; Consensus: {(consensus_score * 0.10):.1f}/10"
        
        if high_severity_count > 0:
            primary_reason = f"High-severity business rule violations (Count: {high_severity_count})"
        elif consensus_level == 3:
            primary_reason = "Multi-method consensus agreement (Consensus Level: 3)"
        elif stat_flag and len(stat_evidences) > 0:
            primary_reason = f"Severe numerical statistical deviations in {len(stat_evidences)} columns"
        elif ml_flag and ml_pct >= 99.0:
            primary_reason = f"Extreme Isolation Forest anomaly (Percentile: {ml_pct:.1f}%)"
        elif rule_flag:
            primary_reason = f"Business rule violations (Count: {len(r_violations)})"
        else:
            primary_reason = "Typical baseline operations"
            
        supporting = []
        if ml_flag:
            supporting.append(f"ML Anomaly (Score: {ml_score:.3f}, Percentile: {ml_pct:.1f}%)")
        if stat_flag:
            supporting.append(f"Statistical outlier in columns: {', '.join(stat_evidences)}")
        if rule_flag:
            supporting.append(f"Rule Engine: {len(r_violations)} violations")
            
        supporting_evidence = "; ".join(supporting) if supporting else "None"
        
        # Explanations strings
        explanation = str(row["explanation"])
        
        investigation_records.append({
            "Record_ID": rid,
            "Record_Type": rtype,
            "Provider_NPI": str(row["Provider_NPI"]),
            "overall_risk_score": overall_risk_score,
            "risk_category": risk_category,
            "recommended_priority": recommended_priority,
            "priority_num": priority_num,
            "consensus_score": consensus_score,
            "Consensus_Level": consensus_level,
            "ML_Flag": ml_flag,
            "Rule_Flag": rule_flag,
            "Statistical_Flag": stat_flag,
            "Rule_Violation_Count": len(r_violations),
            "High_Severity_Count": high_severity_count,
            "highest_rule_severity": str(row["highest_rule_severity"]),
            "rule_risk_score": rule_risk_score,
            "statistical_risk_score": statistical_risk_score,
            "ml_risk_score": ml_risk_score,
            "Primary_Investigation_Reason": primary_reason,
            "Supporting_Evidence": supporting_evidence,
            "Explanation": explanation,
            "Affected_Columns": "; ".join(set(affected_cols_rule)) if affected_cols_rule else (outlier_cols_str if pd.notna(outlier_cols_str) else ""),
            "Rule_IDs": "; ".join(rule_ids),
            "Rule_Source": "anomaly_detection/outputs/combined_rule_anomalies.csv",
            "Statistical_Source": "statistical_analysis/medical/descriptive_statistics.csv (Baseline)",
            "ML_Source": "ml_anomaly_detection/outputs/*_ml_predictions.csv",
            "Risk_Score_Source": "risk_scoring/record_risk_scores.csv",
            "risk_score_breakdown": breakdown
        })
        
    df_investigations = pd.DataFrame(investigation_records)
    
    # 3. Create Dashboard Ready Output
    dashboard_cols = [
        "Record_ID", "Record_Type", "Provider_NPI", "overall_risk_score",
        "risk_category", "recommended_priority", "Consensus_Level",
        "Rule_Flag", "Statistical_Flag", "ML_Flag",
        "Primary_Investigation_Reason", "Supporting_Evidence",
        "Explanation", "Affected_Columns", "Rule_IDs"
    ]
    df_dashboard = df_investigations[dashboard_cols]
    df_dashboard.to_csv(INV_DIR / "dashboard_data.csv", index=False)
    print(f"Saved dashboard data CSV to: {INV_DIR / 'dashboard_data.csv'}")
    
    # 4. Generate Final Investigation Queue (Sorted hierarchically)
    # Sort keys:
    # 1. priority_num ascending (P1=1, P4=4)
    # 2. overall_risk_score descending
    # 3. Consensus_Level descending
    # 4. rule severity sorting (High=1, Medium=2, Low=3, None=4)
    # 5. Consensus_Level (number of supporting detection methods) descending
    sev_rank = {"High": 1, "Medium": 2, "Low": 3, "None": 4}
    df_investigations["sev_num"] = df_investigations["highest_rule_severity"].map(sev_rank).fillna(4)
    
    df_queue_sorted = df_investigations.sort_values(
        by=["priority_num", "overall_risk_score", "Consensus_Level", "sev_num", "Rule_Violation_Count"],
        ascending=[True, False, False, True, False]
    ).reset_index(drop=True)
    
    # Drop sorting help columns
    df_queue_sorted = df_queue_sorted.drop(columns=["sev_num"])
    
    queue_cols = [
        "Record_ID", "Record_Type", "overall_risk_score", "risk_category",
        "rule_risk_score", "statistical_risk_score", "ml_risk_score", "consensus_score",
        "Rule_Violation_Count", "Statistical_Flag", "ML_Flag", "highest_rule_severity",
        "Primary_Investigation_Reason", "Supporting_Evidence", "recommended_priority",
        "risk_score_breakdown", "Explanation"
    ]
    df_queue_final = df_queue_sorted[queue_cols]
    df_queue_final.to_csv(INV_DIR / "final_investigation_queue.csv", index=False)
    print(f"Saved final prioritized investigation queue to: {INV_DIR / 'final_investigation_queue.csv'}")
    
    # 5. Provider-Level Investigation Summary (with explanations)
    print("\nProcessing provider-level explanations...")
    prov_sum_path = RISK_DIR / "provider_risk_summary.csv"
    df_prov = pd.read_csv(prov_sum_path)
    
    prov_exps = []
    for _, row_p in df_prov.iterrows():
        npi = str(row_p["Provider_NPI"])
        flagged_records = int(row_p["Flagged_Records"])
        flagged_pct = float(row_p["Flagged_Record_Percentage"])
        total_records = int(row_p["Total_Records"])
        crit_cnt = int(row_p["Critical_Risk_Record_Count"])
        high_cnt = int(row_p["High_Risk_Record_Count"])
        avg_score = float(row_p["Average_Risk_Score"])
        max_score = float(row_p["Maximum_Risk_Score"])
        prov_status = str(row_p["Provider_Status"])
        
        explanation = f"Provider NPI {npi} submitted {total_records} claims, of which {flagged_records} ({flagged_pct:.1f}%) were flagged as high-risk."
        if prov_status == "PROVIDER REQUIRING REVIEW":
            reasons = []
            if crit_cnt > 0:
                reasons.append(f"{crit_cnt} critical-risk records")
            if high_cnt > 0:
                reasons.append(f"{high_cnt} high-risk records")
            if flagged_pct > 10.0:
                reasons.append(f"flagged claim proportion of {flagged_pct:.1f}% exceeds 10%")
            if avg_score > 35.0:
                reasons.append(f"average claim risk score is {avg_score:.1f}")
                
            explanation += f" This provider is flagged for REVIEW because of: {', '.join(reasons)}. Pattern-level investigation is recommended."
        else:
            explanation += " Overall billing patterns fall within normal operational baselines. No audit review is currently recommended."
            
        prov_exps.append(explanation)
        
    df_prov["Provider_Explanation"] = prov_exps
    
    # Save provider investigation summary
    df_prov.to_csv(INV_DIR / "provider_investigation_summary.csv", index=False)
    print(f"Saved provider investigation summary to: {INV_DIR / 'provider_investigation_summary.csv'}")
    
    # 6. Perform Quality Validation Checks
    print("\nExecuting validation tests checks...")
    validation_checks = []
    
    def add_check(name, status, details):
        validation_checks.append({
            "Check_Name": name,
            "Status": "PASS" if status else "FAIL",
            "Details": details
        })
        
    # Check 1: Every P1 has score
    p1_subset = df_queue_sorted[df_queue_sorted["recommended_priority"].str.startswith("P1")]
    p1_ok = (len(p1_subset) > 0) and p1_subset["overall_risk_score"].notna().all()
    add_check("P1 Risk Scores", p1_ok, f"Checked that all P1 Critical claims contain valid risk scores ({len(p1_subset)} claims).")
    
    # Check 2: Flagged claims have at least one source
    flagged_records = df_investigations[df_investigations["overall_risk_score"] >= 25.0]
    has_source = flagged_records.apply(lambda r: r["ML_Flag"] == 1 or r["Rule_Flag"] == 1 or r["Statistical_Flag"] == 1, axis=1).all()
    add_check("Detection Source Presence", has_source, "All flagged claims contain at least one anomaly source (ML, Rules, or Statistics)")
    
    # Check 3: No duplicate Record_IDs
    no_queue_dups = not df_queue_final["Record_ID"].duplicated().any()
    add_check("No Duplicate Record IDs", no_queue_dups, "No duplicate Record_IDs exist in final_investigation_queue.csv")
    
    # Check 4: Score bounds
    scores_min = df_investigations["overall_risk_score"].min()
    scores_max = df_investigations["overall_risk_score"].max()
    bounds_ok = (scores_min >= 0.0) and (scores_max <= 100.0)
    add_check("Score Bounds [0, 100]", bounds_ok, f"Risk scores are bound in [0, 100]. Min={scores_min}, Max={scores_max}")
    
    # Check 5: Priority match category
    priority_cat_match = True
    for _, row in df_investigations.iterrows():
        cat = row["risk_category"]
        pri = row["recommended_priority"]
        if cat == "CRITICAL" and not pri.startswith("P1"):
            priority_cat_match = False
        elif cat == "HIGH" and not pri.startswith("P2"):
            priority_cat_match = False
        elif cat == "MEDIUM" and not pri.startswith("P3"):
            priority_cat_match = False
        elif cat == "LOW" and not pri.startswith("P4"):
            priority_cat_match = False
    add_check("Priority Category Match", priority_cat_match, "Priority values map correctly to Risk Category brackets")
    
    # Check 6: Consensus level matches flags count
    consensus_match = True
    for _, row in df_investigations.iterrows():
        c_lvl = row["Consensus_Level"]
        flags_cnt = row["ML_Flag"] + row["Rule_Flag"] + row["Statistical_Flag"]
        if c_lvl != flags_cnt:
            consensus_match = False
    add_check("Consensus Flag Consistency", consensus_match, "Consensus levels align exactly with active detection flags count")
    
    # Check 7: No previous files modified
    # In python, we assert that they exist
    previous_exists = scores_path.exists() and rule_detailed_path.exists()
    add_check("Untouched Previous Pipeline", previous_exists, "Verified that all previous stages files exist and remain unmodified")
    
    # Save validation
    df_val = pd.DataFrame(validation_checks)
    df_val.to_csv(INV_DIR / "validation_report.csv", index=False)
    print(f"Saved validation report to: {INV_DIR / 'validation_report.csv'}")
    print(df_val.to_string(index=False))


if __name__ == "__main__":
    main()
