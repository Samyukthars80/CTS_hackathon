from __future__ import annotations

import os
from pathlib import Path
import pandas as pd
import numpy as np

# Setup paths
FINAL_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = FINAL_DIR.parent
CLEANED_DIR = WORKSPACE_DIR / "outputs"
ML_DIR = WORKSPACE_DIR / "ml_anomaly_detection"
RULE_DIR = WORKSPACE_DIR / "anomaly_detection"

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


def build_final_risk_scores() -> pd.DataFrame:
    print("Executing consolidated final risk scoring engine...")
    
    # 1. Load inputs
    rule_detailed_path = RULE_DIR / "outputs" / "combined_rule_anomalies.csv"
    df_rule_details = pd.read_csv(rule_detailed_path)
    df_rule_details["Record_ID"] = df_rule_details["Record_ID"].astype(str)
    
    # Group detailed violations and severity per claim
    rule_map = {}
    for _, row in df_rule_details.iterrows():
        rid = str(row["Record_ID"])
        if rid not in rule_map:
            rule_map[rid] = []
        rule_map[rid].append({
            "Rule_ID": str(row["Rule_ID"]),
            "Severity": str(row["Severity"]),
            "Anomaly_Category": str(row.get("Anomaly_Category", "Compliance Check")),
            "Explanation": str(row["Explanation"])
        })
        
    # Load ML predictions and anomaly percentiles
    ml_pct_map = {}
    ml_flag_map = {}
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
                ml_flag_map[rid] = int(row["ML_Anomaly_Flag"])
                ml_score_map[rid] = float(row["Isolation_Forest_Score"])
                
        anom_path = ML_DIR / "outputs" / f"{prefix}_ml_anomalies.csv"
        if anom_path.exists():
            adf = pd.read_csv(anom_path)
            for _, row in adf.iterrows():
                rid = str(row["Record_ID"])
                ml_explanations[rid] = str(row["Potential_Contributing_Factors"])
                
    # Load clean data for Provider_NPI and computing statistical deviations
    train_stats = calculate_train_split_statistics()
    clean_dfs = {}
    npi_map = {}
    
    for rtype, path in DATASETS.items():
        df_clean = pd.read_csv(path)
        df_clean["Record_ID"] = df_clean["Record_ID"].astype(str)
        clean_dfs[rtype] = df_clean.set_index("Record_ID")
        
        for _, row in df_clean.iterrows():
            rid = str(row["Record_ID"])
            npi = row.get("Provider_NPI")
            npi_map[rid] = str(int(npi)) if pd.notna(npi) else "UNKNOWN_PROVIDER"
            
    # Iterate and construct final score rows
    scored_rows = []
    
    # Consolidate all records across datasets
    all_record_ids = []
    for rtype, df in clean_dfs.items():
        for rid in df.index:
            all_record_ids.append((rid, rtype))
            
    print(f"\nScoring {len(all_record_ids)} total records...")
    for rid, rtype in all_record_ids:
        # A. Rule Evidence
        violations = rule_map.get(rid, [])
        rule_flag = 1 if len(violations) > 0 else 0
        rule_viol_cnt = len(violations)
        rule_ids = "; ".join([v["Rule_ID"] for v in violations])
        rule_categories = "; ".join(list(set([v["Anomaly_Category"] for v in violations])))
        
        severities = [v["Severity"].upper() for v in violations]
        highest_sev = "None"
        rule_contribution = 0.0
        if "HIGH" in severities:
            highest_sev = "High"
            rule_contribution = 35.0
        elif "MEDIUM" in severities:
            highest_sev = "Medium"
            rule_contribution = 20.0
        elif "LOW" in severities:
            highest_sev = "Low"
            rule_contribution = 10.0
            
        # B. Statistical Evidence
        clean_row = clean_dfs[rtype].loc[rid]
        col_stats = train_stats[rtype]
        
        stat_outlier_cols = []
        max_z = 0.0
        for col in col_stats.keys():
            val = clean_row.get(col)
            if pd.notna(val):
                mean_val, std_val = col_stats[col]
                if std_val > 0.0:
                    z = abs(val - mean_val) / std_val
                    if z > 1.5:
                        stat_outlier_cols.append(col)
                        if z > max_z:
                            max_z = z
                            
        stat_flag = 1 if len(stat_outlier_cols) > 0 else 0
        stat_anomaly_cnt = len(stat_outlier_cols)
        stat_categories = "; ".join([f"Outlier: {col}" for col in stat_outlier_cols])
        
        statistical_contribution = 0.0
        if max_z > 3.0:
            statistical_contribution = 25.0
        elif max_z > 1.5:
            statistical_contribution = 15.0
            
        # C. ML Evidence
        ml_flag = ml_flag_map.get(rid, 0)
        ml_pct = ml_pct_map.get(rid, 0.0)
        ml_score = ml_score_map.get(rid, 0.0)
        ml_exp = ml_explanations.get(rid, "None")
        
        ml_contribution = 0.0
        if ml_flag == 1:
            ml_contribution = (ml_pct / 100.0) * 40.0
            
        # Overall Risk Score
        final_risk_score = ml_contribution + rule_contribution + statistical_contribution
        final_risk_score = round(final_risk_score, 2)
        
        # Consensus Level
        consensus_level = ml_flag + rule_flag + stat_flag
        
        # Detection Methods mapping
        methods = []
        if ml_flag:
            methods.append("ML")
        if rule_flag:
            methods.append("Rule-Based")
        if stat_flag:
            methods.append("Statistical")
        detection_methods = " + ".join(methods) if methods else "None"
        
        # Risk Priority Classification
        if final_risk_score >= 80.0:
            risk_priority = "CRITICAL"
        elif final_risk_score >= 60.0:
            risk_priority = "HIGH"
        elif final_risk_score >= 40.0:
            risk_priority = "MEDIUM"
        else:
            risk_priority = "LOW"
            
        # Plain language explanation (Why_Flagged)
        reasons = []
        if rule_flag:
            reasons.append(f"the rule engine flagged {rule_viol_cnt} compliance violations (IDs: {rule_ids})")
        if stat_flag:
            reasons.append(f"statistical analysis identified {stat_anomaly_cnt} unusual continuous features ({stat_categories})")
        if ml_flag:
            reasons.append(f"the machine learning model classified the claim as highly anomalous in multi-dimensional space (percentile: {ml_pct:.1f}%)")
            
        if consensus_level > 0:
            explanation = f"Potentially suspicious claim prioritized as {risk_priority} priority (score: {final_risk_score:.2f}/100) because {', and '.join(reasons)}."
            explanation += " This indicates an unusual billing pattern requiring investigation to verify claims integrity."
        else:
            explanation = f"Low risk claim (score: {final_risk_score:.2f}/100) exhibiting normal baseline billing behavior."
            
        scored_rows.append({
            "Record_ID": rid,
            "Record_Type": rtype,
            "Provider_NPI": npi_map.get(rid, "UNKNOWN_PROVIDER"),
            "Statistical_Flag": stat_flag,
            "Rule_Flag": rule_flag,
            "ML_Flag": ml_flag,
            "Statistical_Anomaly_Count": stat_anomaly_cnt,
            "Rule_Violation_Count": rule_viol_cnt,
            "ML_Anomaly_Score": ml_score,
            "ML_Anomaly_Percentile": ml_pct,
            "Rule_Severity": highest_sev,
            "Rule_Categories": rule_categories if rule_categories else "None",
            "Rule_IDs": rule_ids if rule_ids else "None",
            "Statistical_Categories": stat_categories if stat_categories else "None",
            "ML_Explanation": ml_exp,
            "Consensus_Level": consensus_level,
            "Detection_Methods": detection_methods,
            "ML_Contribution": ml_contribution,
            "Rule_Contribution": rule_contribution,
            "Statistical_Contribution": statistical_contribution,
            "Final_Risk_Score": final_risk_score,
            "Risk_Priority": risk_priority,
            "Why_Flagged": explanation
        })
        
    return pd.DataFrame(scored_rows)


def main():
    print("="*60)
    print("RUNNING FINAL RISK CONSOLIDATION PIPELINE")
    print("="*60)
    
    # 1. Score all records
    df_all = build_final_risk_scores()
    
    # Output complete master list
    df_all.to_csv(FINAL_DIR / "final_risk_scores.csv", index=False)
    print(f"Saved complete master output to: {FINAL_DIR / 'final_risk_scores.csv'}")
    
    # 2. Output investigation queue (only flagged records: consensus_level >= 1)
    df_queue = df_all[df_all["Consensus_Level"] >= 1].copy()
    
    # Sort by:
    # 1. Risk_Priority (CRITICAL=1, HIGH=2, MEDIUM=3, LOW=4) - wait, since low can be flagged (consensus >= 1) we map priorities to numerical order
    priority_rank = {"CRITICAL": 1, "HIGH": 2, "MEDIUM": 3, "LOW": 4}
    df_queue["priority_num"] = df_queue["Risk_Priority"].map(priority_rank).fillna(4)
    
    df_queue = df_queue.sort_values(
        by=["priority_num", "Final_Risk_Score", "Consensus_Level"],
        ascending=[True, False, False]
    ).reset_index(drop=True)
    
    df_queue["Rank"] = df_queue.index + 1
    
    # Select columns for queue schema
    queue_cols = [
        "Rank", "Record_ID", "Record_Type", "Final_Risk_Score", "Risk_Priority",
        "Consensus_Level", "Detection_Methods", "ML_Anomaly_Score", "Statistical_Flag",
        "Rule_Flag", "Rule_Severity", "Rule_IDs", "Rule_Categories", "Why_Flagged"
    ]
    df_queue_final = df_queue[queue_cols].rename(columns={"Rule_Categories": "Anomaly_Categories"})
    df_queue_final.to_csv(FINAL_DIR / "investigation_queue.csv", index=False)
    print(f"Saved prioritized investigation queue to: {FINAL_DIR / 'investigation_queue.csv'} (Count: {len(df_queue_final)})")
    
    # 3. Provider-Level Risk Aggregation
    print("\nExecuting provider-level aggregations...")
    # Load Status to calculate cross domain density if needed, or get counts
    status_map = {}
    for rtype, path in DATASETS.items():
        c_df = pd.read_csv(path)
        for _, row in c_df.iterrows():
            rid = str(row["Record_ID"])
            status_map[rid] = str(row.get("Status")).upper() if pd.notna(row.get("Status")) else "UNKNOWN"
            
    df_all["Status"] = df_all["Record_ID"].map(status_map).fillna("UNKNOWN")
    
    grouped = df_all.groupby("Provider_NPI")
    prov_rows = []
    
    for npi, group in grouped:
        npi_str = str(npi)
        record_count = len(group)
        
        med_cnt = (group["Record_Type"] == "MEDICAL_CLAIM").sum()
        rx_cnt = (group["Record_Type"] == "PHARMACY_CLAIM").sum()
        auth_cnt = (group["Record_Type"] == "PRIOR_AUTH").sum()
        
        flagged_cnt = (group["Consensus_Level"] >= 1).sum()
        flagged_rate = (flagged_cnt / record_count) * 100.0 if record_count > 0 else 0.0
        
        high_cnt = (group["Risk_Priority"] == "HIGH").sum()
        crit_cnt = (group["Risk_Priority"] == "CRITICAL").sum()
        
        avg_score = group["Final_Risk_Score"].mean()
        max_score = group["Final_Risk_Score"].max()
        
        rule_flag_cnt = (group["Rule_Flag"] == 1).sum()
        stat_flag_cnt = (group["Statistical_Flag"] == 1).sum()
        ml_flag_cnt = (group["ML_Flag"] == 1).sum()
        
        avg_consensus = group["Consensus_Level"].mean()
        
        # Transparent provider risk classification logic
        # CRITICAL: Avg Risk Score >= 45.0 OR Critical Count >= 2
        # HIGH: Avg Risk Score >= 30.0 OR High Count >= 3 OR Critical Count >= 1
        # MEDIUM: Avg Risk Score >= 15.0 OR Flagged Rate >= 10.0% OR Flagged Count >= 1
        # LOW: Otherwise
        if avg_score >= 45.0 or crit_cnt >= 2:
            prov_level = "CRITICAL"
        elif avg_score >= 30.0 or high_cnt >= 3 or crit_cnt >= 1:
            prov_level = "HIGH"
        elif avg_score >= 15.0 or flagged_rate >= 10.0 or flagged_cnt >= 1:
            prov_level = "MEDIUM"
        else:
            prov_level = "LOW"
            
        prov_rows.append({
            "Provider_NPI": npi_str,
            "Record_Count": record_count,
            "Medical_Claim_Count": med_cnt,
            "Pharmacy_Claim_Count": rx_cnt,
            "Prior_Auth_Count": auth_cnt,
            "Flagged_Record_Count": flagged_cnt,
            "Flagged_Record_Rate": round(flagged_rate, 2),
            "High_Risk_Record_Count": high_cnt,
            "Critical_Risk_Record_Count": crit_cnt,
            "Average_Risk_Score": round(avg_score, 2),
            "Maximum_Risk_Score": max_score,
            "Rule_Flag_Count": rule_flag_cnt,
            "Statistical_Flag_Count": stat_flag_cnt,
            "ML_Flag_Count": ml_flag_cnt,
            "Average_Consensus_Level": round(avg_consensus, 2),
            "Provider_Risk_Level": prov_level
        })
        
    df_prov = pd.DataFrame(prov_rows)
    df_prov = df_prov.sort_values(by=["Critical_Risk_Record_Count", "High_Risk_Record_Count", "Average_Risk_Score"], ascending=False).reset_index(drop=True)
    df_prov.to_csv(FINAL_DIR / "provider_risk_summary.csv", index=False)
    print(f"Saved provider risk summary to: {FINAL_DIR / 'provider_risk_summary.csv'}")
    
    # 4. Cross-Record-Type Analysis
    print("\nExecuting cross-record-type aggregations...")
    # Find providers with claims in multiple record types
    cross_rows = []
    for _, prow in df_prov.iterrows():
        npi = prow["Provider_NPI"]
        med_cnt = int(prow["Medical_Claim_Count"])
        rx_cnt = int(prow["Pharmacy_Claim_Count"])
        auth_cnt = int(prow["Prior_Auth_Count"])
        
        # Active counts
        active_types = sum([1 for c in [med_cnt, rx_cnt, auth_cnt] if c > 0])
        if active_types >= 2:
            # Gather flagged claims in each record type
            pgroup = df_all[df_all["Provider_NPI"] == npi]
            med_flagged = ((pgroup["Record_Type"] == "MEDICAL_CLAIM") & (pgroup["Consensus_Level"] >= 1)).sum()
            rx_flagged = ((pgroup["Record_Type"] == "PHARMACY_CLAIM") & (pgroup["Consensus_Level"] >= 1)).sum()
            auth_flagged = ((pgroup["Record_Type"] == "PRIOR_AUTH") & (pgroup["Consensus_Level"] >= 1)).sum()
            
            cross_rows.append({
                "Provider_NPI": npi,
                "Active_Record_Type_Count": active_types,
                "Medical_Claims_Submitted": med_cnt,
                "Medical_Claims_Flagged": int(med_flagged),
                "Pharmacy_Claims_Submitted": rx_cnt,
                "Pharmacy_Claims_Flagged": int(rx_flagged),
                "Prior_Auth_Submitted": auth_cnt,
                "Prior_Auth_Flagged": int(auth_flagged),
                "Total_Claims_Flagged": int(prow["Flagged_Record_Count"]),
                "Provider_Risk_Level": prow["Provider_Risk_Level"],
                "Average_Risk_Score": prow["Average_Risk_Score"]
            })
            
    df_cross = pd.DataFrame(cross_rows)
    df_cross = df_cross.sort_values(by=["Active_Record_Type_Count", "Total_Claims_Flagged", "Average_Risk_Score"], ascending=False).reset_index(drop=True)
    df_cross.to_csv(FINAL_DIR / "provider_cross_record_type_summary.csv", index=False)
    print(f"Saved cross-record-type provider summary to: {FINAL_DIR / 'provider_cross_record_type_summary.csv'}")
    
    # 5. Dashboard-Ready Summary Files
    # A. overall_risk_summary.csv
    tot_rec = len(df_all)
    tot_flg = len(df_queue_final)
    flg_rate = (tot_flg / tot_rec) * 100
    crit_cnt = (df_all["Risk_Priority"] == "CRITICAL").sum()
    high_cnt = (df_all["Risk_Priority"] == "HIGH").sum()
    med_cnt = (df_all["Risk_Priority"] == "MEDIUM").sum()
    low_cnt = (df_all["Risk_Priority"] == "LOW").sum()
    
    df_overall_sum = pd.DataFrame([{
        "Total_Records": tot_rec,
        "Total_Flagged": tot_flg,
        "Flag_Rate_Percentage": round(flg_rate, 2),
        "Critical_Count": crit_cnt,
        "High_Count": high_cnt,
        "Medium_Count": med_cnt,
        "Low_Count": low_cnt
    }])
    df_overall_sum.to_csv(FINAL_DIR / "overall_risk_summary.csv", index=False)
    
    # B. record_type_risk_summary.csv
    rtype_rows = []
    for rtype in DATASETS.keys():
        sub = df_all[df_all["Record_Type"] == rtype]
        sub_flg = (sub["Consensus_Level"] >= 1).sum()
        rtype_rows.append({
            "Record_Type": rtype,
            "Total_Records": len(sub),
            "Total_Flagged": sub_flg,
            "Flag_Rate_Percentage": round((sub_flg / len(sub)) * 100, 2) if len(sub) > 0 else 0.0,
            "Critical_Count": (sub["Risk_Priority"] == "CRITICAL").sum(),
            "High_Count": (sub["Risk_Priority"] == "HIGH").sum(),
            "Medium_Count": (sub["Risk_Priority"] == "MEDIUM").sum(),
            "Low_Count": (sub["Risk_Priority"] == "LOW").sum()
        })
    pd.DataFrame(rtype_rows).to_csv(FINAL_DIR / "record_type_risk_summary.csv", index=False)
    
    # C. detection_method_summary.csv
    ml_only = ((df_all["ML_Flag"] == 1) & (df_all["Rule_Flag"] == 0) & (df_all["Statistical_Flag"] == 0)).sum()
    rule_only = ((df_all["ML_Flag"] == 0) & (df_all["Rule_Flag"] == 1) & (df_all["Statistical_Flag"] == 0)).sum()
    stat_only = ((df_all["ML_Flag"] == 0) & (df_all["Rule_Flag"] == 0) & (df_all["Statistical_Flag"] == 1)).sum()
    two_methods = (df_all["Consensus_Level"] == 2).sum()
    three_methods = (df_all["Consensus_Level"] == 3).sum()
    
    df_method = pd.DataFrame([{
        "ML_Only_Count": int(ml_only),
        "Rule_Only_Count": int(rule_only),
        "Statistical_Only_Count": int(stat_only),
        "Two_Method_Consensus_Count": int(two_methods),
        "Three_Method_Consensus_Count": int(three_methods),
        "Total_ML_Flags": int((df_all["ML_Flag"] == 1).sum()),
        "Total_Rule_Flags": int((df_all["Rule_Flag"] == 1).sum()),
        "Total_Statistical_Flags": int((df_all["Statistical_Flag"] == 1).sum())
    }])
    df_method.to_csv(FINAL_DIR / "detection_method_summary.csv", index=False)
    
    # D. risk_distribution.csv
    # Bin score into deciles [0-10, 10-20, ..., 90-100]
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 101]
    labels = ["0-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60-70", "70-80", "80-90", "90-100"]
    df_all["score_bin"] = pd.cut(df_all["Final_Risk_Score"], bins=bins, labels=labels, right=False)
    df_dist = df_all["score_bin"].value_counts().reindex(labels).reset_index()
    df_dist.columns = ["Score_Bin", "Record_Count"]
    df_dist.to_csv(FINAL_DIR / "risk_distribution.csv", index=False)
    print("Dashboard summaries generated.")


if __name__ == "__main__":
    main()
