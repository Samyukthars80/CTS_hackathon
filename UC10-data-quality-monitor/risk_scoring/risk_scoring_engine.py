from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

# Setup directories
RISK_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = RISK_DIR.parent
CLEANED_DIR = WORKSPACE_DIR / "outputs"
ML_DIR = WORKSPACE_DIR / "ml_anomaly_detection"
RULE_DIR = WORKSPACE_DIR / "anomaly_detection"
OUTPUTS_DIR = RISK_DIR / "outputs"
REPORTS_DIR = RISK_DIR / "reports"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Datasets
DATASETS = {
    "MEDICAL_CLAIM": CLEANED_DIR / "medical_claim_cleaned.csv",
    "PHARMACY_CLAIM": CLEANED_DIR / "pharmacy_claim_cleaned.csv",
    "PRIOR_AUTH": CLEANED_DIR / "authorization_cleaned.csv"
}

# Features lists to compute statistical outliers
NUMERICAL_COLS = {
    "MEDICAL_CLAIM": ["Billed_Amount", "Allowed_Amount", "Paid_Amount", "Patient_Responsibility", "Retry_Count", "Processing_Latency_Days", "SLA_Target_Days", "Beneficiary_Record_Count", "Provider_Total_Records", "Provider_Denial_Rate"],
    "PHARMACY_CLAIM": ["Billed_Amount", "Allowed_Amount", "Paid_Amount", "Patient_Responsibility", "Retry_Count", "Processing_Latency_Days", "SLA_Target_Days", "Days_Supply", "Quantity_Dispensed", "Beneficiary_Record_Count", "Provider_Total_Records", "Provider_Denial_Rate"],
    "PRIOR_AUTH": ["Retry_Count", "Processing_Latency_Days", "SLA_Target_Days", "Provider_Total_Records", "Provider_Denial_Rate", "Beneficiary_Record_Count"]
}


def compute_iqr_outliers_detailed(df: pd.DataFrame, rtype: str) -> dict[str, list[str]]:
    """Compute IQR outliers and return mapping of Record_ID to list of outlier columns."""
    outliers_map = {}
    cols = NUMERICAL_COLS[rtype]
    
    # Initialize empty list for all records
    for rid in df["Record_ID"].astype(str):
        outliers_map[rid] = []
        
    for col in cols:
        if col in df.columns:
            non_nulls = df[col].dropna()
            if len(non_nulls) == 0:
                continue
            q1 = non_nulls.quantile(0.25)
            q3 = non_nulls.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            outliers_df = df[mask.fillna(False)]
            for rid in outliers_df["Record_ID"].astype(str):
                outliers_map[rid].append(col)
                
    return outliers_map


def build_risk_score_dataset() -> pd.DataFrame:
    print("Consolidating record level anomaly signals from all stages...")
    
    # 1. Load validated consensus file
    consensus_path = ML_DIR / "model_validation" / "cross_method_consensus.csv"
    if not consensus_path.exists():
        raise FileNotFoundError(f"Consensus file not found: {consensus_path}")
    consensus_df = pd.read_csv(consensus_path)
    print(f"Loaded consensus file. Shape: {consensus_df.shape}")
    
    # Convert consensus columns
    consensus_df["Record_ID"] = consensus_df["Record_ID"].astype(str)
    consensus_df["Record_Type"] = consensus_df["Record_Type"].astype(str)
    
    # Set up container fields for consolidation
    npi_map = {}
    ml_score_map = {}
    ml_pct_map = {}
    ml_factors_map = {}
    
    # 2. Load ML Predictions and Anomalies to get scores, percentiles, factors
    ml_prefixes = {
        "MEDICAL_CLAIM": "medical",
        "PHARMACY_CLAIM": "pharmacy",
        "PRIOR_AUTH": "prior_auth"
    }
    
    for rtype, prefix in ml_prefixes.items():
        # Load predictions
        pred_path = ML_DIR / "outputs" / f"{prefix}_ml_predictions.csv"
        if pred_path.exists():
            pdf = pd.read_csv(pred_path)
            for _, row in pdf.iterrows():
                rid = str(row["Record_ID"])
                ml_score_map[rid] = float(row["Isolation_Forest_Score"])
                ml_pct_map[rid] = float(row["Anomaly_Percentile"])
        
        # Load anomaly explanations
        anom_path = ML_DIR / "outputs" / f"{prefix}_ml_anomalies.csv"
        if anom_path.exists():
            adf = pd.read_csv(anom_path)
            for _, row in adf.iterrows():
                rid = str(row["Record_ID"])
                ml_factors_map[rid] = str(row["Potential_Contributing_Factors"])
                
    # 3. Load Rule Violations summaries
    rule_sum_path = RULE_DIR / "outputs" / "record_level_summary.csv"
    rule_sum_map = {}
    if rule_sum_path.exists():
        rs_df = pd.read_csv(rule_sum_path)
        for _, row in rs_df.iterrows():
            rid = str(row["Record_ID"])
            rule_sum_map[rid] = {
                "Total_Rule_Violations": int(row["Total_Rule_Violations"]),
                "High_Severity_Rule_Count": int(row["High"])
            }
            
    # Load detailed rule violations
    detailed_rule_path = RULE_DIR / "outputs" / "combined_rule_anomalies.csv"
    detailed_rule_map = {}
    if detailed_rule_path.exists():
        dr_df = pd.read_csv(detailed_rule_path)
        for _, row in dr_df.iterrows():
            rid = str(row["Record_ID"])
            if rid not in detailed_rule_map:
                detailed_rule_map[rid] = {
                    "Rule_IDs": [],
                    "Explanations": []
                }
            detailed_rule_map[rid]["Rule_IDs"].append(str(row["Rule_ID"]))
            detailed_rule_map[rid]["Explanations"].append(str(row["Explanation"]))
            
    # 4. Load cleaned source data to get Provider_NPI and calculate IQR outliers
    stat_outliers_map = {}
    for rtype, path in DATASETS.items():
        df_clean = pd.read_csv(path)
        # NPI mapping
        for _, row in df_clean.iterrows():
            rid = str(row["Record_ID"])
            npi = row.get("Provider_NPI")
            npi_map[rid] = str(int(npi)) if pd.notna(npi) else "UNKNOWN_PROVIDER"
            
        # Statistical IQR outliers calculation
        rtype_outliers = compute_iqr_outliers_detailed(df_clean, rtype)
        stat_outliers_map.update(rtype_outliers)
        
    # 5. Populate consolidated rows
    consolidated_rows = []
    for _, row in consensus_df.iterrows():
        rid = row["Record_ID"]
        rtype = row["Record_Type"]
        
        # ML values
        ml_flag = int(row["ML_Anomaly"])
        ml_score = ml_score_map.get(rid, 0.0)
        ml_pct = ml_pct_map.get(rid, 0.0)
        ml_explanations = ml_factors_map.get(rid, "")
        
        # Rule values
        rule_flag = int(row["Rule_Anomaly"])
        r_sum = rule_sum_map.get(rid, {"Total_Rule_Violations": 0, "High_Severity_Rule_Count": 0})
        total_rule_violations = r_sum["Total_Rule_Violations"]
        high_severity_rule_count = r_sum["High_Severity_Rule_Count"]
        
        rule_detail = detailed_rule_map.get(rid, {"Rule_IDs": [], "Explanations": []})
        rule_ids = "; ".join(rule_detail["Rule_IDs"])
        rule_explanations = "; ".join(rule_detail["Explanations"])
        
        # Statistical values
        stat_flag = int(row["Statistical_Anomaly"])
        outlier_cols = stat_outliers_map.get(rid, [])
        outlier_column_count = len(outlier_cols)
        statistical_anomaly_info = "; ".join(outlier_cols)
        
        # Consensus
        consensus_level = int(row["Consensus_Level"])
        npi = npi_map.get(rid, "UNKNOWN_PROVIDER")
        
        # Compute Risk Score explicitly
        # A. ML Contribution
        ml_cont = 15.0 * ml_flag + 10.0 * (ml_pct / 100.0)
        
        # B. Statistical Contribution
        stat_cont = 15.0 * stat_flag
        if outlier_column_count > 1:
            stat_cont += 5.0
            
        # C. Rule-Based Contribution
        rule_cont = 10.0 * rule_flag + min(15.0, 5.0 * total_rule_violations) + min(20.0, 10.0 * high_severity_rule_count)
        
        # D. Consensus Contribution
        consensus_cont = 0.0
        if consensus_level == 2:
            consensus_cont = 10.0
        elif consensus_level == 3:
            consensus_cont = 20.0
            
        raw_score = ml_cont + stat_cont + rule_cont + consensus_cont
        risk_score = round(min(100.0, raw_score), 2)
        
        # Determine Risk Level
        if risk_score < 25.0:
            risk_level = "LOW"
            priority = 4
        elif risk_score < 50.0:
            risk_level = "MEDIUM"
            priority = 3
        elif risk_score < 75.0:
            risk_level = "HIGH"
            priority = 2
        else:
            risk_level = "CRITICAL"
            priority = 1
            
        # Identify Primary Risk Factors (human-readable)
        factors = []
        if ml_flag:
            factors.append(f"ML Anomaly (Percentile: {ml_pct:.1f}%)")
        if stat_flag:
            factors.append(f"Statistical outlier in {len(outlier_cols)} columns ({statistical_anomaly_info})")
        if rule_flag:
            factors.append(f"Violates {total_rule_violations} business rules")
            if high_severity_rule_count > 0:
                factors.append(f"High-severity rule violations count: {high_severity_rule_count}")
        if consensus_level >= 2:
            factors.append(f"Multi-method agreement (Consensus Level: {consensus_level})")
            
        primary_risk_factors = "; ".join(factors) if factors else "No significant risk factors flagged"
        
        # Generate Deterministic Plain-Language Explanation (Honest, non-defamatory)
        methods = []
        if ml_flag:
            methods.append("Machine Learning Isolation Forest")
        if stat_flag:
            methods.append("Statistical Outlier Analysis")
        if rule_flag:
            methods.append("Rule-Based Violation Checks")
            
        explanation = f"This {rtype.lower().replace('_', ' ')} was classified as {risk_level} risk ({risk_score:.1f}/100)"
        if consensus_level > 0:
            explanation += f" because it was flagged by {', and '.join(methods)} detectors."
            if rule_flag:
                explanation += f" It exhibited {total_rule_violations} rule violations (IDs: {rule_ids}) with explanations: {rule_explanations}."
            if stat_flag:
                explanation += f" It was identified as an outlier in the following numerical features: {statistical_anomaly_info}."
            if ml_flag:
                explanation += f" The Isolation Forest isolated this record in feature space (percentile: {ml_pct:.1f}%) with contributing factors: {ml_explanations}."
            explanation += " This record requires investigation to verify the integrity of the submitted claims."
        else:
            explanation += " because it was not flagged by any of the independent detection methods."
            
        consolidated_rows.append({
            "Record_ID": rid,
            "Record_Type": rtype,
            "Provider_NPI": npi,
            "ML_Anomaly_Flag": ml_flag,
            "ML_Anomaly_Score": ml_score,
            "Anomaly_Percentile": ml_pct,
            "ML_Explanation": ml_explanations,
            "Statistical_Anomaly_Flag": stat_flag,
            "Statistical_Outlier_Columns": statistical_anomaly_info,
            "Outlier_Column_Count": outlier_column_count,
            "Rule_Anomaly_Flag": rule_flag,
            "Rule_Violation_Count": total_rule_violations,
            "High_Severity_Rule_Count": high_severity_rule_count,
            "Rule_IDs": rule_ids,
            "Rule_Explanations": rule_explanations,
            "Consensus_Level": consensus_level,
            "Risk_Score": round(risk_score, 2),
            "Risk_Level": risk_level,
            "Priority": priority,
            "Primary_Risk_Factors": primary_risk_factors,
            "Explanation": explanation
        })
        
    consolidated_df = pd.DataFrame(consolidated_rows)
    
    # Save Outputs
    scores_path = OUTPUTS_DIR / "record_risk_scores.csv"
    consolidated_df.to_csv(scores_path, index=False)
    print(f"Saved consolidated record risk scores to: {scores_path}")
    
    return consolidated_df


def generate_risk_factor_summary(df: pd.DataFrame):
    print("Generating risk factor summary statistics...")
    
    # Let's count occurrence of different risk flags and rule violations
    total_records = len(df)
    
    factors = [
        ("ML Anomaly", df["ML_Anomaly_Flag"] == 1, "Medium"),
        ("Statistical Outlier", df["Statistical_Anomaly_Flag"] == 1, "Medium"),
        ("Rule-Based Anomaly", df["Rule_Anomaly_Flag"] == 1, "High"),
        ("Level 3 Consensus", df["Consensus_Level"] == 3, "Critical"),
        ("Level 2 Consensus", df["Consensus_Level"] == 2, "High"),
        ("High Severity Rule Violation", df["High_Severity_Rule_Count"] > 0, "Critical"),
        ("Multiple Rule Violations", df["Rule_Violation_Count"] > 1, "High")
    ]
    
    # Scan detailed rule IDs to compile counts of specific checks
    detailed_rule_path = RULE_DIR / "outputs" / "combined_rule_anomalies.csv"
    if detailed_rule_path.exists():
        dr_df = pd.read_csv(detailed_rule_path)
        rule_counts = dr_df["Rule_ID"].value_counts()
        for rule_id, cnt in rule_counts.items():
            # Get severity
            sev = dr_df[dr_df["Rule_ID"] == rule_id]["Severity"].iloc[0]
            # Map severity to High/Medium/Critical
            m_sev = "High" if sev == "High" else ("Medium" if sev == "Medium" else "Critical")
            
            # Find affected record types
            rtypes = dr_df[dr_df["Rule_ID"] == rule_id]["Record_Type"].unique().tolist()
            
            # Add to factors
            # We must mask on df: checking if rule_id is in Rule_IDs
            mask = df["Rule_IDs"].fillna("").apply(lambda x: rule_id in x.split("; "))
            factors.append((f"Rule Violation: {rule_id}", mask, m_sev))
            
    summary_rows = []
    for name, mask, importance in factors:
        cnt = mask.sum()
        pct = (cnt / total_records) * 100
        
        # Affected record types
        affected_types = df[mask]["Record_Type"].unique().tolist()
        rtypes_str = "; ".join(affected_types) if affected_types else "None"
        
        summary_rows.append({
            "Risk_Factor": name,
            "Record_Count": int(cnt),
            "Percentage": round(pct, 2),
            "Record_Types_Affected": rtypes_str,
            "Severity_Importance": importance
        })
        
    summary_df = pd.DataFrame(summary_rows)
    summary_path = OUTPUTS_DIR / "risk_factor_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved risk factor summary CSV to: {summary_path}")


def main():
    df = build_risk_score_dataset()
    generate_risk_factor_summary(df)


if __name__ == "__main__":
    main()
