from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

# Setup paths
RISK_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = RISK_DIR.parent
CLEANED_DIR = WORKSPACE_DIR / "outputs"
OUTPUTS_DIR = RISK_DIR / "outputs"


def build_provider_risk_summary():
    print("\nAggregating risk metrics at the provider level...")
    
    # 1. Load record risk scores
    scores_path = OUTPUTS_DIR / "record_risk_scores.csv"
    if not scores_path.exists():
        raise FileNotFoundError(f"Consolidated record risk scores not found: {scores_path}")
    df = pd.read_csv(scores_path)
    
    # 2. Load cleaned datasets to extract statuses for denial rates
    status_map = {}
    datasets = {
        "MEDICAL_CLAIM": CLEANED_DIR / "medical_claim_cleaned.csv",
        "PHARMACY_CLAIM": CLEANED_DIR / "pharmacy_claim_cleaned.csv",
        "PRIOR_AUTH": CLEANED_DIR / "authorization_cleaned.csv"
    }
    
    for rtype, path in datasets.items():
        clean_df = pd.read_csv(path)
        for _, row in clean_df.iterrows():
            rid = str(row["Record_ID"])
            status_map[rid] = str(row.get("Status")).upper() if pd.notna(row.get("Status")) else "UNKNOWN"
            
    # 3. Add status back to record df
    df["Status"] = df["Record_ID"].map(status_map).fillna("UNKNOWN")
    
    # 4. Group by Provider_NPI
    grouped = df.groupby("Provider_NPI")
    
    provider_rows = []
    for npi, group in grouped:
        npi_str = str(npi)
        
        # Predominant record type
        rtypes = group["Record_Type"].value_counts()
        pred_type = rtypes.index[0] if len(rtypes) > 0 else "UNKNOWN"
        
        total_records = len(group)
        flagged_records = (group["Risk_Score"] >= 25.0).sum()
        flagged_pct = (flagged_records / total_records) * 100
        
        ml_anom_cnt = (group["ML_Anomaly_Flag"] == 1).sum()
        stat_anom_cnt = (group["Statistical_Anomaly_Flag"] == 1).sum()
        rule_viol_cnt = group["Rule_Violation_Count"].sum()
        
        high_risk_cnt = (group["Risk_Level"] == "HIGH").sum()
        crit_risk_cnt = (group["Risk_Level"] == "CRITICAL").sum()
        
        avg_score = group["Risk_Score"].mean()
        max_score = group["Risk_Score"].max()
        
        # Denial Rate calculation
        # Sum of Denied or Rejected / Total non-pending
        denied_flags = ["DENIED", "REJECTED"]
        non_pend_group = group[group["Status"] != "PENDING"]
        total_non_pending = len(non_pend_group)
        
        denied_cnt = non_pend_group["Status"].isin(denied_flags).sum()
        denied_rate = (denied_cnt / total_non_pending) * 100 if total_non_pending > 0 else 0.0
        
        # Assign Provider status
        prov_status = "NORMAL"
        if crit_risk_cnt > 0 or high_risk_cnt > 0 or flagged_pct > 10.0 or avg_score > 35.0:
            prov_status = "PROVIDER REQUIRING REVIEW"
            
        # Compile Primary Reasons
        reasons = []
        if crit_risk_cnt > 0:
            reasons.append(f"{crit_risk_cnt} critical-risk records")
        if high_risk_cnt > 0:
            reasons.append(f"{high_risk_cnt} high-risk records")
        if flagged_pct > 10.0:
            reasons.append(f"{flagged_pct:.1f}% flagged records (Threshold: 10%)")
        if avg_score > 35.0:
            reasons.append(f"High average claim risk score of {avg_score:.1f}")
        if denied_rate > 20.0:
            reasons.append(f"High claim denial/rejection rate of {denied_rate:.1f}%")
            
        primary_reasons = "; ".join(reasons) if reasons else "Normal claim patterns"
        
        provider_rows.append({
            "Provider_NPI": npi_str,
            "Predominant_Record_Type": pred_type,
            "Total_Records": int(total_records),
            "Flagged_Records": int(flagged_records),
            "Flagged_Record_Percentage": round(flagged_pct, 2),
            "ML_Anomaly_Count": int(ml_anom_cnt),
            "Statistical_Anomaly_Count": int(stat_anom_cnt),
            "Rule_Violation_Count": int(rule_viol_cnt),
            "High_Risk_Record_Count": int(high_risk_cnt),
            "Critical_Risk_Record_Count": int(crit_risk_cnt),
            "Average_Risk_Score": round(float(avg_score), 2),
            "Maximum_Risk_Score": round(float(max_score), 2),
            "Denial_Rate_Percentage": round(denied_rate, 2),
            "Provider_Risk_Level": "HIGH-RISK PROVIDER" if prov_status == "PROVIDER REQUIRING REVIEW" else "LOW-RISK PROVIDER",
            "Provider_Status": prov_status,
            "Primary_Risk_Factors": primary_reasons
        })
        
    provider_df = pd.DataFrame(provider_rows)
    
    # Sort providers by risk severity
    provider_df = provider_df.sort_values(
        by=["Critical_Risk_Record_Count", "High_Risk_Record_Count", "Average_Risk_Score", "Total_Records"],
        ascending=False
    ).reset_index(drop=True)
    
    provider_path = OUTPUTS_DIR / "provider_risk_summary.csv"
    provider_df.to_csv(provider_path, index=False)
    print(f"Saved provider risk summary to: {provider_path}")


if __name__ == "__main__":
    build_provider_risk_summary()
