import pandas as pd
import numpy as np
import json
import os
import sys
import datetime

# ---------------------------------------------------------
# RULE CATALOG
# ---------------------------------------------------------
RULES = [
    {
        "rule_id": "R001",
        "rule_name": "Record_ID Completeness",
        "dimension": "Completeness",
        "severity": "Critical",
        "applicable_record_types": ["MEDICAL_CLAIM", "PHARMACY_CLAIM", "PRIOR_AUTH"],
        "fields": ["Record_ID"],
        "description": "Record_ID must not be missing.",
        "recommended_fix": "Investigate source system extraction logic."
    },
    {
        "rule_id": "R002",
        "rule_name": "Record_ID Uniqueness",
        "dimension": "Uniqueness",
        "severity": "Critical",
        "applicable_record_types": ["MEDICAL_CLAIM", "PHARMACY_CLAIM", "PRIOR_AUTH"],
        "fields": ["Record_ID"],
        "description": "Record_ID must be unique.",
        "recommended_fix": "Deduplicate records based on Record_ID."
    },
    {
        "rule_id": "R003",
        "rule_name": "Beneficiary and Provider Completeness",
        "dimension": "Completeness",
        "severity": "High",
        "applicable_record_types": ["MEDICAL_CLAIM", "PHARMACY_CLAIM", "PRIOR_AUTH"],
        "fields": ["BENE_ID", "Provider_NPI"],
        "description": "BENE_ID and Provider_NPI must be present for every record type.",
        "recommended_fix": "Ensure patient and provider contexts are fully mapped."
    },
    {
        "rule_id": "R004",
        "rule_name": "Provider_NPI Validity",
        "dimension": "Validity",
        "severity": "High",
        "applicable_record_types": ["MEDICAL_CLAIM", "PHARMACY_CLAIM", "PRIOR_AUTH"],
        "fields": ["Provider_NPI"],
        "description": "Provider_NPI must contain exactly 10 digits when present.",
        "recommended_fix": "Validate NPI format against the NPPES standard."
    },
    {
        "rule_id": "R005",
        "rule_name": "Record_ID Prefix Consistency",
        "dimension": "Consistency",
        "severity": "High",
        "applicable_record_types": ["MEDICAL_CLAIM", "PHARMACY_CLAIM", "PRIOR_AUTH"],
        "fields": ["Record_ID", "Record_Type"],
        "description": "Record_ID prefix must match Record_Type (MC, PH, PA).",
        "recommended_fix": "Check for ID generation errors or mismatched Record_Type assignments."
    },
    {
        "rule_id": "R006",
        "rule_name": "Source System Consistency",
        "dimension": "Consistency",
        "severity": "High",
        "applicable_record_types": ["MEDICAL_CLAIM", "PHARMACY_CLAIM", "PRIOR_AUTH"],
        "fields": ["Record_Type", "Source_System"],
        "description": "Record_Type must map to the correct Source_System.",
        "recommended_fix": "Correct source system mapping tables in the ETL logic."
    },
    {
        "rule_id": "R007",
        "rule_name": "Medical Claim Core Fields Completeness",
        "dimension": "Completeness",
        "severity": "High",
        "applicable_record_types": ["MEDICAL_CLAIM"],
        "fields": ["Service_Date", "Service_End_Date", "Diagnosis_Code", "Billed_Amount", "Allowed_Amount", "Paid_Amount", "Patient_Responsibility"],
        "description": "MEDICAL_CLAIM requires specific dates, codes, and financial amounts.",
        "recommended_fix": "Verify all required medical claim fields are extracted from CARRIER_CLAIMS_SYS."
    },
    {
        "rule_id": "R008",
        "rule_name": "Pharmacy Claim Core Fields Completeness",
        "dimension": "Completeness",
        "severity": "High",
        "applicable_record_types": ["PHARMACY_CLAIM"],
        "fields": ["Service_Date", "Service_End_Date", "NDC_Code", "Drug_Name", "Days_Supply", "Quantity_Dispensed", "Billed_Amount", "Allowed_Amount", "Paid_Amount", "Patient_Responsibility"],
        "description": "PHARMACY_CLAIM requires specific dates, drug details, and financial amounts.",
        "recommended_fix": "Verify all required pharmacy claim fields are extracted from PHARMACY_ADJ_SYS."
    },
    {
        "rule_id": "R009",
        "rule_name": "Prior Auth Core Fields Completeness",
        "dimension": "Completeness",
        "severity": "High",
        "applicable_record_types": ["PRIOR_AUTH"],
        "fields": ["Procedure_Code", "Urgency_Flag", "Processed_Date", "Decision_Date", "Status"],
        "description": "PRIOR_AUTH requires codes and urgency, plus dates if APPROVED or DENIED.",
        "recommended_fix": "Ensure decision dates are captured when prior authorizations leave the PENDING state."
    },
    {
        "rule_id": "R010",
        "rule_name": "Financial Amount Non-Negative Validity",
        "dimension": "Validity",
        "severity": "High",
        "applicable_record_types": ["MEDICAL_CLAIM", "PHARMACY_CLAIM"],
        "fields": ["Billed_Amount", "Allowed_Amount", "Paid_Amount", "Patient_Responsibility"],
        "description": "Financial amounts for claims must not be negative.",
        "recommended_fix": "Review adjustment or reversal logic to ensure final line amounts are non-negative."
    },
    {
        "rule_id": "R011",
        "rule_name": "Pharmacy Supply Validity",
        "dimension": "Validity",
        "severity": "High",
        "applicable_record_types": ["PHARMACY_CLAIM"],
        "fields": ["Days_Supply", "Quantity_Dispensed"],
        "description": "Days_Supply and Quantity_Dispensed must be greater than zero.",
        "recommended_fix": "Investigate pharmacy dispensing data for zero values."
    },
    {
        "rule_id": "R012",
        "rule_name": "Status Validity by Record Type",
        "dimension": "Validity",
        "severity": "High",
        "applicable_record_types": ["MEDICAL_CLAIM", "PHARMACY_CLAIM", "PRIOR_AUTH"],
        "fields": ["Record_Type", "Status"],
        "description": "Status must be a valid value for the given Record_Type.",
        "recommended_fix": "Update allowed value lists or correct status mapping during data ingestion."
    },
    {
        "rule_id": "R013",
        "rule_name": "Denial Reason Completeness",
        "dimension": "Completeness",
        "severity": "Medium",
        "applicable_record_types": ["MEDICAL_CLAIM", "PHARMACY_CLAIM", "PRIOR_AUTH"],
        "fields": ["Status", "Denial_Reason_Code"],
        "description": "DENIED or REJECTED records must have a Denial_Reason_Code.",
        "recommended_fix": "Ensure denial codes are populated whenever a claim or auth is denied."
    },
    {
        "rule_id": "R014",
        "rule_name": "Service Dates Consistency",
        "dimension": "Consistency",
        "severity": "High",
        "applicable_record_types": ["MEDICAL_CLAIM", "PHARMACY_CLAIM"],
        "fields": ["Service_Date", "Service_End_Date"],
        "description": "Service_End_Date must be on or after Service_Date.",
        "recommended_fix": "Fix date entry errors where end date precedes start date."
    },
    {
        "rule_id": "R015",
        "rule_name": "Submission vs Service Date Consistency",
        "dimension": "Consistency",
        "severity": "High",
        "applicable_record_types": ["MEDICAL_CLAIM", "PHARMACY_CLAIM", "PRIOR_AUTH"],
        "fields": ["Service_Date", "Submission_Date"],
        "description": "Submission_Date must not be before Service_Date.",
        "recommended_fix": "Investigate time zone issues or data entry errors causing submissions prior to service."
    },
    {
        "rule_id": "R016",
        "rule_name": "Processed vs Submission Date Consistency",
        "dimension": "Consistency",
        "severity": "Critical",
        "applicable_record_types": ["MEDICAL_CLAIM", "PHARMACY_CLAIM", "PRIOR_AUTH"],
        "fields": ["Submission_Date", "Processed_Date"],
        "description": "Processed_Date must not be before Submission_Date.",
        "recommended_fix": "Investigate system clock sync or ETL latency causing processed date anomalies."
    },
    {
        "rule_id": "R017",
        "rule_name": "Auth Link Consistency",
        "dimension": "Consistency",
        "severity": "Critical",
        "applicable_record_types": ["MEDICAL_CLAIM", "PHARMACY_CLAIM"],
        "fields": ["Auth_Required_Flag", "Auth_Linked_ID"],
        "description": "If Auth_Required_Flag is Y, Auth_Linked_ID must exist.",
        "recommended_fix": "Ensure auth numbers are carried over into claims processing systems."
    },
    {
        "rule_id": "R018",
        "rule_name": "Auth Link Referential Integrity",
        "dimension": "Consistency",
        "severity": "High",
        "applicable_record_types": ["MEDICAL_CLAIM", "PHARMACY_CLAIM"],
        "fields": ["Auth_Linked_ID"],
        "description": "If Auth_Linked_ID exists, it must match an existing PRIOR_AUTH record.",
        "recommended_fix": "Check for orphaned claims where the prior authorization is missing from the dataset."
    },
    {
        "rule_id": "R019",
        "rule_name": "SLA Breach Flag Timeliness",
        "dimension": "Timeliness",
        "severity": "High",
        "applicable_record_types": ["MEDICAL_CLAIM", "PHARMACY_CLAIM", "PRIOR_AUTH"],
        "fields": ["Processed_Date", "SLA_Breach_Flag", "Processing_Latency_Days", "SLA_Target_Days"],
        "description": "SLA_Breach_Flag must accurately reflect latency vs target.",
        "recommended_fix": "Fix SLA flag calculation logic in the data warehouse."
    },
    {
        "rule_id": "R020",
        "rule_name": "Volume Trend Consistency",
        "dimension": "Consistency",
        "severity": "Medium",
        "applicable_record_types": ["MEDICAL_CLAIM", "PHARMACY_CLAIM", "PRIOR_AUTH"],
        "fields": ["Batch_Volume", "Rolling_7D_Avg_Volume", "Volume_Vs_Trend_Ratio"],
        "description": "Volume_Vs_Trend_Ratio must equal Batch_Volume / Rolling_7D_Avg_Volume within 0.005 margin.",
        "recommended_fix": "Verify the calculation script for batch volume aggregates."
    },
    {
        "rule_id": "R021",
        "rule_name": "SLA Breach Rate Trend Consistency",
        "dimension": "Consistency",
        "severity": "Medium",
        "applicable_record_types": ["MEDICAL_CLAIM", "PHARMACY_CLAIM", "PRIOR_AUTH"],
        "fields": ["Batch_SLA_Breach_Rate", "Rolling_7D_Avg_SLA_Breach_Rate", "SLA_Breach_Rate_Vs_Trend_Diff"],
        "description": "SLA_Breach_Rate_Vs_Trend_Diff must accurately reflect the difference within 0.0001.",
        "recommended_fix": "Verify the calculation script for SLA breach rate aggregates."
    }
]

# ---------------------------------------------------------
# PROFILING ENGINE
# ---------------------------------------------------------
def generate_profile(df):
    profile = {
        "total_records": len(df),
        "total_columns": len(df.columns),
        "exact_duplicate_rows": int(df.duplicated().sum()),
        "duplicate_record_ids": int(df["Record_ID"].duplicated().sum()) if "Record_ID" in df.columns else 0,
        "columns": {}
    }
    
    for col in df.columns:
        col_data = df[col]
        missing_count = int(col_data.isnull().sum())
        missing_pct = float(missing_count / len(df) * 100) if len(df) > 0 else 0.0
        
        col_profile = {
            "data_type": str(col_data.dtype),
            "missing_count": missing_count,
            "missing_percentage": missing_pct,
            "unique_count": int(col_data.nunique(dropna=True))
        }
        
        if pd.api.types.is_numeric_dtype(col_data):
            col_profile["min"] = float(col_data.min()) if pd.notnull(col_data.min()) else None
            col_profile["max"] = float(col_data.max()) if pd.notnull(col_data.max()) else None
            col_profile["mean"] = float(col_data.mean()) if pd.notnull(col_data.mean()) else None
            
        profile["columns"][col] = col_profile
        
    return profile

# ---------------------------------------------------------
# QUALITY ENGINE
# ---------------------------------------------------------
def run_quality_checks(df, rules):
    results = []
    
    for rule in rules:
        rule_id = rule["rule_id"]
        applicable_types = rule["applicable_record_types"]
        mask = df["Record_Type"].isin(applicable_types)
        applicable_df = df[mask]
        
        total_applicable = len(applicable_df)
        if total_applicable == 0:
            continue
            
        failed_mask = pd.Series(False, index=applicable_df.index)
        
        if rule_id == "R001":
            failed_mask = applicable_df["Record_ID"].isnull() | (applicable_df["Record_ID"] == "")
        elif rule_id == "R002":
            failed_mask = applicable_df.duplicated(subset=["Record_ID"], keep=False)
        elif rule_id == "R003":
            failed_mask = applicable_df["BENE_ID"].isnull() | applicable_df["Provider_NPI"].isnull() | (applicable_df["BENE_ID"] == "") | (applicable_df["Provider_NPI"] == "")
        elif rule_id == "R004":
            # Provider_NPI must not be null. After removing trailing .0, it must contain exactly 10 digits.
            npi_str = applicable_df["Provider_NPI"].astype(str).str.replace(r'\.0$', '', regex=True)
            failed_mask = applicable_df["Provider_NPI"].isnull() | (applicable_df["Provider_NPI"] == "") | (npi_str.str.len() != 10) | ~npi_str.str.isdigit()
        elif rule_id == "R005":
            mc_fail = (applicable_df["Record_Type"] == "MEDICAL_CLAIM") & ~applicable_df["Record_ID"].astype(str).str.startswith("MC")
            ph_fail = (applicable_df["Record_Type"] == "PHARMACY_CLAIM") & ~applicable_df["Record_ID"].astype(str).str.startswith("PH")
            pa_fail = (applicable_df["Record_Type"] == "PRIOR_AUTH") & ~applicable_df["Record_ID"].astype(str).str.startswith("PA")
            failed_mask = mc_fail | ph_fail | pa_fail
        elif rule_id == "R006":
            mc_fail = (applicable_df["Record_Type"] == "MEDICAL_CLAIM") & (applicable_df["Source_System"] != "CARRIER_CLAIMS_SYS")
            ph_fail = (applicable_df["Record_Type"] == "PHARMACY_CLAIM") & (applicable_df["Source_System"] != "PHARMACY_ADJ_SYS")
            pa_fail = (applicable_df["Record_Type"] == "PRIOR_AUTH") & (applicable_df["Source_System"] != "AUTH_MGMT_SYS")
            failed_mask = mc_fail | ph_fail | pa_fail
        elif rule_id == "R007":
            cols = ["Service_Date", "Service_End_Date", "Diagnosis_Code", "Billed_Amount", "Allowed_Amount", "Paid_Amount", "Patient_Responsibility"]
            failed_mask = applicable_df[cols].isnull().any(axis=1)
        elif rule_id == "R008":
            cols = ["Service_Date", "Service_End_Date", "NDC_Code", "Drug_Name", "Days_Supply", "Quantity_Dispensed", "Billed_Amount", "Allowed_Amount", "Paid_Amount", "Patient_Responsibility"]
            failed_mask = applicable_df[cols].isnull().any(axis=1)
        elif rule_id == "R009":
            missing_base = applicable_df[["Procedure_Code", "Urgency_Flag"]].isnull().any(axis=1)
            needs_dates = applicable_df["Status"].isin(["APPROVED", "DENIED"])
            missing_dates = needs_dates & applicable_df[["Processed_Date", "Decision_Date"]].isnull().any(axis=1)
            failed_mask = missing_base | missing_dates
        elif rule_id == "R010":
            cols = ["Billed_Amount", "Allowed_Amount", "Paid_Amount", "Patient_Responsibility"]
            failed_mask = (applicable_df[cols] < 0).any(axis=1)
        elif rule_id == "R011":
            cols = ["Days_Supply", "Quantity_Dispensed"]
            failed_mask = (applicable_df[cols] <= 0).any(axis=1)
        elif rule_id == "R012":
            mc_fail = (applicable_df["Record_Type"] == "MEDICAL_CLAIM") & ~applicable_df["Status"].isin(["PAID", "DENIED", "PENDING"])
            ph_fail = (applicable_df["Record_Type"] == "PHARMACY_CLAIM") & ~applicable_df["Status"].isin(["PAID", "REJECTED", "PENDING"])
            pa_fail = (applicable_df["Record_Type"] == "PRIOR_AUTH") & ~applicable_df["Status"].isin(["APPROVED", "DENIED", "PENDING"])
            failed_mask = mc_fail | ph_fail | pa_fail
        elif rule_id == "R013":
            needs_reason = applicable_df["Status"].isin(["DENIED", "REJECTED"])
            missing_reason = applicable_df["Denial_Reason_Code"].isnull() | (applicable_df["Denial_Reason_Code"] == "")
            failed_mask = needs_reason & missing_reason
        elif rule_id == "R014":
            has_dates = applicable_df["Service_Date"].notnull() & applicable_df["Service_End_Date"].notnull()
            failed_mask = has_dates & (applicable_df["Service_End_Date"] < applicable_df["Service_Date"])
        elif rule_id == "R015":
            has_dates = applicable_df["Service_Date"].notnull() & applicable_df["Submission_Date"].notnull()
            failed_mask = has_dates & (applicable_df["Submission_Date"] < applicable_df["Service_Date"])
        elif rule_id == "R016":
            has_dates = applicable_df["Submission_Date"].notnull() & applicable_df["Processed_Date"].notnull()
            failed_mask = has_dates & (applicable_df["Processed_Date"] < applicable_df["Submission_Date"])
        elif rule_id == "R017":
            auth_req = applicable_df["Auth_Required_Flag"] == "Y"
            auth_missing = applicable_df["Auth_Linked_ID"].isnull() | (applicable_df["Auth_Linked_ID"] == "")
            failed_mask = auth_req & auth_missing
        elif rule_id == "R018":
            has_auth = applicable_df["Auth_Linked_ID"].notnull() & (applicable_df["Auth_Linked_ID"] != "")
            valid_auths = df[df["Record_Type"] == "PRIOR_AUTH"]["Record_ID"].unique()
            failed_mask = has_auth & ~applicable_df["Auth_Linked_ID"].isin(valid_auths)
        elif rule_id == "R019":
            has_processed = applicable_df["Processed_Date"].notnull()
            breach_expected = applicable_df["Processing_Latency_Days"] > applicable_df["SLA_Target_Days"]
            breach_actual = applicable_df["SLA_Breach_Flag"] == "Y"
            fail_has_processed = has_processed & (breach_expected != breach_actual)
            
            no_processed = applicable_df["Processed_Date"].isnull()
            fail_no_processed = no_processed & (applicable_df["SLA_Breach_Flag"] != "UNKNOWN_NO_DATE")
            
            failed_mask = fail_has_processed | fail_no_processed
        elif rule_id == "R020":
            diff = abs(applicable_df["Batch_Volume"] / applicable_df["Rolling_7D_Avg_Volume"] - applicable_df["Volume_Vs_Trend_Ratio"])
            failed_mask = diff > 0.005
        elif rule_id == "R021":
            diff = abs((applicable_df["Batch_SLA_Breach_Rate"] - applicable_df["Rolling_7D_Avg_SLA_Breach_Rate"]) - applicable_df["SLA_Breach_Rate_Vs_Trend_Diff"])
            failed_mask = diff > 0.0001
            
        affected_count = int(failed_mask.sum())
        failure_rate = (affected_count / total_applicable) * 100 if total_applicable > 0 else 0
        
        status = "PASSED" if affected_count == 0 else "FAILED"
        sample_ids = applicable_df[failed_mask]["Record_ID"].head(10).tolist() if "Record_ID" in applicable_df.columns else []
        
        results.append({
            "rule_id": rule_id,
            "rule_name": rule["rule_name"],
            "dimension": rule["dimension"],
            "severity": rule["severity"],
            "status": status,
            "total_applicable_records": total_applicable,
            "affected_records": affected_count,
            "failure_rate_pct": failure_rate,
            "sample_record_ids": sample_ids,
            "fields": rule["fields"],
            "message": f"{affected_count} records failed the rule." if affected_count > 0 else "All records passed.",
            "description": rule["description"],
            "recommended_fix": rule["recommended_fix"]
        })
        
    return results

# ---------------------------------------------------------
# SCORING & REPORTING ENGINE
# ---------------------------------------------------------
def calculate_scores_and_risk(rule_results, df):
    dimensions = ["Completeness", "Validity", "Uniqueness", "Consistency", "Timeliness"]
    dim_scores = {dim: 100.0 for dim in dimensions}
    dim_rates = {dim: [] for dim in dimensions}
    critical_failures = 0
    top_failed_rules = []
    
    for res in rule_results:
        dim = res["dimension"]
        if dim not in dim_rates:
            dim = "Consistency"
        dim_rates[dim].append(res["failure_rate_pct"])
        
        if res["severity"] == "Critical" and res["status"] == "FAILED":
            critical_failures += 1
            
        if res["status"] == "FAILED":
            top_failed_rules.append(res)
            
    for dim, rates in dim_rates.items():
        if rates:
            avg_rate = sum(rates) / len(rates)
            dim_scores[dim] = max(0.0, 100.0 - avg_rate)
            
    overall_score = (
        0.25 * dim_scores.get("Completeness", 100) +
        0.25 * dim_scores.get("Validity", 100) +
        0.20 * dim_scores.get("Uniqueness", 100) +
        0.20 * dim_scores.get("Consistency", 100) +
        0.10 * dim_scores.get("Timeliness", 100)
    )
    
    if overall_score >= 90 and critical_failures == 0:
        overall_risk = "LOW"
    elif overall_score >= 75:
        overall_risk = "MEDIUM"
        if overall_score >= 90 and critical_failures > 0:
            overall_risk = "MEDIUM"
    elif overall_score >= 50:
        overall_risk = "HIGH"
    else:
        overall_risk = "CRITICAL"
        
    top_failed_rules.sort(key=lambda x: x["failure_rate_pct"], reverse=True)
    
    if len(df) > 0:
        batch_id = df["Batch_ID"].iloc[0] if "Batch_ID" in df.columns else "UNKNOWN_BATCH"
        batch_volume = len(df)
        breach_count = len(df[df["SLA_Breach_Flag"] == "Y"]) if "SLA_Breach_Flag" in df.columns else 0
        sla_breach_rate = breach_count / batch_volume if batch_volume > 0 else 0
        retry_count = int(df["Retry_Count"].sum()) if "Retry_Count" in df.columns else 0
        pipeline_gap_count = int((df["Pipeline_Gap_Flag"] == "Y").sum()) if "Pipeline_Gap_Flag" in df.columns else 0
        rolling_vol = float(df["Rolling_7D_Avg_Volume"].iloc[0]) if "Rolling_7D_Avg_Volume" in df.columns else 0.0
        vol_ratio = float(df["Volume_Vs_Trend_Ratio"].iloc[0]) if "Volume_Vs_Trend_Ratio" in df.columns else 0.0
        rolling_sla = float(df["Rolling_7D_Avg_SLA_Breach_Rate"].iloc[0]) if "Rolling_7D_Avg_SLA_Breach_Rate" in df.columns else 0.0
        sla_diff = float(df["SLA_Breach_Rate_Vs_Trend_Diff"].iloc[0]) if "SLA_Breach_Rate_Vs_Trend_Diff" in df.columns else 0.0
    else:
        batch_id, batch_volume, breach_count, sla_breach_rate = "UNKNOWN_BATCH", 0, 0, 0
        retry_count, pipeline_gap_count = 0, 0
        rolling_vol, vol_ratio, rolling_sla, sla_diff = 0.0, 0.0, 0.0, 0.0
        
    sla_breach_rate_pct = sla_breach_rate * 100
    
    if sla_breach_rate_pct < 5:
        sla_risk = "LOW"
    elif sla_breach_rate_pct < 15:
        sla_risk = "MEDIUM"
    elif sla_breach_rate_pct < 30:
        sla_risk = "HIGH"
    else:
        sla_risk = "CRITICAL"
        
    anomaly_signals = {
        "Batch_ID": batch_id,
        "Batch_Volume": batch_volume,
        "Rolling_7D_Avg_Volume": rolling_vol,
        "Volume_Vs_Trend_Ratio": vol_ratio,
        "Batch_SLA_Breach_Rate": sla_breach_rate,
        "Rolling_7D_Avg_SLA_Breach_Rate": rolling_sla,
        "SLA_Breach_Rate_Vs_Trend_Diff": sla_diff,
        "Retry_Count": retry_count,
        "Pipeline_Gap_Flag": pipeline_gap_count,
        "quality_score": overall_score,
        "critical_issue_count": critical_failures
    }
    
    quality_report = {
        "run_timestamp": datetime.datetime.now().isoformat(),
        "records_scanned": batch_volume,
        "dimension_scores": dim_scores,
        "overall_quality_score": overall_score,
        "overall_risk_level": overall_risk,
        "critical_issue_count": critical_failures,
        "all_rule_results": rule_results,
        "top_failed_rules": top_failed_rules[:5],
        "anomaly_signals": anomaly_signals,
        "batch_sla_risk_level": sla_risk,
        "sla_breach_rate": sla_breach_rate
    }
    
    return quality_report

# ---------------------------------------------------------
# FORMATTED OUTPUT
# ---------------------------------------------------------
def print_and_save_report(profile, report, output_txt_path):
    lines = []
    lines.append("="*60)
    lines.append(" UC10 DATA QUALITY MONITOR - TERMINAL REPORT ")
    lines.append("="*60)
    
    # 1. Profiling Baseline
    lines.append("\n--- 1. PROFILING BASELINE ---")
    lines.append(f"Total Records Scanned: {profile['total_records']}")
    lines.append(f"Total Columns: {profile['total_columns']}")
    lines.append(f"Exact Duplicate Rows: {profile['exact_duplicate_rows']}")
    lines.append(f"Duplicate Record IDs: {profile['duplicate_record_ids']}")
    
    # 2. Quality Dimensions & Overall Score
    lines.append("\n--- 2. QUALITY SCORES ---")
    lines.append(f"Overall Quality Score: {report['overall_quality_score']:.2f} / 100")
    lines.append(f"Overall Risk Level: {report['overall_risk_level']}")
    lines.append(f"SLA Risk Level: {report['batch_sla_risk_level']} (Breach Rate: {report['sla_breach_rate']*100:.2f}%)")
    lines.append("\nDimension Scores:")
    for dim, score in report["dimension_scores"].items():
        lines.append(f"  - {dim}: {score:.2f}%")
        
    # 3. Rule Execution Table
    lines.append("\n--- 3. RULE EXECUTION SUMMARY ---")
    lines.append(f"{'RULE ID':<8} | {'SEVERITY':<10} | {'STATUS':<8} | {'AFFECTED':<10} | {'FAIL RATE':<10} | {'RULE NAME'}")
    lines.append("-" * 90)
    for res in report["all_rule_results"]:
        lines.append(f"{res['rule_id']:<8} | {res['severity']:<10} | {res['status']:<8} | {res['affected_records']:<10} | {res['failure_rate_pct']:<9.2f}% | {res['rule_name']}")
        
    # 4. Top Failed Rules Details
    lines.append("\n--- 4. TOP FAILED RULES DETAILS ---")
    if not report["top_failed_rules"]:
        lines.append("No rules failed! Perfect data quality.")
    else:
        for issue in report["top_failed_rules"]:
            lines.append(f"\n[ {issue['rule_id']} ] {issue['rule_name']} ({issue['severity']})")
            lines.append(f"  Description      : {issue['description']}")
            lines.append(f"  Affected Records : {issue['affected_records']} ({issue['failure_rate_pct']:.2f}%)")
            lines.append(f"  Sample IDs       : {', '.join([str(x) for x in issue['sample_record_ids']])}")
            lines.append(f"  Recommended Fix  : {issue['recommended_fix']}")
            
    # 5. Anomaly Signals
    lines.append("\n--- 5. ANOMALY SIGNALS (HANDOFF) ---")
    for key, val in report["anomaly_signals"].items():
        if isinstance(val, float):
            lines.append(f"  {key}: {val:.4f}")
        else:
            lines.append(f"  {key}: {val}")
            
    lines.append("\n" + "="*60 + "\n")
    
    full_report_text = "\n".join(lines)
    print(full_report_text)
    
    with open(output_txt_path, "w") as f:
        f.write(full_report_text)

# ---------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------
def main():
    data_path = "data/claims_pharmacy_auth_monitor_dataset_features.csv"
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}")
        sys.exit(1)
        
    print(f"[*] Loading data from {data_path} without modification...")
    
    dtype_spec = {
        "Record_ID": str,
        "BENE_ID": str,
        "Provider_NPI": str,
        "Auth_Linked_ID": str
    }
    
    df = pd.read_csv(data_path, dtype=dtype_spec)
    
    date_columns = ["Service_Date", "Service_End_Date", "Processed_Date", "Decision_Date", "Submission_Date"]
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            
    print("[*] Profiling rows, columns, data types, missing values...")
    profile = generate_profile(df)
    
    print("[*] Running UC10 quality rules...")
    rule_results = run_quality_checks(df, RULES)
    
    print("[*] Calculating scores, risk levels, and anomaly signals...")
    quality_report = calculate_scores_and_risk(rule_results, df)
    
    print("[*] Saving full JSON result...")
    with open(f"{output_dir}/quality_report.json", "w") as f:
        json.dump(quality_report, f, indent=4)
        
    print("[*] Generating formatted terminal report...")
    print_and_save_report(profile, quality_report, f"{output_dir}/quality_check_summary.txt")

if __name__ == "__main__":
    main()
