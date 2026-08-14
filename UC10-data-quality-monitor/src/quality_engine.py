import pandas as pd
import numpy as np

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
            failed_mask = applicable_df["Provider_NPI"].notnull() & (applicable_df["Provider_NPI"] != "") & (applicable_df["Provider_NPI"].astype(str).str.len() != 10)
            
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
