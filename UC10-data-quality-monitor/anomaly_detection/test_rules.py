import pandas as pd
from anomaly_detection.rule_config import RULES_CONFIG
from anomaly_detection.medical_rules import evaluate_medical_rules
from anomaly_detection.pharmacy_rules import evaluate_pharmacy_rules
from anomaly_detection.authorization_rules import evaluate_authorization_rules


def test_medical_rules():
    print("Testing Medical Rules with synthetic cases...")
    
    # 1. Valid Normal Record
    r_valid = {
        "Record_ID": "MC_VALID", "Record_Type": "MEDICAL_CLAIM",
        "Billed_Amount": 100.0, "Allowed_Amount": 80.0, "Paid_Amount": 80.0, "Patient_Responsibility": 0.0,
        "Service_Date": "2025-01-01", "Service_End_Date": "2025-01-02", "Submission_Date": "2025-01-03",
        "Processed_Date": "2025-01-04", "Decision_Date": "2025-01-04",
        "Processing_Latency_Days": 1.0, "SLA_Target_Days": 30.0, "Retry_Count": 0, "Status": "PAID",
        "SLA_Breach_Flag": "N", "BENE_ID": "B1", "Provider_NPI": "1234567890"
    }
    
    # 2. Negative Billed (R_FIN_001)
    r_neg_billed = r_valid.copy()
    r_neg_billed.update({"Record_ID": "MC_NEG_BILLED", "Billed_Amount": -10.0})
    
    # 3. Paid > Allowed (R_FIN_005)
    r_paid_gt_allow = r_valid.copy()
    r_paid_gt_allow.update({"Record_ID": "MC_PAID_GT_ALLOW", "Paid_Amount": 90.0, "Allowed_Amount": 80.0})
    
    # 4. Allowed > Billed (R_FIN_006)
    r_allow_gt_bill = r_valid.copy()
    r_allow_gt_bill.update({"Record_ID": "MC_ALLOW_GT_BILL", "Allowed_Amount": 120.0, "Billed_Amount": 100.0})
    
    # 5. Invalid date sequence: Service_End_Date < Service_Date (R_DATE_002)
    r_inv_date = r_valid.copy()
    r_inv_date.update({"Record_ID": "MC_INV_DATE", "Service_Date": "2025-01-05", "Service_End_Date": "2025-01-01"})
    
    # 6. Negative latency (R_DATE_001)
    r_neg_latency = r_valid.copy()
    r_neg_latency.update({"Record_ID": "MC_NEG_LAT", "Processing_Latency_Days": -2.0})
    
    # 7. Processed before submission (R_DATE_001)
    r_proc_before_sub = r_valid.copy()
    r_proc_before_sub.update({
        "Record_ID": "MC_PROC_SUB", "Processed_Date": "2025-01-01", "Submission_Date": "2025-01-02", "Processing_Latency_Days": -1.0
    })
    
    # 8. Negative retry (R_TECH_001)
    r_neg_retry = r_valid.copy()
    r_neg_retry.update({"Record_ID": "MC_NEG_RETRY", "Retry_Count": -1})
    
    # 9. SLA breach (R_SLA_001)
    r_sla_breach = r_valid.copy()
    r_sla_breach.update({"Record_ID": "MC_SLA", "Processing_Latency_Days": 35.0, "SLA_Target_Days": 30.0})
    
    # 10. Pending claim without Processed_Date (Valid, should not breach SLA or trigger date mismatch)
    r_pending = r_valid.copy()
    r_pending.update({
        "Record_ID": "MC_PENDING", "Status": "PENDING", "Processed_Date": None, "Processing_Latency_Days": None
    })
    
    df = pd.DataFrame([
        r_valid, r_neg_billed, r_paid_gt_allow, r_allow_gt_bill,
        r_inv_date, r_neg_latency, r_proc_before_sub, r_neg_retry, r_sla_breach, r_pending
    ])
    
    anoms = evaluate_medical_rules(df, RULES_CONFIG)
    anom_ids = [a["Record_ID"] for a in anoms]
    rules_triggered = [a["Rule_ID"] for a in anoms]
    
    print(f"Triggered anomaly IDs: {anom_ids}")
    print(f"Triggered rule IDs: {rules_triggered}")
    
    # Asserts
    assert "MC_VALID" not in anom_ids, "Error: Valid record was flagged!"
    assert "MC_PENDING" not in anom_ids, "Error: Valid pending record was flagged!"
    
    assert "MC_NEG_BILLED" in anom_ids
    assert "R_FIN_001" in rules_triggered
    
    assert "MC_PAID_GT_ALLOW" in anom_ids
    assert "R_FIN_005" in rules_triggered
    
    assert "MC_ALLOW_GT_BILL" in anom_ids
    assert "R_FIN_006" in rules_triggered
    
    assert "MC_INV_DATE" in anom_ids
    assert "R_DATE_002" in rules_triggered
    
    # Negative latency must be R_DATE_001
    assert "MC_NEG_LAT" in anom_ids
    assert "MC_PROC_SUB" in anom_ids
    assert "R_DATE_001" in rules_triggered
    
    # Verify deduplication for MC_PROC_SUB: it should only have ONE anomaly event
    proc_sub_anoms = [a for a in anoms if a["Record_ID"] == "MC_PROC_SUB"]
    assert len(proc_sub_anoms) == 1, f"Error: Timeline anomaly not deduplicated! Expected 1, got {len(proc_sub_anoms)}"
    
    assert "MC_NEG_RETRY" in anom_ids
    assert "R_TECH_001" in rules_triggered
    
    assert "MC_SLA" in anom_ids
    assert "R_SLA_001" in rules_triggered
    
    print("Success: All Medical Rules tests passed successfully!\n")


def test_pharmacy_rules():
    print("Testing Pharmacy Rules with synthetic cases...")
    
    r_valid = {
        "Record_ID": "PH_VALID", "Record_Type": "PHARMACY_CLAIM",
        "Billed_Amount": 50.0, "Allowed_Amount": 40.0, "Paid_Amount": 40.0, "Patient_Responsibility": 0.0,
        "Days_Supply": 30.0, "Quantity_Dispensed": 45.0,
        "Service_Date": "2025-01-01", "Service_End_Date": "2025-01-01", "Submission_Date": "2025-01-02",
        "Processed_Date": "2025-01-03", "Decision_Date": "2025-01-03",
        "Processing_Latency_Days": 1.0, "SLA_Target_Days": 30.0, "Retry_Count": 0, "Status": "PAID",
        "SLA_Breach_Flag": "N", "BENE_ID": "B1", "Provider_NPI": "1234567890"
    }
    
    # Negative quantity
    r_neg_qty = r_valid.copy()
    r_neg_qty.update({"Record_ID": "PH_NEG_QTY", "Quantity_Dispensed": -5.0})
    
    # Days supply <= 0
    r_zero_days = r_valid.copy()
    r_zero_days.update({"Record_ID": "PH_ZERO_DAYS", "Days_Supply": 0.0})
    
    df = pd.DataFrame([r_valid, r_neg_qty, r_zero_days])
    
    anoms = evaluate_pharmacy_rules(df, RULES_CONFIG)
    anom_ids = [a["Record_ID"] for a in anoms]
    
    assert "PH_VALID" not in anom_ids
    assert "PH_NEG_QTY" in anom_ids
    assert "PH_ZERO_DAYS" in anom_ids
    
    print("Success: All Pharmacy Rules tests passed successfully!\n")


def test_authorization_rules():
    print("Testing Authorization Rules with synthetic cases...")
    
    r_valid = {
        "Record_ID": "PA_VALID", "Record_Type": "PRIOR_AUTH",
        "Retry_Count": 0, "Processing_Latency_Days": 2.0, "SLA_Target_Days": 14.0,
        "Submission_Date": "2025-01-01", "Processed_Date": "2025-01-03", "Decision_Date": "2025-01-03",
        "Status": "APPROVED", "SLA_Breach_Flag": "N", "BENE_ID": "B1", "Provider_NPI": "1234567890"
    }
    
    # Decision date before submission
    r_dec_before_sub = r_valid.copy()
    r_dec_before_sub.update({"Record_ID": "PA_DEC_SUB", "Decision_Date": "2024-12-30"})
    
    df = pd.DataFrame([r_valid, r_dec_before_sub])
    
    anoms = evaluate_authorization_rules(df, RULES_CONFIG)
    anom_ids = [a["Record_ID"] for a in anoms]
    
    assert "PA_VALID" not in anom_ids
    assert "PA_DEC_SUB" in anom_ids
    
    print("Success: All Authorization Rules tests passed successfully!\n")


if __name__ == "__main__":
    test_medical_rules()
    test_pharmacy_rules()
    test_authorization_rules()
    print("All synthetic test suites completed successfully!")
