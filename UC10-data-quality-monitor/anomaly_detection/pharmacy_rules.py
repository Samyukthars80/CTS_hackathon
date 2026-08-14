from __future__ import annotations

import pandas as pd
from anomaly_detection.anomaly_utils import create_anomaly_record


def evaluate_pharmacy_rules(df: pd.DataFrame, config: dict) -> list[dict]:
    anomalies = []
    source = "pharmacy_claim_cleaned.csv"
    
    # Date Conversions
    date_cols = ["Service_Date", "Service_End_Date", "Submission_Date", "Processed_Date", "Decision_Date"]
    for col in date_cols:
        if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col], errors='coerce')
            
    # Helper to log violations
    def add_anomalies_from_mask(mask, rule_id, affected_cols, observed_expr, expected, explanation_func):
        if rule_id not in config or not config[rule_id].get("enabled", True):
            return
        
        violators = df[mask]
        for _, row in violators.iterrows():
            record_id = str(row.get("Record_ID", "UNKNOWN"))
            rec_type = str(row.get("Record_Type", "PHARMACY_CLAIM"))
            observed = observed_expr(row)
            explanation = explanation_func(row)
            rule_info = config[rule_id]
            
            anomalies.append(create_anomaly_record(
                record_id=record_id,
                record_type=rec_type,
                rule_id=rule_id,
                category=rule_info["category"],
                affected_cols=affected_cols,
                observed=observed,
                expected=expected,
                severity=rule_info["severity"],
                explanation=explanation,
                source_dataset=source
            ))

    # R_FIN_001: Negative Billed Amount
    if "Billed_Amount" in df.columns:
        mask = df["Billed_Amount"] < 0
        add_anomalies_from_mask(
            mask, "R_FIN_001", "Billed_Amount",
            lambda r: f"Billed_Amount = {r['Billed_Amount']}",
            "Billed_Amount >= 0",
            lambda r: f"Billed amount (${r['Billed_Amount']:.2f}) is negative on this pharmacy claim."
        )

    # R_FIN_002: Negative Allowed Amount
    if "Allowed_Amount" in df.columns:
        mask = df["Allowed_Amount"] < 0
        add_anomalies_from_mask(
            mask, "R_FIN_002", "Allowed_Amount",
            lambda r: f"Allowed_Amount = {r['Allowed_Amount']}",
            "Allowed_Amount >= 0",
            lambda r: f"Allowed amount (${r['Allowed_Amount']:.2f}) is negative."
        )

    # R_FIN_003: Negative Paid Amount
    if "Paid_Amount" in df.columns:
        mask = df["Paid_Amount"] < 0
        add_anomalies_from_mask(
            mask, "R_FIN_003", "Paid_Amount",
            lambda r: f"Paid_Amount = {r['Paid_Amount']}",
            "Paid_Amount >= 0",
            lambda r: f"Paid amount (${r['Paid_Amount']:.2f}) is negative."
        )

    # R_FIN_004: Negative Patient Responsibility
    if "Patient_Responsibility" in df.columns:
        mask = df["Patient_Responsibility"] < 0
        add_anomalies_from_mask(
            mask, "R_FIN_004", "Patient_Responsibility",
            lambda r: f"Patient_Responsibility = {r['Patient_Responsibility']}",
            "Patient_Responsibility >= 0",
            lambda r: f"Patient responsibility (${r['Patient_Responsibility']:.2f}) is negative."
        )

    # R_FIN_005: Paid > Allowed
    if "Paid_Amount" in df.columns and "Allowed_Amount" in df.columns:
        mask = df["Paid_Amount"] > df["Allowed_Amount"]
        add_anomalies_from_mask(
            mask, "R_FIN_005", "Paid_Amount, Allowed_Amount",
            lambda r: f"Paid = {r['Paid_Amount']}, Allowed = {r['Allowed_Amount']}",
            "Paid_Amount <= Allowed_Amount",
            lambda r: f"Paid amount (${r['Paid_Amount']:.2f}) exceeds the allowed amount (${r['Allowed_Amount']:.2f}) on this pharmacy claim."
        )

    # R_FIN_006: Allowed > Billed
    if "Allowed_Amount" in df.columns and "Billed_Amount" in df.columns:
        mask = df["Allowed_Amount"] > df["Billed_Amount"]
        add_anomalies_from_mask(
            mask, "R_FIN_006", "Allowed_Amount, Billed_Amount",
            lambda r: f"Allowed = {r['Allowed_Amount']}, Billed = {r['Billed_Amount']}",
            "Allowed_Amount <= Billed_Amount",
            lambda r: f"Allowed amount (${r['Allowed_Amount']:.2f}) exceeds the billed amount (${r['Billed_Amount']:.2f})."
        )

    # R_FIN_007: Paid > Billed
    if "Paid_Amount" in df.columns and "Billed_Amount" in df.columns:
        mask = df["Paid_Amount"] > df["Billed_Amount"]
        add_anomalies_from_mask(
            mask, "R_FIN_007", "Paid_Amount, Billed_Amount",
            lambda r: f"Paid = {r['Paid_Amount']}, Billed = {r['Billed_Amount']}",
            "Paid_Amount <= Billed_Amount",
            lambda r: f"Paid amount (${r['Paid_Amount']:.2f}) exceeds the billed amount (${r['Billed_Amount']:.2f})."
        )

    # R_DATE_001: Invalid Processing Timeline (Deduplicated)
    # Check if processed before submission OR latency is negative
    mask_date_order = df["Processed_Date"].notna() & df["Submission_Date"].notna() & (df["Processed_Date"] < df["Submission_Date"])
    mask_latency = df["Processing_Latency_Days"].notna() & (df["Processing_Latency_Days"] < 0)
    mask_timeline = mask_date_order | mask_latency
    
    def observed_timeline(row):
        obs = []
        if row["Processed_Date"] < row["Submission_Date"]:
            obs.append(f"Processed = {row['Processed_Date'].strftime('%Y-%m-%d')}, Submission = {row['Submission_Date'].strftime('%Y-%m-%d')}")
        if row["Processing_Latency_Days"] < 0:
            obs.append(f"Latency = {row['Processing_Latency_Days']}")
        return "; ".join(obs)
        
    def explanation_timeline(row):
        exp = []
        if row["Processed_Date"] < row["Submission_Date"]:
            exp.append(f"Pharmacy claim was processed ({row['Processed_Date'].strftime('%Y-%m-%d')}) before its submission date ({row['Submission_Date'].strftime('%Y-%m-%d')})")
        if row["Processing_Latency_Days"] < 0:
            exp.append(f"Processing latency is negative ({row['Processing_Latency_Days']} days)")
        return ". ".join(exp) + ". This indicates an invalid processing timeline."

    add_anomalies_from_mask(
        mask_timeline, "R_DATE_001", "Submission_Date, Processed_Date, Processing_Latency_Days",
        observed_timeline,
        "Processed_Date >= Submission_Date AND Processing_Latency_Days >= 0",
        explanation_timeline
    )

    # R_DATE_002: Service_Date > Service_End_Date
    if "Service_Date" in df.columns and "Service_End_Date" in df.columns:
        mask = df["Service_Date"].notna() & df["Service_End_Date"].notna() & (df["Service_Date"] > df["Service_End_Date"])
        add_anomalies_from_mask(
            mask, "R_DATE_002", "Service_Date, Service_End_Date",
            lambda r: f"Service = {r['Service_Date'].strftime('%Y-%m-%d')}, End = {r['Service_End_Date'].strftime('%Y-%m-%d')}",
            "Service_Date <= Service_End_Date",
            lambda r: f"Pharmacy service date ({r['Service_Date'].strftime('%Y-%m-%d')}) is after the service end date ({r['Service_End_Date'].strftime('%Y-%m-%d')})."
        )
        
    # R_DATE_003: Submission_Date < Service_Date
    if "Submission_Date" in df.columns and "Service_Date" in df.columns:
        mask = df["Submission_Date"].notna() & df["Service_Date"].notna() & (df["Submission_Date"] < df["Service_Date"])
        add_anomalies_from_mask(
            mask, "R_DATE_003", "Submission_Date, Service_Date",
            lambda r: f"Submission = {r['Submission_Date'].strftime('%Y-%m-%d')}, Service = {r['Service_Date'].strftime('%Y-%m-%d')}",
            "Submission_Date >= Service_Date",
            lambda r: f"Pharmacy claim submission date ({r['Submission_Date'].strftime('%Y-%m-%d')}) occurs before the service date ({r['Service_Date'].strftime('%Y-%m-%d')})."
        )

    # R_DATE_004: Decision_Date < Submission_Date
    if "Decision_Date" in df.columns and "Submission_Date" in df.columns:
        mask = df["Decision_Date"].notna() & df["Submission_Date"].notna() & (df["Decision_Date"] < df["Submission_Date"])
        add_anomalies_from_mask(
            mask, "R_DATE_004", "Decision_Date, Submission_Date",
            lambda r: f"Decision = {r['Decision_Date'].strftime('%Y-%m-%d')}, Submission = {r['Submission_Date'].strftime('%Y-%m-%d')}",
            "Decision_Date >= Submission_Date",
            lambda r: f"Pharmacy claim decision date ({r['Decision_Date'].strftime('%Y-%m-%d')}) occurs before the submission date ({r['Submission_Date'].strftime('%Y-%m-%d')})."
        )

    # R_MET_001: Days Supply <= 0
    if "Days_Supply" in df.columns:
        mask = df["Days_Supply"] <= 0
        add_anomalies_from_mask(
            mask, "R_MET_001", "Days_Supply",
            lambda r: f"Days_Supply = {r['Days_Supply']}",
            "Days_Supply > 0",
            lambda r: f"Pharmacy claim has an invalid days supply value of {r['Days_Supply']}. Days supply must be positive."
        )

    # R_MET_002: Quantity Dispensed <= 0
    if "Quantity_Dispensed" in df.columns:
        mask = df["Quantity_Dispensed"] <= 0
        add_anomalies_from_mask(
            mask, "R_MET_002", "Quantity_Dispensed",
            lambda r: f"Quantity_Dispensed = {r['Quantity_Dispensed']}",
            "Quantity_Dispensed > 0",
            lambda r: f"Pharmacy claim has an invalid dispensed quantity value of {r['Quantity_Dispensed']}. Quantity must be positive."
        )

    # R_TECH_001: Negative Retry
    if "Retry_Count" in df.columns:
        mask = df["Retry_Count"] < 0
        add_anomalies_from_mask(
            mask, "R_TECH_001", "Retry_Count",
            lambda r: f"Retry_Count = {r['Retry_Count']}",
            "Retry_Count >= 0",
            lambda r: f"Retry count ({r['Retry_Count']}) is negative."
        )

    # R_TECH_002: Negative SLA Target Days
    if "SLA_Target_Days" in df.columns:
        mask = df["SLA_Target_Days"] < 0
        add_anomalies_from_mask(
            mask, "R_TECH_002", "SLA_Target_Days",
            lambda r: f"SLA_Target_Days = {r['SLA_Target_Days']}",
            "SLA_Target_Days >= 0",
            lambda r: f"SLA target days ({r['SLA_Target_Days']}) is negative."
        )

    # R_SLA_001: SLA Breach
    if "Processing_Latency_Days" in df.columns and "SLA_Target_Days" in df.columns:
        mask = df["Processing_Latency_Days"].notna() & df["SLA_Target_Days"].notna() & (df["Processing_Latency_Days"] > df["SLA_Target_Days"])
        add_anomalies_from_mask(
            mask, "R_SLA_001", "Processing_Latency_Days, SLA_Target_Days",
            lambda r: f"Latency = {r['Processing_Latency_Days']}, SLA_Target = {r['SLA_Target_Days']}",
            "Processing_Latency_Days <= SLA_Target_Days",
            lambda r: f"Processing latency ({r['Processing_Latency_Days']} days) exceeded the pharmacy claim SLA target days ({r['SLA_Target_Days']} days)."
        )

    # R_MET_003: Quantity/Days-Supply consistency
    if "Quantity_Dispensed" in df.columns and "Days_Supply" in df.columns:
        mask = df["Quantity_Dispensed"].isna() | df["Days_Supply"].isna()
        add_anomalies_from_mask(
            mask, "R_MET_003", "Quantity_Dispensed, Days_Supply",
            lambda r: f"Quantity = {r['Quantity_Dispensed']}, Days = {r['Days_Supply']}",
            "Quantity_Dispensed and Days_Supply must both be present",
            lambda r: f"Pharmacy claim has incomplete dispensing metrics. Quantity = {r['Quantity_Dispensed']}, Days Supply = {r['Days_Supply']}."
        )

    # R_CAT_001: Invalid Status
    if "Status" in df.columns:
        mask = ~df["Status"].isin(["PAID", "REJECTED", "PENDING"])
        add_anomalies_from_mask(
            mask, "R_CAT_001", "Status",
            lambda r: f"Status = {r['Status']}",
            "Status in ['PAID', 'REJECTED', 'PENDING']",
            lambda r: f"Pharmacy claim Status ({r['Status']}) contains an unexpected code not conforming to standard statuses (PAID, REJECTED, PENDING)."
        )

    # R_CAT_002: Status/Date consistency
    if "Status" in df.columns and "Processed_Date" in df.columns:
        mask1 = (df["Status"] == "PENDING") & df["Processed_Date"].notna()
        add_anomalies_from_mask(
            mask1, "R_CAT_002", "Status, Processed_Date",
            lambda r: f"Status = PENDING, Processed_Date = {r['Processed_Date'].strftime('%Y-%m-%d')}",
            "Processed_Date must be missing for PENDING status",
            lambda r: f"Pharmacy claim status is PENDING but contains a populated processed date ({r['Processed_Date'].strftime('%Y-%m-%d')})."
        )
        
        mask2 = df["Status"].isin(["PAID", "REJECTED"]) & df["Processed_Date"].isna()
        add_anomalies_from_mask(
            mask2, "R_CAT_002", "Status, Processed_Date",
            lambda r: f"Status = {r['Status']}, Processed_Date = NaN",
            "Processed_Date must be present for completed status",
            lambda r: f"Pharmacy claim has a completed status ({r['Status']}) but its required processed date is missing (NaN)."
        )

    return anomalies
