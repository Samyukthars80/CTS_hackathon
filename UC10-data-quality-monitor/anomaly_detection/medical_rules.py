from __future__ import annotations

import pandas as pd
from anomaly_detection.anomaly_utils import create_anomaly_record


def evaluate_medical_rules(df: pd.DataFrame, config: dict) -> list[dict]:
    anomalies = []
    source = "medical_claim_cleaned.csv"
    
    # 1. Date Conversions
    date_cols = ["Service_Date", "Service_End_Date", "Submission_Date", "Processed_Date", "Decision_Date"]
    for col in date_cols:
        if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col], errors='coerce')
            
    # Helper to log violations
    def add_anomalies_from_mask(mask, rule_id, affected_cols, observed_expr, expected, explanation_func):
        if rule_id not in config or not config[rule_id].get("enabled", True):
            return
        
        # Apply mask and select matching rows
        violators = df[mask]
        for _, row in violators.iterrows():
            record_id = str(row.get("Record_ID", "UNKNOWN"))
            rec_type = str(row.get("Record_Type", "MEDICAL_CLAIM"))
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

    # MED-M001: Negative Billed Amount
    if "Billed_Amount" in df.columns:
        mask = df["Billed_Amount"] < 0
        add_anomalies_from_mask(
            mask, "MED-M001", "Billed_Amount",
            lambda r: f"Billed_Amount = {r['Billed_Amount']}",
            "Billed_Amount >= 0",
            lambda r: f"Billed amount (${r['Billed_Amount']:.2f}) is negative. Under financial semantics, billed amounts must be non-negative."
        )

    # MED-M002: Negative Allowed Amount
    if "Allowed_Amount" in df.columns:
        mask = df["Allowed_Amount"] < 0
        add_anomalies_from_mask(
            mask, "MED-M002", "Allowed_Amount",
            lambda r: f"Allowed_Amount = {r['Allowed_Amount']}",
            "Allowed_Amount >= 0",
            lambda r: f"Allowed amount (${r['Allowed_Amount']:.2f}) is negative. Under financial semantics, allowed amounts must be non-negative."
        )

    # MED-M003: Negative Paid Amount
    if "Paid_Amount" in df.columns:
        mask = df["Paid_Amount"] < 0
        add_anomalies_from_mask(
            mask, "MED-M003", "Paid_Amount",
            lambda r: f"Paid_Amount = {r['Paid_Amount']}",
            "Paid_Amount >= 0",
            lambda r: f"Paid amount (${r['Paid_Amount']:.2f}) is negative. Paid claim lines must have non-negative payments."
        )

    # MED-M004: Negative Patient Responsibility
    if "Patient_Responsibility" in df.columns:
        mask = df["Patient_Responsibility"] < 0
        add_anomalies_from_mask(
            mask, "MED-M004", "Patient_Responsibility",
            lambda r: f"Patient_Responsibility = {r['Patient_Responsibility']}",
            "Patient_Responsibility >= 0",
            lambda r: f"Patient responsibility (${r['Patient_Responsibility']:.2f}) is negative. Patient responsibility amounts must be non-negative."
        )

    # MED-M005: Paid > Allowed
    if "Paid_Amount" in df.columns and "Allowed_Amount" in df.columns:
        mask = df["Paid_Amount"] > df["Allowed_Amount"]
        add_anomalies_from_mask(
            mask, "MED-M005", "Paid_Amount, Allowed_Amount",
            lambda r: f"Paid = {r['Paid_Amount']}, Allowed = {r['Allowed_Amount']}",
            "Paid_Amount <= Allowed_Amount",
            lambda r: f"Paid amount (${r['Paid_Amount']:.2f}) exceeds the allowed amount (${r['Allowed_Amount']:.2f}). Under logical financial conventions, payment must not exceed allowed benefits."
        )

    # MED-M006: Allowed > Billed
    if "Allowed_Amount" in df.columns and "Billed_Amount" in df.columns:
        mask = df["Allowed_Amount"] > df["Billed_Amount"]
        add_anomalies_from_mask(
            mask, "MED-M006", "Allowed_Amount, Billed_Amount",
            lambda r: f"Allowed = {r['Allowed_Amount']}, Billed = {r['Billed_Amount']}",
            "Allowed_Amount <= Billed_Amount",
            lambda r: f"Allowed amount (${r['Allowed_Amount']:.2f}) exceeds the provider's billed amount (${r['Billed_Amount']:.2f}). Insurers do not negotiate contract allowances exceeding original charges."
        )

    # MED-M007: Paid > Billed
    if "Paid_Amount" in df.columns and "Billed_Amount" in df.columns:
        mask = df["Paid_Amount"] > df["Billed_Amount"]
        add_anomalies_from_mask(
            mask, "MED-M007", "Paid_Amount, Billed_Amount",
            lambda r: f"Paid = {r['Paid_Amount']}, Billed = {r['Billed_Amount']}",
            "Paid_Amount <= Billed_Amount",
            lambda r: f"Paid amount (${r['Paid_Amount']:.2f}) exceeds the provider's billed amount (${r['Billed_Amount']:.2f}). Reimbursements cannot exceed the initial claim submission."
        )

    # MED-M008: Service_Date > Service_End_Date
    if "Service_Date" in df.columns and "Service_End_Date" in df.columns:
        mask = df["Service_Date"].notna() & df["Service_End_Date"].notna() & (df["Service_Date"] > df["Service_End_Date"])
        add_anomalies_from_mask(
            mask, "MED-M008", "Service_Date, Service_End_Date",
            lambda r: f"Service = {r['Service_Date'].strftime('%Y-%m-%d')}, End = {r['Service_End_Date'].strftime('%Y-%m-%d')}",
            "Service_Date <= Service_End_Date",
            lambda r: f"Service start date ({r['Service_Date'].strftime('%Y-%m-%d')}) occurs after service end date ({r['Service_End_Date'].strftime('%Y-%m-%d')})."
        )

    # MED-M009: Submission_Date < Service_Date
    if "Submission_Date" in df.columns and "Service_Date" in df.columns:
        mask = df["Submission_Date"].notna() & df["Service_Date"].notna() & (df["Submission_Date"] < df["Service_Date"])
        add_anomalies_from_mask(
            mask, "MED-M009", "Submission_Date, Service_Date",
            lambda r: f"Submission = {r['Submission_Date'].strftime('%Y-%m-%d')}, Service = {r['Service_Date'].strftime('%Y-%m-%d')}",
            "Submission_Date >= Service_Date",
            lambda r: f"Claim was submitted ({r['Submission_Date'].strftime('%Y-%m-%d')}) prior to the service start date ({r['Service_Date'].strftime('%Y-%m-%d')})."
        )

    # MED-M010: Processed_Date < Submission_Date
    if "Processed_Date" in df.columns and "Submission_Date" in df.columns:
        mask = df["Processed_Date"].notna() & df["Submission_Date"].notna() & (df["Processed_Date"] < df["Submission_Date"])
        add_anomalies_from_mask(
            mask, "MED-M010", "Processed_Date, Submission_Date",
            lambda r: f"Processed = {r['Processed_Date'].strftime('%Y-%m-%d')}, Submission = {r['Submission_Date'].strftime('%Y-%m-%d')}",
            "Processed_Date >= Submission_Date",
            lambda r: f"Claim processing date ({r['Processed_Date'].strftime('%Y-%m-%d')}) is before the claim submission date ({r['Submission_Date'].strftime('%Y-%m-%d')})."
        )

    # MED-M011: Decision_Date < Submission_Date
    if "Decision_Date" in df.columns and "Submission_Date" in df.columns:
        mask = df["Decision_Date"].notna() & df["Submission_Date"].notna() & (df["Decision_Date"] < df["Submission_Date"])
        add_anomalies_from_mask(
            mask, "MED-M011", "Decision_Date, Submission_Date",
            lambda r: f"Decision = {r['Decision_Date'].strftime('%Y-%m-%d')}, Submission = {r['Submission_Date'].strftime('%Y-%m-%d')}",
            "Decision_Date >= Submission_Date",
            lambda r: f"Decision date ({r['Decision_Date'].strftime('%Y-%m-%d')}) occurs before the submission date ({r['Submission_Date'].strftime('%Y-%m-%d')})."
        )

    # MED-M012: Negative Latency
    if "Processing_Latency_Days" in df.columns:
        mask = df["Processing_Latency_Days"] < 0
        add_anomalies_from_mask(
            mask, "MED-M012", "Processing_Latency_Days",
            lambda r: f"Processing_Latency_Days = {r['Processing_Latency_Days']}",
            "Processing_Latency_Days >= 0",
            lambda r: f"Processing latency ({r['Processing_Latency_Days']} days) is negative, representing an impossible chronological gap."
        )

    # MED-M013: Negative Retry
    if "Retry_Count" in df.columns:
        mask = df["Retry_Count"] < 0
        add_anomalies_from_mask(
            mask, "MED-M013", "Retry_Count",
            lambda r: f"Retry_Count = {r['Retry_Count']}",
            "Retry_Count >= 0",
            lambda r: f"Retry count ({r['Retry_Count']}) is negative, which is technically invalid."
        )

    # MED-M014: SLA Breach
    if "Processing_Latency_Days" in df.columns and "SLA_Target_Days" in df.columns:
        # Ignore cases where latency is missing (e.g. pending claims)
        mask = df["Processing_Latency_Days"].notna() & df["SLA_Target_Days"].notna() & (df["Processing_Latency_Days"] > df["SLA_Target_Days"])
        add_anomalies_from_mask(
            mask, "MED-M014", "Processing_Latency_Days, SLA_Target_Days",
            lambda r: f"Latency = {r['Processing_Latency_Days']}, SLA_Target = {r['SLA_Target_Days']}",
            "Processing_Latency_Days <= SLA_Target_Days",
            lambda r: f"Processing latency ({r['Processing_Latency_Days']} days) has exceeded the designated SLA target limit ({r['SLA_Target_Days']} days)."
        )

    # MED-M015: Invalid Status
    if "Status" in df.columns:
        mask = ~df["Status"].isin(["PAID", "DENIED", "PENDING"])
        add_anomalies_from_mask(
            mask, "MED-M015", "Status",
            lambda r: f"Status = {r['Status']}",
            "Status in ['PAID', 'DENIED', 'PENDING']",
            lambda r: f"Claim Status ({r['Status']}) contains an unexpected code not conforming to standard statuses (PAID, DENIED, PENDING)."
        )

    # MED-M016: Status/Date consistency
    if "Status" in df.columns and "Processed_Date" in df.columns:
        # PENDING claims must not have Processed_Date
        mask1 = (df["Status"] == "PENDING") & df["Processed_Date"].notna()
        add_anomalies_from_mask(
            mask1, "MED-M016", "Status, Processed_Date",
            lambda r: f"Status = PENDING, Processed_Date = {r['Processed_Date'].strftime('%Y-%m-%d')}",
            "Processed_Date must be missing for PENDING status",
            lambda r: f"Claim status is PENDING but contains a populated processed date ({r['Processed_Date'].strftime('%Y-%m-%d')})."
        )
        
        # PAID or DENIED claims must have Processed_Date
        mask2 = df["Status"].isin(["PAID", "DENIED"]) & df["Processed_Date"].isna()
        add_anomalies_from_mask(
            mask2, "MED-M016", "Status, Processed_Date",
            lambda r: f"Status = {r['Status']}, Processed_Date = NaN",
            "Processed_Date must be present for completed status",
            lambda r: f"Claim has a completed status ({r['Status']}) but its required processed date is missing (NaN)."
        )

    return anomalies
