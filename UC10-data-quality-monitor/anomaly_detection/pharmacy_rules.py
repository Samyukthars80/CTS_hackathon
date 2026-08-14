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

    # PH-M001: Negative Billed Amount
    if "Billed_Amount" in df.columns:
        mask = df["Billed_Amount"] < 0
        add_anomalies_from_mask(
            mask, "PH-M001", "Billed_Amount",
            lambda r: f"Billed_Amount = {r['Billed_Amount']}",
            "Billed_Amount >= 0",
            lambda r: f"Billed amount (${r['Billed_Amount']:.2f}) is negative on this pharmacy claim."
        )

    # PH-M002: Negative Allowed Amount
    if "Allowed_Amount" in df.columns:
        mask = df["Allowed_Amount"] < 0
        add_anomalies_from_mask(
            mask, "PH-M002", "Allowed_Amount",
            lambda r: f"Allowed_Amount = {r['Allowed_Amount']}",
            "Allowed_Amount >= 0",
            lambda r: f"Allowed amount (${r['Allowed_Amount']:.2f}) is negative."
        )

    # PH-M003: Negative Paid Amount
    if "Paid_Amount" in df.columns:
        mask = df["Paid_Amount"] < 0
        add_anomalies_from_mask(
            mask, "PH-M003", "Paid_Amount",
            lambda r: f"Paid_Amount = {r['Paid_Amount']}",
            "Paid_Amount >= 0",
            lambda r: f"Paid amount (${r['Paid_Amount']:.2f}) is negative."
        )

    # PH-M004: Negative Patient Responsibility
    if "Patient_Responsibility" in df.columns:
        mask = df["Patient_Responsibility"] < 0
        add_anomalies_from_mask(
            mask, "PH-M004", "Patient_Responsibility",
            lambda r: f"Patient_Responsibility = {r['Patient_Responsibility']}",
            "Patient_Responsibility >= 0",
            lambda r: f"Patient responsibility (${r['Patient_Responsibility']:.2f}) is negative."
        )

    # PH-M005: Paid > Allowed
    if "Paid_Amount" in df.columns and "Allowed_Amount" in df.columns:
        mask = df["Paid_Amount"] > df["Allowed_Amount"]
        add_anomalies_from_mask(
            mask, "PH-M005", "Paid_Amount, Allowed_Amount",
            lambda r: f"Paid = {r['Paid_Amount']}, Allowed = {r['Allowed_Amount']}",
            "Paid_Amount <= Allowed_Amount",
            lambda r: f"Paid amount (${r['Paid_Amount']:.2f}) exceeds the allowed amount (${r['Allowed_Amount']:.2f}) on this pharmacy claim."
        )

    # PH-M006: Allowed > Billed
    if "Allowed_Amount" in df.columns and "Billed_Amount" in df.columns:
        mask = df["Allowed_Amount"] > df["Billed_Amount"]
        add_anomalies_from_mask(
            mask, "PH-M006", "Allowed_Amount, Billed_Amount",
            lambda r: f"Allowed = {r['Allowed_Amount']}, Billed = {r['Billed_Amount']}",
            "Allowed_Amount <= Billed_Amount",
            lambda r: f"Allowed amount (${r['Allowed_Amount']:.2f}) exceeds the billed amount (${r['Billed_Amount']:.2f})."
        )

    # PH-M007: Paid > Billed
    if "Paid_Amount" in df.columns and "Billed_Amount" in df.columns:
        mask = df["Paid_Amount"] > df["Billed_Amount"]
        add_anomalies_from_mask(
            mask, "PH-M007", "Paid_Amount, Billed_Amount",
            lambda r: f"Paid = {r['Paid_Amount']}, Billed = {r['Billed_Amount']}",
            "Paid_Amount <= Billed_Amount",
            lambda r: f"Paid amount (${r['Paid_Amount']:.2f}) exceeds the billed amount (${r['Billed_Amount']:.2f})."
        )

    # PH-M008: Days Supply <= 0
    if "Days_Supply" in df.columns:
        mask = df["Days_Supply"] <= 0
        add_anomalies_from_mask(
            mask, "PH-M008", "Days_Supply",
            lambda r: f"Days_Supply = {r['Days_Supply']}",
            "Days_Supply > 0",
            lambda r: f"Pharmacy claim has an invalid days supply value of {r['Days_Supply']}. Days supply must be positive."
        )

    # PH-M009: Quantity Dispensed <= 0
    if "Quantity_Dispensed" in df.columns:
        mask = df["Quantity_Dispensed"] <= 0
        add_anomalies_from_mask(
            mask, "PH-M009", "Quantity_Dispensed",
            lambda r: f"Quantity_Dispensed = {r['Quantity_Dispensed']}",
            "Quantity_Dispensed > 0",
            lambda r: f"Pharmacy claim has an invalid dispensed quantity value of {r['Quantity_Dispensed']}. Quantity must be positive."
        )

    # PH-M010: Negative Latency
    if "Processing_Latency_Days" in df.columns:
        mask = df["Processing_Latency_Days"] < 0
        add_anomalies_from_mask(
            mask, "PH-M010", "Processing_Latency_Days",
            lambda r: f"Processing_Latency_Days = {r['Processing_Latency_Days']}",
            "Processing_Latency_Days >= 0",
            lambda r: f"Processing latency ({r['Processing_Latency_Days']} days) is negative."
        )

    # PH-M011: Negative Retry
    if "Retry_Count" in df.columns:
        mask = df["Retry_Count"] < 0
        add_anomalies_from_mask(
            mask, "PH-M011", "Retry_Count",
            lambda r: f"Retry_Count = {r['Retry_Count']}",
            "Retry_Count >= 0",
            lambda r: f"Retry count ({r['Retry_Count']}) is negative."
        )

    # PH-M012: Date Order
    if "Service_Date" in df.columns and "Service_End_Date" in df.columns:
        mask1 = df["Service_Date"].notna() & df["Service_End_Date"].notna() & (df["Service_Date"] > df["Service_End_Date"])
        add_anomalies_from_mask(
            mask1, "PH-M012", "Service_Date, Service_End_Date",
            lambda r: f"Service = {r['Service_Date'].strftime('%Y-%m-%d')}, End = {r['Service_End_Date'].strftime('%Y-%m-%d')}",
            "Service_Date <= Service_End_Date",
            lambda r: f"Pharmacy service date ({r['Service_Date'].strftime('%Y-%m-%d')}) is after the service end date ({r['Service_End_Date'].strftime('%Y-%m-%d')})."
        )
        
    if "Submission_Date" in df.columns and "Service_Date" in df.columns:
        mask2 = df["Submission_Date"].notna() & df["Service_Date"].notna() & (df["Submission_Date"] < df["Service_Date"])
        add_anomalies_from_mask(
            mask2, "PH-M012", "Submission_Date, Service_Date",
            lambda r: f"Submission = {r['Submission_Date'].strftime('%Y-%m-%d')}, Service = {r['Service_Date'].strftime('%Y-%m-%d')}",
            "Submission_Date >= Service_Date",
            lambda r: f"Pharmacy claim submission date ({r['Submission_Date'].strftime('%Y-%m-%d')}) occurs before the service date ({r['Service_Date'].strftime('%Y-%m-%d')})."
        )
        
    if "Processed_Date" in df.columns and "Submission_Date" in df.columns:
        mask3 = df["Processed_Date"].notna() & df["Submission_Date"].notna() & (df["Processed_Date"] < df["Submission_Date"])
        add_anomalies_from_mask(
            mask3, "PH-M012", "Processed_Date, Submission_Date",
            lambda r: f"Processed = {r['Processed_Date'].strftime('%Y-%m-%d')}, Submission = {r['Submission_Date'].strftime('%Y-%m-%d')}",
            "Processed_Date >= Submission_Date",
            lambda r: f"Pharmacy claim processing date ({r['Processed_Date'].strftime('%Y-%m-%d')}) is before the claim submission date ({r['Submission_Date'].strftime('%Y-%m-%d')})."
        )

    # PH-M013: SLA Breach
    if "Processing_Latency_Days" in df.columns and "SLA_Target_Days" in df.columns:
        mask = df["Processing_Latency_Days"].notna() & df["SLA_Target_Days"].notna() & (df["Processing_Latency_Days"] > df["SLA_Target_Days"])
        add_anomalies_from_mask(
            mask, "PH-M013", "Processing_Latency_Days, SLA_Target_Days",
            lambda r: f"Latency = {r['Processing_Latency_Days']}, SLA_Target = {r['SLA_Target_Days']}",
            "Processing_Latency_Days <= SLA_Target_Days",
            lambda r: f"Processing latency ({r['Processing_Latency_Days']} days) exceeded the pharmacy claim SLA target days ({r['SLA_Target_Days']} days)."
        )

    # PH-M014: Quantity/Days-Supply consistency
    if "Quantity_Dispensed" in df.columns and "Days_Supply" in df.columns:
        # Check if either value is null (legitimate missingness check)
        mask = df["Quantity_Dispensed"].isna() | df["Days_Supply"].isna()
        add_anomalies_from_mask(
            mask, "PH-M014", "Quantity_Dispensed, Days_Supply",
            lambda r: f"Quantity = {r['Quantity_Dispensed']}, Days = {r['Days_Supply']}",
            "Quantity_Dispensed and Days_Supply must both be present",
            lambda r: f"Pharmacy claim has incomplete dispensing metrics. Quantity = {r['Quantity_Dispensed']}, Days Supply = {r['Days_Supply']}."
        )

    return anomalies
