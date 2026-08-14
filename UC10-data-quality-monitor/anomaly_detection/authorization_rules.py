from __future__ import annotations

import pandas as pd
from anomaly_detection.anomaly_utils import create_anomaly_record


def evaluate_authorization_rules(df: pd.DataFrame, config: dict) -> list[dict]:
    anomalies = []
    source = "authorization_cleaned.csv"
    
    # Date Conversions
    date_cols = ["Submission_Date", "Processed_Date", "Decision_Date"]
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
            rec_type = str(row.get("Record_Type", "PRIOR_AUTH"))
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

    # AUTH-M001: Negative Retry Count
    if "Retry_Count" in df.columns:
        mask = df["Retry_Count"] < 0
        add_anomalies_from_mask(
            mask, "AUTH-M001", "Retry_Count",
            lambda r: f"Retry_Count = {r['Retry_Count']}",
            "Retry_Count >= 0",
            lambda r: f"Retry count ({r['Retry_Count']}) is negative."
        )

    # AUTH-M002: Negative Latency
    if "Processing_Latency_Days" in df.columns:
        mask = df["Processing_Latency_Days"] < 0
        add_anomalies_from_mask(
            mask, "AUTH-M002", "Processing_Latency_Days",
            lambda r: f"Processing_Latency_Days = {r['Processing_Latency_Days']}",
            "Processing_Latency_Days >= 0",
            lambda r: f"Processing latency ({r['Processing_Latency_Days']} days) is negative."
        )

    # AUTH-M003: Invalid SLA Target
    if "SLA_Target_Days" in df.columns:
        mask = df["SLA_Target_Days"] < 0
        add_anomalies_from_mask(
            mask, "AUTH-M003", "SLA_Target_Days",
            lambda r: f"SLA_Target_Days = {r['SLA_Target_Days']}",
            "SLA_Target_Days >= 0",
            lambda r: f"SLA target days ({r['SLA_Target_Days']}) is negative."
        )

    # AUTH-M004: Processed before Submission
    if "Processed_Date" in df.columns and "Submission_Date" in df.columns:
        mask = df["Processed_Date"].notna() & df["Submission_Date"].notna() & (df["Processed_Date"] < df["Submission_Date"])
        add_anomalies_from_mask(
            mask, "AUTH-M004", "Processed_Date, Submission_Date",
            lambda r: f"Processed = {r['Processed_Date'].strftime('%Y-%m-%d')}, Submission = {r['Submission_Date'].strftime('%Y-%m-%d')}",
            "Processed_Date >= Submission_Date",
            lambda r: f"Authorization processed date ({r['Processed_Date'].strftime('%Y-%m-%d')}) occurs before the submission date ({r['Submission_Date'].strftime('%Y-%m-%d')})."
        )

    # AUTH-M005: Decision before Submission
    if "Decision_Date" in df.columns and "Submission_Date" in df.columns:
        mask = df["Decision_Date"].notna() & df["Submission_Date"].notna() & (df["Decision_Date"] < df["Submission_Date"])
        add_anomalies_from_mask(
            mask, "AUTH-M005", "Decision_Date, Submission_Date",
            lambda r: f"Decision = {r['Decision_Date'].strftime('%Y-%m-%d')}, Submission = {r['Submission_Date'].strftime('%Y-%m-%d')}",
            "Decision_Date >= Submission_Date",
            lambda r: f"Decision date ({r['Decision_Date'].strftime('%Y-%m-%d')}) occurs before the submission date ({r['Submission_Date'].strftime('%Y-%m-%d')})."
        )

    # AUTH-M006: Decision before Processing
    if "Decision_Date" in df.columns and "Processed_Date" in df.columns:
        mask = df["Decision_Date"].notna() & df["Processed_Date"].notna() & (df["Decision_Date"] < df["Processed_Date"])
        add_anomalies_from_mask(
            mask, "AUTH-M006", "Decision_Date, Processed_Date",
            lambda r: f"Decision = {r['Decision_Date'].strftime('%Y-%m-%d')}, Processed = {r['Processed_Date'].strftime('%Y-%m-%d')}",
            "Decision_Date >= Processed_Date",
            lambda r: f"Decision date ({r['Decision_Date'].strftime('%Y-%m-%d')}) is before the processed date ({r['Processed_Date'].strftime('%Y-%m-%d')})."
        )

    # AUTH-M007: SLA Breach
    if "Processing_Latency_Days" in df.columns and "SLA_Target_Days" in df.columns:
        mask = df["Processing_Latency_Days"].notna() & df["SLA_Target_Days"].notna() & (df["Processing_Latency_Days"] > df["SLA_Target_Days"])
        add_anomalies_from_mask(
            mask, "AUTH-M007", "Processing_Latency_Days, SLA_Target_Days",
            lambda r: f"Latency = {r['Processing_Latency_Days']}, SLA_Target = {r['SLA_Target_Days']}",
            "Processing_Latency_Days <= SLA_Target_Days",
            lambda r: f"Processing latency ({r['Processing_Latency_Days']} days) exceeded the authorization SLA target days ({r['SLA_Target_Days']} days)."
        )

    # AUTH-M008: Status/Decision Consistency
    if "Status" in df.columns and "Decision_Date" in df.columns:
        # PENDING claims must not have Decision_Date
        mask1 = (df["Status"] == "PENDING") & df["Decision_Date"].notna()
        add_anomalies_from_mask(
            mask1, "AUTH-M008", "Status, Decision_Date",
            lambda r: f"Status = PENDING, Decision_Date = {r['Decision_Date'].strftime('%Y-%m-%d')}",
            "Decision_Date must be missing for PENDING status",
            lambda r: f"Authorization has status PENDING but contains a populated decision date ({r['Decision_Date'].strftime('%Y-%m-%d')})."
        )
        
        # APPROVED or DENIED claims must have Decision_Date
        mask2 = df["Status"].isin(["APPROVED", "DENIED"]) & df["Decision_Date"].isna()
        add_anomalies_from_mask(
            mask2, "AUTH-M008", "Status, Decision_Date",
            lambda r: f"Status = {r['Status']}, Decision_Date = NaN",
            "Decision_Date must be present for completed status",
            lambda r: f"Authorization status is {r['Status']} but its required decision date is missing (NaN)."
        )

    # AUTH-M009: Missing Auth Linkage (Disabled in config by default, but implemented here for completeness)
    if "Auth_Linked_ID" in df.columns:
        mask = df["Auth_Linked_ID"].isna()
        add_anomalies_from_mask(
            mask, "AUTH-M009", "Auth_Linked_ID",
            lambda r: "Auth_Linked_ID = NaN",
            "Auth_Linked_ID must be present",
            lambda r: "Prior Authorization is missing the required linkage identifier."
        )

    return anomalies
