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

    # R_TECH_001: Negative Retry Count
    if "Retry_Count" in df.columns:
        mask = df["Retry_Count"] < 0
        add_anomalies_from_mask(
            mask, "R_TECH_001", "Retry_Count",
            lambda r: f"Retry_Count = {r['Retry_Count']}",
            "Retry_Count >= 0",
            lambda r: f"Retry count ({r['Retry_Count']}) is negative."
        )

    # R_TECH_002: SLA target days must not be negative
    if "SLA_Target_Days" in df.columns:
        mask = df["SLA_Target_Days"] < 0
        add_anomalies_from_mask(
            mask, "R_TECH_002", "SLA_Target_Days",
            lambda r: f"SLA_Target_Days = {r['SLA_Target_Days']}",
            "SLA_Target_Days >= 0",
            lambda r: f"SLA target days ({r['SLA_Target_Days']}) is negative."
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
            exp.append(f"Authorization processed date ({row['Processed_Date'].strftime('%Y-%m-%d')}) is before submission date ({row['Submission_Date'].strftime('%Y-%m-%d')})")
        if row["Processing_Latency_Days"] < 0:
            exp.append(f"Processing latency is negative ({row['Processing_Latency_Days']} days)")
        return ". ".join(exp) + ". This indicates an invalid processing timeline."

    add_anomalies_from_mask(
        mask_timeline, "R_DATE_001", "Submission_Date, Processed_Date, Processing_Latency_Days",
        observed_timeline,
        "Processed_Date >= Submission_Date AND Processing_Latency_Days >= 0",
        explanation_timeline
    )

    # R_DATE_004: Decision before Submission
    if "Decision_Date" in df.columns and "Submission_Date" in df.columns:
        mask = df["Decision_Date"].notna() & df["Submission_Date"].notna() & (df["Decision_Date"] < df["Submission_Date"])
        add_anomalies_from_mask(
            mask, "R_DATE_004", "Decision_Date, Submission_Date",
            lambda r: f"Decision = {r['Decision_Date'].strftime('%Y-%m-%d')}, Submission = {r['Submission_Date'].strftime('%Y-%m-%d')}",
            "Decision_Date >= Submission_Date",
            lambda r: f"Decision date ({r['Decision_Date'].strftime('%Y-%m-%d')}) occurs before the submission date ({r['Submission_Date'].strftime('%Y-%m-%d')})."
        )

    # R_DATE_005: Decision before Processing
    if "Decision_Date" in df.columns and "Processed_Date" in df.columns:
        mask = df["Decision_Date"].notna() & df["Processed_Date"].notna() & (df["Decision_Date"] < df["Processed_Date"])
        add_anomalies_from_mask(
            mask, "R_DATE_005", "Decision_Date, Processed_Date",
            lambda r: f"Decision = {r['Decision_Date'].strftime('%Y-%m-%d')}, Processed = {r['Processed_Date'].strftime('%Y-%m-%d')}",
            "Decision_Date >= Processed_Date",
            lambda r: f"Decision date ({r['Decision_Date'].strftime('%Y-%m-%d')}) occurs before the processed date ({r['Processed_Date'].strftime('%Y-%m-%d')})."
        )

    # R_SLA_001: SLA Breach
    if "Processing_Latency_Days" in df.columns and "SLA_Target_Days" in df.columns:
        mask = df["Processing_Latency_Days"].notna() & df["SLA_Target_Days"].notna() & (df["Processing_Latency_Days"] > df["SLA_Target_Days"])
        add_anomalies_from_mask(
            mask, "R_SLA_001", "Processing_Latency_Days, SLA_Target_Days",
            lambda r: f"Latency = {r['Processing_Latency_Days']}, SLA_Target = {r['SLA_Target_Days']}",
            "Processing_Latency_Days <= SLA_Target_Days",
            lambda r: f"Processing latency ({r['Processing_Latency_Days']} days) exceeded the authorization SLA target days ({r['SLA_Target_Days']} days)."
        )

    # R_CAT_001: Invalid Status
    if "Status" in df.columns:
        mask = ~df["Status"].isin(["APPROVED", "DENIED", "PENDING"])
        add_anomalies_from_mask(
            mask, "R_CAT_001", "Status",
            lambda r: f"Status = {r['Status']}",
            "Status in ['APPROVED', 'DENIED', 'PENDING']",
            lambda r: f"Authorization status ({r['Status']}) contains an unexpected code not conforming to standard statuses (APPROVED, DENIED, PENDING)."
        )

    # R_CAT_002: Status/Decision Consistency
    if "Status" in df.columns and "Decision_Date" in df.columns:
        mask1 = (df["Status"] == "PENDING") & df["Decision_Date"].notna()
        add_anomalies_from_mask(
            mask1, "R_CAT_002", "Status, Decision_Date",
            lambda r: f"Status = PENDING, Decision_Date = {r['Decision_Date'].strftime('%Y-%m-%d')}",
            "Decision_Date must be missing for PENDING status",
            lambda r: f"Authorization has status PENDING but contains a populated decision date ({r['Decision_Date'].strftime('%Y-%m-%d')})."
        )
        
        mask2 = df["Status"].isin(["APPROVED", "DENIED"]) & df["Decision_Date"].isna()
        add_anomalies_from_mask(
            mask2, "R_CAT_002", "Status, Decision_Date",
            lambda r: f"Status = {r['Status']}, Decision_Date = NaN",
            "Decision_Date must be present for completed status",
            lambda r: f"Authorization status is {r['Status']} but its required decision date is missing (NaN)."
        )

    return anomalies
