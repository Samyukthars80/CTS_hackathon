import datetime

def create_anomaly_record(
    record_id: str,
    record_type: str,
    rule_id: str,
    category: str,
    affected_cols: str,
    observed: str,
    expected: str,
    severity: str,
    explanation: str,
    source_dataset: str
) -> dict:
    """Instantiate a standardized anomaly record."""
    return {
        "Record_ID": record_id,
        "Record_Type": record_type,
        "Detection_Method": "Rule-Based",
        "Anomaly_Category": category,
        "Affected_Columns": affected_cols,
        "Observed_Value": observed,
        "Expected_Condition": expected,
        "Severity": severity,
        "Explanation": explanation,
        "Rule_ID": rule_id,
        "Source_Dataset": source_dataset,
        "Detection_Timestamp": datetime.datetime.now().isoformat()
    }
