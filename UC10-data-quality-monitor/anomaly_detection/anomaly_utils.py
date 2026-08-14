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
        "record_identifier": record_id,
        "record_type": record_type,
        "rule_id": rule_id,
        "anomaly_category": category,
        "affected_columns": affected_cols,
        "observed_value": observed,
        "expected_condition": expected,
        "severity": severity,
        "explanation": explanation,
        "source_dataset": source_dataset,
        "detection_timestamp": datetime.datetime.now().isoformat()
    }
