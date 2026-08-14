# Centralized Rule Configurations

RULES_CONFIG = {
    # ==========================================
    # MEDICAL CLAIM RULES (MED-M001 to MED-M016)
    # ==========================================
    "MED-M001": {
        "rule_id": "MED-M001",
        "record_type": "MEDICAL_CLAIM",
        "category": "Invalid Financial Value",
        "severity": "HIGH",
        "description": "Billed amount must not be negative.",
        "expected_condition": "Billed_Amount >= 0",
        "enabled": True
    },
    "MED-M002": {
        "rule_id": "MED-M002",
        "record_type": "MEDICAL_CLAIM",
        "category": "Invalid Financial Value",
        "severity": "HIGH",
        "description": "Allowed amount must not be negative.",
        "expected_condition": "Allowed_Amount >= 0",
        "enabled": True
    },
    "MED-M003": {
        "rule_id": "MED-M003",
        "record_type": "MEDICAL_CLAIM",
        "category": "Invalid Financial Value",
        "severity": "HIGH",
        "description": "Paid amount must not be negative.",
        "expected_condition": "Paid_Amount >= 0",
        "enabled": True
    },
    "MED-M004": {
        "rule_id": "MED-M004",
        "record_type": "MEDICAL_CLAIM",
        "category": "Invalid Financial Value",
        "severity": "HIGH",
        "description": "Patient responsibility must not be negative.",
        "expected_condition": "Patient_Responsibility >= 0",
        "enabled": True
    },
    "MED-M005": {
        "rule_id": "MED-M005",
        "record_type": "MEDICAL_CLAIM",
        "category": "Logical Inconsistency",
        "severity": "HIGH",
        "description": "Paid amount must not exceed the allowed amount.",
        "expected_condition": "Paid_Amount <= Allowed_Amount",
        "enabled": True
    },
    "MED-M006": {
        "rule_id": "MED-M006",
        "record_type": "MEDICAL_CLAIM",
        "category": "Logical Inconsistency",
        "severity": "MEDIUM",
        "description": "Allowed amount must not exceed the billed amount.",
        "expected_condition": "Allowed_Amount <= Billed_Amount",
        "enabled": True
    },
    "MED-M007": {
        "rule_id": "MED-M007",
        "record_type": "MEDICAL_CLAIM",
        "category": "Logical Inconsistency",
        "severity": "HIGH",
        "description": "Paid amount must not exceed the billed amount.",
        "expected_condition": "Paid_Amount <= Billed_Amount",
        "enabled": True
    },
    "MED-M008": {
        "rule_id": "MED-M008",
        "record_type": "MEDICAL_CLAIM",
        "category": "Chronological Anomaly",
        "severity": "HIGH",
        "description": "Service date must be on or before service end date.",
        "expected_condition": "Service_Date <= Service_End_Date",
        "enabled": True
    },
    "MED-M009": {
        "rule_id": "MED-M009",
        "record_type": "MEDICAL_CLAIM",
        "category": "Chronological Anomaly",
        "severity": "MEDIUM",
        "description": "Submission date must be on or after service date.",
        "expected_condition": "Submission_Date >= Service_Date",
        "enabled": True
    },
    "MED-M010": {
        "rule_id": "MED-M010",
        "record_type": "MEDICAL_CLAIM",
        "category": "Chronological Anomaly",
        "severity": "HIGH",
        "description": "Processed date must be on or after submission date.",
        "expected_condition": "Processed_Date >= Submission_Date",
        "enabled": True
    },
    "MED-M011": {
        "rule_id": "MED-M011",
        "record_type": "MEDICAL_CLAIM",
        "category": "Chronological Anomaly",
        "severity": "HIGH",
        "description": "Decision date must be on or after submission date.",
        "expected_condition": "Decision_Date >= Submission_Date",
        "enabled": True
    },
    "MED-M012": {
        "rule_id": "MED-M012",
        "record_type": "MEDICAL_CLAIM",
        "category": "Invalid Technical Value",
        "severity": "HIGH",
        "description": "Processing latency must not be negative.",
        "expected_condition": "Processing_Latency_Days >= 0",
        "enabled": True
    },
    "MED-M013": {
        "rule_id": "MED-M013",
        "record_type": "MEDICAL_CLAIM",
        "category": "Invalid Technical Value",
        "severity": "HIGH",
        "description": "Retry count must not be negative.",
        "expected_condition": "Retry_Count >= 0",
        "enabled": True
    },
    "MED-M014": {
        "rule_id": "MED-M014",
        "record_type": "MEDICAL_CLAIM",
        "category": "SLA Violation",
        "severity": "MEDIUM",
        "description": "Processing latency must not exceed SLA target days.",
        "expected_condition": "Processing_Latency_Days <= SLA_Target_Days",
        "enabled": True
    },
    "MED-M015": {
        "rule_id": "MED-M015",
        "record_type": "MEDICAL_CLAIM",
        "category": "Invalid Categorical Value",
        "severity": "MEDIUM",
        "description": "Status value must be standard (PAID, DENIED, PENDING).",
        "expected_condition": "Status in ['PAID', 'DENIED', 'PENDING']",
        "enabled": True
    },
    "MED-M016": {
        "rule_id": "MED-M016",
        "record_type": "MEDICAL_CLAIM",
        "category": "Status/Date Inconsistency",
        "severity": "MEDIUM",
        "description": "Dates must be consistent with record status.",
        "expected_condition": "PENDING status requires missing Processed_Date; completed claims require valid dates.",
        "enabled": True
    },

    # ==========================================
    # PHARMACY CLAIM RULES (PH-M001 to PH-M014)
    # ==========================================
    "PH-M001": {
        "rule_id": "PH-M001",
        "record_type": "PHARMACY_CLAIM",
        "category": "Invalid Financial Value",
        "severity": "HIGH",
        "description": "Billed amount must not be negative.",
        "expected_condition": "Billed_Amount >= 0",
        "enabled": True
    },
    "PH-M002": {
        "rule_id": "PH-M002",
        "record_type": "PHARMACY_CLAIM",
        "category": "Invalid Financial Value",
        "severity": "HIGH",
        "description": "Allowed amount must not be negative.",
        "expected_condition": "Allowed_Amount >= 0",
        "enabled": True
    },
    "PH-M003": {
        "rule_id": "PH-M003",
        "record_type": "PHARMACY_CLAIM",
        "category": "Invalid Financial Value",
        "severity": "HIGH",
        "description": "Paid amount must not be negative.",
        "expected_condition": "Paid_Amount >= 0",
        "enabled": True
    },
    "PH-M004": {
        "rule_id": "PH-M004",
        "record_type": "PHARMACY_CLAIM",
        "category": "Invalid Financial Value",
        "severity": "HIGH",
        "description": "Patient responsibility must not be negative.",
        "expected_condition": "Patient_Responsibility >= 0",
        "enabled": True
    },
    "PH-M005": {
        "rule_id": "PH-M005",
        "record_type": "PHARMACY_CLAIM",
        "category": "Logical Inconsistency",
        "severity": "HIGH",
        "description": "Paid amount must not exceed the allowed amount.",
        "expected_condition": "Paid_Amount <= Allowed_Amount",
        "enabled": True
    },
    "PH-M006": {
        "rule_id": "PH-M006",
        "record_type": "PHARMACY_CLAIM",
        "category": "Logical Inconsistency",
        "severity": "MEDIUM",
        "description": "Allowed amount must not exceed the billed amount.",
        "expected_condition": "Allowed_Amount <= Billed_Amount",
        "enabled": True
    },
    "PH-M007": {
        "rule_id": "PH-M007",
        "record_type": "PHARMACY_CLAIM",
        "category": "Logical Inconsistency",
        "severity": "HIGH",
        "description": "Paid amount must not exceed the billed amount.",
        "expected_condition": "Paid_Amount <= Billed_Amount",
        "enabled": True
    },
    "PH-M008": {
        "rule_id": "PH-M008",
        "record_type": "PHARMACY_CLAIM",
        "category": "Invalid Pharmacy Metric",
        "severity": "HIGH",
        "description": "Days supply must be positive.",
        "expected_condition": "Days_Supply > 0",
        "enabled": True
    },
    "PH-M009": {
        "rule_id": "PH-M009",
        "record_type": "PHARMACY_CLAIM",
        "category": "Invalid Pharmacy Metric",
        "severity": "HIGH",
        "description": "Quantity dispensed must be positive.",
        "expected_condition": "Quantity_Dispensed > 0",
        "enabled": True
    },
    "PH-M010": {
        "rule_id": "PH-M010",
        "record_type": "PHARMACY_CLAIM",
        "category": "Invalid Technical Value",
        "severity": "HIGH",
        "description": "Processing latency must not be negative.",
        "expected_condition": "Processing_Latency_Days >= 0",
        "enabled": True
    },
    "PH-M011": {
        "rule_id": "PH-M011",
        "record_type": "PHARMACY_CLAIM",
        "category": "Invalid Technical Value",
        "severity": "HIGH",
        "description": "Retry count must not be negative.",
        "expected_condition": "Retry_Count >= 0",
        "enabled": True
    },
    "PH-M012": {
        "rule_id": "PH-M012",
        "record_type": "PHARMACY_CLAIM",
        "category": "Chronological Anomaly",
        "severity": "HIGH",
        "description": "Dates must follow standard ordering (Service <= Service_End, Submission >= Service, Processed >= Submission).",
        "expected_condition": "Dates follow standard sequential progression.",
        "enabled": True
    },
    "PH-M013": {
        "rule_id": "PH-M013",
        "record_type": "PHARMACY_CLAIM",
        "category": "SLA Violation",
        "severity": "MEDIUM",
        "description": "Processing latency must not exceed SLA target days.",
        "expected_condition": "Processing_Latency_Days <= SLA_Target_Days",
        "enabled": True
    },
    "PH-M014": {
        "rule_id": "PH-M014",
        "record_type": "PHARMACY_CLAIM",
        "category": "Invalid Pharmacy Metric",
        "severity": "HIGH",
        "description": "Dispensed quantity and days supply must represent a mathematically valid relation.",
        "expected_condition": "Quantity_Dispensed > 0 and Days_Supply > 0",
        "enabled": True
    },

    # ==========================================
    # AUTHORIZATION RULES (AUTH-M001 to AUTH-M009)
    # ==========================================
    "AUTH-M001": {
        "rule_id": "AUTH-M001",
        "record_type": "PRIOR_AUTH",
        "category": "Invalid Technical Value",
        "severity": "HIGH",
        "description": "Retry count must not be negative.",
        "expected_condition": "Retry_Count >= 0",
        "enabled": True
    },
    "AUTH-M002": {
        "rule_id": "AUTH-M002",
        "record_type": "PRIOR_AUTH",
        "category": "Invalid Technical Value",
        "severity": "HIGH",
        "description": "Processing latency must not be negative.",
        "expected_condition": "Processing_Latency_Days >= 0",
        "enabled": True
    },
    "AUTH-M003": {
        "rule_id": "AUTH-M003",
        "record_type": "PRIOR_AUTH",
        "category": "Invalid Technical Value",
        "severity": "HIGH",
        "description": "SLA target days must not be negative.",
        "expected_condition": "SLA_Target_Days >= 0",
        "enabled": True
    },
    "AUTH-M004": {
        "rule_id": "AUTH-M004",
        "record_type": "PRIOR_AUTH",
        "category": "Chronological Anomaly",
        "severity": "HIGH",
        "description": "Processed date must be on or after submission date.",
        "expected_condition": "Processed_Date >= Submission_Date",
        "enabled": True
    },
    "AUTH-M005": {
        "rule_id": "AUTH-M005",
        "record_type": "PRIOR_AUTH",
        "category": "Chronological Anomaly",
        "severity": "HIGH",
        "description": "Decision date must be on or after submission date.",
        "expected_condition": "Decision_Date >= Submission_Date",
        "enabled": True
    },
    "AUTH-M006": {
        "rule_id": "AUTH-M006",
        "record_type": "PRIOR_AUTH",
        "category": "Chronological Anomaly",
        "severity": "HIGH",
        "description": "Decision date must be on or after processed date.",
        "expected_condition": "Decision_Date >= Processed_Date",
        "enabled": True
    },
    "AUTH-M007": {
        "rule_id": "AUTH-M007",
        "record_type": "PRIOR_AUTH",
        "category": "SLA Violation",
        "severity": "MEDIUM",
        "description": "Processing latency must not exceed SLA target days.",
        "expected_condition": "Processing_Latency_Days <= SLA_Target_Days",
        "enabled": True
    },
    "AUTH-M008": {
        "rule_id": "AUTH-M008",
        "record_type": "PRIOR_AUTH",
        "category": "Status/Decision Inconsistency",
        "severity": "MEDIUM",
        "description": "Decision dates and statuses must be logically consistent.",
        "expected_condition": "PENDING has no Decision_Date; APPROVED/DENIED requires valid dates.",
        "enabled": True
    },
    "AUTH-M009": {
        "rule_id": "AUTH-M009",
        "record_type": "PRIOR_AUTH",
        "category": "Linkage Anomaly",
        "severity": "MEDIUM",
        "description": "Authorization link ID must be populated if expected.",
        "expected_condition": "Auth_Linked_ID should be present if expected by workflow.",
        "enabled": False  # Disabled by default as Auth_Linked_ID is structurally unavailable/null for PRIOR_AUTH records
    }
}
