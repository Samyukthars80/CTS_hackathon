# Centralized Rule Configurations - Unified and Standardized Rule IDs

RULES_CONFIG = {
    # ==========================================
    # FINANCIAL RULES (R_FIN_001 to R_FIN_007)
    # ==========================================
    "R_FIN_001": {
        "rule_id": "R_FIN_001",
        "category": "Invalid Financial Value",
        "severity": "HIGH",
        "description": "Billed amount must not be negative.",
        "expected_condition": "Billed_Amount >= 0",
        "enabled": True
    },
    "R_FIN_002": {
        "rule_id": "R_FIN_002",
        "category": "Invalid Financial Value",
        "severity": "HIGH",
        "description": "Allowed amount must not be negative.",
        "expected_condition": "Allowed_Amount >= 0",
        "enabled": True
    },
    "R_FIN_003": {
        "rule_id": "R_FIN_003",
        "category": "Invalid Financial Value",
        "severity": "HIGH",
        "description": "Paid amount must not be negative.",
        "expected_condition": "Paid_Amount >= 0",
        "enabled": True
    },
    "R_FIN_004": {
        "rule_id": "R_FIN_004",
        "category": "Invalid Financial Value",
        "severity": "HIGH",
        "description": "Patient responsibility must not be negative.",
        "expected_condition": "Patient_Responsibility >= 0",
        "enabled": True
    },
    "R_FIN_005": {
        "rule_id": "R_FIN_005",
        "category": "Logical Inconsistency",
        "severity": "HIGH",
        "description": "Paid amount must not exceed the allowed amount.",
        "expected_condition": "Paid_Amount <= Allowed_Amount",
        "enabled": True
    },
    "R_FIN_006": {
        "rule_id": "R_FIN_006",
        "category": "Logical Inconsistency",
        "severity": "MEDIUM",
        "description": "Allowed amount must not exceed the billed amount.",
        "expected_condition": "Allowed_Amount <= Billed_Amount",
        "enabled": True
    },
    "R_FIN_007": {
        "rule_id": "R_FIN_007",
        "category": "Logical Inconsistency",
        "severity": "HIGH",
        "description": "Paid amount must not exceed the billed amount.",
        "expected_condition": "Paid_Amount <= Billed_Amount",
        "enabled": True
    },

    # ==========================================
    # CHRONOLOGICAL RULES (R_DATE_001 to R_DATE_005)
    # ==========================================
    "R_DATE_001": {
        "rule_id": "R_DATE_001",
        "category": "Invalid Processing Timeline",
        "severity": "HIGH",
        "description": "Processing date must be on or after submission date, and latency must be non-negative.",
        "expected_condition": "Processed_Date >= Submission_Date AND Processing_Latency_Days >= 0",
        "enabled": True
    },
    "R_DATE_002": {
        "rule_id": "R_DATE_002",
        "category": "Chronological Anomaly",
        "severity": "HIGH",
        "description": "Service date must be on or before service end date.",
        "expected_condition": "Service_Date <= Service_End_Date",
        "enabled": True
    },
    "R_DATE_003": {
        "rule_id": "R_DATE_003",
        "category": "Chronological Anomaly",
        "severity": "MEDIUM",
        "description": "Submission date must be on or after service date.",
        "expected_condition": "Submission_Date >= Service_Date",
        "enabled": True
    },
    "R_DATE_004": {
        "rule_id": "R_DATE_004",
        "category": "Chronological Anomaly",
        "severity": "HIGH",
        "description": "Decision date must be on or after submission date.",
        "expected_condition": "Decision_Date >= Submission_Date",
        "enabled": True
    },
    "R_DATE_005": {
        "rule_id": "R_DATE_005",
        "category": "Chronological Anomaly",
        "severity": "HIGH",
        "description": "Decision date must be on or after processed date.",
        "expected_condition": "Decision_Date >= Processed_Date",
        "enabled": True
    },

    # ==========================================
    # TECHNICAL RULES (R_TECH_001 to R_TECH_002)
    # ==========================================
    "R_TECH_001": {
        "rule_id": "R_TECH_001",
        "category": "Invalid Technical Value",
        "severity": "HIGH",
        "description": "Retry count must not be negative.",
        "expected_condition": "Retry_Count >= 0",
        "enabled": True
    },
    "R_TECH_002": {
        "rule_id": "R_TECH_002",
        "category": "Invalid Technical Value",
        "severity": "HIGH",
        "description": "SLA target days must not be negative.",
        "expected_condition": "SLA_Target_Days >= 0",
        "enabled": True
    },

    # ==========================================
    # PHARMACY METRIC RULES (R_MET_001 to R_MET_003)
    # ==========================================
    "R_MET_001": {
        "rule_id": "R_MET_001",
        "category": "Invalid Pharmacy Metric",
        "severity": "HIGH",
        "description": "Days supply must be positive.",
        "expected_condition": "Days_Supply > 0",
        "enabled": True
    },
    "R_MET_002": {
        "rule_id": "R_MET_002",
        "category": "Invalid Pharmacy Metric",
        "severity": "HIGH",
        "description": "Quantity dispensed must be positive.",
        "expected_condition": "Quantity_Dispensed > 0",
        "enabled": True
    },
    "R_MET_003": {
        "rule_id": "R_MET_003",
        "category": "Invalid Pharmacy Metric",
        "severity": "HIGH",
        "description": "Dispensing quantity and days supply must both be populated.",
        "expected_condition": "Quantity_Dispensed and Days_Supply are not null",
        "enabled": True
    },

    # ==========================================
    # SLA RULES (R_SLA_001)
    # ==========================================
    "R_SLA_001": {
        "rule_id": "R_SLA_001",
        "category": "SLA Violation",
        "severity": "MEDIUM",
        "description": "Processing latency must not exceed SLA target days.",
        "expected_condition": "Processing_Latency_Days <= SLA_Target_Days",
        "enabled": True
    },

    # ==========================================
    # CATEGORICAL RULES (R_CAT_001 to R_CAT_002)
    # ==========================================
    "R_CAT_001": {
        "rule_id": "R_CAT_001",
        "category": "Invalid Categorical Value",
        "severity": "MEDIUM",
        "description": "Status value must be standard for the record type.",
        "expected_condition": "Status is standard",
        "enabled": True
    },
    "R_CAT_002": {
        "rule_id": "R_CAT_002",
        "category": "Status/Date Inconsistency",
        "severity": "MEDIUM",
        "description": "Dates must be consistent with record status.",
        "expected_condition": "PENDING claims require missing Processed_Date; completed claims require valid dates.",
        "enabled": True
    }
}
