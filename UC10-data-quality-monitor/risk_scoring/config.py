import os
from pathlib import Path

# Paths
RISK_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = RISK_DIR.parent
CLEANED_DIR = WORKSPACE_DIR / "outputs"
ML_DIR = WORKSPACE_DIR / "ml_anomaly_detection"
RULE_DIR = WORKSPACE_DIR / "anomaly_detection"

# Input paths
CONSENSUS_PATH = ML_DIR / "model_validation" / "cross_method_consensus.csv"
RULE_SUMMARY_PATH = RULE_DIR / "outputs" / "record_level_summary.csv"
RULE_DETAILED_PATH = RULE_DIR / "outputs" / "combined_rule_anomalies.csv"

DATASETS = {
    "MEDICAL_CLAIM": CLEANED_DIR / "medical_claim_cleaned.csv",
    "PHARMACY_CLAIM": CLEANED_DIR / "pharmacy_claim_cleaned.csv",
    "PRIOR_AUTH": CLEANED_DIR / "authorization_cleaned.csv"
}

ML_PREDICTIONS = {
    "MEDICAL_CLAIM": ML_DIR / "outputs" / "medical_ml_predictions.csv",
    "PHARMACY_CLAIM": ML_DIR / "outputs" / "pharmacy_ml_predictions.csv",
    "PRIOR_AUTH": ML_DIR / "outputs" / "prior_auth_ml_predictions.csv"
}

ML_ANOMALIES = {
    "MEDICAL_CLAIM": ML_DIR / "outputs" / "medical_ml_anomalies.csv",
    "PHARMACY_CLAIM": ML_DIR / "outputs" / "pharmacy_ml_anomalies.csv",
    "PRIOR_AUTH": ML_DIR / "outputs" / "prior_auth_ml_anomalies.csv"
}

# Output paths
RECORD_RISK_SCORES_PATH = RISK_DIR / "record_risk_scores.csv"
INVESTIGATION_QUEUE_PATH = RISK_DIR / "investigation_queue.csv"
PROVIDER_RISK_SUMMARY_PATH = RISK_DIR / "provider_risk_summary.csv"
VALIDATION_REPORT_PATH = RISK_DIR / "risk_scoring_validation.csv"
REPORT_MD_PATH = RISK_DIR / "RISK_SCORING_REPORT.md"

# Weightings (overall score components)
WEIGHT_RULE = 0.40
WEIGHT_STATISTICAL = 0.25
WEIGHT_ML = 0.25
WEIGHT_CONSENSUS = 0.10

# Numerical columns for statistical deviation calculation
NUMERICAL_COLS = {
    "MEDICAL_CLAIM": ["Billed_Amount", "Allowed_Amount", "Paid_Amount", "Patient_Responsibility", "Retry_Count", "Processing_Latency_Days", "SLA_Target_Days", "Beneficiary_Record_Count", "Provider_Total_Records", "Provider_Denial_Rate"],
    "PHARMACY_CLAIM": ["Billed_Amount", "Allowed_Amount", "Paid_Amount", "Patient_Responsibility", "Retry_Count", "Processing_Latency_Days", "SLA_Target_Days", "Days_Supply", "Quantity_Dispensed", "Beneficiary_Record_Count", "Provider_Total_Records", "Provider_Denial_Rate"],
    "PRIOR_AUTH": ["Retry_Count", "Processing_Latency_Days", "SLA_Target_Days", "Provider_Total_Records", "Provider_Denial_Rate", "Beneficiary_Record_Count"]
}
