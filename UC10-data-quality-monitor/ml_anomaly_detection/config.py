import os
from pathlib import Path

# Paths
ML_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = ML_DIR.parent
CLEANED_DIR = WORKSPACE_DIR / "outputs"
MODELS_DIR = ML_DIR / "models"
OUTPUTS_DIR = ML_DIR / "outputs"
REPORTS_DIR = ML_DIR / "reports"

# Ensure dirs exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "MEDICAL_CLAIM": CLEANED_DIR / "medical_claim_cleaned.csv",
    "PHARMACY_CLAIM": CLEANED_DIR / "pharmacy_claim_cleaned.csv",
    "PRIOR_AUTH": CLEANED_DIR / "authorization_cleaned.csv"
}

# Features config
FEATURES = {
    "MEDICAL_CLAIM": {
        "numerical": [
            "Billed_Amount",
            "Allowed_Amount",
            "Paid_Amount",
            "Patient_Responsibility",
            "Retry_Count",
            "Processing_Latency_Days",
            "SLA_Target_Days",
            "Beneficiary_Record_Count",
            "Provider_Total_Records",
            "Provider_Denial_Rate"
        ],
        "categorical": [
            "Provider_State",
            "Status",
            "Urgency_Flag"
        ],
        "ratios": [
            "Paid_to_Allowed_Ratio",
            "Paid_to_Billed_Ratio",
            "Allowed_to_Billed_Ratio"
        ]
    },
    "PHARMACY_CLAIM": {
        "numerical": [
            "Billed_Amount",
            "Allowed_Amount",
            "Paid_Amount",
            "Patient_Responsibility",
            "Retry_Count",
            "Processing_Latency_Days",
            "SLA_Target_Days",
            "Days_Supply",
            "Quantity_Dispensed",
            "Beneficiary_Record_Count",
            "Provider_Total_Records",
            "Provider_Denial_Rate"
        ],
        "categorical": [
            "Provider_State",
            "Status"
        ],
        "ratios": [
            "Paid_to_Allowed_Ratio",
            "Paid_to_Billed_Ratio",
            "Allowed_to_Billed_Ratio",
            "Quantity_to_Days_Ratio"
        ]
    },
    "PRIOR_AUTH": {
        "numerical": [
            "Retry_Count",
            "Processing_Latency_Days",
            "SLA_Target_Days",
            "Provider_Total_Records",
            "Provider_Denial_Rate",
            "Beneficiary_Record_Count"
        ],
        "categorical": [
            "Provider_State",
            "Status",
            "Urgency_Flag"
        ],
        "ratios": []
    }
}

# Model Settings
RANDOM_STATE = 42
N_ESTIMATORS = 300
CONTAMINATION_VALUES = [0.01, 0.02, 0.05]
DEFAULT_CONTAMINATION = 0.02
