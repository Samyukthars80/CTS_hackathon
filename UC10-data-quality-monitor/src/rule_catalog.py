RULES = [
    {
        "rule_id": "R001",
        "rule_name": "Record_ID Completeness",
        "dimension": "Completeness",
        "severity": "Critical",
        "applicable_record_types": ["MEDICAL_CLAIM", "PHARMACY_CLAIM", "PRIOR_AUTH"],
        "fields": ["Record_ID"],
        "description": "Record_ID must not be missing.",
        "recommended_fix": "Investigate source system extraction logic. All records must have a primary identifier."
    },
    {
        "rule_id": "R002",
        "rule_name": "Record_ID Uniqueness",
        "dimension": "Uniqueness",
        "severity": "Critical",
        "applicable_record_types": ["MEDICAL_CLAIM", "PHARMACY_CLAIM", "PRIOR_AUTH"],
        "fields": ["Record_ID"],
        "description": "Record_ID must be unique.",
        "recommended_fix": "Deduplicate records based on Record_ID or check if upstream systems are sending multiple updates as new records."
    },
    {
        "rule_id": "R003",
        "rule_name": "Beneficiary and Provider Completeness",
        "dimension": "Completeness",
        "severity": "High",
        "applicable_record_types": ["MEDICAL_CLAIM", "PHARMACY_CLAIM", "PRIOR_AUTH"],
        "fields": ["BENE_ID", "Provider_NPI"],
        "description": "BENE_ID and Provider_NPI must be present for every record type.",
        "recommended_fix": "Ensure patient and provider contexts are fully mapped in the data pipeline."
    },
    {
        "rule_id": "R004",
        "rule_name": "Provider_NPI Validity",
        "dimension": "Validity",
        "severity": "High",
        "applicable_record_types": ["MEDICAL_CLAIM", "PHARMACY_CLAIM", "PRIOR_AUTH"],
        "fields": ["Provider_NPI"],
        "description": "Provider_NPI must contain exactly 10 digits when present.",
        "recommended_fix": "Validate NPI format against the National Plan and Provider Enumeration System standard."
    },
    {
        "rule_id": "R005",
        "rule_name": "Record_ID Prefix Consistency",
        "dimension": "Consistency",
        "severity": "High",
        "applicable_record_types": ["MEDICAL_CLAIM", "PHARMACY_CLAIM", "PRIOR_AUTH"],
        "fields": ["Record_ID", "Record_Type"],
        "description": "Record_ID prefix must match Record_Type (MC, PH, PA).",
        "recommended_fix": "Check for ID generation errors or mismatched Record_Type assignments."
    },
    {
        "rule_id": "R006",
        "rule_name": "Source System Consistency",
        "dimension": "Consistency",
        "severity": "High",
        "applicable_record_types": ["MEDICAL_CLAIM", "PHARMACY_CLAIM", "PRIOR_AUTH"],
        "fields": ["Record_Type", "Source_System"],
        "description": "Record_Type must map to the correct Source_System.",
        "recommended_fix": "Correct source system mapping tables in the ETL logic."
    },
    {
        "rule_id": "R007",
        "rule_name": "Medical Claim Core Fields Completeness",
        "dimension": "Completeness",
        "severity": "High",
        "applicable_record_types": ["MEDICAL_CLAIM"],
        "fields": ["Service_Date", "Service_End_Date", "Diagnosis_Code", "Billed_Amount", "Allowed_Amount", "Paid_Amount", "Patient_Responsibility"],
        "description": "MEDICAL_CLAIM requires specific dates, codes, and financial amounts.",
        "recommended_fix": "Verify that all required medical claim fields are extracted from CARRIER_CLAIMS_SYS."
    },
    {
        "rule_id": "R008",
        "rule_name": "Pharmacy Claim Core Fields Completeness",
        "dimension": "Completeness",
        "severity": "High",
        "applicable_record_types": ["PHARMACY_CLAIM"],
        "fields": ["Service_Date", "Service_End_Date", "NDC_Code", "Drug_Name", "Days_Supply", "Quantity_Dispensed", "Billed_Amount", "Allowed_Amount", "Paid_Amount", "Patient_Responsibility"],
        "description": "PHARMACY_CLAIM requires specific dates, drug details, and financial amounts.",
        "recommended_fix": "Verify that all required pharmacy claim fields are extracted from PHARMACY_ADJ_SYS."
    },
    {
        "rule_id": "R009",
        "rule_name": "Prior Auth Core Fields Completeness",
        "dimension": "Completeness",
        "severity": "High",
        "applicable_record_types": ["PRIOR_AUTH"],
        "fields": ["Procedure_Code", "Urgency_Flag", "Processed_Date", "Decision_Date", "Status"],
        "description": "PRIOR_AUTH requires codes and urgency, plus dates if APPROVED or DENIED.",
        "recommended_fix": "Ensure decision dates are captured when prior authorizations leave the PENDING state."
    },
    {
        "rule_id": "R010",
        "rule_name": "Financial Amount Non-Negative Validity",
        "dimension": "Validity",
        "severity": "High",
        "applicable_record_types": ["MEDICAL_CLAIM", "PHARMACY_CLAIM"],
        "fields": ["Billed_Amount", "Allowed_Amount", "Paid_Amount", "Patient_Responsibility"],
        "description": "Financial amounts for claims must not be negative.",
        "recommended_fix": "Review adjustment or reversal logic to ensure final line amounts are non-negative."
    },
    {
        "rule_id": "R011",
        "rule_name": "Pharmacy Supply Validity",
        "dimension": "Validity",
        "severity": "High",
        "applicable_record_types": ["PHARMACY_CLAIM"],
        "fields": ["Days_Supply", "Quantity_Dispensed"],
        "description": "Days_Supply and Quantity_Dispensed must be greater than zero.",
        "recommended_fix": "Investigate pharmacy dispensing data for zero values."
    },
    {
        "rule_id": "R012",
        "rule_name": "Status Validity by Record Type",
        "dimension": "Validity",
        "severity": "High",
        "applicable_record_types": ["MEDICAL_CLAIM", "PHARMACY_CLAIM", "PRIOR_AUTH"],
        "fields": ["Record_Type", "Status"],
        "description": "Status must be a valid value for the given Record_Type.",
        "recommended_fix": "Update allowed value lists or correct status mapping during data ingestion."
    },
    {
        "rule_id": "R013",
        "rule_name": "Denial Reason Completeness",
        "dimension": "Completeness",
        "severity": "Medium",
        "applicable_record_types": ["MEDICAL_CLAIM", "PHARMACY_CLAIM", "PRIOR_AUTH"],
        "fields": ["Status", "Denial_Reason_Code"],
        "description": "DENIED or REJECTED records must have a Denial_Reason_Code.",
        "recommended_fix": "Ensure denial codes are populated whenever a claim or auth is denied."
    },
    {
        "rule_id": "R014",
        "rule_name": "Service Dates Consistency",
        "dimension": "Consistency",
        "severity": "High",
        "applicable_record_types": ["MEDICAL_CLAIM", "PHARMACY_CLAIM"],
        "fields": ["Service_Date", "Service_End_Date"],
        "description": "Service_End_Date must be on or after Service_Date.",
        "recommended_fix": "Fix date entry errors where end date precedes start date."
    },
    {
        "rule_id": "R015",
        "rule_name": "Submission vs Service Date Consistency",
        "dimension": "Consistency",
        "severity": "High",
        "applicable_record_types": ["MEDICAL_CLAIM", "PHARMACY_CLAIM", "PRIOR_AUTH"],
        "fields": ["Service_Date", "Submission_Date"],
        "description": "Submission_Date must not be before Service_Date.",
        "recommended_fix": "Investigate time zone issues or data entry errors causing submissions prior to service."
    },
    {
        "rule_id": "R016",
        "rule_name": "Processed vs Submission Date Consistency",
        "dimension": "Consistency",
        "severity": "Critical",
        "applicable_record_types": ["MEDICAL_CLAIM", "PHARMACY_CLAIM", "PRIOR_AUTH"],
        "fields": ["Submission_Date", "Processed_Date"],
        "description": "Processed_Date must not be before Submission_Date.",
        "recommended_fix": "Investigate system clock sync or ETL latency causing processed date anomalies."
    },
    {
        "rule_id": "R017",
        "rule_name": "Auth Link Consistency",
        "dimension": "Consistency",
        "severity": "Critical",
        "applicable_record_types": ["MEDICAL_CLAIM", "PHARMACY_CLAIM"],
        "fields": ["Auth_Required_Flag", "Auth_Linked_ID"],
        "description": "If Auth_Required_Flag is Y, Auth_Linked_ID must exist.",
        "recommended_fix": "Ensure auth numbers are carried over into claims processing systems."
    },
    {
        "rule_id": "R018",
        "rule_name": "Auth Link Referential Integrity",
        "dimension": "Consistency", # Can use Referential integrity here? We'll map it to Consistency dimension in scoring
        "severity": "High",
        "applicable_record_types": ["MEDICAL_CLAIM", "PHARMACY_CLAIM"],
        "fields": ["Auth_Linked_ID"],
        "description": "If Auth_Linked_ID exists, it must match an existing PRIOR_AUTH record.",
        "recommended_fix": "Check for orphaned claims where the prior authorization is missing from the dataset."
    },
    {
        "rule_id": "R019",
        "rule_name": "SLA Breach Flag Timeliness",
        "dimension": "Timeliness",
        "severity": "High",
        "applicable_record_types": ["MEDICAL_CLAIM", "PHARMACY_CLAIM", "PRIOR_AUTH"],
        "fields": ["Processed_Date", "SLA_Breach_Flag", "Processing_Latency_Days", "SLA_Target_Days"],
        "description": "SLA_Breach_Flag must accurately reflect latency vs target.",
        "recommended_fix": "Fix SLA flag calculation logic in the data warehouse."
    },
    {
        "rule_id": "R020",
        "rule_name": "Volume Trend Consistency",
        "dimension": "Consistency",
        "severity": "Medium",
        "applicable_record_types": ["MEDICAL_CLAIM", "PHARMACY_CLAIM", "PRIOR_AUTH"],
        "fields": ["Batch_Volume", "Rolling_7D_Avg_Volume", "Volume_Vs_Trend_Ratio"],
        "description": "Volume_Vs_Trend_Ratio must equal Batch_Volume / Rolling_7D_Avg_Volume within 0.005 margin.",
        "recommended_fix": "Verify the calculation script for batch volume aggregates."
    },
    {
        "rule_id": "R021",
        "rule_name": "SLA Breach Rate Trend Consistency",
        "dimension": "Consistency",
        "severity": "Medium",
        "applicable_record_types": ["MEDICAL_CLAIM", "PHARMACY_CLAIM", "PRIOR_AUTH"],
        "fields": ["Batch_SLA_Breach_Rate", "Rolling_7D_Avg_SLA_Breach_Rate", "SLA_Breach_Rate_Vs_Trend_Diff"],
        "description": "SLA_Breach_Rate_Vs_Trend_Diff must accurately reflect the difference within 0.0001.",
        "recommended_fix": "Verify the calculation script for SLA breach rate aggregates."
    }
]
