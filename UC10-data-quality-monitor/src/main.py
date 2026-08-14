import pandas as pd
import os
import sys

from profiler import generate_profile
from rule_catalog import RULES
from quality_engine import run_quality_checks
from scoring import calculate_scores_and_risk

def main():
    data_path = "data/claims_pharmacy_auth_monitor_dataset_features.csv"
    
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}")
        print("Please place the CSV file in the 'data/' directory.")
        sys.exit(1)
        
    print(f"Loading data from {data_path}...")
    
    # Read Record_ID, BENE_ID, Provider_NPI, and Auth_Linked_ID as strings
    dtype_spec = {
        "Record_ID": str,
        "BENE_ID": str,
        "Provider_NPI": str,
        "Auth_Linked_ID": str
    }
    
    df = pd.read_csv(data_path, dtype=dtype_spec)
    
    # Parse date columns safely using errors="coerce"
    date_columns = ["Service_Date", "Service_End_Date", "Processed_Date", "Decision_Date", "Submission_Date"]
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            
    print("Generating data profile...")
    profile = generate_profile(df)
    print("Data profile saved to outputs/data_profile.json")
    
    print("Running quality checks...")
    results = run_quality_checks(df, RULES)
    
    print("Calculating scores and risk...")
    report, sla_risk = calculate_scores_and_risk(results, df)
    
    print("Quality report saved to outputs/quality_report.json")
    print("Batch SLA risk saved to outputs/batch_sla_risk.json")
    
    print(f"Overall Quality Score: {report['overall_quality_score']:.2f}")
    print(f"Overall Risk Level: {report['overall_risk_level']}")
    print(f"Batch SLA Risk: {sla_risk['batch_sla_risk_level']}")
    print("Run successful!")

if __name__ == "__main__":
    main()
