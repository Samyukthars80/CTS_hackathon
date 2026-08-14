from __future__ import annotations

import json
import pickle
import pandas as pd
import numpy as np

from ml_anomaly_detection.config import (
    DATASETS, MODELS_DIR, REPORTS_DIR
)
from ml_anomaly_detection.medical_model import train_medical_model
from ml_anomaly_detection.pharmacy_model import train_pharmacy_model
from ml_anomaly_detection.prior_auth_model import train_prior_auth_model


def main():
    print("="*60)
    print("STARTING ML ANOMALY DETECTION MODEL TRAINING PIPELINE")
    print("="*60)
    
    # 1. Run individual model training
    med_info = train_medical_model()
    pharm_info = train_pharmacy_model()
    auth_info = train_prior_auth_model()
    
    # 2. Collect statistics by scoring the full datasets
    summary_rows = []
    
    for info in [med_info, pharm_info, auth_info]:
        rtype = info["record_type"]
        print(f"\nScoring full dataset for {rtype} summary statistics...")
        
        # Load full cleaned dataset
        df = pd.read_csv(DATASETS[rtype])
        
        # Load fitted preprocessor and model
        model_name_map = {
            "MEDICAL_CLAIM": "medical",
            "PHARMACY_CLAIM": "pharmacy",
            "PRIOR_AUTH": "prior_auth"
        }
        prefix = model_name_map[rtype]
        
        with open(MODELS_DIR / f"{prefix}_isolation_forest.pkl", "rb") as f:
            model = pickle.load(f)
        with open(MODELS_DIR / f"{prefix}_preprocessor.pkl", "rb") as f:
            prep = pickle.load(f)
            
        # Transform full dataset
        X = prep.transform(df)
        
        # Predict on full dataset
        preds = model.predict(X)
        scores = -model.decision_function(X) # Higher score is more anomalous
        
        total_anomalies = (preds == -1).sum()
        anomaly_pct = (total_anomalies / len(df)) * 100
        
        summary_rows.append({
            "Record_Type": rtype,
            "Training_Rows": info["train_rows"],
            "Test_Rows": info["test_rows"],
            "Number_of_Features": info["num_features"],
            "Contamination": info["contamination"],
            "Total_Anomalies": int(total_anomalies),
            "Anomaly_Percentage": round(anomaly_pct, 2),
            "Median_Anomaly_Score": round(float(np.median(scores)), 4),
            "Minimum_Anomaly_Score": round(float(np.min(scores)), 4),
            "Maximum_Anomaly_Score": round(float(np.max(scores)), 4)
        })
        
    # Save ml_model_summary.csv
    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = REPORTS_DIR / "ml_model_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"\nSaved ML model summary CSV to: {summary_csv_path}")
    print(summary_df.to_string(index=False))
    
    print("\nModel training phase completed successfully!")


if __name__ == "__main__":
    main()
