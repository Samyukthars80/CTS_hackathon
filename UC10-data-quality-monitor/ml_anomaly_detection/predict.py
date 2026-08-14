from __future__ import annotations

import json
import pickle
import pandas as pd
import numpy as np

from ml_anomaly_detection.config import (
    DATASETS, MODELS_DIR, OUTPUTS_DIR
)


def explain_anomalies(df: pd.DataFrame, X_scaled: pd.DataFrame, preprocessor) -> list[str]:
    """
    Identify potential contributing features for anomalous records.
    Finds preprocessed features where the scaled absolute value is > 2.0 (Z-score limit).
    """
    explanations = []
    
    # We map preprocessed features back to original columns where helpful
    for idx in range(len(df)):
        row_scaled = X_scaled.iloc[idx]
        row_orig = df.iloc[idx]
        
        # Sort features by absolute scaled value (Z-score deviation)
        devs = []
        for col in X_scaled.columns:
            # Skip missingness indicators themselves to focus on the main variables
            if col.endswith("_is_missing"):
                continue
            val_scaled = float(row_scaled[col])
            devs.append((col, val_scaled, abs(val_scaled)))
            
        devs = sorted(devs, key=lambda x: x[2], reverse=True)
        
        # Select features with Z-score deviation > 2.0
        strong_devs = [d for d in devs if d[2] > 2.0]
        
        if not strong_devs:
            # If no features exceed 2.0, take the single top deviating feature
            strong_devs = devs[:1]
            
        factors = []
        for col, val_scaled, _ in strong_devs[:3]:
            # Try to get original value
            orig_col = col
            # If it is a derived ratio, print ratio value
            if orig_col in row_orig:
                orig_val = row_orig[orig_col]
                if isinstance(orig_val, float):
                    factors.append(f"{orig_col} ({orig_val:.2f}) is unusual")
                else:
                    factors.append(f"{orig_col} ({orig_val}) is unusual")
            elif orig_col in row_scaled:
                # If it's a preprocessed ratio not in the original file
                factors.append(f"{orig_col} is unusual (scaled value: {val_scaled:.2f})")
                
        explanations.append("; ".join(factors))
        
    return explanations


def run_prediction_pipeline():
    print("\n" + "="*50)
    print("RUNNING ANOMALY DETECTION PREDICTIONS PIPELINE")
    print("="*50)
    
    model_name_map = {
        "MEDICAL_CLAIM": "medical",
        "PHARMACY_CLAIM": "pharmacy",
        "PRIOR_AUTH": "prior_auth"
    }
    
    for rtype, file_path in DATASETS.items():
        print(f"\nProcessing {rtype}...")
        
        prefix = model_name_map[rtype]
        
        # Load cleaned data
        df = pd.read_csv(file_path)
        
        # Load artifacts
        with open(MODELS_DIR / f"{prefix}_isolation_forest.pkl", "rb") as f:
            model = pickle.load(f)
        with open(MODELS_DIR / f"{prefix}_preprocessor.pkl", "rb") as f:
            preprocessor = pickle.load(f)
            
        # Preprocess full dataset
        X = preprocessor.transform(df)
        
        # Prediction and Anomaly Scores
        preds = model.predict(X)
        scores = -model.decision_function(X) # Higher score is more anomalous
        
        # Percentile and Ranks
        # Percentile rank (0 to 100) where 100 means most anomalous
        percentiles = pd.Series(scores).rank(pct=True) * 100
        ranks = pd.Series(scores).rank(ascending=False, method="first")
        
        # Build predictions dataframe for ALL records
        all_df = pd.DataFrame({
            "Record_ID": df["Record_ID"],
            "Record_Type": df["Record_Type"] if "Record_Type" in df.columns else rtype,
            "Isolation_Forest_Score": scores,
            "ML_Anomaly_Flag": (preds == -1).astype(int),
            "Anomaly_Percentile": percentiles
        })
        
        # Build anomalies-only dataframe
        anoms_mask = preds == -1
        anoms_df_raw = df[anoms_mask].copy()
        
        anoms_X_scaled = X[anoms_mask].copy()
        
        # Generate explanations for anomalies
        print(f"Generating explanations for {anoms_mask.sum()} anomalies...")
        explanations = explain_anomalies(anoms_df_raw, anoms_X_scaled, preprocessor)
        
        # Severity calculation based on percentile:
        # Top 1% of ALL claims (percentile >= 99) -> High
        # Top 1-5% of ALL claims (percentile >= 95 and < 99) -> Medium
        # Remaining flagged anomalies -> Low
        severities = []
        for pct in percentiles[anoms_mask]:
            if pct >= 99.0:
                severities.append("High")
            elif pct >= 95.0:
                severities.append("Medium")
            else:
                severities.append("Low")
                
        anoms_df = pd.DataFrame({
            "Record_ID": anoms_df_raw["Record_ID"],
            "Record_Type": anoms_df_raw["Record_Type"] if "Record_Type" in anoms_df_raw.columns else rtype,
            "Detection_Method": "Isolation Forest",
            "ML_Anomaly_Flag": 1,
            "Isolation_Forest_Score": scores[anoms_mask],
            "Anomaly_Rank": ranks[anoms_mask].astype(int),
            "Severity": severities,
            "Model_Name": f"{prefix}_isolation_forest",
            "Potential_Contributing_Factors": explanations
        })
        
        # Save output reports
        pred_path = OUTPUTS_DIR / f"{prefix}_ml_predictions.csv"
        anom_path = OUTPUTS_DIR / f"{prefix}_ml_anomalies.csv"
        
        all_df.to_csv(pred_path, index=False)
        anoms_df.to_csv(anom_path, index=False)
        
        print(f"Saved full predictions dataset to: {pred_path}")
        print(f"Saved anomalies-only report to: {anom_path}")


if __name__ == "__main__":
    run_prediction_pipeline()
