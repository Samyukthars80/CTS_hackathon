from __future__ import annotations

import json
import pickle
import pandas as pd
from sklearn.ensemble import IsolationForest

from ml_anomaly_detection.config import (
    DATASETS, FEATURES, RANDOM_STATE, N_ESTIMATORS, DEFAULT_CONTAMINATION,
    MODELS_DIR, CONTAMINATION_VALUES
)
from ml_anomaly_detection.preprocessing import ClaimsPreprocessor


def train_medical_model() -> dict:
    print("\n" + "="*50)
    print("TRAINING MEDICAL CLAIMS ANOMALY MODEL")
    print("="*50)
    
    # 1. Load cleaned data
    file_path = DATASETS["MEDICAL_CLAIM"]
    if not file_path.exists():
        raise FileNotFoundError(f"Cleaned medical claim file not found: {file_path}")
        
    df = pd.read_csv(file_path)
    print(f"Loaded Medical Claims. Shape: {df.shape}")
    
    # Verify required identifiers exist
    assert "Record_ID" in df.columns, "Record_ID column is missing"
    assert "Record_Type" in df.columns, "Record_Type column is missing"
    
    # 2. Sort chronologically and split
    df = df.sort_values(by="Submission_Date").reset_index(drop=True)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    print(f"Chronological Split: {len(train_df)} train rows, {len(test_df)} test rows")
    
    # 3. Initialize Preprocessor
    feat_cfg = FEATURES["MEDICAL_CLAIM"]
    preprocessor = ClaimsPreprocessor(
        numerical_features=feat_cfg["numerical"],
        categorical_features=feat_cfg["categorical"],
        ratio_features=feat_cfg["ratios"]
    )
    
    # Fit preprocessor on training split
    preprocessor.fit(train_df)
    
    # Transform train and test splits
    X_train = preprocessor.transform(train_df)
    X_test = preprocessor.transform(test_df)
    
    print(f"Preprocessed features: {preprocessor.final_feature_names_}")
    print(f"X_train matrix shape: {X_train.shape}")
    print(f"X_test matrix shape: {X_test.shape}")
    
    # 4. Contamination Grid Search comparison
    print("\nComparing Contamination rates:")
    grid_results = {}
    for cont in CONTAMINATION_VALUES:
        iso_forest = IsolationForest(
            n_estimators=N_ESTIMATORS,
            contamination=cont,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        iso_forest.fit(X_train)
        
        preds_train = iso_forest.predict(X_train)
        preds_test = iso_forest.predict(X_test)
        
        # prediction = -1 is anomaly
        anoms_train = (preds_train == -1).sum()
        anoms_test = (preds_test == -1).sum()
        
        print(f"  Contamination: {cont:.2f} -> Train Anomalies: {anoms_train} ({anoms_train/len(X_train)*100:.1f}%), Test Anomalies: {anoms_test} ({anoms_test/len(X_test)*100:.1f}%)")
        grid_results[cont] = {
            "train_anomalies": int(anoms_train),
            "test_anomalies": int(anoms_test)
        }
        
    # 5. Fit model with default contamination
    print(f"\nFitting final Isolation Forest with contamination: {DEFAULT_CONTAMINATION}")
    final_model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=DEFAULT_CONTAMINATION,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    final_model.fit(X_train)
    
    # 6. Save model artifacts
    model_path = MODELS_DIR / "medical_isolation_forest.pkl"
    prep_path = MODELS_DIR / "medical_preprocessor.pkl"
    feat_path = MODELS_DIR / "medical_features.json"
    
    with open(model_path, "wb") as f:
        pickle.dump(final_model, f)
    with open(prep_path, "wb") as f:
        pickle.dump(preprocessor, f)
    with open(feat_path, "w") as f:
        json.dump(preprocessor.final_feature_names_, f, indent=4)
        
    print(f"Saved model to: {model_path}")
    print(f"Saved preprocessor to: {prep_path}")
    print(f"Saved feature list to: {feat_path}")
    
    return {
        "record_type": "MEDICAL_CLAIM",
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "num_features": len(preprocessor.final_feature_names_),
        "contamination": DEFAULT_CONTAMINATION,
        "features": preprocessor.final_feature_names_
    }


if __name__ == "__main__":
    train_medical_model()
