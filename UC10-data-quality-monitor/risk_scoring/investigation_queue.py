from __future__ import annotations

from pathlib import Path
import pandas as pd

# Setup paths
RISK_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = RISK_DIR / "outputs"


def generate_investigation_queue():
    print("\nGenerating prioritized investigation queue...")
    
    # 1. Load record risk scores
    scores_path = OUTPUTS_DIR / "record_risk_scores.csv"
    if not scores_path.exists():
        raise FileNotFoundError(f"Consolidated record risk scores not found: {scores_path}")
    df = pd.read_csv(scores_path)
    
    # 2. Sort hierarchically:
    # Priority ascending (1 is CRITICAL, 4 is LOW)
    # Risk_Score descending
    # Consensus_Level descending
    df_sorted = df.sort_values(
        by=["Priority", "Risk_Score", "Consensus_Level"],
        ascending=[True, False, False]
    ).reset_index(drop=True)
    
    # 3. Create Rank
    df_sorted.insert(0, "Queue_Rank", df_sorted.index + 1)
    
    # 4. Map columns to match requested output schema
    queue_cols = {
        "Queue_Rank": "Queue_Rank",
        "Record_ID": "Record_ID",
        "Record_Type": "Record_Type",
        "Risk_Score": "Risk_Score",
        "Risk_Level": "Risk_Level",
        "Priority": "Priority",
        "Consensus_Level": "Consensus_Level",
        "ML_Anomaly_Flag": "ML_Flag",
        "Rule_Anomaly_Flag": "Rule_Flag",
        "Statistical_Anomaly_Flag": "Statistical_Flag",
        "Rule_Violation_Count": "Rule_Violation_Count",
        "High_Severity_Rule_Count": "High_Severity_Count",
        "Primary_Risk_Factors": "Primary_Risk_Factors",
        "Explanation": "Explanation"
    }
    
    # Check that columns exist
    df_queue = df_sorted[list(queue_cols.keys())].rename(columns=queue_cols)
    
    queue_path = OUTPUTS_DIR / "investigation_queue.csv"
    df_queue.to_csv(queue_path, index=False)
    print(f"Saved ranked investigation queue to: {queue_path}")
    print(f"Total queued records: {len(df_queue)}")
    
    # Print statistics of the queue
    print("\nQueue count by Risk Level:")
    print(df_queue["Risk_Level"].value_counts().to_string())


if __name__ == "__main__":
    generate_investigation_queue()
