"""
test_pipeline.py
----------------
Standalone verification harness for the Data-Quality Anomaly Monitor pipeline.

Usage:  python test_pipeline.py
"""

import os, sys, warnings
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from backend.db_manager  import init_db, get_dental_only_issuer_ids, get_issuer_valid_states
from backend.rule_engine import evaluate_claims
from backend.ml_engine   import train_and_score

# ── Synthetic claims ────────────────────────────────────────────────────────
CLAIM_A = {   # Normal clean claim
    "DESYNPUF_ID": "BENE_001", "CLM_ID": "CLM_A001",
    "CLM_FROM_DT": 20240101,   "CLM_THRU_DT": 20240105,
    "CLM_PMT_AMT": 8500.00,    "BENRES_IP": 1340.00,   "PPPYMT_IP": 0.00,
    "CLM_DRG_CD":  None,       "ICD9_PRCDR_CD_1": "4019",
    "StateCode":   "AL",       "IssuerId": "46944",
}
CLAIM_B = {   # Network mis-routing – dental issuer + inpatient DRG + wrong state
    "DESYNPUF_ID": "BENE_002", "CLM_ID": "CLM_B002",
    "CLM_FROM_DT": 20240210,   "CLM_THRU_DT": 20240215,
    "CLM_PMT_AMT": 22000.00,   "BENRES_IP": 1340.00,   "PPPYMT_IP": 0.00,
    "CLM_DRG_CD":  "470",      "ICD9_PRCDR_CD_1": "8154",
    "StateCode":   "TX",       "IssuerId": "21989",
}
CLAIM_C = {   # Financial outlier – extreme billing spike + bad date
    "DESYNPUF_ID": "BENE_003", "CLM_ID": "CLM_C003",
    "CLM_FROM_DT": 20240301,   "CLM_THRU_DT": 20240345,  # invalid date
    "CLM_PMT_AMT": 985000.00,  "BENRES_IP": 0.01,         "PPPYMT_IP": 0.00,
    "CLM_DRG_CD":  None,       "ICD9_PRCDR_CD_1": None,
    "StateCode":   "AL",       "IssuerId": "46944",
}


def print_report(df: pd.DataFrame) -> None:
    SEP = "=" * 110
    display = df[["CLM_ID","StateCode","IssuerId",
                  "rule_violations","anomaly_risk_score","is_anomaly"]].copy()
    display.columns = ["Claim ID","State","Issuer ID",
                       "Rule Violations","ML Anomaly Score","Is Anomaly"]
    pd.set_option("display.max_colwidth", 55)
    pd.set_option("display.width", 160)
    print(f"\n{SEP}")
    print("  UC10 – Claims & Authorization Data-Quality Anomaly Monitor  |  Pipeline Report")
    print(SEP)
    print(display.to_string(index=False))
    print(SEP)
    for _, row in df.iterrows():
        print(f"\n  ► {row['CLM_ID']}")
        print(f"      Rule Detail  : {row.get('rule_detail','—')}")
        print(f"      Anomaly Score: {row.get('anomaly_risk_score','N/A')}  "
              f"|  Is Anomaly: {row.get('is_anomaly','N/A')}")
    print()


def main():
    print("\n" + "="*60)
    print("  UC10 Anomaly Monitor – Pipeline Verification")
    print("="*60)

    print("\n[Step 1] Initialising DuckDB ...")
    conn = init_db()

    print("[Step 2] Loading network metadata ...")
    dental_only_ids    = get_dental_only_issuer_ids(conn)
    issuer_valid_states = get_issuer_valid_states(conn)
    print(f"         Dental-only issuers: {len(dental_only_ids)}")
    print(f"         Issuers with jurisdiction map: {len(issuer_valid_states)}")

    print("[Step 3] Creating synthetic claim records ...")
    claims_df = pd.DataFrame([CLAIM_A, CLAIM_B, CLAIM_C])

    print("[Step 4] Running Rule Engine ...")
    claims_with_rules = evaluate_claims(claims_df, dental_only_ids, issuer_valid_states)
    violations = claims_with_rules[claims_with_rules["rule_violations"] != "None"].shape[0]
    print(f"         Rule violations in {violations}/3 claims.")

    print("[Step 5] Running ML Anomaly Engine ...")
    final_df = train_and_score(claims_with_rules)
    flagged  = final_df["is_anomaly"].sum()
    print(f"         ML anomalies flagged: {flagged}/3 claims.")

    print_report(final_df)
    conn.close()
    print("[Done] Pipeline verification complete.\n")


if __name__ == "__main__":
    main()
