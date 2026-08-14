"""
run_preprocessing.py
--------------------
Executes the full preprocessing pipeline and prints the complete
summary report to console.

Usage:
    python run_preprocessing.py
"""

import os
import sys
import json
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing import run_pipeline, ML_FEATURE_LIST

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT       = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(ROOT, "Data_sets",
             "claims_pharmacy_auth_monitor_dataset_features (1).csv")
OUTPUT_PATH    = os.path.join(ROOT, "claims_cleaned_for_anomaly_detection.csv")
DQ_REPORT_PATH = os.path.join(ROOT, "data_quality_report.csv")

# ── Run ─────────────────────────────────────────────────────────────────────
df_clean, dq_report, summary = run_pipeline(
    input_path=INPUT_PATH,
    output_path=OUTPUT_PATH,
    dq_report_path=DQ_REPORT_PATH,
)

# ── Print summary ────────────────────────────────────────────────────────────
SEP = "=" * 70

print(f"\n{SEP}")
print("  PIPELINE SUMMARY REPORT")
print(SEP)

# Shape
print(f"\n{'── Shape':─<50}")
print(f"  Raw rows × cols    : {summary['total_records_raw']:>7,} × {summary['columns_raw']}")
print(f"  Cleaned rows × cols: {summary['total_records_cleaned']:>7,} × {summary['columns_cleaned']}")
print(f"  Duplicate rows removed            : {summary['full_duplicate_rows_removed']}")
print(f"  Duplicate Record_ID groups flagged: {summary['duplicate_record_id_groups_flagged']}")

# Missingness
print(f"\n{'── Missingness':─<50}")
for k in ['records_with_missing_BENE_ID','records_with_missing_Provider_NPI',
          'records_with_missing_Service_Date','records_with_missing_Billed_Amount']:
    print(f"  {k:<45}: {summary[k]:>6,}")

# Date anomalies
print(f"\n{'── Date Anomalies':─<50}")
for k in ['invalid_service_date_range','submission_before_service',
          'processing_before_submission','decision_before_submission']:
    print(f"  {k:<45}: {summary[k]:>6,}")

# Processing latency
print(f"\n{'── Processing Latency Anomalies':─<50}")
for k in ['negative_processing_latency','extreme_processing_latency_gt30d',
          'processing_latency_iqr_outliers']:
    print(f"  {k:<45}: {summary[k]:>6,}")

# Financial
print(f"\n{'── Financial Anomalies':─<50}")
for k in ['negative_billed_amount','negative_allowed_amount','negative_paid_amount',
          'paid_exceeds_allowed','allowed_exceeds_billed',
          'zero_billed_amount_non_auth','billed_amount_iqr_outliers']:
    print(f"  {k:<45}: {summary[k]:>6,}")

# Authorization
print(f"\n{'── Authorization Anomalies':─<50}")
for k in ['auth_required_but_link_missing','missing_required_auth_link_existing']:
    print(f"  {k:<45}: {summary[k]:>6,}")

# SLA
print(f"\n{'── SLA Anomalies':─<50}")
for k in ['sla_breach_records','sla_unknown_no_date']:
    print(f"  {k:<45}: {summary[k]:>6,}")

# Pipeline
print(f"\n{'── Pipeline Anomalies':─<50}")
for k in ['pipeline_gap_records','extended_pipeline_gap_records']:
    print(f"  {k:<45}: {summary[k]:>6,}")

# Provider
print(f"\n{'── Provider Anomalies':─<50}")
for k in ['high_denial_rate_providers','high_volume_provider_records','provider_volume_iqr_outliers']:
    print(f"  {k:<45}: {summary[k]:>6,}")

# Beneficiary
print(f"\n{'── Beneficiary Anomalies':─<50}")
for k in ['high_frequency_beneficiary_records','extreme_beneficiary_frequency']:
    print(f"  {k:<45}: {summary[k]:>6,}")

# DQ score
print(f"\n{'── Data-Quality Score':─<50}")
for k in ['records_with_any_dq_issue','avg_dq_issue_count','max_dq_issue_count']:
    print(f"  {k:<45}: {summary[k]}")

# Feature guidance
print(f"\n{SEP}")
print("  FEATURE RECOMMENDATIONS FOR ML")
print(SEP)
print(f"\n  USE for ML ({len(ML_FEATURE_LIST['use_for_ml'])} features):")
for f in ML_FEATURE_LIST['use_for_ml']:
    print(f"    ✓  {f}")

print(f"\n  DO NOT USE for ML ({len(ML_FEATURE_LIST['do_not_use_for_ml'])} columns):")
for f in ML_FEATURE_LIST['do_not_use_for_ml']:
    print(f"    ✗  {f}")

# DQ report sample
print(f"\n{SEP}")
print("  DATA-QUALITY REPORT — TOP 10 COLUMNS BY MISSING PCT")
print(SEP)
top10 = dq_report.sort_values("Missing_Pct", ascending=False).head(10)
print(top10[["Column","Data_Type","Missing_Count","Missing_Pct",
             "Anomaly_Count","Cleaning_Action"]].to_string(index=False))

print(f"\n{SEP}")
print(f"  All outputs saved.")
print(f"  → {OUTPUT_PATH}")
print(f"  → {DQ_REPORT_PATH}")
print(SEP + "\n")
