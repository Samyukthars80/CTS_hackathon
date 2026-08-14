"""
preprocessing.py
================
Production-quality data cleaning and preprocessing pipeline for:
    UC10 – Claims & Authorization Data-Quality Anomaly Monitor

Design principles
-----------------
- NEVER blindly delete anomalous records — flag them.
- NEVER use blanket fillna(0).
- NEVER fabricate IDs.
- NEVER remove outliers — create outlier flags.
- Handle MEDICAL_CLAIM / PHARMACY_CLAIM / PRIOR_AUTH distinctly.
- Preserve ALL existing engineered features.
- Output is ready for ML anomaly detection.

Reproducible entry-point
------------------------
    from preprocessing import run_pipeline
    df_clean, dq_report, summary = run_pipeline(input_path, output_path)
"""

from __future__ import annotations

import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

DATE_COLS = [
    "Service_Date",
    "Service_End_Date",
    "Submission_Date",
    "Batch_Date",
]

DATETIME_COLS = [
    "Processed_Date",
    "Decision_Date",
    "Ingestion_Timestamp",
]

# Identifier columns — must NOT be used as raw numerical ML features
IDENTIFIER_COLS = [
    "Record_ID",
    "BENE_ID",
    "Provider_NPI",
    "Auth_Linked_ID",
    "Batch_ID",
]

# Existing engineered features to unconditionally preserve
EXISTING_ENGINEERED_FEATURES = [
    "Provider_Total_Records",
    "Provider_Denial_Rate",
    "Batch_Volume",
    "Rolling_7D_Avg_Volume",
    "Volume_Vs_Trend_Ratio",
    "Batch_SLA_Breach_Rate",
    "Rolling_7D_Avg_SLA_Breach_Rate",
    "SLA_Breach_Rate_Vs_Trend_Diff",
    "Beneficiary_Record_Count",
    "High_Frequency_Beneficiary_Flag",
    "Missing_Required_Auth_Link",
    "Days_Since_Prev_Batch",
    "Pipeline_Gap_Flag",
    "Submission_Day_Of_Week",
    "DOW_Avg_SLA_Breach_Rate",
    "Record_SLA_Breach_Numeric",
    "SLA_Breach_Vs_DOW_Norm",
]

# Financial columns relevant to claims (not PRIOR_AUTH)
FINANCIAL_COLS = [
    "Billed_Amount",
    "Allowed_Amount",
    "Paid_Amount",
    "Patient_Responsibility",
]

# Fields required by record type (used for missingness indicator generation)
REQUIRED_BY_TYPE: dict[str, list[str]] = {
    "MEDICAL_CLAIM":  ["BENE_ID", "Provider_NPI", "Service_Date", "Diagnosis_Code", "Billed_Amount"],
    "PHARMACY_CLAIM": ["BENE_ID", "Provider_NPI", "Service_Date", "NDC_Code", "Drug_Name", "Billed_Amount"],
    "PRIOR_AUTH":     ["BENE_ID", "Provider_NPI", "Procedure_Code", "Auth_Required_Flag"],
}


# ──────────────────────────────────────────────────────────────────────────────
# STEP 1  —  LOAD
# ──────────────────────────────────────────────────────────────────────────────

def load_raw(path: str) -> pd.DataFrame:
    """Load CSV without any transformation.  Returns the raw DataFrame."""
    df = pd.read_csv(path, low_memory=False)
    print(f"[load]  Loaded {len(df):,} rows × {df.shape[1]} cols from '{os.path.basename(path)}'")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# STEP 2  —  DUPLICATE REMOVAL
# ──────────────────────────────────────────────────────────────────────────────

def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Remove only TRUE duplicate rows (all columns identical).
    Duplicate Record_IDs with different content are FLAGGED, not removed.
    """
    before = len(df)

    # 1. Fully identical rows
    full_dups = df.duplicated()
    df = df[~full_dups].copy()
    full_dup_count = full_dups.sum()

    # 2. Duplicate Record_IDs with differing content → flag, keep both
    dup_ids = df[df.duplicated(subset=["Record_ID"], keep=False)]["Record_ID"].nunique()
    df["Duplicate_Record_ID_Flag"] = df.duplicated(subset=["Record_ID"], keep=False)

    after = len(df)
    info = {
        "full_duplicate_rows_removed": int(full_dup_count),
        "duplicate_record_id_groups_flagged": int(dup_ids),
        "rows_before": before,
        "rows_after": after,
    }
    print(f"[dedup] Removed {full_dup_count} fully-identical rows. "
          f"{dup_ids} Record_ID groups flagged as duplicates.")
    return df, info


# ──────────────────────────────────────────────────────────────────────────────
# STEP 3  —  DATE / DATETIME PARSING
# ──────────────────────────────────────────────────────────────────────────────

def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse date/datetime columns with errors='coerce'.
    Creates a corresponding _ParseFail indicator for any column that has
    at least one failed conversion.
    Does NOT delete rows with failed dates.
    """
    for col in DATE_COLS:
        if col in df.columns:
            parsed = pd.to_datetime(df[col], errors="coerce", format="%Y-%m-%d")
            fail_count = parsed.isna().sum() - df[col].isna().sum()
            fail_count = max(fail_count, 0)
            if fail_count > 0:
                df[f"{col}_ParseFail"] = parsed.isna() & df[col].notna()
            df[col] = parsed

    for col in DATETIME_COLS:
        if col in df.columns:
            parsed = pd.to_datetime(df[col], errors="coerce")
            fail_count = parsed.isna().sum() - df[col].isna().sum()
            fail_count = max(fail_count, 0)
            if fail_count > 0:
                df[f"{col}_ParseFail"] = parsed.isna() & df[col].notna()
            df[col] = parsed

    print("[dates] Date/datetime columns parsed with errors='coerce'.")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# STEP 4  —  MISSINGNESS INDICATORS
# ──────────────────────────────────────────────────────────────────────────────

def create_missingness_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary missing-value flags for key fields.
    Does NOT impute or delete.
    """
    miss_map = {
        "Missing_BENE_ID_Flag":               "BENE_ID",
        "Missing_Provider_NPI_Flag":          "Provider_NPI",
        "Missing_Processing_Latency_Flag":    "Processing_Latency_Days",
        "Missing_Service_Date_Flag":          "Service_Date",
        "Missing_Billed_Amount_Flag":         "Billed_Amount",
        "Missing_Diagnosis_Code_Flag":        "Diagnosis_Code",
        "Missing_Procedure_Code_Flag":        "Procedure_Code",
        "Missing_NDC_Code_Flag":              "NDC_Code",
        "Missing_Decision_Date_Flag":         "Decision_Date",
        "Missing_Processed_Date_Flag":        "Processed_Date",
        "Missing_Auth_Linked_ID_Flag":        "Auth_Linked_ID",
    }
    for flag_col, src_col in miss_map.items():
        if src_col in df.columns:
            df[flag_col] = df[src_col].isna().astype(int)

    print(f"[miss]  Created {len(miss_map)} missingness indicator columns.")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# STEP 5  —  RECORD-TYPE-AWARE CATEGORICAL IMPUTATION
# ──────────────────────────────────────────────────────────────────────────────

def impute_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill categorical fields with 'UNKNOWN' only where the field is
    relevant to the record type and genuinely missing.
    Never fabricates IDs.
    """
    # Status — always relevant
    df["Status"] = df["Status"].fillna("UNKNOWN")

    # Denial_Reason_Code — only relevant when Status == DENIED/REJECTED
    mask_denied = df["Status"].isin(["DENIED", "REJECTED"])
    df.loc[mask_denied & df["Denial_Reason_Code"].isna(), "Denial_Reason_Code"] = "UNKNOWN_DENIAL_REASON"

    # Urgency_Flag — only relevant for PRIOR_AUTH
    mask_auth = df["Record_Type"] == "PRIOR_AUTH"
    df.loc[mask_auth & df["Urgency_Flag"].isna(), "Urgency_Flag"] = "UNKNOWN"

    # Source_System — always present in real data; fill conservatively
    df["Source_System"] = df["Source_System"].fillna("UNKNOWN")

    # Provider_State
    df["Provider_State"] = df["Provider_State"].fillna("UNKNOWN")

    print("[cat]   Categorical fields filled with 'UNKNOWN' where appropriate.")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# STEP 6  —  RECORD-TYPE-AWARE NUMERICAL IMPUTATION
# ──────────────────────────────────────────────────────────────────────────────

def impute_numericals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Median-impute numeric ML features per Record_Type.
    Does NOT impute identifiers (BENE_ID, Provider_NPI, etc.).
    Does NOT impute financial amounts (preserved with missingness flags).
    """
    # Processing_Latency_Days — impute per Record_Type median
    # (already flagged via Missing_Processing_Latency_Flag)
    for rtype in df["Record_Type"].unique():
        mask = df["Record_Type"] == rtype
        median_val = df.loc[mask, "Processing_Latency_Days"].median()
        if pd.notna(median_val):
            df.loc[mask & df["Processing_Latency_Days"].isna(), "Processing_Latency_Days"] = median_val

    # Provider_Total_Records / Provider_Denial_Rate — impute median overall
    for col in ["Provider_Total_Records", "Provider_Denial_Rate"]:
        med = df[col].median()
        if pd.notna(med):
            df[col] = df[col].fillna(med)

    # Beneficiary_Record_Count — impute median overall
    med = df["Beneficiary_Record_Count"].median()
    if pd.notna(med):
        df["Beneficiary_Record_Count"] = df["Beneficiary_Record_Count"].fillna(med)

    # Days_Since_Prev_Batch — impute with 1 (normal cadence) where missing
    df["Days_Since_Prev_Batch"] = df["Days_Since_Prev_Batch"].fillna(1.0)

    print("[num]   Numerical features median-imputed per Record_Type where appropriate.")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# STEP 7  —  DATA-QUALITY FLAGS
# ──────────────────────────────────────────────────────────────────────────────

def create_dq_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create all data-quality / anomaly indicator columns.
    Records are NEVER deleted here.
    """

    # ── A. Date ordering anomalies ──────────────────────────────────────────

    # Service_End_Date < Service_Date
    mask = df["Service_End_Date"].notna() & df["Service_Date"].notna()
    df["Invalid_Service_Date_Flag"] = 0
    df.loc[mask & (df["Service_End_Date"] < df["Service_Date"]),
           "Invalid_Service_Date_Flag"] = 1

    # Submission_Date < Service_Date
    mask = df["Submission_Date"].notna() & df["Service_Date"].notna()
    df["Submission_Before_Service_Flag"] = 0
    df.loc[mask & (df["Submission_Date"] < df["Service_Date"]),
           "Submission_Before_Service_Flag"] = 1

    # Processed_Date < Submission_Date
    mask = df["Processed_Date"].notna() & df["Submission_Date"].notna()
    df["Processing_Before_Submission_Flag"] = 0
    df.loc[mask & (df["Processed_Date"] < df["Submission_Date"]),
           "Processing_Before_Submission_Flag"] = 1

    # Decision_Date < Submission_Date
    mask = df["Decision_Date"].notna() & df["Submission_Date"].notna()
    df["Decision_Before_Submission_Flag"] = 0
    df.loc[mask & (df["Decision_Date"] < df["Submission_Date"]),
           "Decision_Before_Submission_Flag"] = 1

    # ── B. Processing latency anomaly ───────────────────────────────────────

    df["Negative_Processing_Latency_Flag"] = (
        df["Processing_Latency_Days"].notna() &
        (df["Processing_Latency_Days"] < 0)
    ).astype(int)

    # ── C. Financial anomalies ───────────────────────────────────────────────

    df["Negative_Billed_Amount_Flag"]   = ((df["Billed_Amount"].notna())   & (df["Billed_Amount"]   < 0)).astype(int)
    df["Negative_Allowed_Amount_Flag"]  = ((df["Allowed_Amount"].notna())  & (df["Allowed_Amount"]  < 0)).astype(int)
    df["Negative_Paid_Amount_Flag"]     = ((df["Paid_Amount"].notna())     & (df["Paid_Amount"]     < 0)).astype(int)

    # Paid > Allowed
    mask = df["Paid_Amount"].notna() & df["Allowed_Amount"].notna()
    df["Paid_Exceeds_Allowed_Flag"] = 0
    df.loc[mask & (df["Paid_Amount"] > df["Allowed_Amount"]),
           "Paid_Exceeds_Allowed_Flag"] = 1

    # Allowed > Billed
    mask = df["Allowed_Amount"].notna() & df["Billed_Amount"].notna()
    df["Allowed_Exceeds_Billed_Flag"] = 0
    df.loc[mask & (df["Allowed_Amount"] > df["Billed_Amount"]),
           "Allowed_Exceeds_Billed_Flag"] = 1

    # Zero billed amount on a non-PRIOR_AUTH claim
    non_auth = df["Record_Type"] != "PRIOR_AUTH"
    df["Zero_Billed_Amount_Flag"] = (
        non_auth & df["Billed_Amount"].notna() & (df["Billed_Amount"] == 0)
    ).astype(int)

    # ── D. Authorization anomaly ─────────────────────────────────────────────

    # Auth required but no linked auth ID
    df["Missing_Required_Auth_Link_Flag"] = (
        (df["Auth_Required_Flag"] == "Y") & df["Auth_Linked_ID"].isna()
    ).astype(int)

    # ── E. SLA anomaly (supplement to existing features) ────────────────────

    # Extreme latency (> 30 days) as a flag
    df["Extreme_Processing_Latency_Flag"] = (
        df["Processing_Latency_Days"].notna() &
        (df["Processing_Latency_Days"] > 30)
    ).astype(int)

    # ── F. Pipeline anomaly (supplement) ────────────────────────────────────

    # Days_Since_Prev_Batch > 2 considered a pipeline gap supplement
    df["Extended_Pipeline_Gap_Flag"] = (
        df["Days_Since_Prev_Batch"].notna() &
        (df["Days_Since_Prev_Batch"] > 2)
    ).astype(int)

    # ── G. Provider anomaly ──────────────────────────────────────────────────

    df["High_Denial_Rate_Flag"] = (
        df["Provider_Denial_Rate"].notna() &
        (df["Provider_Denial_Rate"] > 0.5)
    ).astype(int)

    df["High_Volume_Provider_Flag"] = (
        df["Provider_Total_Records"].notna() &
        (df["Provider_Total_Records"] > df["Provider_Total_Records"].quantile(0.95))
    ).astype(int)

    # ── H. Beneficiary anomaly ───────────────────────────────────────────────

    # Extreme beneficiary frequency (top 5%)
    df["Extreme_Beneficiary_Frequency_Flag"] = (
        df["Beneficiary_Record_Count"].notna() &
        (df["Beneficiary_Record_Count"] > df["Beneficiary_Record_Count"].quantile(0.95))
    ).astype(int)

    print("[flags] Data-quality / anomaly flags created.")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# STEP 8  —  OUTLIER FLAGS  (IQR-based — flag only, never delete)
# ──────────────────────────────────────────────────────────────────────────────

def create_outlier_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    IQR-based outlier flags for key numerical columns.
    Original values are NEVER modified.
    """
    outlier_targets = {
        "Processing_Latency_Outlier_Flag":  "Processing_Latency_Days",
        "Billed_Amount_Outlier_Flag":       "Billed_Amount",
        "Batch_Volume_Outlier_Flag":        "Batch_Volume",
        "Provider_Volume_Outlier_Flag":     "Provider_Total_Records",
        "Paid_Amount_Outlier_Flag":         "Paid_Amount",
    }
    for flag_col, src_col in outlier_targets.items():
        if src_col not in df.columns:
            continue
        s = df[src_col].dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        df[flag_col] = (
            df[src_col].notna() &
            ((df[src_col] < lower) | (df[src_col] > upper))
        ).astype(int)

    print("[outlier] IQR outlier flags created (values preserved).")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# STEP 9  —  DATA-QUALITY SCORE
# ──────────────────────────────────────────────────────────────────────────────

def compute_dq_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rule-based Data_Quality_Issue_Count and normalised Data_Quality_Risk_Score.
    Combines all flag columns created above.
    Kept separate from ML anomaly score.
    """
    flag_cols = [c for c in df.columns if c.endswith("_Flag") or c.endswith("_Flag")]

    # Only binary / integer flag columns (exclude booleans that are existing features)
    dq_flag_cols = [
        "Invalid_Service_Date_Flag",
        "Submission_Before_Service_Flag",
        "Processing_Before_Submission_Flag",
        "Decision_Before_Submission_Flag",
        "Negative_Processing_Latency_Flag",
        "Negative_Billed_Amount_Flag",
        "Negative_Allowed_Amount_Flag",
        "Negative_Paid_Amount_Flag",
        "Paid_Exceeds_Allowed_Flag",
        "Allowed_Exceeds_Billed_Flag",
        "Zero_Billed_Amount_Flag",
        "Missing_Required_Auth_Link_Flag",
        "Extreme_Processing_Latency_Flag",
        "Extended_Pipeline_Gap_Flag",
        "High_Denial_Rate_Flag",
        "Missing_BENE_ID_Flag",
        "Missing_Provider_NPI_Flag",
        "Missing_Processing_Latency_Flag",
        "Missing_Service_Date_Flag",
        "Missing_Billed_Amount_Flag",
        "Processing_Latency_Outlier_Flag",
        "Billed_Amount_Outlier_Flag",
        "Batch_Volume_Outlier_Flag",
        "Duplicate_Record_ID_Flag",
    ]
    # Keep only columns that actually exist
    present = [c for c in dq_flag_cols if c in df.columns]

    df["Data_Quality_Issue_Count"] = df[present].sum(axis=1)
    max_count = df["Data_Quality_Issue_Count"].max()
    if max_count > 0:
        df["Data_Quality_Risk_Score"] = (df["Data_Quality_Issue_Count"] / max_count).round(4)
    else:
        df["Data_Quality_Risk_Score"] = 0.0

    print(f"[dq_score] Data_Quality_Issue_Count and Data_Quality_Risk_Score computed "
          f"from {len(present)} flag columns.")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# STEP 10  —  FINAL TYPE CLEAN-UP
# ──────────────────────────────────────────────────────────────────────────────

def finalise_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure bool columns produced by pandas comparisons become int (0/1)
    for consistent ML consumption.
    """
    for col in df.select_dtypes(include="bool").columns:
        df[col] = df[col].astype(int)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# STEP 11  —  DATA-QUALITY REPORT GENERATION
# ──────────────────────────────────────────────────────────────────────────────

def generate_dq_report(df_raw: pd.DataFrame, df_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Per-column data-quality report comparing raw vs. cleaned dataset.
    """
    rows = []
    for col in df_raw.columns:
        raw_miss = int(df_raw[col].isna().sum())
        raw_miss_pct = round(raw_miss / len(df_raw) * 100, 2)
        raw_unique = int(df_raw[col].nunique(dropna=True))
        dtype_str = str(df_raw[col].dtype)

        # Min/max for numerics
        if pd.api.types.is_numeric_dtype(df_raw[col]):
            col_min = df_raw[col].min()
            col_max = df_raw[col].max()
        else:
            col_min = col_max = None

        # Count invalid dates for date columns
        invalid_dates = 0
        if col in DATE_COLS + DATETIME_COLS:
            parsed = pd.to_datetime(df_raw[col], errors="coerce")
            invalid_dates = int((parsed.isna() & df_raw[col].notna()).sum())

        # Count anomaly-flagged records in cleaned data
        flag_col = col.replace("_Days", "").replace("_Amount", "") + "_Flag"
        anomaly_count = 0
        if flag_col in df_clean.columns:
            anomaly_count = int(df_clean[flag_col].sum())

        # Cleaning action taken
        if col in IDENTIFIER_COLS:
            action = "Preserved as identifier; excluded from numerical ML features"
        elif col in DATE_COLS + DATETIME_COLS:
            action = "Parsed to datetime with errors='coerce'; ParseFail indicator created"
        elif raw_miss > 0 and col in ["Status", "Source_System", "Provider_State"]:
            action = "Filled with UNKNOWN"
        elif raw_miss > 0 and col == "Processing_Latency_Days":
            action = "Missing flagged; median-imputed per Record_Type"
        elif raw_miss > 0 and col in ["Diagnosis_Code", "NDC_Code", "Drug_Name"]:
            action = "Missing expected for non-applicable Record_Type; missingness flag created"
        elif raw_miss > 0:
            action = "Missingness indicator created; no imputation of identifiers"
        else:
            action = "No action required"

        rows.append({
            "Column":               col,
            "Data_Type":            dtype_str,
            "Missing_Count":        raw_miss,
            "Missing_Pct":          raw_miss_pct,
            "Unique_Count":         raw_unique,
            "Invalid_Count":        invalid_dates,
            "Anomaly_Count":        anomaly_count,
            "Min":                  col_min,
            "Max":                  col_max,
            "Example_Value":        str(df_raw[col].dropna().iloc[0]) if raw_miss < len(df_raw) else "ALL_NULL",
            "Cleaning_Action":      action,
        })

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# STEP 12  —  SUMMARY REPORT
# ──────────────────────────────────────────────────────────────────────────────

def generate_summary(df_raw: pd.DataFrame, df_clean: pd.DataFrame,
                     dedup_info: dict) -> dict:
    """Produce the anomaly-category summary report."""

    def _flag_sum(col: str) -> int:
        return int(df_clean[col].sum()) if col in df_clean.columns else 0

    summary = {
        "total_records_raw":                   len(df_raw),
        "total_records_cleaned":               len(df_clean),
        "columns_raw":                         df_raw.shape[1],
        "columns_cleaned":                     df_clean.shape[1],
        "full_duplicate_rows_removed":         dedup_info["full_duplicate_rows_removed"],
        "duplicate_record_id_groups_flagged":  dedup_info["duplicate_record_id_groups_flagged"],

        # Missingness
        "records_with_missing_BENE_ID":        _flag_sum("Missing_BENE_ID_Flag"),
        "records_with_missing_Provider_NPI":   _flag_sum("Missing_Provider_NPI_Flag"),
        "records_with_missing_Service_Date":   _flag_sum("Missing_Service_Date_Flag"),
        "records_with_missing_Billed_Amount":  _flag_sum("Missing_Billed_Amount_Flag"),

        # Date anomalies
        "invalid_service_date_range":          _flag_sum("Invalid_Service_Date_Flag"),
        "submission_before_service":           _flag_sum("Submission_Before_Service_Flag"),
        "processing_before_submission":        _flag_sum("Processing_Before_Submission_Flag"),
        "decision_before_submission":          _flag_sum("Decision_Before_Submission_Flag"),

        # Processing latency
        "negative_processing_latency":         _flag_sum("Negative_Processing_Latency_Flag"),
        "extreme_processing_latency_gt30d":    _flag_sum("Extreme_Processing_Latency_Flag"),
        "processing_latency_iqr_outliers":     _flag_sum("Processing_Latency_Outlier_Flag"),

        # Financial
        "negative_billed_amount":              _flag_sum("Negative_Billed_Amount_Flag"),
        "negative_allowed_amount":             _flag_sum("Negative_Allowed_Amount_Flag"),
        "negative_paid_amount":                _flag_sum("Negative_Paid_Amount_Flag"),
        "paid_exceeds_allowed":                _flag_sum("Paid_Exceeds_Allowed_Flag"),
        "allowed_exceeds_billed":              _flag_sum("Allowed_Exceeds_Billed_Flag"),
        "zero_billed_amount_non_auth":         _flag_sum("Zero_Billed_Amount_Flag"),
        "billed_amount_iqr_outliers":          _flag_sum("Billed_Amount_Outlier_Flag"),

        # Authorization
        "auth_required_but_link_missing":      _flag_sum("Missing_Required_Auth_Link_Flag"),
        "missing_required_auth_link_existing": _flag_sum("Missing_Required_Auth_Link"),

        # SLA
        "sla_breach_records":                  int((df_clean["SLA_Breach_Flag"] == "Y").sum())
                                               if "SLA_Breach_Flag" in df_clean.columns else 0,
        "sla_unknown_no_date":                 int((df_clean["SLA_Breach_Flag"] == "UNKNOWN_NO_DATE").sum())
                                               if "SLA_Breach_Flag" in df_clean.columns else 0,

        # Pipeline
        "pipeline_gap_records":                int(df_clean["Pipeline_Gap_Flag"].sum())
                                               if "Pipeline_Gap_Flag" in df_clean.columns else 0,
        "extended_pipeline_gap_records":       _flag_sum("Extended_Pipeline_Gap_Flag"),

        # Provider
        "high_denial_rate_providers":          _flag_sum("High_Denial_Rate_Flag"),
        "high_volume_provider_records":        _flag_sum("High_Volume_Provider_Flag"),
        "provider_volume_iqr_outliers":        _flag_sum("Provider_Volume_Outlier_Flag"),

        # Beneficiary
        "high_frequency_beneficiary_records":  int(df_clean["High_Frequency_Beneficiary_Flag"].sum())
                                               if "High_Frequency_Beneficiary_Flag" in df_clean.columns else 0,
        "extreme_beneficiary_frequency":       _flag_sum("Extreme_Beneficiary_Frequency_Flag"),

        # DQ score
        "records_with_any_dq_issue":           int((df_clean["Data_Quality_Issue_Count"] > 0).sum())
                                               if "Data_Quality_Issue_Count" in df_clean.columns else 0,
        "avg_dq_issue_count":                  round(float(df_clean["Data_Quality_Issue_Count"].mean()), 3)
                                               if "Data_Quality_Issue_Count" in df_clean.columns else 0,
        "max_dq_issue_count":                  int(df_clean["Data_Quality_Issue_Count"].max())
                                               if "Data_Quality_Issue_Count" in df_clean.columns else 0,
    }
    return summary


# ──────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    input_path: str,
    output_path: str,
    dq_report_path: str = "data_quality_report.csv",
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Execute the full preprocessing pipeline.

    Parameters
    ----------
    input_path    : Path to the raw CSV file.
    output_path   : Destination for the cleaned CSV.
    dq_report_path: Destination for the per-column DQ report CSV.

    Returns
    -------
    (df_clean, dq_report_df, summary_dict)
    """
    print("\n" + "=" * 70)
    print("  UC10 — Data Cleaning & Preprocessing Pipeline")
    print("=" * 70 + "\n")

    # Load raw — keep a clean copy for reporting
    df_raw = load_raw(input_path)
    df = df_raw.copy()

    # Pipeline stages
    df, dedup_info = remove_duplicates(df)
    df = parse_dates(df)
    df = create_missingness_indicators(df)
    df = impute_categoricals(df)
    df = impute_numericals(df)
    df = create_dq_flags(df)
    df = create_outlier_flags(df)
    df = compute_dq_score(df)
    df = finalise_types(df)

    # Reports
    dq_report_df = generate_dq_report(df_raw, df)
    summary = generate_summary(df_raw, df, dedup_info)

    # Save outputs
    df.to_csv(output_path, index=False)
    dq_report_df.to_csv(dq_report_path, index=False)

    print(f"\n[save]  Cleaned dataset → '{output_path}' ({len(df):,} rows × {df.shape[1]} cols)")
    print(f"[save]  DQ report       → '{dq_report_path}' ({len(dq_report_df)} columns documented)")

    return df, dq_report_df, summary


# ──────────────────────────────────────────────────────────────────────────────
# FEATURE RECOMMENDATIONS  (for reference — not executed at runtime)
# ──────────────────────────────────────────────────────────────────────────────

ML_FEATURE_LIST = {
    "use_for_ml": [
        # Financial ratios
        "Billed_Amount", "Allowed_Amount", "Paid_Amount", "Patient_Responsibility",
        # Latency & SLA
        "Processing_Latency_Days", "SLA_Target_Days", "Record_SLA_Breach_Numeric",
        "SLA_Breach_Vs_DOW_Norm",
        # Existing engineered batch/provider/beneficiary features
        "Provider_Total_Records", "Provider_Denial_Rate",
        "Batch_Volume", "Rolling_7D_Avg_Volume", "Volume_Vs_Trend_Ratio",
        "Batch_SLA_Breach_Rate", "Rolling_7D_Avg_SLA_Breach_Rate",
        "SLA_Breach_Rate_Vs_Trend_Diff", "Beneficiary_Record_Count",
        "Days_Since_Prev_Batch", "DOW_Avg_SLA_Breach_Rate",
        # Pharmacy-specific (when Record_Type==PHARMACY_CLAIM)
        "Days_Supply", "Quantity_Dispensed",
        # Retry
        "Retry_Count",
        # All DQ flags (new)
        "Invalid_Service_Date_Flag", "Submission_Before_Service_Flag",
        "Processing_Before_Submission_Flag", "Decision_Before_Submission_Flag",
        "Negative_Processing_Latency_Flag", "Negative_Billed_Amount_Flag",
        "Negative_Allowed_Amount_Flag", "Negative_Paid_Amount_Flag",
        "Paid_Exceeds_Allowed_Flag", "Allowed_Exceeds_Billed_Flag",
        "Zero_Billed_Amount_Flag", "Missing_Required_Auth_Link_Flag",
        "Extreme_Processing_Latency_Flag", "Extended_Pipeline_Gap_Flag",
        "High_Denial_Rate_Flag", "High_Volume_Provider_Flag",
        "High_Frequency_Beneficiary_Flag", "Extreme_Beneficiary_Frequency_Flag",
        "Processing_Latency_Outlier_Flag", "Billed_Amount_Outlier_Flag",
        "Provider_Volume_Outlier_Flag",
        # DQ score (as a meta-feature, not as target)
        "Data_Quality_Issue_Count", "Data_Quality_Risk_Score",
    ],
    "do_not_use_for_ml": [
        # Raw identifiers — high cardinality, no predictive structure
        "Record_ID", "BENE_ID", "Provider_NPI", "Auth_Linked_ID", "Batch_ID",
        # Free-text / code fields — use encoded versions instead
        "Diagnosis_Code", "Procedure_Code", "NDC_Code", "Drug_Name",
        "Denial_Reason_Code",
        # Raw date columns — use derived features instead
        "Service_Date", "Service_End_Date", "Submission_Date",
        "Processed_Date", "Decision_Date", "Ingestion_Timestamp", "Batch_Date",
        # High-cardinality categoricals without encoding
        "Provider_State", "Source_System", "Submission_Day_Of_Week",
        # Status/flag as string — encode before use
        "SLA_Breach_Flag", "Auth_Required_Flag", "Urgency_Flag", "Status",
    ],
}
