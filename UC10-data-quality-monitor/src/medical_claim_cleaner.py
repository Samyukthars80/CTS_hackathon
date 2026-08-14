from __future__ import annotations

from pathlib import Path
import pandas as pd

ID_LIKE_COLUMNS = {"Record_ID", "BENE_ID", "Provider_NPI", "Auth_Linked_ID", "Batch_ID", "Record_Type"}
IDENTIFIER_COLUMNS = {"Record_ID", "BENE_ID", "Provider_NPI", "Auth_Linked_ID", "Batch_ID"}
DATE_COLUMNS = [
    "Service_Date",
    "Service_End_Date",
    "Submission_Date",
    "Processed_Date",
    "Decision_Date",
    "Batch_Date",
    "Ingestion_Timestamp",
]

PRESERVE_STRING_COLUMNS = {
    "Record_ID", "BENE_ID", "Provider_NPI", "Auth_Linked_ID", "Batch_ID", "Record_Type",
    "NDC_Code", "Diagnosis_Code", "Procedure_Code", "Drug_Name", "Denial_Reason_Code",
    "Status", "Urgency_Flag", "Auth_Required_Flag", "Provider_State", "SLA_Breach_Flag",
    "Source_System", "Submission_Day_Of_Week"
}


def _clean_base_record_type(df: pd.DataFrame, record_type: str) -> tuple[pd.DataFrame, int, int, int, int]:
    # 1. Filter by record type
    df_filtered = df.loc[df["Record_Type"] == record_type].copy()

    # 2. Remove exact duplicates
    dup_removed = int(df_filtered.duplicated().sum())
    df_filtered = df_filtered.drop_duplicates(keep="first").copy()

    # 3. Trim whitespace from all string columns
    whitespace_trimmed_fields = 0
    for col in df_filtered.columns:
        if df_filtered[col].dtype == "object" or pd.api.types.is_string_dtype(df_filtered[col]):
            trimmed = df_filtered[col].map(
                lambda value: value.strip() if isinstance(value, str) else value
            )
            # Count non-null fields that changed due to stripping
            changed_mask = (df_filtered[col] != trimmed) & df_filtered[col].notna()
            whitespace_trimmed_fields += int(changed_mask.sum())
            df_filtered[col] = trimmed

    # 4. Standardize dates
    dates_standardized = 0
    for col in DATE_COLUMNS:
        if col in df_filtered.columns:
            converted = pd.to_datetime(df_filtered[col], errors="coerce")
            non_null_before = df_filtered[col].notna()
            dates_standardized += int(non_null_before.sum())
            df_filtered[col] = converted

    # 5. Convert numeric columns represented as object/string to numeric format
    numeric_conversions = 0
    for col in df_filtered.columns:
        if col in PRESERVE_STRING_COLUMNS or col in DATE_COLUMNS:
            continue

        if pd.api.types.is_numeric_dtype(df_filtered[col]):
            continue

        converted = pd.to_numeric(df_filtered[col], errors="coerce")
        if converted.notna().sum() > 0:
            non_null_before = df_filtered[col].notna()
            numeric_conversions += int(non_null_before.sum())
            df_filtered[col] = converted

    return df_filtered, dup_removed, whitespace_trimmed_fields, dates_standardized, numeric_conversions


def compile_cleaning_summary(df_before_clean: pd.DataFrame, df_cleaned: pd.DataFrame, dup_removed: int) -> dict:
    total_nulls = int(df_cleaned.isna().sum().sum())
    nulls_per_col = df_cleaned.isna().sum()
    cols_with_nulls = sorted(list(nulls_per_col[nulls_per_col > 0].index))
    
    preserved_details = {}
    for col in cols_with_nulls:
        preserved_details[col] = int(nulls_per_col[col])
        
    summary = {
        "Number of rows before cleaning": len(df_before_clean),
        "Number of rows after cleaning": len(df_cleaned),
        "Number of columns": len(df_cleaned.columns),
        "Remaining missing values": total_nulls,
        "Columns with remaining missing values": cols_with_nulls,
        "Imputation performed": "None (imputations disabled to preserve data characteristics)",
        "Missing values intentionally preserved": preserved_details,
        "Rows removed, if any": dup_removed,
        "Reason for any row removal": "Exact duplicates removed" if dup_removed > 0 else "None",
    }
    return summary


def clean_medical_claims(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    raw_filtered = df[df["Record_Type"] == "MEDICAL_CLAIM"]
    
    cleaned_df, dup_removed, trimmed, dates_std, num_conv = _clean_base_record_type(df, "MEDICAL_CLAIM")
    
    # Create missingness indicators
    cleaned_df["BENE_ID_MISSING_FLAG"] = cleaned_df["BENE_ID"].isna()
    cleaned_df["Provider_NPI_MISSING_FLAG"] = cleaned_df["Provider_NPI"].isna()
    
    summary = compile_cleaning_summary(raw_filtered, cleaned_df, dup_removed)
    return cleaned_df, summary


def clean_pharmacy_claims(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    raw_filtered = df[df["Record_Type"] == "PHARMACY_CLAIM"]
    
    cleaned_df, dup_removed, trimmed, dates_std, num_conv = _clean_base_record_type(df, "PHARMACY_CLAIM")
    
    # Create missingness indicators
    cleaned_df["BENE_ID_MISSING_FLAG"] = cleaned_df["BENE_ID"].isna()
    cleaned_df["Provider_NPI_MISSING_FLAG"] = cleaned_df["Provider_NPI"].isna()
    cleaned_df["NDC_CODE_MISSING_FLAG"] = cleaned_df["NDC_Code"].isna()
    
    summary = compile_cleaning_summary(raw_filtered, cleaned_df, dup_removed)
    return cleaned_df, summary


def clean_authorizations(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    # PRIOR_AUTH corresponds to AUTHORIZATION
    raw_filtered = df[df["Record_Type"] == "PRIOR_AUTH"]
    
    cleaned_df, dup_removed, trimmed, dates_std, num_conv = _clean_base_record_type(df, "PRIOR_AUTH")
    
    # Create missingness indicators
    cleaned_df["BENE_ID_MISSING_FLAG"] = cleaned_df["BENE_ID"].isna()
    cleaned_df["Provider_NPI_MISSING_FLAG"] = cleaned_df["Provider_NPI"].isna()
    cleaned_df["Auth_Linked_ID_MISSING_FLAG"] = cleaned_df["Auth_Linked_ID"].isna()
    
    summary = compile_cleaning_summary(raw_filtered, cleaned_df, dup_removed)
    return cleaned_df, summary


def clean_medical_claims_dataset(csv_path: str | Path) -> tuple[pd.DataFrame, dict]:
    # Wrapper function for backward compatibility
    df = pd.read_csv(csv_path)
    df.columns = [str(col).strip() for col in df.columns]
    if "Record_Type" in df.columns:
        df["Record_Type"] = df["Record_Type"].astype(str).str.strip()
    return clean_medical_claims(df)


def clean_all_datasets(csv_path: str | Path) -> dict[str, dict]:
    df = pd.read_csv(csv_path)
    df.columns = [str(col).strip() for col in df.columns]
    if "Record_Type" in df.columns:
        df["Record_Type"] = df["Record_Type"].astype(str).str.strip()
        
    # Clean each record type
    med_df, med_summary = clean_medical_claims(df)
    pharm_df, pharm_summary = clean_pharmacy_claims(df)
    auth_df, auth_summary = clean_authorizations(df)
    
    output_dir = Path(csv_path).resolve().parent.parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the files
    med_df.to_csv(output_dir / "medical_claim_cleaned.csv", index=False)
    pharm_df.to_csv(output_dir / "pharmacy_claim_cleaned.csv", index=False)
    auth_df.to_csv(output_dir / "authorization_cleaned.csv", index=False)
    
    all_summaries = {
        "medical_claim": med_summary,
        "pharmacy_claim": pharm_summary,
        "authorization": auth_summary,
    }
    return all_summaries


if __name__ == "__main__":
    default_path = Path(__file__).resolve().parent.parent / "data" / "claims_pharmacy_auth_monitor_dataset_features.csv"
    summaries = clean_all_datasets(default_path)
    
    output_dir = Path(__file__).resolve().parent.parent / "outputs"
    print(f"Cleaned datasets written to: {output_dir}\n")
    for name, summary in summaries.items():
        print(f"=== Cleaning Summary for: {name.upper()} ===")
        for k, v in summary.items():
            if k == "Missing values intentionally preserved":
                print(f"  {k}:")
                for col, count in v.items():
                    print(f"    - {col}: {count}")
            else:
                print(f"  {k}: {v}")
        print()
