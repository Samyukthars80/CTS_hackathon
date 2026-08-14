from __future__ import annotations

import os
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure matplotlib runs in headless mode
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for plots
sns.set_theme(style="whitegrid")

# Column groups based on specifications
MEDICAL_NUM_COLUMNS = [
    "Billed_Amount", "Allowed_Amount", "Paid_Amount", "Patient_Responsibility",
    "Retry_Count", "Processing_Latency_Days", "SLA_Target_Days"
]
MEDICAL_CAT_COLUMNS = ["Provider_State", "Status", "Auth_Required_Flag", "SLA_Breach_Flag"]

PHARMACY_NUM_COLUMNS = [
    "Billed_Amount", "Allowed_Amount", "Paid_Amount", "Patient_Responsibility",
    "Retry_Count", "Processing_Latency_Days", "SLA_Target_Days", "Days_Supply", "Quantity_Dispensed"
]
PHARMACY_CAT_COLUMNS = ["Provider_State", "Status", "Drug_Name", "Auth_Required_Flag", "SLA_Breach_Flag"]

AUTH_NUM_COLUMNS = ["Retry_Count", "Processing_Latency_Days", "SLA_Target_Days"]
AUTH_CAT_COLUMNS = ["Provider_State", "Status", "Urgency_Flag", "Auth_Required_Flag", "SLA_Breach_Flag"]


def compute_descriptive_stats(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for col in columns:
        if col in df.columns:
            series = df[col].dropna()
            cnt = int(series.count())
            missing = int(df[col].isna().sum())
            mean = float(series.mean()) if cnt > 0 else np.nan
            median = float(series.median()) if cnt > 0 else np.nan
            std = float(series.std()) if cnt > 1 else np.nan
            min_val = float(series.min()) if cnt > 0 else np.nan
            max_val = float(series.max()) if cnt > 0 else np.nan
            q1 = float(series.quantile(0.25)) if cnt > 0 else np.nan
            q3 = float(series.quantile(0.75)) if cnt > 0 else np.nan
            iqr = q3 - q1 if cnt > 0 else np.nan
            p1 = float(series.quantile(0.01)) if cnt > 0 else np.nan
            p5 = float(series.quantile(0.05)) if cnt > 0 else np.nan
            p25 = float(series.quantile(0.25)) if cnt > 0 else np.nan
            p50 = float(series.quantile(0.50)) if cnt > 0 else np.nan
            p75 = float(series.quantile(0.75)) if cnt > 0 else np.nan
            p95 = float(series.quantile(0.95)) if cnt > 0 else np.nan
            p99 = float(series.quantile(0.99)) if cnt > 0 else np.nan
            skewness = float(series.skew()) if cnt > 2 else np.nan
            
            rows.append({
                "column": col,
                "count": cnt,
                "missing_count": missing,
                "mean": mean,
                "median": median,
                "std": std,
                "min": min_val,
                "max": max_val,
                "Q1": q1,
                "Q3": q3,
                "IQR": iqr,
                "1st_percentile": p1,
                "5th_percentile": p5,
                "25th_percentile": p25,
                "50th_percentile": p50,
                "75th_percentile": p75,
                "95th_percentile": p95,
                "99th_percentile": p99,
                "skewness": skewness
            })
    return pd.DataFrame(rows)


def compute_correlation_matrix(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid_cols = [c for c in columns if c in df.columns]
    if not valid_cols:
        return pd.DataFrame(), pd.DataFrame()
    pearson = df[valid_cols].corr(method='pearson')
    spearman = df[valid_cols].corr(method='spearman')
    return pearson, spearman


def compute_outlier_stats(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for col in columns:
        if col in df.columns:
            data = df[col].dropna()
            total = len(data)
            if total == 0:
                continue
            
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            iqr_outliers = ((data < lower_bound) | (data > upper_bound)).sum()
            
            mean = data.mean()
            std = data.std()
            z_outliers = 0
            if std > 0:
                z_scores = (data - mean) / std
                z_outliers = (z_scores.abs() > 3).sum()
                
            p99 = data.quantile(0.99)
            p1 = data.quantile(0.01)
            p99_outliers = (data > p99).sum()
            p1_outliers = (data < p1).sum()
            
            rows.append({
                "column": col,
                "total_non_null_records": int(total),
                "Q1": float(Q1),
                "Q3": float(Q3),
                "IQR": float(IQR),
                "iqr_lower_bound": float(lower_bound),
                "iqr_upper_bound": float(upper_bound),
                "iqr_outlier_count": int(iqr_outliers),
                "iqr_outlier_percentage": float((iqr_outliers / total) * 100) if total > 0 else 0.0,
                "zscore_outlier_count_3sd": int(z_outliers),
                "percentile_1st_lower_count": int(p1_outliers),
                "percentile_99th_upper_count": int(p99_outliers)
            })
    return pd.DataFrame(rows)


def compute_categorical_stats(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for col in columns:
        if col in df.columns:
            total = len(df)
            val_counts = df[col].value_counts(dropna=False)
            unique_cnt = df[col].nunique(dropna=True)
            for val, cnt in val_counts.items():
                rows.append({
                    "column": col,
                    "unique_count": unique_cnt,
                    "value": str(val),
                    "count": cnt,
                    "percentage": (cnt / total) * 100 if total > 0 else 0.0
                })
    return pd.DataFrame(rows)


def compute_missingness_stats(df: pd.DataFrame) -> pd.DataFrame:
    total = len(df)
    rows = []
    for col in df.columns:
        cnt = int(df[col].isna().sum())
        pct = (cnt / total) * 100 if total > 0 else 0.0
        rows.append({
            "column": col,
            "missing_count": cnt,
            "missing_percentage": pct
        })
    return pd.DataFrame(rows)


def save_plots(df: pd.DataFrame, columns: list[str], output_dir: Path):
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    for col in columns:
        if col in df.columns:
            data = df[col].dropna()
            if len(data) == 0:
                continue
            
            # Histogram
            plt.figure(figsize=(8, 5))
            sns.histplot(data, kde=True, color='skyblue')
            plt.title(f'Distribution Histogram of {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')
            plt.tight_layout()
            plt.savefig(plots_dir / f'{col}_histogram.png', dpi=150)
            plt.close()
            
            # Boxplot
            plt.figure(figsize=(8, 3))
            sns.boxplot(x=data, color='lightgreen')
            plt.title(f'Box Plot of {col}')
            plt.xlabel(col)
            plt.tight_layout()
            plt.savefig(plots_dir / f'{col}_boxplot.png', dpi=150)
            plt.close()


# Custom analyses
def analyze_medical_sla(df: pd.DataFrame) -> pd.DataFrame:
    results = {}
    if "Processing_Latency_Days" in df.columns and "SLA_Target_Days" in df.columns:
        latency = df["Processing_Latency_Days"].dropna()
        
        results["mean_processing_latency"] = float(latency.mean()) if len(latency) > 0 else np.nan
        results["median_processing_latency"] = float(latency.median()) if len(latency) > 0 else np.nan
        results["std_processing_latency"] = float(latency.std()) if len(latency) > 1 else np.nan
        results["90th_percentile_latency"] = float(latency.quantile(0.90)) if len(latency) > 0 else np.nan
        results["95th_percentile_latency"] = float(latency.quantile(0.95)) if len(latency) > 0 else np.nan
        results["99th_percentile_latency"] = float(latency.quantile(0.99)) if len(latency) > 0 else np.nan
        
        exceeding_count = int((df["Processing_Latency_Days"] > df["SLA_Target_Days"]).sum())
        total_valid = int((df["Processing_Latency_Days"].notna() & df["SLA_Target_Days"].notna()).sum())
        results["proportion_exceeding_sla_target"] = float(exceeding_count / total_valid) if total_valid > 0 else np.nan
        
        if "SLA_Breach_Flag" in df.columns:
            y_group = df[df["SLA_Breach_Flag"] == "Y"]
            n_group = df[df["SLA_Breach_Flag"] == "N"]
            results["y_count"] = len(y_group)
            results["n_count"] = len(n_group)
            results["y_mean_latency"] = float(y_group["Processing_Latency_Days"].mean()) if len(y_group) > 0 else np.nan
            results["n_mean_latency"] = float(n_group["Processing_Latency_Days"].mean()) if len(n_group) > 0 else np.nan
            results["y_mean_retry"] = float(y_group["Retry_Count"].mean()) if len(y_group) > 0 else np.nan
            results["n_mean_retry"] = float(n_group["Retry_Count"].mean()) if len(n_group) > 0 else np.nan
            
    return pd.DataFrame([results])


def analyze_auth_sla_retry(df: pd.DataFrame) -> pd.DataFrame:
    results = []
    if "Status" in df.columns:
        status_groups = df.groupby("Status")
        for name, group in status_groups:
            latency_mean = group["Processing_Latency_Days"].mean() if "Processing_Latency_Days" in group.columns else np.nan
            retry_mean = group["Retry_Count"].mean() if "Retry_Count" in group.columns else np.nan
            sla_breach_rate = (group["SLA_Breach_Flag"] == "Y").mean() if "SLA_Breach_Flag" in group.columns else np.nan
            results.append({
                "Status": name,
                "record_count": len(group),
                "average_processing_latency": latency_mean,
                "average_retry_count": retry_mean,
                "sla_breach_rate": sla_breach_rate
            })
    return pd.DataFrame(results)


def analyze_pharmacy_quantity_days(df: pd.DataFrame) -> pd.DataFrame:
    results = {}
    if "Quantity_Dispensed" in df.columns and "Days_Supply" in df.columns:
        qd = df["Quantity_Dispensed"].dropna()
        ds = df["Days_Supply"].dropna()
        mask = (df["Quantity_Dispensed"].notna()) & (df["Days_Supply"].notna()) & (df["Days_Supply"] > 0)
        
        if len(qd) > 0 and len(ds) > 0:
            results["pearson_correlation"] = float(df[mask]["Quantity_Dispensed"].corr(df[mask]["Days_Supply"], method='pearson'))
            results["spearman_correlation"] = float(df[mask]["Quantity_Dispensed"].corr(df[mask]["Days_Supply"], method='spearman'))
            
            qpd = df[mask]["Quantity_Dispensed"] / df[mask]["Days_Supply"]
            results["mean_quantity_per_day"] = float(qpd.mean())
            results["median_quantity_per_day"] = float(qpd.median())
            results["min_quantity_per_day"] = float(qpd.min())
            results["max_quantity_per_day"] = float(qpd.max())
            results["std_quantity_per_day"] = float(qpd.std())
            results["iqr_quantity_per_day"] = float(qpd.quantile(0.75) - qpd.quantile(0.25))
            
    return pd.DataFrame([results])


def analyze_medical(df: pd.DataFrame, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Descriptive stats
    desc = compute_descriptive_stats(df, MEDICAL_NUM_COLUMNS)
    desc.to_csv(output_dir / "descriptive_statistics.csv", index=False)
    
    # 2. Financial relationships (correlation)
    pearson, spearman = compute_correlation_matrix(df, MEDICAL_NUM_COLUMNS)
    pearson.to_csv(output_dir / "correlation_matrix_pearson.csv")
    spearman.to_csv(output_dir / "correlation_matrix_spearman.csv")
    
    # 3. Provider statistics
    if "Provider_NPI" in df.columns:
        valid_prov = df[df["Provider_NPI"].notna()]
        prov_stats = valid_prov.groupby("Provider_NPI").agg(
            total_claims=("Record_ID", "count"),
            total_billed_amount=("Billed_Amount", "sum"),
            average_billed_amount=("Billed_Amount", "mean"),
            median_billed_amount=("Billed_Amount", "median"),
            total_allowed_amount=("Allowed_Amount", "sum"),
            average_allowed_amount=("Allowed_Amount", "mean"),
            total_paid_amount=("Paid_Amount", "sum"),
            average_paid_amount=("Paid_Amount", "mean"),
            average_processing_latency=("Processing_Latency_Days", "mean"),
            average_retry_count=("Retry_Count", "mean"),
            denial_count=("Status", lambda s: (s == "DENIED").sum()),
            denial_rate=("Status", lambda s: (s == "DENIED").mean()),
            SLA_breach_count=("SLA_Breach_Flag", lambda s: (s == "Y").sum()),
            SLA_breach_rate=("SLA_Breach_Flag", lambda s: (s == "Y").mean())
        ).reset_index()
        prov_stats.to_csv(output_dir / "provider_statistics.csv", index=False)
        
    # 4. Beneficiary statistics
    if "BENE_ID" in df.columns:
        valid_bene = df[df["BENE_ID"].notna()]
        bene_stats = valid_bene.groupby("BENE_ID").agg(
            number_of_claims=("Record_ID", "count"),
            total_billed_amount=("Billed_Amount", "sum"),
            average_billed_amount=("Billed_Amount", "mean"),
            total_paid_amount=("Paid_Amount", "sum"),
            average_paid_amount=("Paid_Amount", "mean"),
            average_processing_latency=("Processing_Latency_Days", "mean"),
            number_of_providers=("Provider_NPI", "nunique")
        ).reset_index()
        bene_stats.to_csv(output_dir / "beneficiary_statistics.csv", index=False)
        
    # 5. Categorical statistics
    cat = compute_categorical_stats(df, MEDICAL_CAT_COLUMNS)
    cat.to_csv(output_dir / "categorical_statistics.csv", index=False)
    
    # 6. Missingness statistics
    miss = compute_missingness_stats(df)
    miss.to_csv(output_dir / "missingness_statistics.csv", index=False)
    
    # 7. Outliers
    outliers = compute_outlier_stats(df, MEDICAL_NUM_COLUMNS)
    outliers.to_csv(output_dir / "outlier_statistics.csv", index=False)
    
    # 8. Custom SLA Analysis
    sla = analyze_medical_sla(df)
    sla.to_csv(output_dir / "sla_analysis.csv", index=False)
    
    # 9. Plots
    save_plots(df, MEDICAL_NUM_COLUMNS, output_dir)


def analyze_pharmacy(df: pd.DataFrame, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Descriptive stats
    desc = compute_descriptive_stats(df, PHARMACY_NUM_COLUMNS)
    desc.to_csv(output_dir / "descriptive_statistics.csv", index=False)
    
    # 2. Relationships (correlation)
    pearson, spearman = compute_correlation_matrix(df, PHARMACY_NUM_COLUMNS)
    pearson.to_csv(output_dir / "correlation_matrix_pearson.csv")
    spearman.to_csv(output_dir / "correlation_matrix_spearman.csv")
    
    # 3. Provider statistics
    if "Provider_NPI" in df.columns:
        valid_prov = df[df["Provider_NPI"].notna()]
        prov_stats = valid_prov.groupby("Provider_NPI").agg(
            prescription_claim_count=("Record_ID", "count"),
            total_billed_amount=("Billed_Amount", "sum"),
            average_billed_amount=("Billed_Amount", "mean"),
            total_paid_amount=("Paid_Amount", "sum"),
            average_paid_amount=("Paid_Amount", "mean"),
            average_days_supply=("Days_Supply", "mean"),
            average_quantity_dispensed=("Quantity_Dispensed", "mean"),
            average_processing_latency=("Processing_Latency_Days", "mean"),
            average_retry_count=("Retry_Count", "mean"),
            denial_rate=("Status", lambda s: (s == "REJECTED").mean()),
            SLA_breach_rate=("SLA_Breach_Flag", lambda s: (s == "Y").mean())
        ).reset_index()
        prov_stats.to_csv(output_dir / "provider_statistics.csv", index=False)
        
    # 4. Beneficiary statistics
    if "BENE_ID" in df.columns:
        valid_bene = df[df["BENE_ID"].notna()]
        bene_stats = valid_bene.groupby("BENE_ID").agg(
            pharmacy_claim_count=("Record_ID", "count"),
            total_pharmacy_spending=("Paid_Amount", "sum"),
            average_claim_amount=("Paid_Amount", "mean"),
            number_of_unique_providers=("Provider_NPI", "nunique"),
            number_of_unique_drugs=("NDC_Code", "nunique"),
            average_days_supply=("Days_Supply", "mean"),
            average_quantity_dispensed=("Quantity_Dispensed", "mean")
        ).reset_index()
        bene_stats.to_csv(output_dir / "beneficiary_statistics.csv", index=False)
        
    # 5. Drug Statistics
    if "NDC_Code" in df.columns:
        valid_ndc = df[df["NDC_Code"].notna()]
        drug_stats = valid_ndc.groupby("NDC_Code").agg(
            number_of_prescriptions=("Record_ID", "count"),
            drug_name=("Drug_Name", lambda s: s.dropna().iloc[0] if s.dropna().any() else np.nan),
            average_billed_amount=("Billed_Amount", "mean"),
            median_billed_amount=("Billed_Amount", "median"),
            average_paid_amount=("Paid_Amount", "mean"),
            average_allowed_amount=("Allowed_Amount", "mean"),
            average_days_supply=("Days_Supply", "mean"),
            average_quantity_dispensed=("Quantity_Dispensed", "mean")
        ).reset_index()
        drug_stats.to_csv(output_dir / "drug_statistics.csv", index=False)
        
    # 6. Categorical statistics
    cat = compute_categorical_stats(df, PHARMACY_CAT_COLUMNS)
    cat.to_csv(output_dir / "categorical_statistics.csv", index=False)
    
    # 7. Missingness statistics
    miss = compute_missingness_stats(df)
    miss.to_csv(output_dir / "missingness_statistics.csv", index=False)
    
    # 8. Outliers
    outliers = compute_outlier_stats(df, PHARMACY_NUM_COLUMNS)
    outliers.to_csv(output_dir / "outlier_statistics.csv", index=False)
    
    # 9. Custom Quantity vs Days Supply
    qds = analyze_pharmacy_quantity_days(df)
    qds.to_csv(output_dir / "quantity_days_analysis.csv", index=False)
    
    # 10. Plots
    save_plots(df, PHARMACY_NUM_COLUMNS, output_dir)


def analyze_authorization(df: pd.DataFrame, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Descriptive stats
    desc = compute_descriptive_stats(df, AUTH_NUM_COLUMNS)
    desc.to_csv(output_dir / "descriptive_statistics.csv", index=False)
    
    # 2. Relationships (correlation)
    pearson, spearman = compute_correlation_matrix(df, AUTH_NUM_COLUMNS)
    pearson.to_csv(output_dir / "correlation_matrix_pearson.csv")
    spearman.to_csv(output_dir / "correlation_matrix_spearman.csv")
    
    # 3. Provider statistics
    if "Provider_NPI" in df.columns:
        valid_prov = df[df["Provider_NPI"].notna()]
        prov_stats = valid_prov.groupby("Provider_NPI").agg(
            authorization_count=("Record_ID", "count"),
            approval_count=("Status", lambda s: (s == "APPROVED").sum()),
            denial_count=("Status", lambda s: (s == "DENIED").sum()),
            pending_count=("Status", lambda s: (s == "PENDING").sum()),
            approval_rate=("Status", lambda s: (s == "APPROVED").mean()),
            denial_rate=("Status", lambda s: (s == "DENIED").mean()),
            average_processing_latency=("Processing_Latency_Days", "mean"),
            SLA_breach_rate=("SLA_Breach_Flag", lambda s: (s == "Y").mean()),
            average_retry_count=("Retry_Count", "mean")
        ).reset_index()
        prov_stats.to_csv(output_dir / "provider_statistics.csv", index=False)
        
    # 4. Beneficiary statistics
    if "BENE_ID" in df.columns:
        valid_bene = df[df["BENE_ID"].notna()]
        bene_stats = valid_bene.groupby("BENE_ID").agg(
            authorization_count=("Record_ID", "count"),
            approval_count=("Status", lambda s: (s == "APPROVED").sum()),
            denial_count=("Status", lambda s: (s == "DENIED").sum()),
            pending_count=("Status", lambda s: (s == "PENDING").sum()),
            average_processing_latency=("Processing_Latency_Days", "mean"),
            average_retry_count=("Retry_Count", "mean"),
            number_of_providers=("Provider_NPI", "nunique")
        ).reset_index()
        bene_stats.to_csv(output_dir / "beneficiary_statistics.csv", index=False)
        
    # 5. Categorical statistics
    cat = compute_categorical_stats(df, AUTH_CAT_COLUMNS)
    cat.to_csv(output_dir / "categorical_statistics.csv", index=False)
    
    # 6. Missingness statistics
    miss = compute_missingness_stats(df)
    miss.to_csv(output_dir / "missingness_statistics.csv", index=False)
    
    # 7. Outliers
    outliers = compute_outlier_stats(df, AUTH_NUM_COLUMNS)
    outliers.to_csv(output_dir / "outlier_statistics.csv", index=False)
    
    # 8. Custom SLA Status analysis
    sla_status = analyze_auth_sla_retry(df)
    sla_status.to_csv(output_dir / "sla_status_analysis.csv", index=False)
    
    # 9. Plots
    save_plots(df, AUTH_NUM_COLUMNS, output_dir)


def main():
    workspace_dir = Path(__file__).resolve().parent.parent
    outputs_dir = workspace_dir / "outputs"
    stat_dir = workspace_dir / "statistical_analysis"
    
    print("Loading cleaned datasets...")
    med_df = pd.read_csv(outputs_dir / "medical_claim_cleaned.csv")
    pharm_df = pd.read_csv(outputs_dir / "pharmacy_claim_cleaned.csv")
    auth_df = pd.read_csv(outputs_dir / "authorization_cleaned.csv")
    
    print("Performing Medical Claims Statistical Analysis...")
    analyze_medical(med_df, stat_dir / "medical")
    
    print("Performing Pharmacy Claims Statistical Analysis...")
    analyze_pharmacy(pharm_df, stat_dir / "pharmacy")
    
    print("Performing Authorizations Statistical Analysis...")
    analyze_authorization(auth_df, stat_dir / "authorization")
    
    print(f"\nAll statistical analyses successfully completed! Results saved under: {stat_dir}")


if __name__ == "__main__":
    main()
