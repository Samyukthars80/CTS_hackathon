import json
import os
import datetime

def calculate_scores_and_risk(rule_results, df, output_dir="outputs"):
    # Group results by dimension
    dimensions = ["Completeness", "Validity", "Uniqueness", "Consistency", "Timeliness"]
    dim_scores = {dim: 100.0 for dim in dimensions}
    
    dim_rates = {dim: [] for dim in dimensions}
    critical_failures = 0
    top_failed_rules = []
    
    for res in rule_results:
        dim = res["dimension"]
        # In case we mapped Referential Integrity to Consistency in scoring but named it Referential integrity in catalog
        if dim not in dim_rates:
            dim = "Consistency" # Fallback
            
        dim_rates[dim].append(res["failure_rate_pct"])
        
        if res["severity"] == "Critical" and res["status"] == "FAILED":
            critical_failures += 1
            
        if res["status"] == "FAILED":
            top_failed_rules.append(res)
            
    # Calculate dimension scores
    for dim, rates in dim_rates.items():
        if rates:
            avg_rate = sum(rates) / len(rates)
            dim_scores[dim] = max(0.0, 100.0 - avg_rate)
            
    # Overall Quality Score
    overall_score = (
        0.25 * dim_scores.get("Completeness", 100) +
        0.25 * dim_scores.get("Validity", 100) +
        0.20 * dim_scores.get("Uniqueness", 100) +
        0.20 * dim_scores.get("Consistency", 100) +
        0.10 * dim_scores.get("Timeliness", 100)
    )
    
    # Risk classification
    if overall_score >= 90 and critical_failures == 0:
        overall_risk_level = "LOW"
    elif overall_score >= 75:
        overall_risk_level = "MEDIUM" if critical_failures == 0 else "MEDIUM"
        if overall_score >= 90 and critical_failures > 0:
            overall_risk_level = "MEDIUM"
    elif overall_score >= 50:
        overall_risk_level = "HIGH"
    else:
        overall_risk_level = "CRITICAL"
        
    # Top failed rules sorted by failure rate
    top_failed_rules.sort(key=lambda x: x["failure_rate_pct"], reverse=True)
    
    # SLA Risk Assessment and Anomaly Signals
    if len(df) > 0:
        batch_id = df["Batch_ID"].iloc[0] if "Batch_ID" in df.columns else "UNKNOWN_BATCH"
        batch_volume = len(df)
        breach_count = len(df[df["SLA_Breach_Flag"] == "Y"]) if "SLA_Breach_Flag" in df.columns else 0
        sla_breach_rate = breach_count / batch_volume if batch_volume > 0 else 0
        
        retry_count = int(df["Retry_Count"].sum()) if "Retry_Count" in df.columns else 0
        pipeline_gap_count = int((df["Pipeline_Gap_Flag"] == "Y").sum()) if "Pipeline_Gap_Flag" in df.columns else 0
        
        rolling_vol = float(df["Rolling_7D_Avg_Volume"].iloc[0]) if "Rolling_7D_Avg_Volume" in df.columns else 0.0
        vol_ratio = float(df["Volume_Vs_Trend_Ratio"].iloc[0]) if "Volume_Vs_Trend_Ratio" in df.columns else 0.0
        rolling_sla = float(df["Rolling_7D_Avg_SLA_Breach_Rate"].iloc[0]) if "Rolling_7D_Avg_SLA_Breach_Rate" in df.columns else 0.0
        sla_diff = float(df["SLA_Breach_Rate_Vs_Trend_Diff"].iloc[0]) if "SLA_Breach_Rate_Vs_Trend_Diff" in df.columns else 0.0
    else:
        batch_id = "UNKNOWN_BATCH"
        batch_volume = 0
        breach_count = 0
        sla_breach_rate = 0
        retry_count = 0
        pipeline_gap_count = 0
        rolling_vol = 0.0
        vol_ratio = 0.0
        rolling_sla = 0.0
        sla_diff = 0.0
        
    sla_breach_rate_pct = sla_breach_rate * 100
    
    if sla_breach_rate_pct < 5:
        sla_risk = "LOW"
    elif sla_breach_rate_pct < 15:
        sla_risk = "MEDIUM"
    elif sla_breach_rate_pct < 30:
        sla_risk = "HIGH"
    else:
        sla_risk = "CRITICAL"
        
    anomaly_signals = {
        "Batch_ID": batch_id,
        "Batch_Volume": batch_volume,
        "Rolling_7D_Avg_Volume": rolling_vol,
        "Volume_Vs_Trend_Ratio": vol_ratio,
        "Batch_SLA_Breach_Rate": sla_breach_rate,
        "Rolling_7D_Avg_SLA_Breach_Rate": rolling_sla,
        "SLA_Breach_Rate_Vs_Trend_Diff": sla_diff,
        "Retry_Count": retry_count,
        "Pipeline_Gap_Flag": pipeline_gap_count,
        "quality_score": overall_score,
        "critical_issue_count": critical_failures
    }
    
    quality_report = {
        "run_timestamp": datetime.datetime.now().isoformat(),
        "records_scanned": batch_volume,
        "dimension_scores": dim_scores,
        "overall_quality_score": overall_score,
        "overall_risk_level": overall_risk_level,
        "critical_issue_count": critical_failures,
        "all_rule_results": rule_results,
        "top_failed_rules": top_failed_rules[:5],
        "anomaly_signals": anomaly_signals
    }
    
    batch_sla_risk = {
        "batch_id": batch_id,
        "batch_record_count": batch_volume,
        "breach_count": breach_count,
        "sla_breach_rate": sla_breach_rate,
        "retry_count_summary": retry_count,
        "pipeline_gap_flag_count": pipeline_gap_count,
        "batch_sla_risk_level": sla_risk
    }
    
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f"{output_dir}/quality_report.json", "w") as f:
        json.dump(quality_report, f, indent=4)
        
    with open(f"{output_dir}/batch_sla_risk.json", "w") as f:
        json.dump(batch_sla_risk, f, indent=4)
        
    return quality_report, batch_sla_risk
