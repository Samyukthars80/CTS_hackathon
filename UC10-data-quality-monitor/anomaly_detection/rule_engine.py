from __future__ import annotations

import json
import os
from pathlib import Path
import pandas as pd

from anomaly_detection.rule_config import RULES_CONFIG
from anomaly_detection.medical_rules import evaluate_medical_rules
from anomaly_detection.pharmacy_rules import evaluate_pharmacy_rules
from anomaly_detection.authorization_rules import evaluate_authorization_rules


def build_record_level_summary(anomalies_list: list[dict], all_record_ids: list[str]) -> pd.DataFrame:
    # Build a lookup/dict to hold counts
    summary = {rid: {"Total_Rule_Violations": 0, "High": 0, "Medium": 0, "Low": 0} for rid in all_record_ids}
    
    for anom in anomalies_list:
        rid = anom["record_identifier"]
        if rid not in summary:
            summary[rid] = {"Total_Rule_Violations": 0, "High": 0, "Medium": 0, "Low": 0}
        
        summary[rid]["Total_Rule_Violations"] += 1
        sev = str(anom["severity"]).capitalize()
        if sev == "High":
            summary[rid]["High"] += 1
        elif sev == "Medium":
            summary[rid]["Medium"] += 1
        elif sev == "Low":
            summary[rid]["Low"] += 1
            
    # Convert to DataFrame
    rows = []
    for rid, stats in summary.items():
        rows.append({
            "Record_ID": rid,
            "Total_Rule_Violations": stats["Total_Rule_Violations"],
            "High": stats["High"],
            "Medium": stats["Medium"],
            "Low": stats["Low"]
        })
        
    return pd.DataFrame(rows)


def build_anomaly_stats(
    anomalies_list: list[dict],
    med_total: int,
    pharm_total: int,
    auth_total: int
) -> dict:
    total_records = med_total + pharm_total + auth_total
    total_violations = len(anomalies_list)
    flagged_records = len(set(a["record_identifier"] for a in anomalies_list))
    
    violations_by_rule = {}
    violations_by_category = {}
    violations_by_severity = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    violations_by_type = {"MEDICAL_CLAIM": 0, "PHARMACY_CLAIM": 0, "PRIOR_AUTH": 0}
    
    for anom in anomalies_list:
        rule_id = anom["rule_id"]
        category = anom["anomaly_category"]
        severity = anom["severity"].upper()
        rec_type = anom["record_type"]
        
        violations_by_rule[rule_id] = violations_by_rule.get(rule_id, 0) + 1
        violations_by_category[category] = violations_by_category.get(category, 0) + 1
        violations_by_severity[severity] = violations_by_severity.get(severity, 0) + 1
        violations_by_type[rec_type] = violations_by_type.get(rec_type, 0) + 1
        
    stats = {
        "global_summary": {
            "total_records_analyzed": total_records,
            "total_records_flagged": flagged_records,
            "total_rule_violations": total_violations,
            "overall_anomaly_rate_pct": (flagged_records / total_records) * 100 if total_records > 0 else 0.0
        },
        "by_record_type": {
            "MEDICAL_CLAIM": {
                "analyzed": med_total,
                "flagged": len(set(a["record_identifier"] for a in anomalies_list if a["record_type"] == "MEDICAL_CLAIM")),
                "violations": violations_by_type["MEDICAL_CLAIM"]
            },
            "PHARMACY_CLAIM": {
                "analyzed": pharm_total,
                "flagged": len(set(a["record_identifier"] for a in anomalies_list if a["record_type"] == "PHARMACY_CLAIM")),
                "violations": violations_by_type["PHARMACY_CLAIM"]
            },
            "PRIOR_AUTH": {
                "analyzed": auth_total,
                "flagged": len(set(a["record_identifier"] for a in anomalies_list if a["record_type"] == "PRIOR_AUTH")),
                "violations": violations_by_type["PRIOR_AUTH"]
            }
        },
        "violations_by_rule": violations_by_rule,
        "violations_by_category": violations_by_category,
        "violations_by_severity": violations_by_severity
    }
    return stats


def main():
    # Resolve directories
    anomaly_dir = Path(__file__).resolve().parent
    workspace_dir = anomaly_dir.parent
    outputs_dir = workspace_dir / "outputs"
    anomaly_outputs_dir = anomaly_dir / "outputs"
    anomaly_outputs_dir.mkdir(parents=True, exist_ok=True)
    
    # Load cleaned datasets
    print("Loading cleaned datasets for anomaly evaluation...")
    med_df = pd.read_csv(outputs_dir / "medical_claim_cleaned.csv")
    pharm_df = pd.read_csv(outputs_dir / "pharmacy_claim_cleaned.csv")
    auth_df = pd.read_csv(outputs_dir / "authorization_cleaned.csv")
    
    med_cnt = len(med_df)
    pharm_cnt = len(pharm_df)
    auth_cnt = len(auth_df)
    
    # Collect all Record_IDs to generate an exhaustive summary
    all_record_ids = (
        med_df["Record_ID"].astype(str).tolist() +
        pharm_df["Record_ID"].astype(str).tolist() +
        auth_df["Record_ID"].astype(str).tolist()
    )
    
    # Run evaluations
    print("Evaluating Medical Claims rules...")
    med_anoms = evaluate_medical_rules(med_df, RULES_CONFIG)
    
    print("Evaluating Pharmacy Claims rules...")
    pharm_anoms = evaluate_pharmacy_rules(pharm_df, RULES_CONFIG)
    
    print("Evaluating Prior Authorizations rules...")
    auth_anoms = evaluate_authorization_rules(auth_df, RULES_CONFIG)
    
    # Create DataFrames
    med_anoms_df = pd.DataFrame(med_anoms)
    pharm_anoms_df = pd.DataFrame(pharm_anoms)
    auth_anoms_df = pd.DataFrame(auth_anoms)
    
    combined_anoms = med_anoms + pharm_anoms + auth_anoms
    combined_anoms_df = pd.DataFrame(combined_anoms)
    
    # Save separate detailed reports
    med_anoms_df.to_csv(anomaly_outputs_dir / "medical_rule_anomalies.csv", index=False)
    pharm_anoms_df.to_csv(anomaly_outputs_dir / "pharmacy_rule_anomalies.csv", index=False)
    auth_anoms_df.to_csv(anomaly_outputs_dir / "authorization_rule_anomalies.csv", index=False)
    combined_anoms_df.to_csv(anomaly_outputs_dir / "combined_rule_anomalies.csv", index=False)
    
    # Save record-level summary
    print("Generating record-level anomaly summary...")
    rec_summary_df = build_record_level_summary(combined_anoms, all_record_ids)
    rec_summary_df.to_csv(anomaly_outputs_dir / "record_level_summary.csv", index=False)
    
    # Save stats JSON
    stats = build_anomaly_stats(combined_anoms, med_cnt, pharm_cnt, auth_cnt)
    with open(anomaly_outputs_dir / "anomaly_statistics.json", "w") as f:
        json.dump(stats, f, indent=4)
        
    print(f"\nRule-based anomaly detection completed successfully!")
    print(f"Detailed anomaly reports written to: {anomaly_outputs_dir}")
    print(f"Total rule violations logged: {len(combined_anoms)}")
    print(f"Flagged records: {stats['global_summary']['total_records_flagged']} out of {stats['global_summary']['total_records_analyzed']}")


if __name__ == "__main__":
    main()
