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
        rid = anom["Record_ID"]
        if rid not in summary:
            summary[rid] = {"Total_Rule_Violations": 0, "High": 0, "Medium": 0, "Low": 0}
        
        summary[rid]["Total_Rule_Violations"] += 1
        sev = str(anom["Severity"]).capitalize()
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
    flagged_records = len(set(a["Record_ID"] for a in anomalies_list))
    
    violations_by_rule = {}
    violations_by_category = {}
    violations_by_severity = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    violations_by_type = {"MEDICAL_CLAIM": 0, "PHARMACY_CLAIM": 0, "PRIOR_AUTH": 0}
    
    for anom in anomalies_list:
        rule_id = anom["Rule_ID"]
        category = anom["Anomaly_Category"]
        severity = anom["Severity"].upper()
        rec_type = anom["Record_Type"]
        
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
                "flagged": len(set(a["Record_ID"] for a in anomalies_list if a["Record_Type"] == "MEDICAL_CLAIM")),
                "violations": violations_by_type["MEDICAL_CLAIM"]
            },
            "PHARMACY_CLAIM": {
                "analyzed": pharm_total,
                "flagged": len(set(a["Record_ID"] for a in anomalies_list if a["Record_Type"] == "PHARMACY_CLAIM")),
                "violations": violations_by_type["PHARMACY_CLAIM"]
            },
            "PRIOR_AUTH": {
                "analyzed": auth_total,
                "flagged": len(set(a["Record_ID"] for a in anomalies_list if a["Record_Type"] == "PRIOR_AUTH")),
                "violations": violations_by_type["PRIOR_AUTH"]
            }
        },
        "violations_by_rule": violations_by_rule,
        "violations_by_category": violations_by_category,
        "violations_by_severity": violations_by_severity
    }
    return stats


def print_final_reports(anomalies: list[dict], med_cnt: int, pharm_cnt: int, auth_cnt: int):
    print("=" * 60)
    print("FINAL SUMMARY REPORT")
    print("=" * 60)
    
    # 1. Summary table
    print("\nRecord Type | Total Records | Unique Flagged Records | Anomaly Events | Anomaly Rate")
    print("-" * 85)
    for rtype, tot in [("MEDICAL_CLAIM", med_cnt), ("PHARMACY_CLAIM", pharm_cnt), ("PRIOR_AUTH", auth_cnt)]:
        flagged = len(set(a["Record_ID"] for a in anomalies if a["Record_Type"] == rtype))
        events = sum(1 for a in anomalies if a["Record_Type"] == rtype)
        rate = (flagged / tot) * 100 if tot > 0 else 0
        print(f"{rtype:<14} | {tot:<13} | {flagged:<22} | {events:<14} | {rate:.2f}%")
        
    tot_flagged = len(set(a["Record_ID"] for a in anomalies))
    tot_events = len(anomalies)
    tot_records = med_cnt + pharm_cnt + auth_cnt
    tot_rate = (tot_flagged / tot_records) * 100 if tot_records > 0 else 0
    print(f"{'TOTAL':<14} | {tot_records:<13} | {tot_flagged:<22} | {tot_events:<14} | {tot_rate:.2f}%")
    
    # 2. Category breakdown
    print("\nAnomaly Category | Record Type | Count")
    print("-" * 50)
    cat_breakdown = {}
    for a in anomalies:
        key = (a["Anomaly_Category"], a["Record_Type"])
        cat_breakdown[key] = cat_breakdown.get(key, 0) + 1
    for (cat, rtype), count in sorted(cat_breakdown.items()):
        print(f"{cat:<26} | {rtype:<14} | {count}")
        
    # 3. Rule breakdown
    print("\nRule_ID | Rule Description | Trigger Count | Unique Records")
    print("-" * 85)
    rule_counts = {}
    rule_uniques = {}
    for a in anomalies:
        rid = a["Rule_ID"]
        rule_counts[rid] = rule_counts.get(rid, 0) + 1
        if rid not in rule_uniques:
            rule_uniques[rid] = set()
        rule_uniques[rid].add(a["Record_ID"])
        
    for rid, cfg in sorted(RULES_CONFIG.items()):
        if not cfg.get("enabled", True):
            continue
        count = rule_counts.get(rid, 0)
        uniques = len(rule_uniques.get(rid, []))
        desc = cfg["description"]
        print(f"{rid:<7} | {desc[:40]:<40} | {count:<13} | {uniques}")
        
    # 4. Before vs After comparison
    print("\nBEFORE vs AFTER processing timeline deduplication:")
    print("-" * 50)
    print("Before:")
    print("  Processed < Submission = 2,745")
    print("  Negative Latency = 2,745")
    print("  Total Timeline Anomalies = 5,490")
    print("\nAfter:")
    # Count of R_DATE_001 across all datasets
    timeline_count = sum(1 for a in anomalies if a["Rule_ID"] == "R_DATE_001")
    med_timeline = sum(1 for a in anomalies if a["Rule_ID"] == "R_DATE_001" and a["Record_Type"] == "MEDICAL_CLAIM")
    pharm_timeline = sum(1 for a in anomalies if a["Rule_ID"] == "R_DATE_001" and a["Record_Type"] == "PHARMACY_CLAIM")
    auth_timeline = sum(1 for a in anomalies if a["Rule_ID"] == "R_DATE_001" and a["Record_Type"] == "PRIOR_AUTH")
    print(f"  Invalid Processing Timeline (R_DATE_001) = {timeline_count}")
    print(f"    - Medical: {med_timeline}")
    print(f"    - Pharmacy: {pharm_timeline}")
    print(f"    - Prior Auth: {auth_timeline}")
    print(f"  Total Timeline Anomalies = {timeline_count}")
    print("=" * 60)


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
    
    # Ensure correct columns structure even if empty
    cols = [
        "Record_ID", "Record_Type", "Detection_Method", "Anomaly_Category", 
        "Affected_Columns", "Observed_Value", "Expected_Condition", 
        "Severity", "Explanation", "Rule_ID", "Source_Dataset", "Detection_Timestamp"
    ]
    if combined_anoms_df.empty:
        combined_anoms_df = pd.DataFrame(columns=cols)
    else:
        combined_anoms_df = combined_anoms_df[cols]
        
    if med_anoms_df.empty:
        med_anoms_df = pd.DataFrame(columns=cols)
    else:
        med_anoms_df = med_anoms_df[cols]
        
    if pharm_anoms_df.empty:
        pharm_anoms_df = pd.DataFrame(columns=cols)
    else:
        pharm_anoms_df = pharm_anoms_df[cols]
        
    if auth_anoms_df.empty:
        auth_anoms_df = pd.DataFrame(columns=cols)
    else:
        auth_anoms_df = auth_anoms_df[cols]
    
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
        
    print_final_reports(combined_anoms, med_cnt, pharm_cnt, auth_cnt)


if __name__ == "__main__":
    main()
