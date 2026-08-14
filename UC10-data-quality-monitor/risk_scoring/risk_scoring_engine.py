from __future__ import annotations

import pandas as pd
import numpy as np
from risk_scoring.config import (
    CONSENSUS_PATH, RULE_SUMMARY_PATH, RULE_DETAILED_PATH,
    DATASETS, ML_PREDICTIONS, ML_ANOMALIES,
    WEIGHT_RULE, WEIGHT_STATISTICAL, WEIGHT_ML, WEIGHT_CONSENSUS,
    NUMERICAL_COLS
)


def calculate_split_statistics() -> dict[str, dict[str, tuple[float, float]]]:
    """Calculate mean and standard deviation for each numerical column strictly on the training split (80%)."""
    stats = {}
    for rtype, path in DATASETS.items():
        df = pd.read_csv(path)
        # Sort chronologically by Submission_Date
        df = df.sort_values(by="Submission_Date").reset_index(drop=True)
        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx]
        
        stats[rtype] = {}
        for col in NUMERICAL_COLS[rtype]:
            if col in train_df.columns:
                col_vals = train_df[col].dropna()
                mean_val = float(col_vals.mean()) if len(col_vals) > 0 else 0.0
                std_val = float(col_vals.std()) if len(col_vals) > 1 else 0.0
                stats[rtype][col] = (mean_val, std_val)
    return stats


def calculate_rule_risk(rid: str, rule_sum_df: pd.DataFrame, rule_detailed_df: pd.DataFrame) -> tuple[float, int, str]:
    """Compute rule risk score (0-100) and return violations count and highest severity."""
    # Find summary record
    r_sum = rule_sum_df[rule_sum_df["Record_ID"] == rid]
    if r_sum.empty:
        return 0.0, 0, "None"
        
    total_violations = int(r_sum["Total_Rule_Violations"].iloc[0])
    
    # Detailed violations to check severity
    r_details = rule_detailed_df[rule_detailed_df["Record_ID"] == rid]
    if r_details.empty:
        return 0.0, 0, "None"
        
    # Points assignment
    points = 0.0
    severities = []
    for _, row in r_details.iterrows():
        sev = str(row["Severity"]).strip().capitalize()
        severities.append(sev)
        if sev == "High":
            points += 10.0
        elif sev == "Medium":
            points += 5.0
        else: # Low or others
            points += 2.0
            
    # Cap total points at 20.0
    rule_score = min(100.0, 100.0 * (points / 20.0))
    
    # Determine highest severity
    highest_sev = "None"
    if "High" in severities:
        highest_sev = "High"
    elif "Medium" in severities:
        highest_sev = "Medium"
    elif "Low" in severities:
        highest_sev = "Low"
        
    return rule_score, total_violations, highest_sev


def calculate_statistical_risk(
    row: pd.Series, rtype: str, train_stats: dict[str, dict[str, tuple[float, float]]]
) -> tuple[float, int, str]:
    """Calculate statistical risk score (0-100) based on Z-score magnitudes, returns score, count, and outlier cols."""
    points = 0.0
    outlier_cols = []
    
    col_stats = train_stats[rtype]
    for col in col_stats.keys():
        val = row.get(col)
        if pd.notna(val):
            mean_val, std_val = col_stats[col]
            if std_val > 0.0:
                z = abs(val - mean_val) / std_val
                # Magnitude check
                if z > 3.0:
                    points += 6.0
                    outlier_cols.append(col)
                elif z > 1.5:
                    points += 3.0
                    outlier_cols.append(col)
                    
    # Cap total points at 12.0
    stat_score = min(100.0, 100.0 * (points / 12.0))
    stat_flag = 1 if len(outlier_cols) > 0 else 0
    
    return stat_score, stat_flag, "; ".join(outlier_cols)


def build_record_level_risk_scores() -> pd.DataFrame:
    print("Executing consolidated risk scoring engine...")
    
    # 1. Load inputs
    df_consensus = pd.read_csv(CONSENSUS_PATH)
    df_consensus["Record_ID"] = df_consensus["Record_ID"].astype(str)
    df_consensus["Record_Type"] = df_consensus["Record_Type"].astype(str)
    
    # Load rules outputs
    df_rule_sum = pd.read_csv(RULE_SUMMARY_PATH)
    df_rule_sum["Record_ID"] = df_rule_sum["Record_ID"].astype(str)
    
    df_rule_details = pd.read_csv(RULE_DETAILED_PATH)
    df_rule_details["Record_ID"] = df_rule_details["Record_ID"].astype(str)
    
    # Group rule IDs and explanations
    rule_ids_map = {}
    rule_exps_map = {}
    for _, row in df_rule_details.iterrows():
        rid = str(row["Record_ID"])
        if rid not in rule_ids_map:
            rule_ids_map[rid] = []
            rule_exps_map[rid] = []
        rule_ids_map[rid].append(str(row["Rule_ID"]))
        rule_exps_map[rid].append(str(row["Explanation"]))
        
    # Load ML predictions
    ml_pct_map = {}
    ml_flag_map = {}
    ml_score_map = {}
    ml_explanations = {}
    
    for rtype, path in ML_PREDICTIONS.items():
        pdf = pd.read_csv(path)
        for _, row in pdf.iterrows():
            rid = str(row["Record_ID"])
            ml_pct_map[rid] = float(row["Anomaly_Percentile"])
            ml_flag_map[rid] = int(row["ML_Anomaly_Flag"])
            ml_score_map[rid] = float(row["Isolation_Forest_Score"])
            
    for rtype, path in ML_ANOMALIES.items():
        adf = pd.read_csv(path)
        for _, row in adf.iterrows():
            rid = str(row["Record_ID"])
            ml_explanations[rid] = str(row["Potential_Contributing_Factors"])
            
    # Load clean data for NPIs and computing statistical deviations
    train_stats = calculate_split_statistics()
    npi_map = {}
    df_clean_map = {}
    
    for rtype, path in DATASETS.items():
        df_clean = pd.read_csv(path)
        df_clean["Record_ID"] = df_clean["Record_ID"].astype(str)
        df_clean_map[rtype] = df_clean.set_index("Record_ID")
        
        # NPI mapping
        for _, row in df_clean.iterrows():
            rid = str(row["Record_ID"])
            npi = row.get("Provider_NPI")
            npi_map[rid] = str(int(npi)) if pd.notna(npi) else "UNKNOWN_PROVIDER"
            
    # 2. Iterate and score
    scored_rows = []
    for _, row in df_consensus.iterrows():
        rid = row["Record_ID"]
        rtype = row["Record_Type"]
        
        # A. Rule Component
        rule_risk_score, num_rule_violations, highest_severity = calculate_rule_risk(rid, df_rule_sum, df_rule_details)
        rule_ids = "; ".join(rule_ids_map.get(rid, []))
        rule_exps = "; ".join(rule_exps_map.get(rid, []))
        
        # B. Statistical Component
        clean_row = df_clean_map[rtype].loc[rid]
        stat_risk_score, stat_flag, stat_cols = calculate_statistical_risk(clean_row, rtype, train_stats)
        
        # C. ML Component
        ml_flag = ml_flag_map.get(rid, 0)
        ml_score = ml_score_map.get(rid, 0.0)
        ml_pct = ml_pct_map.get(rid, 0.0)
        ml_risk_score = ml_pct # Percentile rank directly represents the normalized ML risk score (0-100)
        ml_exp = ml_explanations.get(rid, "")
        
        # D. Consensus Component
        consensus_level = int(row["Consensus_Level"])
        # Map consensus level to score
        if consensus_level == 3:
            consensus_score = 100.0
        elif consensus_level == 2:
            consensus_score = 60.0
        elif consensus_level == 1:
            consensus_score = 20.0
        else:
            consensus_score = 0.0
            
        # Weighted Overall Score
        overall_risk_score = (
            WEIGHT_RULE * rule_risk_score +
            WEIGHT_STATISTICAL * stat_risk_score +
            WEIGHT_ML * ml_risk_score +
            WEIGHT_CONSENSUS * consensus_score
        )
        overall_risk_score = round(overall_risk_score, 2)
        
        # Risk Categories
        if overall_risk_score < 25.0:
            risk_category = "LOW"
            recommended_priority = "P4 - Low"
            priority_num = 4
        elif overall_risk_score < 50.0:
            risk_category = "MEDIUM"
            recommended_priority = "P3 - Medium"
            priority_num = 3
        elif overall_risk_score < 75.0:
            risk_category = "HIGH"
            recommended_priority = "P2 - High"
            priority_num = 2
        else:
            risk_category = "CRITICAL"
            recommended_priority = "P1 - Critical"
            priority_num = 1
            
        # Risk Score Breakdown format
        w_rule = rule_risk_score * WEIGHT_RULE
        w_stat = stat_risk_score * WEIGHT_STATISTICAL
        w_ml = ml_risk_score * WEIGHT_ML
        w_cons = consensus_score * WEIGHT_CONSENSUS
        breakdown = f"Rule: {w_rule:.1f}/40; Statistical: {w_stat:.1f}/25; ML: {w_ml:.1f}/25; Consensus: {w_cons:.1f}/10"
        
        # Determine Primary & Secondary risk reasons (based on weighted score contribution)
        components = [
            ("Rule Engine Violations", w_rule),
            ("Statistical Outliers", w_stat),
            ("Isolation Forest ML Anomaly", w_ml),
            ("Consensus Agreement", w_cons)
        ]
        components_sorted = sorted(components, key=lambda x: x[1], reverse=True)
        primary_risk_reason = f"{components_sorted[0][0]} (weighted contribution: {components_sorted[0][1]:.1f})"
        secondary_risk_reason = f"{components_sorted[1][0]} (weighted contribution: {components_sorted[1][1]:.1f})"
        
        # Plain-language explanation
        reasons = []
        if rule_risk_score > 0:
            reasons.append("it violated explicit business rule validations")
        if stat_risk_score > 0:
            reasons.append("it was identified as statistically unusual compared to normal claims distributions")
        if ml_risk_score >= 95.0:
            reasons.append("the machine learning model flagged it as highly anomalous in multi-dimensional space")
            
        explanation = f"This {rtype.lower().replace('_', ' ')} was classified as {risk_category} risk ({overall_risk_score:.1f}/100)"
        if reasons:
            explanation += f" because {', and '.join(reasons)}."
            if num_rule_violations > 0:
                explanation += f" It violated {num_rule_violations} rules (IDs: {rule_ids}) with explanations: {rule_exps}."
            if stat_flag:
                explanation += f" Statistically anomalous columns include: {stat_cols}."
            if ml_flag:
                explanation += f" ML contributing factors: {ml_exp}."
            explanation += " This record requires investigation to verify claims integrity."
        else:
            explanation += " because it did not exhibit significant anomalies across the rule engine, statistical benchmarks, or machine learning models."
            
        scored_rows.append({
            "Record_ID": rid,
            "Record_Type": rtype,
            "Provider_NPI": npi_map.get(rid, "UNKNOWN_PROVIDER"),
            "overall_risk_score": overall_risk_score,
            "risk_category": risk_category,
            "rule_risk_score": round(rule_risk_score, 2),
            "statistical_risk_score": round(stat_risk_score, 2),
            "ml_risk_score": round(ml_risk_score, 2),
            "consensus_score": round(consensus_score, 2),
            "number_of_rule_violations": num_rule_violations,
            "statistical_flag": stat_flag,
            "ml_flag": ml_flag,
            "highest_rule_severity": highest_severity,
            "primary_risk_reason": primary_risk_reason,
            "secondary_risk_reason": secondary_risk_reason,
            "recommended_priority": recommended_priority,
            "priority_num": priority_num,
            "risk_score_breakdown": breakdown,
            "explanation": explanation
        })
        
    return pd.DataFrame(scored_rows)
